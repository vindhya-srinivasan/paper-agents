"""Durable conversation state.

STATE_BACKEND=firestore (Cloud Run): each session is one Firestore document
    sessions/{session_id} = {created, updated, turns: [{role, content, ts, meta}]}
so history survives instance recycling, scale-out (any instance can serve any session) and redeploys.

STATE_BACKEND=memory (local dev / tests only): a process dict. Not durable, and never claimed as such.
"""
from __future__ import annotations

import asyncio
import time
from typing import Protocol

from src import config

MAX_STORED_TURNS = 40


class Store(Protocol):
    name: str
    async def load(self, session_id: str) -> list[dict]: ...
    async def append(self, session_id: str, turns: list[dict]) -> None: ...
    async def clear(self, session_id: str) -> None: ...


class MemoryStore:
    name = "memory (local dev only, not durable)"

    def __init__(self):
        self._d: dict[str, list[dict]] = {}

    async def load(self, session_id: str) -> list[dict]:
        return list(self._d.get(session_id, []))

    async def append(self, session_id: str, turns: list[dict]) -> None:
        self._d.setdefault(session_id, []).extend(turns)
        self._d[session_id] = self._d[session_id][-MAX_STORED_TURNS:]

    async def clear(self, session_id: str) -> None:
        self._d.pop(session_id, None)


class FirestoreStore:
    def __init__(self):
        from google.cloud import firestore

        self._fs = firestore
        self.client = firestore.Client(project=config.FIRESTORE_PROJECT_ID or None)
        self.col = self.client.collection(config.FIRESTORE_COLLECTION)
        self.name = f"firestore (project={self.client.project}, collection={config.FIRESTORE_COLLECTION})"

    async def load(self, session_id: str) -> list[dict]:
        snap = await asyncio.to_thread(self.col.document(session_id).get)
        return (snap.to_dict() or {}).get("turns", []) if snap.exists else []

    async def append(self, session_id: str, turns: list[dict]) -> None:
        ref = self.col.document(session_id)

        def _tx():
            snap = ref.get()
            existing = (snap.to_dict() or {}).get("turns", []) if snap.exists else []
            ref.set({"turns": (existing + turns)[-MAX_STORED_TURNS:], "updated": time.time(),
                     "created": (snap.to_dict() or {}).get("created", time.time()) if snap.exists else time.time()})
        await asyncio.to_thread(_tx)

    async def clear(self, session_id: str) -> None:
        await asyncio.to_thread(self.col.document(session_id).delete)


def make_store() -> Store:
    if config.STATE_BACKEND == "firestore":
        return FirestoreStore()
    return MemoryStore()

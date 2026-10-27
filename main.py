"""FastAPI entry point: SSE chat endpoint + static vanilla HTML/JS frontend.

Run locally:  uvicorn main:app --reload --port 8080
On Cloud Run the container CMD binds uvicorn to $PORT.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config
from src.llm import router
from src.state_store import MemoryStore, make_store
from src.tools import tavily_mcp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

STATIC = Path(__file__).parent / "static"
INSTANCE_ID = uuid.uuid4().hex[:8]          # changes whenever Cloud Run starts a new instance
STARTED_AT = time.time()
STATUS: dict = {}


def _warm_rag() -> str:
    from src.rag import retriever
    from src.rag.embeddings import get_embeddings

    get_embeddings().embed_query("warm up")
    n = retriever._collection().count()
    return f"ok ({n} chunks)"


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    try:
        STATUS["rag"] = await asyncio.to_thread(_warm_rag)
    except Exception as e:  # noqa: BLE001 - app still starts; research tools report the error per call
        STATUS["rag"] = f"unavailable: {type(e).__name__}: {e}"
    STATUS["tavily_mcp"] = await tavily_mcp.load()
    try:
        app.state.store = make_store()
    except Exception as e:  # noqa: BLE001
        log.error("State store init failed, falling back to memory: %s", e)
        app.state.store = MemoryStore()
    STATUS["state_store"] = app.state.store.name
    STATUS["providers"] = {p: (router.model_name(p) if router.provider_available(p) else "not configured")
                           for p in router.PROVIDERS}
    STATUS["startup_s"] = round(time.time() - t0, 2)
    log.info("Startup complete: %s", STATUS)
    yield


app = FastAPI(title="CSCI 599 A2 - Multi-agent RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    debug_fail_agent: str | None = Field(default=None, description="Test hook: force this worker to fail.")


class SessionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


# /health, not /healthz: Cloud Run's Google front end reserves /healthz and answers it with its own 404.
@app.get("/health")
@app.get("/healthz", include_in_schema=False)   # local-dev alias only; unreachable on Cloud Run
async def health():
    return {"status": "ok", "instance_id": INSTANCE_ID, "revision": os.getenv("K_REVISION", "local"),
            "uptime_s": round(time.time() - STARTED_AT, 1), **STATUS}


@app.get("/api/history")
async def history(session_id: str):
    SessionRequest(session_id=session_id)
    return {"session_id": session_id, "instance_id": INSTANCE_ID,
            "turns": await app.state.store.load(session_id)}


@app.post("/api/reset")
async def reset(req: SessionRequest):
    await app.state.store.clear(req.session_id)
    return {"ok": True}


async def run_turn(req: ChatRequest):
    from src.graph.graph import GRAPH

    t0 = time.perf_counter()
    store = app.state.store
    yield sse("trace", {"kind": "request", "node": "api", "instance_id": INSTANCE_ID,
                        "revision": os.getenv("K_REVISION", "local"), "state_store": store.name, "t": time.time()})
    try:
        hist = await store.load(req.session_id)
    except Exception as e:  # noqa: BLE001
        hist = []
        yield sse("error", {"message": f"Could not load history: {e}", "node": "api"})
    yield sse("trace", {"kind": "history_loaded", "node": "api", "turns": len(hist), "t": time.time()})

    state = {"session_id": req.session_id, "query": req.query.strip(),
             "history": [{"role": t["role"], "content": t["content"]} for t in hist],
             "force_fail": req.debug_fail_agent, "completed": [], "failed": [], "notes": [], "citations": [],
             "steps": 0, "plan": None}
    answer, ttft_ms = [], None
    llm_ms = retrieval_ms = tool_ms = 0.0
    usage = {"input_tokens": 0, "output_tokens": 0}
    cost = 0.0
    providers: dict[str, str] = {}
    plan = None
    try:
        async for chunk in GRAPH.astream(state, stream_mode="custom"):
            ev, data = chunk["event"], chunk["data"]
            if ev == "token":
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                answer.append(data["text"])
            elif ev == "metric":
                if data["kind"] == "llm":
                    llm_ms += data["ms"]
                    usage["input_tokens"] += data["input_tokens"]
                    usage["output_tokens"] += data["output_tokens"]
                    cost += data["cost_usd"]
                    providers[data["role"]] = f"{data['provider']}:{data['model']}"
                elif data["tool_kind"] in ("rag_retrieval", "paper_read"):
                    retrieval_ms += data["ms"]
                else:
                    tool_ms += data["ms"]
                continue          # metrics are summarized in the done event
            elif ev == "trace" and data.get("kind") == "plan":
                plan = data.get("plan")
            yield sse(ev, data)
    except Exception as e:  # noqa: BLE001
        log.exception("graph failed")
        yield sse("error", {"message": f"{type(e).__name__}: {e}", "node": "graph"})

    text = "".join(answer)
    if text:
        try:
            now = time.time()
            await store.append(req.session_id, [
                {"role": "user", "content": req.query.strip(), "ts": now},
                {"role": "assistant", "content": text, "ts": now, "meta": {"plan": plan, "providers": providers}},
            ])
        except Exception as e:  # noqa: BLE001
            yield sse("error", {"message": f"Could not save history: {e}", "node": "api"})
    total_ms = (time.perf_counter() - t0) * 1000
    yield sse("done", {
        "instance_id": INSTANCE_ID, "plan": plan, "providers": providers,
        "timings_ms": {"total": round(total_ms, 1), "ttft": ttft_ms and round(ttft_ms, 1),
                       "llm": round(llm_ms, 1), "retrieval": round(retrieval_ms, 1), "tools": round(tool_ms, 1)},
        "usage": usage, "cost_usd": round(cost, 6),
    })


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.debug_fail_agent and req.debug_fail_agent not in ("research", "data", "summary"):
        raise HTTPException(400, "debug_fail_agent must be research, data or summary")
    return StreamingResponse(run_turn(req), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))

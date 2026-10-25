"""LangGraph state schema (documented in docs/architecture.md)."""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

AgentName = Literal["research", "data", "summary"]


class GraphState(TypedDict, total=False):
    # Inputs (set by the API per turn)
    session_id: str
    query: str
    history: list[dict]                 # prior turns [{role, content}] loaded from the durable store
    force_fail: str | None              # test hook: name of a worker to fail on purpose

    # Supervisor fields - routing reads ONLY these, so every transition is inspectable
    plan: list[str] | None              # ordered workers chosen by the supervisor LLM
    complexity: str                     # "simple" | "complex"  (feeds the LLM router)
    plan_reason: str
    next: str                           # "research" | "data" | "summary" | "respond"
    steps: int                          # worker invocations so far (guard against loops)

    # Worker outputs (append-only reducers so workers never overwrite each other)
    completed: Annotated[list[str], operator.add]
    failed: Annotated[list[dict], operator.add]      # [{agent, error}]
    notes: Annotated[list[dict], operator.add]       # [{agent, text}]
    citations: Annotated[list[dict], operator.add]   # corpus chunks + web results

    # Output
    answer: str

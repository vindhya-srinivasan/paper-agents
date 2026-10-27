"""Supervisor-pattern LangGraph.

    START -> supervisor --(state.next)--> research | data | summary --> supervisor ... -> respond -> END

* supervisor: first visit asks the routed LLM for a structured Plan; every visit then picks the next
  worker from state fields only (plan, completed, failed, steps) and writes the reason to the trace.
* workers: specialized agents with their own prompt + tools. Any exception inside a worker is caught,
  recorded in state.failed, and control returns to the supervisor, so one failing worker never
  crashes the graph.
* respond: augments the generation prompt with numbered sources and streams the answer from the
  provider chosen by the LLM router.

Everything the UI shows comes from `emit()` events written to LangGraph's custom stream.
"""
from __future__ import annotations

import time
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src import config
from src.agents import tools as T
from src.agents.prompts import SUPERVISOR, WORKER_PROMPTS
from src.graph.state import GraphState
from src.llm import router
from src.rag.retriever import build_answer_chain, format_sources

WORKERS = ("research", "data", "summary")


def emit(event: str, data: dict) -> None:
    try:
        get_stream_writer()({"event": event, "data": data})
    except Exception:  # noqa: BLE001 - not streaming (e.g. ainvoke in a script): nothing to emit
        pass


def trace(kind: str, node: str, **data) -> None:
    emit("trace", {"kind": kind, "node": node, "t": time.time(), **data})


def metric_llm(role: str, decision: router.RouteDecision, msg, ms: float, ttft_ms: float | None = None, usage=None):
    usage = usage or router.usage_of(msg)
    emit("metric", {"kind": "llm", "role": role, "provider": decision.provider, "model": decision.model,
                    "ms": round(ms, 1), "ttft_ms": ttft_ms, **usage,
                    "cost_usd": router.cost_usd(decision.provider, usage)})


def on_fallback(node: str):
    return lambda d: trace("llm_route", node, **d.as_dict())


def history_text(state: GraphState, max_chars: int = 600) -> str:
    turns = state.get("history") or []
    return "\n".join(f"{t['role']}: {t['content'][:max_chars]}" for t in turns[-config.HISTORY_TURNS:])


# ---------------------------------------------------------------- supervisor
class Plan(BaseModel):
    agents: list[Literal["research", "data", "summary"]] = Field(description="Ordered workers to run, no repeats.")
    complexity: Literal["simple", "complex"]
    reason: str = Field(description="One sentence naming the features of the question that drove the plan.")


def heuristic_plan(query: str) -> Plan:
    """Used only if both providers fail to return a structured plan; keeps the graph answering."""
    q = query.lower()
    if any(w in q for w in ("how many", "count", "list all", "which year", "authors", "published in", "most ")):
        return Plan(agents=["data"], complexity="simple", reason="Heuristic fallback: metadata-style question.")
    if any(w in q for w in ("summarize", "summary of", "tl;dr", "overview of the")):
        return Plan(agents=["summary"], complexity="simple", reason="Heuristic fallback: summarization request.")
    return Plan(agents=["research"], complexity="simple", reason="Heuristic fallback: default to research.")


async def supervisor(state: GraphState) -> dict:
    update: dict = {}
    plan = state.get("plan")
    if plan is None:
        decision = router.decide("supervisor")
        trace("llm_route", "supervisor", **decision.as_dict())
        messages = [SystemMessage(SUPERVISOR),
                    HumanMessage(f"Conversation so far:\n{history_text(state) or '(new conversation)'}\n\n"
                                 f"User question: {state['query']}")]
        try:
            res, used, ms = await router.invoke(decision, messages, structured=Plan, on_fallback=on_fallback("supervisor"))
            metric_llm("supervisor", used, res["raw"], ms)
            p: Plan = res["parsed"]
        except Exception as e:  # noqa: BLE001
            p = heuristic_plan(state["query"])
            trace("error", "supervisor", message=f"Planner LLM failed ({type(e).__name__}); using heuristic plan.")
        agents = list(dict.fromkeys(a for a in p.agents if a in WORKERS))[:3]
        plan, update = agents, {"plan": agents, "complexity": p.complexity, "plan_reason": p.reason}
        trace("plan", "supervisor", plan=agents, complexity=p.complexity, reason=p.reason)

    completed = state.get("completed", [])
    failed = [f["agent"] for f in state.get("failed", [])]
    steps = state.get("steps", 0)
    remaining = [a for a in plan if a not in completed and a not in failed]
    reason_txt = update.get("plan_reason") or state.get("plan_reason", "")

    if steps >= config.MAX_WORKER_STEPS:
        nxt, why = "respond", f"step limit {config.MAX_WORKER_STEPS} reached"
    elif remaining:
        nxt = remaining[0]
        done = len(plan) - len(remaining)
        why = f"plan step {done + 1}/{len(plan)}: {nxt}. {reason_txt}"
        if failed:
            why = f"{', '.join(failed)} failed -> continuing with remaining plan. " + why
    else:
        nxt = "respond"
        if not plan:
            why = "no worker needed (greeting / meta question)"
        elif failed and not completed:
            why = f"all planned workers failed ({', '.join(failed)}) -> respond with what is available"
        else:
            why = f"all planned workers finished ({', '.join(completed)}) -> respond"
    trace("agent_route", "supervisor", next=nxt, reason=why,
          state={"plan": plan, "completed": completed, "failed": failed, "steps": steps})
    return {**update, "next": nxt}


def route_from_supervisor(state: GraphState) -> str:
    return state["next"]


# ---------------------------------------------------------------- workers
async def _run_worker(agent: str, state: GraphState) -> dict:
    if state.get("force_fail") == agent:
        raise RuntimeError(f"forced failure of the {agent} agent (test hook)")
    decision = router.decide("worker", agent=agent, complexity=state.get("complexity", "simple"))
    trace("llm_route", agent, **decision.as_dict())
    hist = history_text(state)
    messages = [SystemMessage(WORKER_PROMPTS[agent]),
                HumanMessage((f"Conversation so far:\n{hist}\n\n" if hist else "") + f"User question: {state['query']}")]
    tool_specs = T.lc_tools(agent)
    notes, citations = [], []
    for round_no in range(config.WORKER_TOOL_ROUNDS):
        msg, used, ms = await router.invoke(decision, messages, tools=tool_specs, on_fallback=on_fallback(agent))
        decision = used
        metric_llm(f"worker:{agent}", used, msg, ms)
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            text = msg.content if isinstance(msg.content, str) else "".join(
                b.get("text", "") for b in msg.content if isinstance(b, dict))
            if text.strip():
                notes.append(text.strip())
            break
        messages.append(msg)   # keep the original message: Gemini 3 needs its thought signature echoed back
        had_error = False
        for call in calls:
            res = await T.execute(agent, call["name"], call.get("args", {}))
            had_error |= res.error is not None
            trace(res.kind, agent, **{**res.meta, "tool": call["name"], "args": call.get("args", {}),
                                      "ms": round(res.ms, 1), "error": res.error})
            emit("metric", {"kind": "tool", "tool": call["name"], "tool_kind": res.kind, "ms": round(res.ms, 1)})
            notes.append(res.text)
            citations.extend(res.citations)
            messages.append(ToolMessage(content=res.text, tool_call_id=call.get("id") or call["name"]))
        if not had_error:
            break      # evidence gathered; the respond node writes the answer (saves one LLM call)
    return {"completed": [agent], "notes": [{"agent": agent, "text": "\n".join(notes) or "(no output)"}],
            "citations": citations, "steps": state.get("steps", 0) + 1}


def make_worker(agent: str):
    async def node(state: GraphState) -> dict:
        trace("agent_start", agent)
        t0 = time.perf_counter()
        try:
            out = await _run_worker(agent, state)
            trace("agent_end", agent, ms=round((time.perf_counter() - t0) * 1000, 1),
                  citations=len(out["citations"]))
            return out
        except Exception as e:  # noqa: BLE001 - a failing worker must not crash the graph
            msg = f"{type(e).__name__}: {str(e)[:200]}"
            trace("worker_error", agent, error=msg)
            return {"failed": [{"agent": agent, "error": msg}], "steps": state.get("steps", 0) + 1}
    node.__name__ = agent
    return node


# ---------------------------------------------------------------- respond
def number_citations(citations: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in citations:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        out.append({**c, "n": len(out) + 1})
    return out[:12]          # cap prompt size; retrieval already ranked them


async def respond(state: GraphState) -> dict:
    cites = number_citations(state.get("citations", []))
    for c in cites:
        emit("citation", {k: c.get(k) for k in ("n", "kind", "id", "arxiv_id", "title", "year", "url", "agent", "score")}
             | {"snippet": c["text"][:300]})
    notes = state.get("notes", [])
    failed = state.get("failed", [])
    notes_text = "\n\n".join(f"[{n['agent']} agent]\n{n['text']}" for n in notes)
    if failed:
        notes_text += "\n\n" + "\n".join(f"[{f['agent']} agent FAILED: {f['error']}] Tell the user this part "
                                         f"could not be completed." for f in failed)
    inputs = {"query": state["query"], "history_text": history_text(state), "notes_text": notes_text,
              "citations": cites}
    context_chars = len(format_sources(cites)) + len(notes_text)
    decision = router.decide("respond", complexity=state.get("complexity", "simple"),
                             n_sources=len(state.get("completed", [])), context_chars=context_chars)
    trace("llm_route", "respond", **decision.as_dict())
    parts: list[str] = []
    try:
        async for kind, payload in router.stream(decision, inputs, build=build_answer_chain,
                                                 on_fallback=on_fallback("respond")):
            if kind == "token":
                parts.append(payload)
                emit("token", {"text": payload})
            else:
                metric_llm("respond", payload["decision"], None, payload["ms"], payload["ttft_ms"], payload["usage"])
                trace("respond_done", "respond", provider=payload["decision"].provider,
                      ms=round(payload["ms"], 1), ttft_ms=payload["ttft_ms"] and round(payload["ttft_ms"], 1))
    except Exception as e:  # noqa: BLE001
        msg = f"Answer generation failed on both providers ({type(e).__name__}: {str(e)[:150]})."
        emit("error", {"message": msg, "node": "respond"})
        if not parts:
            fallback = "I couldn't generate an answer right now. " + msg
            parts.append(fallback)
            emit("token", {"text": fallback})
    return {"answer": "".join(parts)}


# ---------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("supervisor", supervisor)
    for w in WORKERS:
        g.add_node(w, make_worker(w))
        g.add_edge(w, "supervisor")
    g.add_node("respond", respond)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_from_supervisor,
                            {"research": "research", "data": "data", "summary": "summary", "respond": "respond"})
    g.add_edge("respond", END)
    return g.compile()


GRAPH = build_graph()

"""Rubric smoke test against a running service (local or Cloud Run).

    python scripts/smoke_test.py http://localhost:8080
    python scripts/smoke_test.py https://multi-agent-rag-xxxx.us-west1.run.app

Each check maps to a grading line. Prints PASS/FAIL per check and saves the raw events to
eval_results/smoke_<host>.json for the evaluation write-up.
"""
from __future__ import annotations

import json
import re
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def chat(base: str, query: str, session: str, fail: str | None = None) -> dict:
    body = {"query": query, "session_id": session}
    if fail:
        body["debug_fail_agent"] = fail
    events, t0, ttft = [], time.perf_counter(), None
    with httpx.stream("POST", f"{base}/api/chat", json=body, timeout=300) as r:
        r.raise_for_status()
        buf = ""
        for piece in r.iter_text():
            buf += piece
            while "\n\n" in buf:
                frame, buf = buf.split("\n\n", 1)
                ev = next(line[7:] for line in frame.splitlines() if line.startswith("event: "))
                data = json.loads(next(line[6:] for line in frame.splitlines() if line.startswith("data: ")))
                if ev == "token" and ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                events.append({"event": ev, "data": data})
    tokens = [e for e in events if e["event"] == "token"]
    traces = [e["data"] for e in events if e["event"] == "trace"]
    return {
        "query": query, "events": events, "answer": "".join(e["data"]["text"] for e in tokens),
        "n_tokens_events": len(tokens), "client_ttft_ms": ttft, "client_total_ms": (time.perf_counter() - t0) * 1000,
        "traces": traces, "citations": [e["data"] for e in events if e["event"] == "citation"],
        "errors": [e["data"] for e in events if e["event"] == "error"],
        "done": next((e["data"] for e in events if e["event"] == "done"), {}),
    }


def kinds(res, kind):
    return [t for t in res["traces"] if t["kind"] == kind]


def main(base: str) -> int:
    base = base.rstrip("/")
    sid = "smoke-" + uuid.uuid4().hex[:6]
    results, checks = {}, []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    h = httpx.get(f"{base}/health", timeout=300).json()
    print("health:", json.dumps(h))
    check("health: RAG index loaded", str(h.get("rag", "")).startswith("ok"), h.get("rag"))
    check("health: Tavily MCP connected", str(h.get("tavily_mcp", "")).startswith("ok"), h.get("tavily_mcp"))
    check("health: both providers configured", "not configured" not in json.dumps(h.get("providers")), h.get("providers"))

    r = results["rag"] = chat(base, "How does Self-RAG decide when to retrieve, and what are reflection tokens?", sid)
    check("RAG path: corpus retrieval traced", kinds(r, "rag_retrieval") or kinds(r, "paper_read"))
    check("RAG path: corpus citations with arXiv URLs", any(c["kind"] == "corpus" and "arxiv.org" in c["url"] for c in r["citations"]))
    check("RAG path: answer cites [n]", re.search(r"\[\d+\]", r["answer"]), r["answer"][:120])
    check("Streaming: many token events", r["n_tokens_events"] > 5, f"{r['n_tokens_events']} token events")

    r = results["live"] = chat(base, "What are the most recent announcements about AI agents this week?", sid)
    mcp = kinds(r, "mcp_tool_call")
    check("Live path: Tavily MCP called by research", any(t["node"] == "research" and not t.get("error") for t in mcp),
          [t.get("error") for t in mcp])
    check("Live path: web citations", any(c["kind"] == "web" for c in r["citations"]))

    r = results["data"] = chat(base, "How many papers in the corpus were published each year? Which year has the most?", sid)
    check("Data agent: SQL executed", any(not t.get("error") for t in kinds(r, "sql_query")))

    r = results["summary"] = chat(base, "Summarize the ReAct paper and the follow-up work it inspired.", sid)
    sm = kinds(r, "mcp_tool_call")
    check("Summary agent: read_paper", kinds(r, "paper_read"))
    check("MCP sharing: summary agent also calls Tavily", any(t["node"] == "summary" for t in sm))

    r = results["complex"] = chat(base, "Compare Reflexion and Tree of Thoughts, and tell me which paper is older.", sid)
    plan = kinds(r, "plan")
    check("Supervisor: multi-worker plan", plan and len(plan[0]["plan"]) >= 2, plan and plan[0]["plan"])

    providers = set()
    for res in results.values():
        for t in kinds(res, "llm_route"):
            if t["node"] == "respond":
                providers.add(t["provider"])
    check("Routing: respond step used BOTH providers across queries", providers >= {"gemini", "nvidia"}, providers)
    check("Routing: every llm_route has a reason", all(t.get("reason") for res in results.values() for t in kinds(res, "llm_route")))

    r = results["memory"] = chat(base, "What was the first question I asked you in this conversation?", sid)
    check("Memory: history loaded", kinds(r, "history_loaded") and kinds(r, "history_loaded")[0]["turns"] >= 2)
    check("Memory: answer refers to Self-RAG", "self-rag" in r["answer"].lower() or "reflection" in r["answer"].lower(), r["answer"][:120])

    # Break the data agent on a question the supervisor reliably sends to it.
    r = results["failure"] = chat(base, "How many papers in the corpus were published in 2023?",
                                  "smoke-fail-" + uuid.uuid4().hex[:4], fail="data")
    check("Failure: worker_error traced", kinds(r, "worker_error"))
    check("Failure: graph still answers + done event", r["answer"] and r["done"], r["answer"][:80])

    for k, res in results.items():
        d = res["done"]
        print(f"  {k:8s} ttft={res['client_ttft_ms'] and round(res['client_ttft_ms'])}ms total={round(res['client_total_ms'])}ms "
              f"server={d.get('timings_ms')} cost=${d.get('cost_usd')} providers={d.get('providers')}")

    out = ROOT / "eval_results" / f"smoke_{base.split('//')[-1].split('.')[0].replace(':', '_')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"base": base, "health": h, "checks": checks, "results": results}, indent=1, default=str))
    failed = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed. Raw events: {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"))

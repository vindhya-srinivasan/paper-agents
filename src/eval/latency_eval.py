"""End-to-end latency against the DEPLOYED service, measured from the client.

Warm (steady state; run right after any request so an instance is up):
    python -m src.eval.latency_eval https://<service-url> --label warm
Cold (first request to an idle service): wait 20+ minutes with no traffic so Cloud Run scales to
zero, then:
    python -m src.eval.latency_eval https://<service-url> --label cold --n 1
Repeat the cold run 2-3 times across the day; each run appends to eval_results/latency_cold.json.

Records client TTFT and total, plus the server-side split (retrieval / LLM / tools), tokens, cost
and providers from the SSE `done` event, and whether the instance id changed (proves a cold start).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.smoke_test import chat  # noqa: E402

QUERIES = [
    "How does RAG-Token differ from RAG-Sequence?",                                  # research, simple
    "How many papers in the corpus are about multi-agent systems?",                  # data
    "Summarize the MemGPT paper.",                                                   # summary (long context)
    "What happened in AI agent research news this week?",                            # research + live web
    "Compare HyDE and DPR as retrieval approaches, and which came first?",           # complex, multi-agent
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--label", choices=["warm", "cold"], required=True)
    ap.add_argument("--n", type=int, default=len(QUERIES))
    args = ap.parse_args()
    base = args.url.rstrip("/")
    out = ROOT / "eval_results" / f"latency_{args.label}.json"
    out.parent.mkdir(exist_ok=True)
    runs = json.loads(out.read_text()) if out.exists() and args.label == "cold" else []

    for i, q in enumerate(QUERIES[: args.n]):
        r = chat(base, q, f"lat-{args.label}-{int(time.time())}-{i}")
        d = r["done"]
        row = {"label": args.label, "when": time.strftime("%Y-%m-%d %H:%M"), "query": q,
               "client_ttft_ms": round(r["client_ttft_ms"] or 0), "client_total_ms": round(r["client_total_ms"]),
               "server": d.get("timings_ms"), "usage": d.get("usage"), "cost_usd": d.get("cost_usd"),
               "providers": d.get("providers"), "plan": d.get("plan"), "instance_id": d.get("instance_id"),
               "errors": r["errors"]}
        runs.append(row)
        print(json.dumps(row))
    out.write_text(json.dumps(runs, indent=1))

    warm = [r for r in runs if r["label"] == args.label]
    print(f"\n| {args.label} (n={len(warm)}) | median | max |\n|---|---|---|")
    for name, get in [("client TTFT ms", lambda r: r["client_ttft_ms"]), ("client total ms", lambda r: r["client_total_ms"]),
                      ("server retrieval ms", lambda r: r["server"]["retrieval"]), ("server LLM ms", lambda r: r["server"]["llm"]),
                      ("server tools ms", lambda r: r["server"]["tools"])]:
        xs = [get(r) for r in warm if r.get("server")]
        if xs:
            print(f"| {name} | {statistics.median(xs):.0f} | {max(xs):.0f} |")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

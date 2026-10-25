"""RAGAS evaluation over the fixed question set (src/eval/questions.json).

    python -m src.eval.ragas_eval                         # routed system, all 4 metrics
    python -m src.eval.ragas_eval --respond-provider gemini --metrics faithfulness,answer_relevancy
    python -m src.eval.ragas_eval --respond-provider nvidia --metrics faithfulness,answer_relevancy

Runs the real LangGraph in-process (local Chroma index + your API keys), collects each answer with
the exact chunks retrieved for it, then scores with RAGAS:
  faithfulness, answer_relevancy (reference-free), context_precision, context_recall (vs reference).
Judge LLM: Gemini (via LangchainLLMWrapper). Embeddings: the same local model used for retrieval.
Also records per-question latency (retrieval / LLM / TTFT / total), tokens and estimated cost.
Outputs eval_results/ragas_<label>.json and a markdown table to paste into docs/evaluation.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from src import config
from src.llm import router

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = Path(__file__).with_name("questions.json")


def force_respond(provider: str | None):
    """Eval-only override so the same questions can be answered by each provider for comparison."""
    if not provider:
        return
    original = router.decide

    def decide(role, **kw):
        d = original(role, **kw)
        if role == "respond":
            return router.RouteDecision("respond", provider, router.model_name(provider), "forced(eval)",
                                        f"Evaluation run: respond forced to {provider}.", d.signals)
        return d
    router.decide = decide
    import src.graph.graph as g
    g.router.decide = decide


async def answer(q: str) -> dict:
    from src.graph.graph import GRAPH

    state = {"session_id": "eval", "query": q, "history": [], "completed": [], "failed": [], "notes": [],
             "citations": [], "steps": 0, "plan": None}
    t0 = time.perf_counter()
    ttft, parts, cites = None, [], []
    m = {"llm_ms": 0.0, "retrieval_ms": 0.0, "tool_ms": 0.0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    providers, plan, errors = {}, None, []
    async for ch in GRAPH.astream(state, stream_mode="custom"):
        ev, d = ch["event"], ch["data"]
        if ev == "token":
            ttft = ttft or (time.perf_counter() - t0) * 1000
            parts.append(d["text"])
        elif ev == "citation":
            cites.append(d)
        elif ev == "metric" and d["kind"] == "llm":
            m["llm_ms"] += d["ms"]
            m["input_tokens"] += d["input_tokens"]
            m["output_tokens"] += d["output_tokens"]
            m["cost_usd"] += d["cost_usd"]
            providers[d["role"]] = d["provider"]
        elif ev == "metric":
            m["retrieval_ms" if d["tool_kind"] in ("rag_retrieval", "paper_read") else "tool_ms"] += d["ms"]
        elif ev == "trace" and d["kind"] == "plan":
            plan = d["plan"]
        elif ev in ("error",) or (ev == "trace" and d["kind"] == "worker_error"):
            errors.append(d)
    return {"answer": "".join(parts), "citations": cites, "ttft_ms": ttft, "total_ms": (time.perf_counter() - t0) * 1000,
            "providers": providers, "plan": plan, "errors": errors, **m}


def contexts_for(citations: list[dict]) -> list[str]:
    """Full text of the corpus chunks that were placed in the prompt (what RAGAS should judge)."""
    from src.rag.retriever import get_chunk

    out = []
    for c in citations:
        if c["kind"] == "corpus":
            ch = get_chunk(c["id"])
            if ch:
                out.append(ch["text"])
        else:
            out.append(f"{c['title']}: {c.get('snippet', '')}")
    return out


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--respond-provider", choices=["gemini", "nvidia"], default=None)
    ap.add_argument("--metrics", default="faithfulness,answer_relevancy,context_precision,context_recall")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-ragas", action="store_true", help="only collect answers + latency")
    args = ap.parse_args()
    label = args.respond_provider or "routed"
    force_respond(args.respond_provider)

    qs = json.loads(QUESTIONS.read_text())
    if args.limit:
        qs = qs[: args.limit]

    async def collect():
        from src.tools import tavily_mcp
        await tavily_mcp.load()
        rows = []
        for q in qs:
            r = await answer(q["question"])
            r.update(id=q["id"], question=q["question"], reference=q["reference"], contexts=contexts_for(r["citations"]))
            print(f"{q['id']} {r['total_ms']:.0f}ms plan={r['plan']} providers={r['providers']} cites={len(r['citations'])}")
            rows.append(r)
            await asyncio.sleep(1)       # gentle on free-tier rate limits
        return rows

    rows = asyncio.run(collect())
    scores = {}
    if not args.skip_ragas:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas import EvaluationDataset, RunConfig, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas import metrics as M

        from src.rag.embeddings import get_embeddings

        chosen = [getattr(M, name.strip()) for name in args.metrics.split(",")]
        ds = EvaluationDataset.from_list([{"user_input": r["question"], "response": r["answer"],
                                           "retrieved_contexts": r["contexts"] or ["(none)"], "reference": r["reference"]}
                                          for r in rows])
        judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, google_api_key=config.GOOGLE_API_KEY,
                                                           temperature=0))
        res = evaluate(ds, metrics=chosen, llm=judge, embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
                       run_config=RunConfig(max_workers=2, timeout=180, max_retries=6))
        df = res.to_pandas()
        for name in [c.name for c in chosen]:
            col = df[name].dropna().tolist()
            scores[name] = round(statistics.mean(col), 3) if col else None
            for r, v in zip(rows, df[name].tolist()):
                r.setdefault("scores", {})[name] = None if v != v else round(float(v), 3)

    out = ROOT / "eval_results" / f"ragas_{label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"label": label, "date": time.strftime("%Y-%m-%d"), "gemini_model": config.GEMINI_MODEL,
                               "nvidia_model": config.NVIDIA_MODEL, "judge": config.GEMINI_MODEL,
                               "embedding_model": config.EMBEDDING_MODEL, "k": config.RETRIEVAL_K,
                               "scores": scores, "rows": rows}, indent=1, default=str))

    print(f"\n### RAGAS ({label}, n={len(rows)}, judge={config.GEMINI_MODEL})\n")
    print("| Metric | Score |\n|---|---|")
    for k, v in scores.items():
        print(f"| {k} | {v} |")
    print("\n| Latency (in-process, warm) | median | p90 |\n|---|---|---|")
    for key in ("ttft_ms", "retrieval_ms", "llm_ms", "total_ms"):
        xs = [r[key] for r in rows]
        print(f"| {key} | {pct(xs, .5):.0f} | {pct(xs, .9):.0f} |")
    tok_in = statistics.mean(r["input_tokens"] for r in rows)
    tok_out = statistics.mean(r["output_tokens"] for r in rows)
    cost = statistics.mean(r["cost_usd"] for r in rows)
    print(f"\nMean tokens/query: {tok_in:.0f} in + {tok_out:.0f} out; mean est. cost/query ${cost:.5f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

# Evaluation

All numbers below come from scripts in this repo; raw outputs are in `eval_results/`.
Run date: ⟪FILL date⟫ · Gemini model: `⟪FILL⟫` · NVIDIA model: `⟪FILL⟫` · Embeddings: `BAAI/bge-small-en-v1.5` · k = 5 · Corpus: ⟪FILL⟫ papers, ⟪FILL⟫ chunks.

## 1. RAGAS (fixed question set)

`python -m src.eval.ragas_eval` runs the **full multi-agent graph in-process** on the 15 fixed questions in `src/eval/questions.json`. Each question targets a specific paper and has a hand-written reference answer. RAGAS 0.4.3 then scores the answer against the exact chunks that were placed in the generation prompt. Judge LLM: `⟪FILL⟫`. RAGAS embeddings: the same local BGE model used for retrieval.

| Metric | What it measures | Score (routed system) |
|---|---|---|
| Faithfulness | Share of answer claims supported by the retrieved context | ⟪FILL⟫ |
| Answer relevancy | How directly the answer addresses the question | ⟪FILL⟫ |
| Context precision | Whether relevant chunks are ranked above irrelevant ones | ⟪FILL⟫ |
| Context recall | Share of the reference answer covered by the retrieved context | ⟪FILL⟫ |

Per-question scores are in `eval_results/ragas_routed.json`. Weakest questions and why: ⟪FILL: e.g. q13 RAGAS paper: retrieval returned the evaluation-results section rather than the metric definitions⟫.

## 2. Latency

### Deployed service, warm (steady state)

`python -m src.eval.latency_eval $URL --label warm`: five queries covering each path (research, data, summary, live web, complex multi-agent), measured from the client over the public internet. Server-side split from the `done` event.

| Metric (n = ⟪FILL⟫) | median | max |
|---|---|---|
| Time to first token (client) | ⟪FILL⟫ ms | ⟪FILL⟫ ms |
| Total (client) | ⟪FILL⟫ ms | ⟪FILL⟫ ms |
| Retrieval (server: embedding + Chroma query) | ⟪FILL⟫ ms | ⟪FILL⟫ ms |
| LLM calls (server, summed) | ⟪FILL⟫ ms | ⟪FILL⟫ ms |
| Tools (server: Tavily MCP + SQL) | ⟪FILL⟫ ms | ⟪FILL⟫ ms |

**Representative warm latency:** TTFT ⟪FILL⟫ s, total ⟪FILL⟫ s.

### Deployed service, cold start (reported separately)

`--label cold --n 1` after 20+ minutes with no traffic (scale to zero, `--min-instances 0`). A new `instance_id` in the `done` event confirms a fresh instance.

| Run | instance_id | TTFT | Total |
|---|---|---|---|
| ⟪FILL⟫ | ⟪FILL⟫ | ⟪FILL⟫ s | ⟪FILL⟫ s |
| ⟪FILL⟫ | ⟪FILL⟫ | ⟪FILL⟫ s | ⟪FILL⟫ s |

The cold-start overhead (⟪FILL⟫ s) comes from instance start-up, loading torch and the embedding model, opening the Chroma index and connecting to Tavily MCP. No cold-start request failed: the request is held until the instance is ready. Where TTFT goes when warm: the planner call (⟪FILL⟫ ms) and the worker tool-selection call (⟪FILL⟫ ms) run before the answer starts streaming; retrieval itself is ⟪FILL⟫ ms.

## 3. Cost per query

Estimated from the token counts each provider returns (`usage_metadata`) multiplied by the list prices in `src/config.py` (checked 2026-10-24). Cross-checked against the providers' billing pages on ⟪FILL date⟫.

| Path | Mean input tokens | Mean output tokens | Est. cost/query |
|---|---|---|---|
| All 15 RAGAS questions (routed) | ⟪FILL⟫ | ⟪FILL⟫ | $⟪FILL⟫ |
| Warm latency set (5 mixed queries) | ⟪FILL⟫ | ⟪FILL⟫ | $⟪FILL⟫ |

Actual bills: Gemini ⟪FILL⟫ (AI Studio usage page), NVIDIA $0 (developer access, ⟪FILL⟫ credits used), Tavily ⟪FILL⟫ of 1,000 free credits used, Google Cloud ⟪FILL⟫ (Billing → Reports, before credits).

## 4. Provider comparison

The same 15 questions, with the answer step forced to each provider (`--respond-provider gemini|nvidia`; planning and tool selection unchanged).

| Provider (model) | Faithfulness | Answer relevancy | Median TTFT | Median total | Mean output tokens | Est. cost/query |
|---|---|---|---|---|---|---|
| Gemini (`⟪FILL⟫`) | ⟪FILL⟫ | ⟪FILL⟫ | ⟪FILL⟫ ms | ⟪FILL⟫ ms | ⟪FILL⟫ | $⟪FILL⟫ |
| NVIDIA (`⟪FILL⟫`) | ⟪FILL⟫ | ⟪FILL⟫ | ⟪FILL⟫ ms | ⟪FILL⟫ ms | ⟪FILL⟫ | $⟪FILL⟫ |
| Routed (hybrid) | ⟪FILL⟫ | ⟪FILL⟫ | ⟪FILL⟫ ms | ⟪FILL⟫ ms | ⟪FILL⟫ | $⟪FILL⟫ |

Takeaway: ⟪FILL: e.g. which provider was faster, which was more faithful, and whether routing complex questions to NVIDIA paid off⟫.

## 5. Orchestration, routing and failure handling (live URL)

`python scripts/smoke_test.py $URL`: ⟪FILL⟫/19 checks passed (`eval_results/smoke_*.json`).

| Behaviour | Evidence |
|---|---|
| Three agents with distinct tools | research → `rag_retrieval` + `mcp_tool_call`; data → `sql_query`; summary → `paper_read` + `mcp_tool_call` traces |
| Same MCP tool used by two agents | `mcp_tool_call` traces with `node=research` and `node=summary` |
| Routing uses both providers | respond step: ⟪FILL⟫ of ⟪FILL⟫ smoke queries on NVIDIA, ⟪FILL⟫ on Gemini, each with its reason |
| Worker failure does not crash the graph | `debug_fail_agent=research`: `worker_error` traced, supervisor reason "research failed → …", answer still streamed, `done` sent |
| Provider failure | ⟪FILL: any `fallback` decisions observed, e.g. Gemini 429 → NVIDIA⟫ |

## 6. Durable state

| Step | Result |
|---|---|
| Ask a question in session `⟪FILL⟫`, served by instance `⟪FILL⟫` | answered, 2 turns written to Firestore `sessions/⟪FILL⟫` |
| `gcloud run services update … --update-env-vars RESTART_TOKEN=…` (new revision, new instance) | `/healthz` instance_id `⟪FILL⟫` (different) |
| Reload the page / ask "what was my first question?" | history restored, answer quoted the first question |

## 7. Resource usage

Cloud Run → Metrics, memory utilization peak during the latency run: ⟪FILL⟫ % of 2 GiB (≈ ⟪FILL⟫ MB). This is why 2 GiB is configured instead of 1 GiB.

## 8. Limitations

- RAGAS uses an LLM judge (Gemini), which may favor Gemini-style answers in the provider comparison.
- 15 questions is a small fixed set focused on single-paper facts. Multi-hop questions across papers are covered only by the smoke test.
- Only the first ~8 chunks (abstract + introduction) feed the Summary agent, so summaries skip late-paper results.
- ⟪FILL: anything else observed⟫

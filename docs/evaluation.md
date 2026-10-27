# Evaluation

All numbers below come from scripts in this repo; raw outputs are in `eval_results/`.
Run date: 2026-10-26 · Gemini model: `gemini-3.5-flash-lite` (free tier) · NVIDIA model: `nvidia/nemotron-3-ultra-550b-a55b` · Embeddings: `BAAI/bge-small-en-v1.5` (local) · k = 5 · Corpus: 50 papers, 3,465 chunks.

## 1. RAGAS (fixed question set)

`python -m src.eval.ragas_eval` runs the **full multi-agent graph in-process** on the 15 fixed questions in `src/eval/questions.json`. Each question targets a specific paper and has a hand-written reference answer. RAGAS 0.4.3 then scores each answer against the exact chunks that were placed in its generation prompt. Judge LLM: `gemini-3.5-flash-lite`. RAGAS embeddings: the same local BGE model used for retrieval. Answer relevancy was re-scored on the saved answers (`--rescore`) after fixing an embedding-wrapper bug; the answers themselves were not regenerated.

| Metric | What it measures | Score (routed system, n = 15) |
|---|---|---|
| Faithfulness | Share of answer claims supported by the retrieved context | **0.983** |
| Answer relevancy | How directly the answer addresses the question | **0.890** |
| Context precision | Whether relevant chunks are ranked above irrelevant ones | **0.668** |
| Context recall | Share of the reference answer covered by the retrieved context | **0.822** |

**Interpretation.**
- Faithfulness is near ceiling: 13 of 15 answers scored 1.0. The strict citation prompt ("answer only from the numbered sources") works.
- Context precision is the weakest metric. Retrieval usually finds the right paper, but not always the most relevant chunk first. The worst case is q10 (Tree of Thoughts / Game of 24, precision 0.0, recall 0.5): none of the top-5 chunks contains the 74% vs 4% result; they cover the task table, the abstract and the method. q08 (Toolformer, 0.33) ranked related-work passages and a chunk from a different paper (an agent survey, `2309.07864`) above the method section that lists the tools.
- Context recall is lowest on q15 (chain-of-thought, 0.33) and q01/q02/q10/q11 (0.5), where the reference answer combines facts from different sections of the paper than the ones retrieved.
- Questions answered by the Summary agent (q03, q07, q09), which loads a paper's opening chunks in order, have higher recall (1.0 each), because abstracts and introductions contain the headline facts.
- Likely improvements: a reranker on the top-20 candidates, or a higher k for "what result did X report" questions.

Per-question scores, answers and retrieved chunk IDs: `eval_results/ragas_routed.json`.

## 2. Latency

### In-process, warm (RAGAS runs, n = 15 each)

Measured inside the graph from the `metric` events. Retrieval = query embedding + Chroma search. LLM = all LLM calls for the turn, summed. The first question of each run (q01) includes about 10 s of one-time embedding-model warm-up and is excluded from the "clean" column. "Clean" also excludes questions where a Gemini call hit the free-tier per-minute limit and fell back to NVIDIA.

| Metric | median (all 15) | p90 (all 15) | median (clean) |
|---|---|---|---|
| Time to first token | 2,865 ms | 10,922 ms | 2,257 ms |
| Retrieval | 35 ms | 45 ms | 35 ms |
| LLM calls | 2,878 ms | 11,062 ms | n/a |
| Total | 3,257 ms | 12,439 ms | 2,669 ms |

The routed system's p90 is set by the 6 questions whose answers were routed to NVIDIA, as designed (see §4). Retrieval is negligible: under 60 ms in every warm query.

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

The cold-start overhead (⟪FILL⟫ s) comes from instance start-up, importing torch, loading the embedding model, opening the Chroma index and connecting to Tavily MCP. No cold-start request failed: the request is held until the instance is ready. When warm, TTFT is mostly the planner call plus one worker tool-selection call (each ~1 s on Gemini Flash-Lite) before the answer starts streaming. Retrieval itself is ~35 ms.

## 3. Cost per query

Estimated from the token counts each provider returns (`usage_metadata`) multiplied by list prices (`src/config.py`): Gemini 3.5 Flash-Lite at $0.30 / 1M input and $2.50 / 1M output (paid-tier list price), and NVIDIA API Catalog at $0 (free developer access). **Actual spend was $0**: the Gemini key is on the free tier, and NVIDIA developer access is free.

| Run | Mean input tokens | Mean output tokens | Est. cost/query |
|---|---|---|---|
| Routed (15 RAGAS questions) | 3,038 | 469 | $0.00097 |
| Answers forced to Gemini | 3,154 | 325 | $0.00143 |
| Answers forced to NVIDIA | 3,090 | 854 | $0.00031 (only Gemini planning/tool calls priced) |
| Live smoke test, 7 mixed queries (local) | n/a | n/a | $0.00036-$0.00158 per query |

So a question costs **about a tenth of a cent** at list price. Routing complex and long-context answers to NVIDIA reduces the priced cost by about a third compared with Gemini-only (routed $0.00097 vs forced-Gemini $0.00143). Actual bills: Gemini $0.00 (AI Studio spend page, free tier); NVIDIA $0 (developer access); Tavily ⟪FILL⟫ of 1,000 free monthly credits used; Google Cloud ⟪FILL⟫ (Billing → Reports, before credits).

## 4. Provider comparison

The same 15 questions, with the answer step forced to each provider (`--respond-provider gemini|nvidia`). Planning and tool selection stay on Gemini in all runs. "Clean" medians exclude q01 (warm-up) and questions where a planning/tool call fell back because of Gemini rate limits.

| Provider (model) | Median TTFT (clean) | Median total (clean) | Mean output tokens | Est. cost/query | Faithfulness* | Answer relevancy* |
|---|---|---|---|---|---|---|
| Gemini (`gemini-3.5-flash-lite`) | 2,138 ms | 2,665 ms | 243 | $0.00158 | 0.984 | 0.881 |
| NVIDIA (`nemotron-3-ultra-550b-a55b`) | 10,151 ms | 12,288 ms | 818 | $0.00039 | 0.982 | 0.903 |
| Routed (hybrid) | 2,257 ms | 2,669 ms | 401 | $0.00113 | 0.983 | 0.890 |

\* RAGAS scores per provider come from the routed run, split by which provider wrote the answer (9 Gemini, 6 NVIDIA). They were not re-scored on the forced runs, to stay within the Gemini free-tier quota.

**Takeaways.**
- **Speed:** Gemini Flash-Lite is about 4.6× faster to finish. Nemotron Ultra is a reasoning model: it spends several seconds reasoning before its first visible token, and its answers are about 3.4× longer.
- **Faithfulness:** effectively identical (0.984 vs 0.982). Both follow the citation rules once the rule is repeated at the end of the prompt. Before that fix, Nemotron answers often omitted `[n]` citations (smoke test, first run).
- **Depth:** NVIDIA's answers were slightly more relevant (0.903 vs 0.881) and more complete on the multi-part and summary questions it was routed. This supports sending complex, multi-source and long-context synthesis there.
- **Routing result:** the hybrid router keeps the median latency at Gemini's level (2.7 s), because most questions are single-source, and spends NVIDIA's extra latency only on the 6/15 questions that met a capability trigger.
- **Free-tier effect:** 3 of 15 routed questions (q07, q14, q15) had a Gemini planning or tool-selection call hit the per-minute limit and fall back to NVIDIA, which was traced as `fallback`. The router now tries other Gemini models (`gemini-3.6-flash`, `gemini-3.5-flash`) before leaving the provider.

## 5. Orchestration, routing and failure handling

`python scripts/smoke_test.py`: 19 checks covering each graded behaviour. Raw events are in `eval_results/smoke_*.json`.

| Run | Result |
|---|---|
| Local, first run | 16/19. Failures: NVIDIA answers had no `[n]` citations; memory forgot early turns (6-message window); failure test targeted an agent the supervisor didn't route to |
| Local, after fixes | **19/19** |
| Live Cloud Run URL | ⟪FILL⟫/19 |

| Behaviour | Evidence |
|---|---|
| Three agents with distinct tools | research → `rag_retrieval` + `mcp_tool_call`; data → `sql_query`; summary → `paper_read` + `mcp_tool_call` traces |
| Same MCP tool used by two agents | `mcp_tool_call` traces with `node=research` (live-news query) and `node=summary` (ReAct follow-up work) |
| Supervisor plans multi-agent work | "Compare Reflexion and Tree of Thoughts…" → plan `[research, data]`, complexity `complex` |
| Routing uses both providers | smoke test: the respond step used Gemini for live/data/memory queries and NVIDIA for summary/complex queries, each with its reason |
| Worker failure does not crash the graph | `debug_fail_agent=data`: `worker_error` traced, supervisor reason "data failed → …", answer still streamed ("The data agent failed to retrieve…"), `done` sent |
| Provider failure | Gemini 429 `RESOURCE_EXHAUSTED` during evaluation → `fallback` to NVIDIA, visible in the Routing tab; answers completed |

## 6. Durable state

| Step | Result |
|---|---|
| Ask a question in session `⟪FILL⟫`, served by instance `⟪FILL⟫` | answered, 2 turns written to Firestore `sessions/⟪FILL⟫` |
| `gcloud run services update … --update-env-vars RESTART_TOKEN=…` (new revision, new instance) | `/health` instance_id `⟪FILL⟫` (different) |
| Reload the page / ask "what was my first question?" | ⟪FILL⟫ |

## 7. Resource usage

Cloud Run → Metrics, memory utilization peak during the latency run: ⟪FILL⟫ % of 2 GiB (≈ ⟪FILL⟫ MB).

## 8. Limitations

- RAGAS uses an LLM judge (Gemini Flash-Lite), the same family as one of the two answer models, which may favour Gemini-style answers.
- 15 questions is a small fixed set focused on single-paper facts. Multi-hop questions across papers are covered only by the smoke test.
- The Summary agent reads only a paper's first 8 chunks (abstract + introduction), so summaries can miss late-paper results.
- Free-tier Gemini rate limits affect latency under bursts. Evaluation runs and live traffic share the same quota.
- Nemotron Ultra's reasoning time makes NVIDIA-routed answers slow (10-30 s). A smaller NIM model (e.g. `nemotron-3-super-120b`) would trade some depth for speed.

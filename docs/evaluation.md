# Evaluation

All numbers below come from scripts in this repo; raw outputs are in `eval_results/`.
Run date: 2026-10-26 · Gemini model: `gemini-3.5-flash-lite` (free tier) · NVIDIA model: `nvidia/nemotron-3-ultra-550b-a55b` · Embeddings: `BAAI/bge-small-en-v1.5` (local) · k = 5 · Corpus: 50 papers, 3,465 chunks.

## 1. RAGAS (fixed question set)

`python -m src.eval.ragas_eval` runs the **full multi-agent graph in-process** on the 15 fixed questions in `src/eval/questions.json`. Each question targets a specific paper and has a hand-written reference answer. RAGAS 0.4.3 then scores each answer against the exact chunks that were placed in its generation prompt. Judge LLM: `gemini-3.5-flash-lite`. RAGAS embeddings: the same local BGE model used for retrieval. Answer relevancy was re-scored on the saved answers (`--rescore`) after fixing an embedding-wrapper bug; the answers themselves were not regenerated.

| Metric            | What it measures                                               | Score (routed system, n = 15) |
| ----------------- | -------------------------------------------------------------- | ----------------------------- |
| Faithfulness      | Share of answer claims supported by the retrieved context      | **0.983**                     |
| Answer relevancy  | How directly the answer addresses the question                 | **0.890**                     |
| Context precision | Whether relevant chunks are ranked above irrelevant ones       | **0.668**                     |
| Context recall    | Share of the reference answer covered by the retrieved context | **0.822**                     |

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

| Metric              | median (all 15) | p90 (all 15) | median (clean) |
| ------------------- | --------------- | ------------ | -------------- |
| Time to first token | 2,865 ms        | 10,922 ms    | 2,257 ms       |
| Retrieval           | 35 ms           | 45 ms        | 35 ms          |
| LLM calls           | 2,878 ms        | 11,062 ms    | n/a            |
| Total               | 3,257 ms        | 12,439 ms    | 2,669 ms       |

The routed system's p90 is set by the 6 questions whose answers were routed to NVIDIA, as designed (see §4). Retrieval is negligible: under 60 ms in every warm query.

### Deployed service, warm (steady state)

`python -m src.eval.latency_eval $URL --label warm`, run 2026-10-27 00:27-00:28: five queries covering each path (research, data, summary, live web, multi-part comparison), measured from my laptop over the public internet, all on instance `f001eebc`. The server-side split comes from the `done` event. Raw rows: `eval_results/latency_warm.json`.

| Metric (n = 5)                               | median   | max       |
| -------------------------------------------- | -------- | --------- |
| Time to first token (client)                 | 4,015 ms | 14,791 ms |
| Total (client)                               | 4,172 ms | 16,306 ms |
| Retrieval (server: embedding + Chroma query) | 10 ms    | 115 ms    |
| LLM calls (server, summed)                   | 2,573 ms | 8,402 ms  |
| Tools (server: Tavily MCP + SQL)             | 0 ms     | 1,959 ms  |

| Query                                                               | Path                   | Answer by | Client TTFT | Client total |
| ------------------------------------------------------------------- | ---------------------- | --------- | ----------- | ------------ |
| How does RAG-Token differ from RAG-Sequence?                        | research               | Gemini    | 4.2 s       | 4.5 s        |
| How many papers in the corpus are about multi-agent systems?        | data (SQL)             | Gemini    | 1.9 s       | 2.0 s        |
| Summarize the MemGPT paper.                                         | summary (long context) | NVIDIA    | 14.8 s      | 16.3 s       |
| What happened in AI agent research news this week?                  | research + Tavily MCP  | Gemini    | 4.0 s       | 4.2 s        |
| Compare HyDE and DPR as retrieval approaches, and which came first? | research               | Gemini    | 2.5 s       | 2.9 s        |

**Representative warm latency:** TTFT about 4.0 s, total about 4.2 s for Gemini-routed questions; 15-16 s when the answer is routed to NVIDIA (the long-context summary). Retrieval is negligible (≤ 115 ms). The time goes to the planner and worker LLM calls before streaming starts, plus about 2 s for the Tavily web search on live questions.

### Deployed service, cold start (reported separately)

`--label cold --n 1` at 2026-10-27 00:27, after more than an hour with no traffic (scale to zero, `--min-instances 0`). The response came from a brand-new instance, `f001eebc`. Raw row: `eval_results/latency_cold.json`.

| Run                                                              | instance_id      | Client TTFT | Client total | Server total (after the instance was up) |
| ---------------------------------------------------------------- | ---------------- | ----------- | ------------ | ---------------------------------------- |
| 2026-10-27 00:27, "How does RAG-Token differ from RAG-Sequence?" | `f001eebc` (new) | 62.5 s      | 62.8 s       | 4.3 s                                    |

The cold-start overhead is about **58.5 s** (client total minus server time). It matches the instance's own `startup_s` of 53.7 s in `/health`, which is the time spent importing torch, loading the embedding model, opening the 3,465-chunk Chroma index and connecting to Tavily MCP, plus container scheduling. The same question warm took 4.5 s. The cold request **did not fail**: Cloud Run held it until the instance was ready (request timeout 300 s). Setting `--min-instances 1` would remove this delay, at the cost of a continuously billed instance, so it was left at 0 for coursework.

## 3. Cost per query

Estimated from the token counts each provider returns (`usage_metadata`) multiplied by list prices (`src/config.py`): Gemini 3.5 Flash-Lite at $0.30 / 1M input and $2.50 / 1M output (paid-tier list price), and NVIDIA API Catalog at $0 (free developer access). **Actual spend was $0**: the Gemini key is on the free tier, and NVIDIA developer access is free.

| Run                                      | Mean input tokens | Mean output tokens | Est. cost/query                                   |
| ---------------------------------------- | ----------------- | ------------------ | ------------------------------------------------- |
| Routed (15 RAGAS questions)              | 3,038             | 469                | $0.00097                                          |
| Answers forced to Gemini                 | 3,154             | 325                | $0.00143                                          |
| Answers forced to NVIDIA                 | 3,090             | 854                | $0.00031 (only Gemini planning/tool calls priced) |
| Live smoke test, 7 mixed queries (local) | n/a               | n/a                | $0.00036-$0.00158 per query                       |
| Deployed warm latency set (5 queries)    | 2,482             | 265                | $0.00092                                          |

So a question costs **about a tenth of a cent** at list price. Routing complex and long-context answers to NVIDIA reduces the priced cost by about a third compared with Gemini-only (routed $0.00097 vs forced-Gemini $0.00143). Actual bills, checked 2026-10-26:

- **Gemini:** $0.00 (AI Studio spend page, free tier).
- **NVIDIA:** $0 (developer access).
- **Tavily:** 17 credits used this month on the Researcher plan, including every live-web query from development, the smoke tests and the evaluations.
- **Google Cloud:** US$0.00 net for 1-26 October (Billing → Reports). That's about $0.03 of usage, offset by savings/credits, across both of my course projects (this one and Assignment 1). Billing data lags up to a day, so the last few Cloud Builds may not be counted yet. At about $0.10-0.20 per build, the total stays under $1.

## 4. Provider comparison

The same 15 questions, with the answer step forced to each provider (`--respond-provider gemini|nvidia`). Planning and tool selection stay on Gemini in all runs. "Clean" medians exclude q01 (warm-up) and questions where a planning/tool call fell back because of Gemini rate limits.

| Provider (model)                      | Median TTFT (clean) | Median total (clean) | Mean output tokens | Est. cost/query | Faithfulness\* | Answer relevancy\* |
| ------------------------------------- | ------------------- | -------------------- | ------------------ | --------------- | -------------- | ------------------ |
| Gemini (`gemini-3.5-flash-lite`)      | 2,138 ms            | 2,665 ms             | 243                | $0.00158        | 0.984          | 0.881              |
| NVIDIA (`nemotron-3-ultra-550b-a55b`) | 10,151 ms           | 12,288 ms            | 818                | $0.00039        | 0.982          | 0.903              |
| Routed (hybrid)                       | 2,257 ms            | 2,669 ms             | 401                | $0.00113        | 0.983          | 0.890              |

\* RAGAS scores per provider come from the routed run, split by which provider wrote the answer (9 Gemini, 6 NVIDIA). They were not re-scored on the forced runs, to stay within the Gemini free-tier quota.

**Takeaways.**

- **Speed:** Gemini Flash-Lite is about 4.6× faster to finish. Nemotron Ultra is a reasoning model: it spends several seconds reasoning before its first visible token, and its answers are about 3.4× longer.
- **Faithfulness:** effectively identical (0.984 vs 0.982). Both follow the citation rules once the rule is repeated at the end of the prompt. Before that fix, Nemotron answers often omitted `[n]` citations (smoke test, first run).
- **Depth:** NVIDIA's answers were slightly more relevant (0.903 vs 0.881) and more complete on the multi-part and summary questions it was routed. This supports sending complex, multi-source and long-context synthesis there.
- **Routing result:** the hybrid router keeps the median latency at Gemini's level (2.7 s), because most questions are single-source, and spends NVIDIA's extra latency only on the 6/15 questions that met a capability trigger.
- **Free-tier effect:** 3 of 15 routed questions (q07, q14, q15) had a Gemini planning or tool-selection call hit the per-minute limit and fall back to NVIDIA, which was traced as `fallback`. The router now tries other Gemini models (`gemini-3.6-flash`, `gemini-3.5-flash`) before leaving the provider.

## 5. Orchestration, routing and failure handling

`python scripts/smoke_test.py`: 19 checks covering each graded behaviour. Raw events are in `eval_results/smoke_*.json`.

| Run                                                | Result                                                                                                                                                                                                                                     |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Local, first run                                   | 16/19. Failures: NVIDIA answers had no `[n]` citations; memory forgot early turns (6-message window); failure test targeted an agent the supervisor didn't route to                                                                        |
| Local, after fixes                                 | **19/19**                                                                                                                                                                                                                                  |
| Live URL, 2026-10-26 23:07                         | 18/19. The "multi-worker plan" check failed: after the evening's evaluation runs, all three Gemini models were rate-limited, the planner fell back to NVIDIA, and NVIDIA planned the comparison with one agent. The answer still completed |
| Live URL, 2026-10-27 00:28 (rate limits recovered) | **19/19**; the comparison was planned as `[research, data]` and answered by NVIDIA, and simple queries were answered by Gemini                                                                                                             |

| Behaviour                               | Evidence                                                                                                                                                       |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Three agents with distinct tools        | research → `rag_retrieval` + `mcp_tool_call`; data → `sql_query`; summary → `paper_read` + `mcp_tool_call` traces                                              |
| Same MCP tool used by two agents        | `mcp_tool_call` traces with `node=research` (live-news query) and `node=summary` (ReAct follow-up work)                                                        |
| Supervisor plans multi-agent work       | "Compare Reflexion and Tree of Thoughts…" → plan `[research, data]`, complexity `complex`                                                                      |
| Routing uses both providers             | smoke test: the respond step used Gemini for live/data/memory queries and NVIDIA for summary/complex queries, each with its reason                             |
| Worker failure does not crash the graph | `debug_fail_agent=data`: `worker_error` traced, supervisor reason "data failed → …", answer still streamed ("The data agent failed to retrieve…"), `done` sent |
| Provider failure                        | Gemini 429 `RESOURCE_EXHAUSTED` during evaluation → `fallback` to NVIDIA, visible in the Routing tab; answers completed                                        |

## 6. Durable state

Tested on the live service on 2026-10-26, with one browser session. The session ID is generated by the page and kept in `localStorage`.

| Step                                                                                             | Result                                                                                                                             |
| ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| ~21:46: first questions in the session ("yoooo", "what's ReAct"), on the first deployed revision | answered; turns written to Firestore `sessions/<session_id>`                                                                       |
| ~23:13: asked "ReAct?" on revision `multi-agent-rag-00004`, instance `e36e572f`                  | answered; history now spans several redeploys                                                                                      |
| `gcloud run services update multi-agent-rag --update-env-vars RESTART_TOKEN=…`                   | new revision, so a brand-new instance: `8b3cc864` (was `e36e572f`)                                                                 |
| Hard-reloaded the page                                                                           | the full earlier conversation was restored from Firestore by the new instance                                                      |
| Asked "What was my first question?"                                                              | _"Your first message in this conversation was "yoooo" (a greeting), followed by your first substantive question: "what's ReAct"."_ |

The answer quotes turns written about 1.5 hours earlier, before several redeploys (revisions up to `00004`) and a forced instance replacement, so the state lives in Firestore, not in process memory. `/health` and the trace's `request` event both report `state_store: firestore (project=csci-599-assignment-1-vindhya, collection=sessions)`.

## 7. Resource usage

Cloud Run → Metrics → Container memory utilisation, peak across tonight's live tests: **about 47% of 2 GiB (≈ 960 MB)**, steady at about 42-45% between requests. That fits under 1 GiB, but only with about 6% to spare. 2 GiB keeps a safe margin for concurrent requests (concurrency 10) and for larger answers.

## 8. Limitations

- RAGAS uses an LLM judge (Gemini Flash-Lite), the same family as one of the two answer models, which may favour Gemini-style answers.
- 15 questions is a small fixed set focused on single-paper facts. Multi-hop questions across papers are covered only by the smoke test.
- The Summary agent reads only a paper's first 8 chunks (abstract + introduction), so summaries can miss late-paper results.
- Free-tier Gemini rate limits affect latency under bursts. Evaluation runs and live traffic share the same quota.
- Nemotron Ultra's reasoning time makes NVIDIA-routed answers slow (10-30 s). A smaller NIM model (e.g. `nemotron-3-super-120b`) would trade some depth for speed.

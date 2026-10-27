# Paper Agents: Multi-Agent RAG with Orchestration (CSCI 599 Assignment 2)

Paper Agents is a research assistant for 50 arXiv papers on LLM agents and retrieval-augmented generation. You ask a question, and a LangGraph supervisor decides which of three agents should handle it:

- **Research agent:** searches the papers (Chroma RAG) and, for anything recent, the live web through Tavily Remote MCP.
- **Data agent:** answers questions about the papers' metadata (authors, years, topics) with read-only SQL.
- **Summary agent:** reads one paper and summarizes it, and can also look up follow-up work on the web with the same Tavily tool.

The final answer is written by either Google Gemini or NVIDIA NIM, depending on how hard the question is, and streams to the browser over Server-Sent Events. The page shows which agents ran, which model handled each step and why, the full trace, and the sources behind every `[n]` citation. Chat history is saved in Firestore, so it survives Cloud Run restarts.

- **Live URL:** https://multi-agent-rag-697124537718.us-west1.run.app (Cloud Run, `us-west1`). The first request after the service has been idle is slower while a new instance starts; cold vs warm numbers are in `docs/evaluation.md`.
- **Docs:** [`docs/architecture.md`](docs/architecture.md) · [`docs/evaluation.md`](docs/evaluation.md) · [`PROCESS_LOG.md`](PROCESS_LOG.md)
- **Frontend:** plain HTML/CSS/JS in `static/`, served by FastAPI. No framework, no HTMX, no build step.

## What to ask (for graders)

The corpus covers 50 papers from 2017-2024:

| Topic                   | Papers                                                                                                                                                                                        |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RAG & retrieval         | RAG (Lewis et al.), REALM, DPR, Fusion-in-Decoder, RETRO, Atlas, HyDE, Lost in the Middle, RAGAS, RAG survey (Gao et al.), Self-RAG, Corrective RAG, GraphRAG, FLARE, IRCoT, Self-Ask, WebGPT |
| Reasoning               | Chain-of-Thought, Self-Consistency, Tree of Thoughts, Plan-and-Solve, Language Agent Tree Search                                                                                              |
| Tool use & agents       | ReAct, Toolformer, MRKL, Gorilla, ToolLLM, HuggingGPT, Reflexion, Voyager, Generative Agents, MemGPT                                                                                          |
| Multi-agent systems     | AutoGen, MetaGPT, CAMEL, Multiagent Debate                                                                                                                                                    |
| Surveys & benchmarks    | two LLM-agent surveys, AgentBench, SWE-bench, GAIA                                                                                                                                            |
| Foundations & alignment | Attention Is All You Need, GPT-3, InstructGPT, Constitutional AI, DPO, LoRA, Flan scaling, LLaMA, Llama 2                                                                                     |

The arXiv IDs are listed in `scripts/fetch_corpus.py`, and the metadata is in `data/papers.json`.

Some questions that exercise each part of the system (nothing is hard-coded, so similar questions work too):

| To see                     | Try asking                                                                 | What to look for                                                                                         |
| -------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| RAG with citations         | "How does Self-RAG decide when to retrieve?"                               | `rag_retrieval` in the Trace tab, arXiv chunks in Sources, `[n]` in the answer                           |
| Live web (Tavily MCP)      | "What are the latest developments in AI agents this week?"                 | `mcp_tool_call` from the research agent, web sources                                                     |
| Structured data            | "How many papers per year are in the corpus?"                              | the data agent's `sql_query`, with the SQL shown                                                         |
| Summary + shared MCP tool  | "Summarize the ReAct paper and the follow-up work it inspired"             | `paper_read` plus `mcp_tool_call` from the summary agent, the same Tavily tool used by a different agent |
| Multi-agent plan + routing | "Compare Reflexion and Tree of Thoughts, and which paper is older?"        | a two-agent plan, with the answer routed to NVIDIA and the reason shown                                  |
| Cheap routing path         | "What does DPR stand for?"                                                 | the answer routed to Gemini, with the reason shown                                                       |
| Memory                     | ask anything, then "Tell me more about the second point"                   | `history_loaded` > 0; history also comes back after a page reload                                        |
| Failure handling           | open the page with `?fail=data`, then ask "How many papers are from 2023?" | `worker_error` in Trace; the supervisor carries on and the answer still completes                        |
| Out of scope               | "What's a good pasta recipe?"                                              | it says the sources don't cover this instead of making something up                                      |

## Repository layout

```text
main.py                 FastAPI app: POST /api/chat (SSE), /health, /api/history, /api/reset, static UI
src/graph/              LangGraph state schema + supervisor graph (routing on state, failure handling)
src/agents/             agent prompts + each agent's tools (search_corpus, read_paper, run_sql, web_search)
src/llm/router.py       two-provider LLM router with traced decisions and fallbacks
src/rag/                ingestion (chunk + embed + Chroma), retrieval, LCEL answer chain with citations
src/data/db.py          SQLite paper metadata + read-only SQL tool
src/tools/tavily_mcp.py Tavily Remote MCP client (Streamable HTTP), shared by Research + Summary
src/state_store.py      Firestore conversation store (in-memory store for local dev only)
src/eval/               RAGAS evaluation, latency evaluation, the fixed question set
static/                 index.html, app.js, styles.css
scripts/                fetch_corpus.py, check_models.py, smoke_test.py, check_submission.py, make_zip.sh
data/corpus/            extracted text of the 50 papers; data/papers.json holds their metadata
eval_results/           raw RAGAS / smoke-test / latency results behind docs/evaluation.md
tests/                  offline tests with fake LLMs (pytest)
Dockerfile, deploy.sh   Cloud Run build and deploy
```

## Provider requirements

| Service                                              | What it's used for                                                       | Getting access                                                                                                                                                                | Settings                                                   |
| ---------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **Google Gemini** (`gemini-3.5-flash-lite`)          | planning, tool selection, simple answers, RAGAS judge                    | key from [Google AI Studio](https://aistudio.google.com/apikey); enable the _Generative Language API_ on that key's project. The free tier works, with rate limits (see Cost) | `GOOGLE_API_KEY`, `GEMINI_MODEL`, `GEMINI_FALLBACK_MODELS` |
| **NVIDIA NIM** (`nvidia/nemotron-3-ultra-550b-a55b`) | complex, multi-source and long-context answers                           | key from [build.nvidia.com](https://build.nvidia.com) (free developer access)                                                                                                 | `NVIDIA_API_KEY`, `NVIDIA_MODEL`, `NVIDIA_BASE_URL`        |
| **Tavily**                                           | live web search over Remote MCP                                          | key from [app.tavily.com](https://app.tavily.com)                                                                                                                             | `TAVILY_API_KEY`                                           |
| **Google Cloud**                                     | Cloud Run, Cloud Build, Artifact Registry, Firestore                     | a project with billing enabled (USC credit covers it)                                                                                                                         | `GCP_PROJECT_ID`, `GCP_REGION`                             |
| Embeddings                                           | `BAAI/bge-small-en-v1.5` via sentence-transformers; runs locally, no key | downloaded automatically                                                                                                                                                      | `EMBEDDING_MODEL`                                          |

The model IDs in `.env.example` were checked on 2026-10-26 with `python scripts/check_models.py`, which lists each provider's models and tests chat and tool calling with your keys. Run it before deploying, because provider catalogs change. On that date `gemini-3.8-flash` and `gemini-3.7-flash` both returned 503 "high demand" on the free tier, while `gemini-3.5-flash-lite` worked, including tool calling. That's why it's the configured Gemini model.

## Local development

You need Python 3.13.

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add the three API keys; keep STATE_BACKEND=memory for local runs

python scripts/check_models.py  # checks model IDs, chat + tool calling on both providers, and Tavily MCP
python scripts/fetch_corpus.py  # downloads the papers into data/corpus (~5-8 min; already in the repo, so optional)
python -m src.rag.ingest        # builds chroma_index/ and data/papers.db (~5 min on a laptop: 3,465 chunks)

python -m uvicorn main:app --port 8080              # then open http://localhost:8080
python scripts/smoke_test.py http://localhost:8080  # 19 checks against the running app
```

The offline tests need no keys or network (they use fake LLMs and embeddings): `pytest -q`.

With `STATE_BACKEND=memory`, history lives only in the process, so it isn't durable. To try Firestore locally, run `gcloud auth application-default login`, then set `STATE_BACKEND=firestore` and `GCP_PROJECT_ID`.

## Deploy to Cloud Run (works on a fresh GCP project)

You need `gcloud` installed and logged in (`gcloud auth login`), and a project with billing enabled.

```bash
# .env needs GCP_PROJECT_ID, GCP_REGION, CLOUD_RUN_SERVICE and the three API keys
./deploy.sh
```

What `deploy.sh` does, in order:

1. Enables the Run, Cloud Build, Artifact Registry, Firestore and Compute APIs.
2. Creates an Artifact Registry Docker repo (`cloud-run`) if there isn't one. Container Registry (`gcr.io`) is shut down for new projects, so the image goes to `REGION-docker.pkg.dev/PROJECT/cloud-run/SERVICE`.
3. Creates the Firestore `(default)` database in Native mode if needed, and gives the Cloud Run service account `roles/datastore.user`.
4. Runs `gcloud builds submit`. The Docker build installs CPU-only torch and the pinned requirements, then builds the Chroma index inside the image from `data/corpus/`. The whole build takes about 15-20 minutes.
5. Runs `gcloud run deploy` with `--memory 2Gi --cpu 1 --cpu-boost --concurrency 10 --timeout 300 --min-instances 0 --max-instances 3 --allow-unauthenticated`. API keys are passed as environment variables from your local `.env`; none are committed to git or baked into the image.

It prints the service URL at the end. To check it:

```bash
curl -s $URL/health        # rag: ok (N chunks), tavily_mcp: ok, state_store: firestore, both providers listed
python scripts/smoke_test.py $URL
```

The service gets 2 GiB instead of the reference 1 GiB because torch, the embedding model and the Chroma index together need more than 1 GiB (peak measured in `docs/evaluation.md`). `--cpu-boost` shortens cold starts. A request that arrives during a cold start waits for the instance to finish loading; it isn't dropped.

**Checking that history survives a restart:**

```bash
# 1. Ask a question in the UI and note the instance id in the top-right corner
# 2. Force a new revision, which means a brand-new instance:
gcloud run services update $CLOUD_RUN_SERVICE --region $GCP_REGION --update-env-vars RESTART_TOKEN=$(date +%s)
# 3. Reload the page. The history comes back from Firestore and the instance id is different.
#    Ask "what was my first question?" and it answers from the stored history.
```

## Cost and access (checked 2026-10-26; prices and free tiers change, so re-check them)

**Per-query estimate.** A typical question makes 3-4 LLM calls: one plan (~500 input tokens), one or two tool-selection calls (~800 each), and the answer (~3,000 input tokens including the retrieved chunks, 100-700 output). In the 19-check smoke test, the estimated cost per query came to **$0.0004-$0.0016**, worked out from the providers' own token counts at the list prices below. Questions answered by NVIDIA come out cheaper, because only their Gemini planning and tool calls are priced.

| Item                                                         | Price used for estimates                                                            | Per query                                                |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Gemini `gemini-3.5-flash-lite` (plan, tools, simple answers) | $0.30 / 1M input, $2.50 / 1M output (paid-tier list price)                          | $0.0004-0.0016 estimated; **$0 actual** on the free tier |
| NVIDIA NIM `nemotron-3-ultra` (complex answers)              | free developer access on build.nvidia.com (rate-limited)                            | $0                                                       |
| Tavily search (only for live questions)                      | 1 credit per basic search; free plan is 1,000 credits/month, then $0.008 per credit | 0-2 credits                                              |
| Embeddings                                                   | local model                                                                         | $0                                                       |
| Cloud Run + Firestore                                        | within the monthly free tier at coursework traffic                                  | about $0                                                 |

So a question costs well under a cent, even at paid prices. A full RAGAS run (15 questions, 4 metrics, Gemini as judge) is a few hundred judge calls: under $0.30 at list price, $0 on the free tier. Each Cloud Build runs on `e2-highcpu-8` to save time, about $0.10-0.20 per build. The measured numbers are in [`docs/evaluation.md`](docs/evaluation.md), and every answer in the UI shows its own token count and estimated cost.

**Gemini free tier.** This deployment runs on Gemini's free tier. Linking my education billing account in AI Studio didn't move the project to a paid tier. The free tier has per-minute and per-day limits, and Gemini returns 429 when you hit them. The router then tries the next Gemini model in `GEMINI_FALLBACK_MODELS` (each model has its own quota), and only falls back to NVIDIA if all of them are out. That fallback shows as a red `fallback` badge, with the reason, in the Routing tab, and the app keeps answering. Running the RAGAS evaluation alongside live traffic is the easiest way to hit the limit, since both share the quota. For heavier use, link a billing account to reach Tier 1.

**Monitor your usage** while testing and after deploying:

- Gemini: [AI Studio usage](https://aistudio.google.com/usage) and [rate limits](https://aistudio.google.com/rate-limit). It's worth setting a budget under Google Cloud Billing → Budgets & alerts.
- NVIDIA: credits and rate limits on your [build.nvidia.com](https://build.nvidia.com) account page.
- Tavily: credits used, on the [Tavily dashboard](https://app.tavily.com).
- Google Cloud: Billing → Reports, filtered to Cloud Run, Cloud Build, Artifact Registry and Firestore. `--max-instances 3` keeps Cloud Run costs from running away.
- Per request: the `done` SSE event and the footer under each answer show tokens and estimated cost.

## Configuration reference

Every setting is an environment variable; see `.env.example`. The ones you're most likely to change: `RETRIEVAL_K` (5), `CHUNK_SIZE` / `CHUNK_OVERLAP` (1200 / 200), `MAX_WORKER_STEPS` (4), `LLM_TIMEOUT_S` (25 s, after which the router gives up on a call and tries the next model), `HISTORY_TURNS` (16 messages of context), `STATE_BACKEND` (`memory` locally, `firestore` on Cloud Run), and `GEMINI_PRICE_IN/OUT` / `NVIDIA_PRICE_IN/OUT` for the cost estimates.

## Testing hooks

- Add `?fail=data` (or `research` / `summary`) to the page URL, or send `"debug_fail_agent": "data"` to the API, to make that agent fail whenever the supervisor routes to it. The graph keeps going, and the trace shows the failure.
- `GET /health` reports the index size, MCP status, state store, configured providers and instance id. It's `/health` rather than `/healthz` because Cloud Run's front end reserves `/healthz` and answers it with Google's own 404.

## Troubleshooting

These are all problems I ran into while building this.

| Symptom                                             | Cause and fix                                                                                                                                                                               |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SSL: CERTIFICATE_VERIFY_FAILED` on macOS           | The python.org installer doesn't include CA certificates. Run `open "/Applications/Python 3.13/Install Certificates.command"`. The repo's scripts also use `certifi` directly               |
| `No module named 'src'`                             | Run from the repo root, and run modules as `python -m src.…` (not `python src/…/file.py`)                                                                                                   |
| `uvicorn: command not found`                        | Activate the venv, or use `python -m uvicorn main:app --port 8080`                                                                                                                          |
| `fetch_corpus.py` prints `HTTP 429 … waiting`       | arXiv is rate limiting. The script backs off and retries on its own. Re-running it resumes, and finished papers are skipped                                                                 |
| Gemini `503 UNAVAILABLE` / `429 RESOURCE_EXHAUSTED` | The model is overloaded or the free-tier quota is used up. The router moves on to other Gemini models, then to NVIDIA. `python scripts/check_models.py` shows which models answer right now |
| The UI header says "status unavailable"             | `/health` didn't return JSON. Open `$URL/health` in the browser to see the error                                                                                                            |

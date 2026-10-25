# Paper Agents: Multi-Agent RAG with Orchestration (CSCI 599 Assignment 2)

A LangGraph **supervisor** delegates each question to three specialized agents (**Research**, **Data**, **Summary**) that search a 50-paper arXiv corpus (Chroma RAG), query paper metadata (SQLite), and search the live web through **Tavily Remote MCP**, a tool shared by two agents. Answer generation is **routed between Google Gemini and NVIDIA NIM** by cost and capability. Answers stream over **SSE** to a vanilla HTML/JS page that shows the routing decision, the orchestration trace and the citations for every turn. Conversation history is stored in **Firestore**.

- **Live URL:** `<CLOUD RUN URL>`
- **Architecture:** [`docs/architecture.md`](docs/architecture.md) · **Evaluation:** [`docs/evaluation.md`](docs/evaluation.md) · **Process log:** [`PROCESS_LOG.md`](PROCESS_LOG.md)
- **Frontend:** vanilla HTML/CSS/JS (no framework, no HTMX, no build step), served by FastAPI from `static/`.

## Repository layout

```text
main.py                 FastAPI app: POST /api/chat (SSE), /healthz, /api/history, /api/reset, static UI
src/graph/              LangGraph state schema + supervisor graph (failure handling, routing on state)
src/agents/             agent system prompts + per-agent tools (search_corpus, read_paper, run_sql, web_search)
src/llm/router.py       inspectable two-provider LLM router with fallback
src/rag/                ingestion (chunk + embed + Chroma), retrieval, LCEL answer chain with citations
src/data/db.py          SQLite paper metadata + read-only SQL tool
src/tools/tavily_mcp.py Tavily Remote MCP client (Streamable HTTP), shared by Research + Summary
src/state_store.py      Firestore conversation store (memory store for local dev only)
src/eval/               RAGAS evaluation, latency evaluation, fixed question set
static/                 index.html, app.js, styles.css
scripts/                fetch_corpus.py, check_models.py, smoke_test.py, check_submission.py
data/corpus/            extracted text of the 50 papers (committed); data/papers.json metadata
tests/                  offline tests with fake LLMs (pytest)
Dockerfile, deploy.sh   Cloud Run build + deploy
```

## Provider requirements

| Service | Used for | How to get access | Key / setting |
|---|---|---|---|
| **Google Gemini API** | supervisor planning, worker tool selection, simple answers, RAGAS judge | Key from [Google AI Studio](https://aistudio.google.com/apikey). **Free tier is too small for this app** (see Cost) | `GOOGLE_API_KEY`, `GEMINI_MODEL` |
| **NVIDIA NIM** (API Catalog) | complex / multi-source / long-context answers | Key from [build.nvidia.com](https://build.nvidia.com) (free developer access) | `NVIDIA_API_KEY`, `NVIDIA_MODEL`, `NVIDIA_BASE_URL` |
| **Tavily** | live web search through Remote MCP | Key from [app.tavily.com](https://app.tavily.com) | `TAVILY_API_KEY` |
| **Google Cloud** | Cloud Run, Cloud Build, Artifact Registry, Firestore | Project with billing enabled (USC credit) | `GCP_PROJECT_ID`, `GCP_REGION` |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers, **runs locally**, no key | downloaded automatically | `EMBEDDING_MODEL` |

Exact model IDs are pinned in `.env.example`. Verify they still exist with `python scripts/check_models.py` before deploying, because provider catalogs change.

## Local development

Requires Python 3.13.

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in the three API keys; keep STATE_BACKEND=memory locally

python scripts/check_models.py  # confirms model IDs, chat + tool calling on both providers, Tavily MCP
python scripts/fetch_corpus.py  # downloads the 50 papers into data/corpus (already committed; skip if present)
python -m src.rag.ingest        # builds chroma_index/ and data/papers.db (about 1-3 min on CPU)

uvicorn main:app --reload --port 8080      # open http://localhost:8080
python scripts/smoke_test.py http://localhost:8080   # rubric checks against the running app
```

Offline tests (no keys or network; fake LLMs and embeddings): `pytest -q`.

Locally, `STATE_BACKEND=memory` keeps history in process memory only, which is **not durable**. To test Firestore locally, run `gcloud auth application-default login`, then set `STATE_BACKEND=firestore` and `GCP_PROJECT_ID`.

## Deploy to Cloud Run (fresh GCP project)

Prerequisites: `gcloud` installed and logged in (`gcloud auth login`), and a project with billing enabled.

```bash
# .env must contain GCP_PROJECT_ID, GCP_REGION, CLOUD_RUN_SERVICE and the three API keys
./deploy.sh
```

`deploy.sh` does the following, in order:
1. `gcloud services enable run cloudbuild artifactregistry firestore compute`
2. Creates an Artifact Registry Docker repo (`cloud-run`) if missing. `gcr.io` / Container Registry is shut down for new projects, so the image goes to `REGION-docker.pkg.dev/PROJECT/cloud-run/SERVICE`.
3. Creates the Firestore `(default)` database in Native mode if missing, and grants the Cloud Run runtime service account `roles/datastore.user`.
4. `gcloud builds submit`: the Docker build installs CPU-only torch and the pinned requirements, then **builds the Chroma index inside the image** from `data/corpus/` (about 8-12 min the first time).
5. `gcloud run deploy` with `--memory 2Gi --cpu 1 --cpu-boost --concurrency 10 --timeout 300 --min-instances 0 --max-instances 3 --allow-unauthenticated`. Keys are passed with `--set-env-vars` from your local `.env`; no secret is committed or baked into the image.

It prints the service URL. Then verify:

```bash
curl -s $URL/healthz        # rag: ok (N chunks), tavily_mcp: ok, state_store: firestore, both providers listed
python scripts/smoke_test.py $URL
```

Why 2 GiB instead of the reference 1 GiB: torch, the embedding model and the Chroma index together need more than 1 GiB resident (measured peak in `docs/evaluation.md`), so 1 GiB would risk out-of-memory restarts. `--cpu-boost` shortens cold starts. Cold start (the first request after scale-to-zero) waits for the instance to load the model and index; the request is held, not failed.

**Durability check** (does history survive instance replacement?):

```bash
# 1. In the UI (or with curl), ask a question and note the session and instance id shown top-right
# 2. Force a brand-new instance:
gcloud run services update $CLOUD_RUN_SERVICE --region $GCP_REGION --update-env-vars RESTART_TOKEN=$(date +%s)
# 3. Reload the page: the history is restored from Firestore and the instance id is different.
#    Ask "what was my first question?" and it answers from the stored history.
```

## Cost and access (checked 2026-10-24; prices and free tiers change, re-check before relying on them)

**Rough per-query estimate.** A typical question makes 3-4 LLM calls: plan (~600 input tokens), 1-2 worker tool-selection calls (~800 each), and the answer (~3,000 input tokens with the retrieved chunks, ~300 output).

| Item | Price used | Per query |
|---|---|---|
| Gemini Flash (plan + workers + simple answers) | $0.375 / 1M input, $1.88 / 1M output (gemini-3.8-flash) | about **$0.001-0.003** |
| NVIDIA NIM (complex answers) | free developer access on build.nvidia.com (rate-limited) | **$0** |
| Tavily search (only when live info is needed) | 1 credit per basic search; free plan 1,000 credits/month, then $0.008/credit | 0-2 credits |
| Embeddings | local model | $0 |
| Cloud Run + Firestore | within the monthly free tier at coursework traffic | about $0 |

So one question costs roughly **a third of a cent or less**. A full RAGAS run (15 questions, 4 metrics, Gemini as judge) is a few hundred judge calls, estimated at **$0.10-0.30**. Each Cloud Build uses `e2-highcpu-8` for speed, about **$0.10-0.20 per build**. Measured numbers from real API usage are in [`docs/evaluation.md`](docs/evaluation.md) §Cost. The UI also shows the estimated cost of every answer, from the providers' token counts.

**Gemini free-tier warning.** As of October 2026 the free tier for `gemini-3.8-flash` is reported at about **20 requests/day**, far too few for 3-4 calls per question. Either link a billing account in AI Studio (Tier 1, instant, costs as above), or set `GEMINI_MODEL` to a Flash-Lite model, which has a larger free daily quota. Without one of these, Gemini calls hit 429 errors and the router falls back to NVIDIA (traced as `fallback`).

**Monitor your usage** while testing and after deploying:
- Gemini: [AI Studio usage](https://aistudio.google.com/usage) and [rate limits](https://aistudio.google.com/rate-limit). Set a budget in Google Cloud Billing → Budgets & alerts.
- NVIDIA: credits and rate limits on your [build.nvidia.com](https://build.nvidia.com) account page.
- Tavily: credits used on the [Tavily dashboard](https://app.tavily.com).
- Google Cloud: Billing → Reports, filtered to Cloud Run, Cloud Build, Artifact Registry and Firestore. `--max-instances 3` caps runaway Cloud Run cost.
- Per request: the `done` SSE event and the UI footer of each answer show tokens and the estimated cost.

## Configuration reference

All settings are environment variables; see `.env.example`. Notable ones: `RETRIEVAL_K` (5), `CHUNK_SIZE` / `CHUNK_OVERLAP` (1200/200), `MAX_WORKER_STEPS` (4), `LLM_TIMEOUT_S` (60), `STATE_BACKEND` (`memory` locally, `firestore` on Cloud Run), `GEMINI_PRICE_IN/OUT` and `NVIDIA_PRICE_IN/OUT` (used for cost estimates).

## Testing hooks

- `?fail=research` (or `data` / `summary`) in the UI URL, or `"debug_fail_agent": "research"` in the API body, forces that worker to fail. This shows that the graph keeps going and the trace records the failure.
- `GET /healthz` reports index size, MCP status, state store, configured providers and the instance id.

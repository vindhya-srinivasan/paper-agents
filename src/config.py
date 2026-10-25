"""Central configuration. Every value comes from the environment (.env locally, --set-env-vars on Cloud Run)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# ---- LLM providers (two different providers are required by the assignment) ----
GOOGLE_API_KEY = env("GOOGLE_API_KEY")
GEMINI_MODEL = env("GEMINI_MODEL", "gemini-3.8-flash")

NVIDIA_API_KEY = env("NVIDIA_API_KEY")
NVIDIA_MODEL = env("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
NVIDIA_BASE_URL = env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# ---- Embeddings (local sentence-transformers, permitted by assignment §3.3) ----
EMBEDDING_MODEL = env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# ---- RAG ----
CORPUS_DIR = ROOT / "data" / "corpus"
PAPERS_JSON = ROOT / "data" / "papers.json"
PAPERS_DB = Path(env("PAPERS_DB", str(ROOT / "data" / "papers.db")))
CHROMA_PERSIST_DIR = Path(env("CHROMA_PERSIST_DIR", str(ROOT / "chroma_index")))
CHROMA_COLLECTION = "corpus"
CHUNK_SIZE = int(env("CHUNK_SIZE", "1200"))        # characters
CHUNK_OVERLAP = int(env("CHUNK_OVERLAP", "200"))   # characters
RETRIEVAL_K = int(env("RETRIEVAL_K", "5"))

# ---- Real-time knowledge: Tavily Remote MCP ----
TAVILY_API_KEY = env("TAVILY_API_KEY")
TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/?tavilyApiKey={key}"

# ---- Durable conversation state ----
STATE_BACKEND = env("STATE_BACKEND", "memory")   # "firestore" on Cloud Run, "memory" for local dev
FIRESTORE_PROJECT_ID = env("FIRESTORE_PROJECT_ID") or env("GCP_PROJECT_ID")
FIRESTORE_COLLECTION = env("FIRESTORE_COLLECTION", "sessions")
HISTORY_TURNS = int(env("HISTORY_TURNS", "6"))

# ---- Orchestration guards ----
MAX_WORKER_STEPS = int(env("MAX_WORKER_STEPS", "4"))
WORKER_TOOL_ROUNDS = int(env("WORKER_TOOL_ROUNDS", "2"))
LLM_TIMEOUT_S = float(env("LLM_TIMEOUT_S", "60"))

# ---- Prices (USD per 1M tokens) used for per-query cost estimates. ----
# Checked 2026-10-24; re-check the provider pricing pages before reporting numbers.
PRICES = {
    "gemini": {"input": float(env("GEMINI_PRICE_IN", "0.375")), "output": float(env("GEMINI_PRICE_OUT", "1.88"))},
    # NVIDIA API Catalog hosted endpoints are free (credit-based) for development use.
    "nvidia": {"input": float(env("NVIDIA_PRICE_IN", "0")), "output": float(env("NVIDIA_PRICE_OUT", "0"))},
}

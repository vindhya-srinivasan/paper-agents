"""RAG stages 3-5: retrieval (top-k from Chroma), augmentation (numbered context block), and the
LCEL generation chain that cites source chunks.
"""
from __future__ import annotations

import time
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src import config
from src.rag.embeddings import get_embeddings


@lru_cache(maxsize=1)
def _collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
    return client.get_collection(config.CHROMA_COLLECTION)


def reset_cache() -> None:
    _collection.cache_clear()


def retrieve(query: str, k: int = config.RETRIEVAL_K, arxiv_id: str | None = None) -> tuple[list[dict], float]:
    """Return (chunks, elapsed_ms). Each chunk: {id, text, score, arxiv_id, title, year, url}."""
    t0 = time.perf_counter()
    vec = get_embeddings().embed_query(query)
    kwargs = {"query_embeddings": [vec], "n_results": k}
    if arxiv_id:
        kwargs["where"] = {"arxiv_id": arxiv_id}
    res = _collection().query(**kwargs)
    chunks = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        chunks.append({
            "id": cid, "text": doc, "score": round(1 - float(dist), 4),
            "arxiv_id": meta["arxiv_id"], "title": meta["title"], "year": meta["year"], "url": meta["url"],
        })
    return chunks, (time.perf_counter() - t0) * 1000


def get_chunk(chunk_id: str) -> dict | None:
    res = _collection().get(ids=[chunk_id])
    if not res["ids"]:
        return None
    meta = res["metadatas"][0]
    return {"id": chunk_id, "text": res["documents"][0], "arxiv_id": meta["arxiv_id"],
            "title": meta["title"], "year": meta["year"], "url": meta["url"]}


def paper_chunks(arxiv_id: str, limit: int = 8) -> list[dict]:
    """First `limit` chunks of a paper in reading order (abstract + introduction)."""
    res = _collection().get(where={"arxiv_id": arxiv_id})
    rows = sorted(zip(res["ids"], res["documents"], res["metadatas"]), key=lambda r: r[2]["chunk"])
    return [{"id": i, "text": d, "arxiv_id": m["arxiv_id"], "title": m["title"], "year": m["year"], "url": m["url"]}
            for i, d, m in rows[:limit]]


# ---- Augmentation: the prompt template documented in docs/architecture.md ----
ANSWER_SYSTEM = """You are the answer writer for a multi-agent research assistant over a corpus of \
arXiv papers on LLM agents and retrieval-augmented generation.

Rules:
1. Answer ONLY from the numbered sources and agent notes below. If they do not contain the answer, say so.
2. Cite every factual sentence with its source number in square brackets, e.g. [1] or [2][4]. \
Never invent a source number. Corpus sources are papers; web sources are live search results.
3. Keep paper names and arXiv IDs exactly as given.
4. Be concise: 1-3 short paragraphs or a short list, unless the user asks for more."""

ANSWER_HUMAN = """Conversation so far:
{history}

Agent notes:
{notes}

Sources:
{sources}

User question: {question}"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages([("system", ANSWER_SYSTEM), ("human", ANSWER_HUMAN)])


def format_sources(citations: list[dict]) -> str:
    if not citations:
        return "(no sources retrieved)"
    lines = []
    for c in citations:
        if c["kind"] == "corpus":
            head = f"[{c['n']}] {c['title']} ({c['year']}), arXiv:{c['arxiv_id']}, chunk {c['id']}"
        else:
            head = f"[{c['n']}] WEB: {c['title']} - {c['url']}"
        lines.append(f"{head}\n{c['text']}")
    return "\n\n".join(lines)


def build_answer_chain(llm):
    """LCEL: dict of state -> prompt -> routed LLM. The router picks `llm`; the chain is the same."""
    to_inputs = RunnableLambda(lambda s: {
        "history": s.get("history_text") or "(new conversation)",
        "notes": s.get("notes_text") or "(none)",
        "sources": format_sources(s.get("citations", [])),
        "question": s["query"],
    })
    return to_inputs | ANSWER_PROMPT | llm

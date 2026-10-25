"""RAG stages 1-2: ingestion (load -> chunk -> embed) and storage (Chroma, embedded/persistent).

Run:  python -m src.rag.ingest            (also runs inside the Docker build)
Re-indexing is idempotent: the collection is dropped and rebuilt so stale chunks never linger.

Chunking strategy: RecursiveCharacterTextSplitter, 1200 chars with 200 overlap, splitting on
paragraph -> line -> sentence -> word boundaries. ~1200 chars is roughly 250-300 tokens: small
enough that top-5 chunks fit easily in the generation prompt, large enough to keep a method
description or result table row together. Overlap keeps sentences that straddle a boundary
retrievable from either side. Each chunk's id is "<arxiv_id>#c<n>" so citations point at an exact
chunk of an exact paper.
"""
from __future__ import annotations

import json
import shutil
import sys
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.rag.embeddings import get_embeddings

MIN_PAPERS = 40


def load_papers() -> list[dict]:
    meta = json.loads(config.PAPERS_JSON.read_text(encoding="utf-8"))
    out = []
    for m in meta:
        path = config.CORPUS_DIR / f"{m['arxiv_id']}.txt"
        if path.exists():
            out.append({**m, "text": path.read_text(encoding="utf-8")})
    return out


def chunk_papers(papers: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",          # sentence-final periods stay with their sentence
    )
    chunks = []
    for p in papers:
        for i, text in enumerate(splitter.split_text(p["text"])):
            if len(text.strip()) < 80:          # drop headers/page-number fragments
                continue
            chunks.append({
                "id": f"{p['arxiv_id']}#c{i}",
                "text": text,
                "metadata": {
                    "arxiv_id": p["arxiv_id"], "title": p["title"], "year": p["year"],
                    "first_author": p["first_author"], "url": p["url"], "chunk": i, "topic": p["topic"],
                },
            })
    return chunks


def build_index(chunks: list[dict]) -> int:
    import chromadb

    if config.CHROMA_PERSIST_DIR.exists():
        shutil.rmtree(config.CHROMA_PERSIST_DIR)
    client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
    col = client.create_collection(name=config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    emb = get_embeddings()
    batch = 128
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        vectors = emb.embed_documents([c["text"] for c in part])
        col.add(ids=[c["id"] for c in part], embeddings=vectors,
                documents=[c["text"] for c in part], metadatas=[c["metadata"] for c in part])
    return col.count()


def main() -> int:
    t0 = time.time()
    papers = load_papers()
    if len(papers) < MIN_PAPERS:
        print(f"ERROR: only {len(papers)} papers found in {config.CORPUS_DIR}. Run scripts/fetch_corpus.py first.")
        return 1
    chunks = chunk_papers(papers)
    n = build_index(chunks)
    from src.data.db import build_db
    build_db(papers, chunks)
    print(f"Indexed {n} chunks from {len(papers)} papers into {config.CHROMA_PERSIST_DIR} "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}) in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

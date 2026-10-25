"""Local sentence-transformers embeddings behind the LangChain Embeddings interface.

Used for corpus ingestion, query embedding, and as the RAGAS embedding model, so all three share
one vector space. BGE models expect a short instruction prefix on queries (not on documents).
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from src import config

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.prefix = QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 200)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([self.prefix + text], normalize_embeddings=True)[0].tolist()


_override: Embeddings | None = None


def set_test_embeddings(emb: Embeddings | None) -> None:
    global _override
    _override = emb
    get_embeddings.cache_clear()


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    return _override if _override is not None else LocalEmbeddings()

import json

import pytest

from src import config
from src.llm import router
from src.rag import embeddings, retriever
from src.tools import tavily_mcp
from tests.fakes import FakeChat, FakeMCPTool, HashEmbeddings, default_script

PAPERS = [
    ("2005.11401", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "rag",
     "RAG combines a parametric seq2seq generator (BART) with a non-parametric dense retriever (DPR) over "
     "a Wikipedia index. We compare RAG-Sequence and RAG-Token formulations. "),
    ("2210.03629", "ReAct: Synergizing Reasoning and Acting in Language Models", "agents",
     "ReAct interleaves reasoning traces and task-specific actions. Evaluated on HotpotQA, FEVER, ALFWorld "
     "and WebShop. "),
    ("2307.03172", "Lost in the Middle: How Language Models Use Long Contexts", "rag",
     "Performance is highest when relevant information is at the beginning or end of the context and "
     "degrades in the middle, a U-shaped curve. "),
]


@pytest.fixture(autouse=True)
def offline_env(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    meta = []
    for aid, title, topic, text in PAPERS:
        (corpus / f"{aid}.txt").write_text(text * 12, encoding="utf-8")
        meta.append({"arxiv_id": aid, "title": title, "authors": ["A. Author", "B. Author"], "first_author": "A. Author",
                     "year": 2020 + len(meta), "published": "2020-05-22", "primary_category": "cs.CL", "topic": topic,
                     "abstract": text, "url": f"https://arxiv.org/abs/{aid}", "chars": len(text * 12)})
    (tmp_path / "papers.json").write_text(json.dumps(meta))
    monkeypatch.setattr(config, "CORPUS_DIR", corpus)
    monkeypatch.setattr(config, "PAPERS_JSON", tmp_path / "papers.json")
    monkeypatch.setattr(config, "PAPERS_DB", tmp_path / "papers.db")
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "STATE_BACKEND", "memory")
    embeddings.set_test_embeddings(HashEmbeddings())
    retriever.reset_cache()

    from src.data.db import build_db
    from src.rag.ingest import build_index, chunk_papers, load_papers
    papers = load_papers()
    chunks = chunk_papers(papers)
    build_index(chunks)
    build_db(papers, chunks)
    retriever.reset_cache()

    gem = FakeChat(name_tag="gemini", script=default_script("gemini"), calls=[])
    nv = FakeChat(name_tag="nvidia", script=default_script("nvidia"), calls=[])
    router.set_test_models({"gemini": gem, "nvidia": nv})
    mcp = FakeMCPTool()
    tavily_mcp.set_test_tools({"tavily_search": mcp})
    yield {"gemini": gem, "nvidia": nv, "mcp": mcp}
    router.set_test_models({})
    embeddings.set_test_embeddings(None)
    retriever.reset_cache()

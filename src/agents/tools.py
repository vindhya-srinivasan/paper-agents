"""Tool implementations. Each agent gets a DIFFERENT subset (see AGENT_TOOLS):

  research : search_corpus (RAG over Chroma)            + web_search (Tavily Remote MCP, shared)
  data     : run_sql (read-only SQLite paper metadata)
  summary  : read_paper (ordered chunks of one paper)   + web_search (Tavily Remote MCP, shared)

Every tool returns a ToolResult carrying text for the LLM, citation records for the UI, and trace
metadata, so the orchestration layer can show exactly what each tool did.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src import config
from src.data import db
from src.rag import retriever
from src.tools import tavily_mcp


@dataclass
class ToolResult:
    text: str
    kind: str                       # rag_retrieval | mcp_tool_call | sql_query | paper_read
    citations: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    ms: float = 0.0
    error: str | None = None


def _corpus_citation(c: dict, agent: str) -> dict:
    return {"kind": "corpus", "id": c["id"], "arxiv_id": c["arxiv_id"], "title": c["title"], "year": c["year"],
            "url": c["url"], "text": c["text"], "score": c.get("score"), "agent": agent}


# ---------- search_corpus ----------
class SearchCorpusArgs(BaseModel):
    query: str = Field(description="A focused search query about methods, results or claims in the papers.")


async def search_corpus(agent: str, query: str) -> ToolResult:
    chunks, ms = await asyncio.to_thread(retriever.retrieve, query, config.RETRIEVAL_K)
    cites = [_corpus_citation(c, agent) for c in chunks]
    listing = "; ".join(f"{c['id']} ({c['title'][:50]}, score {c['score']})" for c in chunks)
    return ToolResult(text=f"search_corpus('{query}') -> {len(chunks)} chunks: {listing}", kind="rag_retrieval",
                      citations=cites, meta={"query": query, "k": config.RETRIEVAL_K,
                                             "chunk_ids": [c["id"] for c in chunks]}, ms=ms)


# ---------- read_paper ----------
class ReadPaperArgs(BaseModel):
    paper: str = Field(description="arXiv ID (e.g. 2210.03629) or a distinctive title phrase (e.g. 'ReAct').")


async def read_paper(agent: str, paper: str) -> ToolResult:
    t0 = time.perf_counter()
    matches = await asyncio.to_thread(db.find_papers, paper)
    if not matches:
        # Fall back to semantic lookup so "the lost-in-the-middle paper" still resolves.
        chunks, _ = await asyncio.to_thread(retriever.retrieve, paper, 1)
        matches = [{"arxiv_id": c["arxiv_id"], "title": c["title"]} for c in chunks]
    if not matches:
        return ToolResult(text=f"read_paper('{paper}'): no matching paper in the corpus.", kind="paper_read",
                          meta={"paper": paper}, ms=(time.perf_counter() - t0) * 1000)
    m = matches[0]
    chunks = await asyncio.to_thread(retriever.paper_chunks, m["arxiv_id"], 8)
    cites = [_corpus_citation(c, agent) for c in chunks]
    return ToolResult(
        text=f"read_paper('{paper}') -> {m['title']} (arXiv:{m['arxiv_id']}), {len(chunks)} opening chunks loaded.",
        kind="paper_read", citations=cites,
        meta={"paper": paper, "arxiv_id": m["arxiv_id"], "chunk_ids": [c["id"] for c in chunks]},
        ms=(time.perf_counter() - t0) * 1000)


# ---------- run_sql ----------
class RunSQLArgs(BaseModel):
    query: str = Field(description="One read-only SQLite SELECT over the papers/authors tables.")


async def run_sql(agent: str, query: str) -> ToolResult:
    t0 = time.perf_counter()
    try:
        rows = await asyncio.to_thread(db.run_sql, query)
    except db.SQLError as e:
        return ToolResult(text=f"SQL error: {e}. Fix the query and try again.", kind="sql_query",
                          meta={"sql": query}, ms=(time.perf_counter() - t0) * 1000, error=str(e))
    return ToolResult(text=f"SQL: {query}\nRows ({len(rows)}): {json.dumps(rows, default=str)[:4000]}",
                      kind="sql_query", meta={"sql": query, "rows": len(rows)}, ms=(time.perf_counter() - t0) * 1000)


# ---------- web_search (Tavily Remote MCP, shared by research + summary) ----------
class WebSearchArgs(BaseModel):
    query: str = Field(description="Web search query for current or post-corpus information.")


def _mcp_text(result) -> str:
    if isinstance(result, tuple):          # (content, artifact)
        result = result[0]
    if isinstance(result, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in result)
    return str(result)


def parse_web_results(text: str, agent: str, limit: int = 5) -> list[dict]:
    """Tavily MCP returns either JSON or 'Title: / URL: / Content:' blocks depending on version."""
    items = []
    try:
        data = json.loads(text)
        for r in (data.get("results", []) if isinstance(data, dict) else data):
            items.append((r.get("title", r.get("url", "")), r.get("url", ""), r.get("content", "")))
    except (json.JSONDecodeError, AttributeError, TypeError):
        for block in re.split(r"\n(?=Title:)", text):
            url = re.search(r"URL:\s*(\S+)", block)
            if not url:
                continue
            title = re.search(r"Title:\s*(.+)", block)
            content = re.search(r"Content:\s*(.+)", block, re.S)
            items.append((title.group(1).strip() if title else url.group(1), url.group(1),
                          (content.group(1).strip() if content else "")))
    return [{"kind": "web", "id": url, "title": title[:150], "url": url, "text": content[:1200], "agent": agent}
            for title, url, content in items[:limit] if url]


async def web_search(agent: str, query: str) -> ToolResult:
    tool = tavily_mcp.search_tool()
    if tool is None:
        return ToolResult(text=f"web_search unavailable ({tavily_mcp.status()}).", kind="mcp_tool_call",
                          meta={"query": query, "server": "tavily"}, error=tavily_mcp.status())
    t0 = time.perf_counter()
    try:
        raw = await asyncio.wait_for(tool.ainvoke({"query": query, "max_results": 5}), timeout=30)
    except Exception as e:  # noqa: BLE001
        return ToolResult(text=f"web_search failed: {e}", kind="mcp_tool_call",
                          meta={"query": query, "server": "tavily", "mcp_tool": tool.name},
                          ms=(time.perf_counter() - t0) * 1000, error=str(e)[:200])
    text = _mcp_text(raw)
    cites = parse_web_results(text, agent)
    return ToolResult(
        text=f"web_search('{query}') -> {len(cites)} results: " + "; ".join(c["title"][:60] for c in cites),
        kind="mcp_tool_call", citations=cites,
        meta={"query": query, "server": "tavily", "mcp_tool": tool.name, "urls": [c["url"] for c in cites]},
        ms=(time.perf_counter() - t0) * 1000)


# ---------- registry ----------
_SPECS = {
    "search_corpus": (search_corpus, SearchCorpusArgs,
                      "Semantic search over the arXiv paper corpus (LLM agents, RAG, reasoning, alignment). "
                      "Returns the top matching chunks with IDs for citation."),
    "read_paper": (read_paper, ReadPaperArgs,
                   "Load the opening sections (abstract, introduction) of ONE paper in the corpus by arXiv ID or title."),
    "run_sql": (run_sql, RunSQLArgs, "Run one read-only SQLite SELECT over paper metadata (papers, authors tables)."),
    "web_search": (web_search, WebSearchArgs,
                   "Live web search via Tavily Remote MCP, for current events, recent releases, or anything "
                   "newer than the paper corpus."),
}

AGENT_TOOLS = {
    "research": ["search_corpus", "web_search"],
    "data": ["run_sql"],
    "summary": ["read_paper", "web_search"],
}


def lc_tools(agent: str) -> list[StructuredTool]:
    """LangChain tool schemas for bind_tools(). Execution goes through execute() so results carry citations."""
    out = []
    for name in AGENT_TOOLS[agent]:
        fn, schema, desc = _SPECS[name]

        async def _noop(**kwargs):  # never called; schema only
            return ""
        out.append(StructuredTool.from_function(coroutine=_noop, name=name, description=desc, args_schema=schema))
    return out


async def execute(agent: str, name: str, args: dict) -> ToolResult:
    if name not in AGENT_TOOLS[agent]:
        return ToolResult(text=f"Tool {name} is not available to the {agent} agent.", kind="error", error="not allowed")
    fn, schema, _ = _SPECS[name]
    try:
        parsed = schema(**args)
    except Exception as e:  # noqa: BLE001
        return ToolResult(text=f"Bad arguments for {name}: {e}", kind="error", error=str(e)[:200])
    return await fn(agent, **parsed.model_dump())

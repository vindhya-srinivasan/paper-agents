"""Deterministic stand-ins for the LLM providers, embeddings and Tavily so tests run offline."""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Callable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda


class HashEmbeddings(Embeddings):
    """Bag-of-words hashed into 256 dims: similar text -> similar vectors, no model download."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * 256
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeChat(BaseChatModel):
    """Scripted chat model. `script(messages, tool_names, structured)` returns an AIMessage or a dict for
    structured output. `fail=True` raises on every call (to test provider fallback)."""

    name_tag: str = "fake"
    script: Callable[..., Any]
    fail: bool = False
    tool_names: list[str] = []
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"tool_names": [t.name for t in tools]})

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        def _run(messages):
            self._check()
            data = self.script(messages, [], schema)
            raw = AIMessage(content=json.dumps(data), usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70})
            parsed = schema(**data)
            return {"raw": raw, "parsed": parsed, "parsing_error": None} if include_raw else parsed
        return RunnableLambda(_run)

    def _check(self):
        self.calls.append(self.name_tag)
        if self.fail:
            raise ConnectionError(f"{self.name_tag} is down")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._check()
        msg = self.script(messages, self.tool_names, None)
        msg.usage_metadata = {"input_tokens": 100, "output_tokens": 30, "total_tokens": 130}
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self._check()
        text = self.script(messages, [], None).content
        words = text.split(" ")
        for i, w in enumerate(words):
            chunk = AIMessageChunk(content=w + (" " if i < len(words) - 1 else ""))
            if i == len(words) - 1:
                chunk.usage_metadata = {"input_tokens": 400, "output_tokens": len(words), "total_tokens": 400 + len(words)}
            yield ChatGenerationChunk(message=chunk)


def default_script(tag: str):
    """Behaves like a sensible model: plans by keyword, calls each agent's tools, answers with citations."""
    def script(messages, tool_names, structured):
        last = messages[-1].content if messages else ""
        system = messages[0].content if messages else ""
        if structured is not None:
            q = last.lower()
            if "how many" in q:
                return {"agents": ["data"], "complexity": "simple", "reason": "counting question -> data"}
            if "summarize" in q:
                return {"agents": ["summary"], "complexity": "simple", "reason": "summary request"}
            if "compare" in q:
                return {"agents": ["research", "data"], "complexity": "complex", "reason": "comparison"}
            if q.strip().endswith("hi"):
                return {"agents": [], "complexity": "simple", "reason": "greeting"}
            return {"agents": ["research"], "complexity": "simple", "reason": "method question"}
        if tool_names:
            already = any(getattr(m, "type", "") == "tool" for m in messages)
            if "run_sql" in tool_names:
                errors = [m for m in messages if getattr(m, "type", "") == "tool" and "SQL error" in m.content]
                sql = "SELECT COUNT(*) AS n FROM papers" if errors or "BADSQL" not in last else "SELEC nope"
                if already and not errors:
                    return AIMessage(content="done")
                return AIMessage(content="", tool_calls=[{"name": "run_sql", "args": {"query": sql}, "id": "s1"}])
            if "search_corpus" in tool_names:
                calls = [{"name": "search_corpus", "args": {"query": "retrieval augmented generation"}, "id": "r1"}]
                if "latest" in last.lower():
                    calls.append({"name": "web_search", "args": {"query": "latest RAG news"}, "id": "w1"})
                return AIMessage(content="", tool_calls=calls)
            if "read_paper" in tool_names:
                return AIMessage(content="", tool_calls=[
                    {"name": "read_paper", "args": {"paper": "ReAct"}, "id": "p1"},
                    {"name": "web_search", "args": {"query": "ReAct follow-up work"}, "id": "w2"}])
        return AIMessage(content=f"Answer from {tag}: retrieval helps [1] and agents act [2].")
    return script


class FakeMCPTool:
    name = "tavily_search"

    def __init__(self):
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        return json.dumps({"results": [
            {"title": "News about RAG", "url": "https://example.com/rag-news", "content": "Fresh RAG news."},
            {"title": "Agents update", "url": "https://example.com/agents", "content": "Agent update."}]})

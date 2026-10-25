"""Tavily Remote MCP (Streamable HTTP) via langchain-mcp-adapters.

ONE MCP client and ONE set of discovered tools is shared by two agents (assignment criterion:
"MCP tool sharing between agents"):
  * Research agent - tavily search for current events / facts newer than the static corpus
  * Summary agent  - tavily search for follow-up work and current context about a paper it summarizes
If the key is missing or the server is unreachable, agents run without live search and the trace
says so; nothing crashes.
"""
from __future__ import annotations

import asyncio
import logging

from src import config

log = logging.getLogger("tavily_mcp")

_tools: dict[str, object] = {}
_status = "not loaded"


def status() -> str:
    return _status


async def load(timeout_s: float = 20.0) -> str:
    global _status
    if not config.TAVILY_API_KEY:
        _status = "disabled: TAVILY_API_KEY not set"
        log.warning(_status)
        return _status
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({
            "tavily": {"url": config.TAVILY_MCP_URL.format(key=config.TAVILY_API_KEY), "transport": "streamable_http"}
        })
        tools = await asyncio.wait_for(client.get_tools(), timeout=timeout_s)
        _tools.clear()
        for t in tools:
            _tools[t.name] = t
        _status = f"ok ({len(_tools)} tools: {', '.join(sorted(_tools))})"
    except Exception as e:  # noqa: BLE001
        _status = f"unavailable: {type(e).__name__}: {str(e)[:150]}"
    log.info("Tavily MCP %s", _status)
    return _status


def set_test_tools(tools: dict[str, object]) -> None:
    global _status
    _tools.clear()
    _tools.update(tools)
    _status = f"ok ({len(_tools)} test tools)"


def search_tool():
    """The shared MCP tool. Tool names have changed across Tavily releases (tavily-search / tavily_search)."""
    for name, tool in _tools.items():
        if "search" in name and "extract" not in name:
            return tool
    return None

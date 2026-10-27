"""Verify every external dependency with YOUR keys before deploying.

    python scripts/check_models.py

1. Lists Gemini and NVIDIA models so you can confirm the exact IDs in .env exist.
2. Sends one tiny chat request to each configured model (and a tool-calling request to Gemini).
3. Connects to Tavily Remote MCP and runs one search.
Paste the output into PROCESS_LOG.md's verification notes and put the date in .env.example.
"""
from __future__ import annotations

import asyncio
import json
import sys
import ssl
import urllib.request
from pathlib import Path

import certifi

# certifi's CA bundle: python.org macOS builds ship without system certificates.
_SSL = ssl.create_default_context(cafile=certifi.where())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402


def get_json(url: str, headers: dict) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30, context=_SSL) as r:
        return json.loads(r.read())


def list_models() -> None:
    print("== Gemini models (flash) ==")
    try:
        data = get_json("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                        {"x-goog-api-key": config.GOOGLE_API_KEY})
        names = [m["name"].removeprefix("models/") for m in data.get("models", [])]
        for n in sorted(x for x in names if "flash" in x):
            print("  ", n, "<-- configured" if n == config.GEMINI_MODEL else "")
        print("   configured model found:", config.GEMINI_MODEL in names)
    except Exception as e:  # noqa: BLE001
        print("   FAILED:", e)
    print("== NVIDIA models (nemotron) ==")
    try:
        data = get_json(f"{config.NVIDIA_BASE_URL}/models", {"Authorization": f"Bearer {config.NVIDIA_API_KEY}"})
        ids = [m["id"] for m in data.get("data", [])]
        for n in sorted(x for x in ids if "nemotron" in x):
            print("  ", n, "<-- configured" if n == config.NVIDIA_MODEL else "")
        print("   configured model found:", config.NVIDIA_MODEL in ids)
    except Exception as e:  # noqa: BLE001
        print("   FAILED:", e)


async def chat_checks() -> None:
    import time

    from langchain_core.tools import tool

    from src.llm import router

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    for p in router.PROVIDERS:
        if not router.provider_available(p):
            print(f"== {p}: SKIPPED, no API key in .env")
            continue
        try:
            llm = router.get_chat(p)
        except Exception as e:  # noqa: BLE001
            print(f"== {p}: client could not be created ({type(e).__name__}: {str(e)[:200]})")
            continue
        t0 = time.perf_counter()
        try:
            msg = await llm.ainvoke("Reply with exactly: OK")
            print(f"== {p} chat: {msg.content!r} in {time.perf_counter() - t0:.1f}s, usage {router.usage_of(msg)}")
        except Exception as e:  # noqa: BLE001
            print(f"== {p} chat FAILED: {e}")
        try:
            msg = await llm.bind_tools([add]).ainvoke("What is 2+3? Use the add tool.")
            print(f"   {p} tool calling: {msg.tool_calls}")
        except Exception as e:  # noqa: BLE001
            print(f"   {p} tool calling FAILED: {e}")


async def tavily_check() -> None:
    from src.tools import tavily_mcp

    print("== Tavily MCP:", await tavily_mcp.load())
    tool = tavily_mcp.search_tool()
    if tool:
        from src.agents.tools import web_search
        res = await web_search("check", "latest news about retrieval-augmented generation")
        print("  ", res.text[:300], "| error:", res.error)


if __name__ == "__main__":
    list_models()
    asyncio.run(chat_checks())
    asyncio.run(tavily_check())

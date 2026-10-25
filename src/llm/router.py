"""Inspectable LLM routing across two providers (Google Gemini and NVIDIA NIM).

Routing dimension: HYBRID.
  * cost        - short structured calls (supervisor planning, worker tool selection, single-source
                  answers) go to the cheaper/faster provider (Gemini Flash).
  * capability  - multi-step synthesis (complex plans, 2+ worker results, or long context such as a
                  full-paper summary) goes to the larger reasoning model (NVIDIA Nemotron).
Every decision is returned as a RouteDecision whose `reason` is streamed to the UI trace panel.
If the chosen provider fails, the call falls back to the other provider and the fallback is traced.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Callable

from src import config

PROVIDERS = ("gemini", "nvidia")

# Thresholds are module constants so the routing table in docs/architecture.md matches the code.
LONG_CONTEXT_CHARS = 8000
MULTI_SOURCE_AGENTS = 2


@dataclass
class RouteDecision:
    role: str
    provider: str
    model: str
    dimension: str            # "cost" | "capability" | "availability" | "fallback"
    reason: str
    signals: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


_test_models: dict[str, Any] = {}
_cache: dict[str, Any] = {}


def set_test_models(models: dict[str, Any]) -> None:
    """Tests inject fake chat models here so the graph runs without network access."""
    _test_models.clear()
    _test_models.update(models)
    _cache.clear()


def model_name(provider: str) -> str:
    return config.GEMINI_MODEL if provider == "gemini" else config.NVIDIA_MODEL


def provider_available(provider: str) -> bool:
    if provider in _test_models:
        return True
    return bool(config.GOOGLE_API_KEY if provider == "gemini" else config.NVIDIA_API_KEY)


def get_chat(provider: str):
    if provider in _test_models:
        return _test_models[provider]
    if provider in _cache:
        return _cache[provider]
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.2,
            timeout=config.LLM_TIMEOUT_S,
            max_retries=1,
        )
    else:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=config.NVIDIA_MODEL,
            api_key=config.NVIDIA_API_KEY,
            base_url=config.NVIDIA_BASE_URL,
            temperature=0.3,
            timeout=config.LLM_TIMEOUT_S,
            max_retries=1,
            stream_usage=True,
        )
    _cache[provider] = llm
    return llm


def _other(provider: str) -> str:
    return "nvidia" if provider == "gemini" else "gemini"


def _apply_availability(d: RouteDecision) -> RouteDecision:
    if provider_available(d.provider):
        return d
    alt = _other(d.provider)
    return RouteDecision(
        role=d.role, provider=alt, model=model_name(alt), dimension="availability",
        reason=f"{d.provider} key not configured; using {alt}. (Preferred: {d.reason})", signals=d.signals,
    )


def decide(role: str, *, agent: str | None = None, complexity: str = "simple",
           n_sources: int = 0, context_chars: int = 0) -> RouteDecision:
    """Pure function: same inputs -> same route. Unit-tested in tests/test_router.py."""
    signals = {"agent": agent, "complexity": complexity, "n_sources": n_sources, "context_chars": context_chars}
    if role == "supervisor":
        d = RouteDecision(role, "gemini", config.GEMINI_MODEL, "cost",
                          "Planning is a short JSON classification -> cheapest, fastest model.", signals)
    elif role == "worker":
        d = RouteDecision(role, "gemini", config.GEMINI_MODEL, "cost",
                          f"{agent} agent: tool selection is a short function-calling step -> Gemini Flash.", signals)
    elif role == "respond":
        triggers = []
        if complexity == "complex":
            triggers.append("supervisor marked the plan complex")
        if n_sources >= MULTI_SOURCE_AGENTS:
            triggers.append(f"{n_sources} agents' results to integrate")
        if context_chars > LONG_CONTEXT_CHARS:
            triggers.append(f"long context ({context_chars} chars > {LONG_CONTEXT_CHARS})")
        if triggers:
            d = RouteDecision(role, "nvidia", config.NVIDIA_MODEL, "capability",
                              "Multi-step synthesis: " + "; ".join(triggers) + " -> larger reasoning model.", signals)
        else:
            d = RouteDecision(role, "gemini", config.GEMINI_MODEL, "cost",
                              f"Single-source short answer ({context_chars} chars) -> cheaper, faster model.", signals)
    else:
        raise ValueError(f"unknown role {role}")
    return _apply_availability(d)


def _fallback(d: RouteDecision, err: Exception) -> RouteDecision:
    alt = _other(d.provider)
    return RouteDecision(d.role, alt, model_name(alt), "fallback",
                         f"{d.provider} failed ({type(err).__name__}: {str(err)[:120]}); retried on {alt}.", d.signals)


def usage_of(msg: Any) -> dict:
    u = getattr(msg, "usage_metadata", None) or {}
    return {"input_tokens": int(u.get("input_tokens", 0) or 0), "output_tokens": int(u.get("output_tokens", 0) or 0)}


def cost_usd(provider: str, usage: dict) -> float:
    p = config.PRICES.get(provider, {"input": 0, "output": 0})
    return (usage["input_tokens"] * p["input"] + usage["output_tokens"] * p["output"]) / 1_000_000


async def invoke(decision: RouteDecision, messages: list, *, tools: list | None = None,
                 structured: type | None = None,
                 on_fallback: Callable[[RouteDecision], None] | None = None):
    """Invoke the routed model; on failure retry once on the other provider.

    Returns (msg, decision_used, ms). With `structured`, msg is {"raw": AIMessage, "parsed": model}.
    """
    attempt = decision
    for i in range(2):
        llm = get_chat(attempt.provider)
        if tools:
            llm = llm.bind_tools(tools)
        if structured:
            llm = llm.with_structured_output(structured, include_raw=True)
        t0 = time.perf_counter()
        try:
            msg = await asyncio.wait_for(llm.ainvoke(messages), timeout=config.LLM_TIMEOUT_S)
            if structured and msg.get("parsed") is None:
                raise ValueError(f"structured output did not parse: {msg.get('parsing_error')}")
            return msg, attempt, (time.perf_counter() - t0) * 1000
        except Exception as e:  # noqa: BLE001 - any provider error triggers fallback
            if i == 1 or not provider_available(_other(attempt.provider)):
                raise
            attempt = _fallback(attempt, e)
            if on_fallback:
                on_fallback(attempt)
    raise RuntimeError("unreachable")


class _ThinkFilter:
    """Drops <think>...</think> reasoning blocks that some NIM models emit inline, across chunk boundaries."""

    def __init__(self):
        self.inside = False
        self.pending = ""          # a possible partial tag held back until the next chunk

    def feed(self, text: str) -> str:
        text = self.pending + text
        self.pending = ""
        tag = "</think>" if self.inside else "<think>"
        for k in range(min(len(tag) - 1, len(text)), 0, -1):    # hold back e.g. "<thi" at the end
            if text.endswith(tag[:k]):
                text, self.pending = text[:-k], text[-k:]
                break
        out = []
        while text:
            if self.inside:
                end = text.find("</think>")
                if end < 0:
                    return "".join(out)
                text, self.inside = text[end + len("</think>"):], False
            else:
                start = text.find("<think>")
                if start < 0:
                    out.append(text)
                    break
                out.append(text[:start])
                text, self.inside = text[start + len("<think>"):], True
        return "".join(out)

    def flush(self) -> str:
        rest, self.pending = ("" if self.inside else self.pending), ""
        return rest


async def stream(decision: RouteDecision, inputs: Any, *, build: Callable[[Any], Any] | None = None,
                 on_fallback: Callable[[RouteDecision], None] | None = None) -> AsyncIterator[tuple[str, Any]]:
    """Yield ("token", text) chunks, then ("final", {decision, usage, ms, ttft_ms}).

    Fallback only happens if the provider fails before the first token, so the user never sees a
    half answer from one model followed by a full answer from another.
    """
    attempt = decision
    for i in range(2):
        llm = get_chat(attempt.provider)
        runnable = build(llm) if build else llm   # e.g. the LCEL answer chain: prompt | llm
        t0 = time.perf_counter()
        ttft = None
        agg = None                 # LangChain chunks are additive; the sum carries total usage
        think = _ThinkFilter()
        try:
            async for chunk in runnable.astream(inputs):
                agg = chunk if agg is None else agg + chunk
                raw = chunk.content if isinstance(chunk.content, str) else "".join(
                    p.get("text", "") for p in chunk.content if isinstance(p, dict))
                text = think.feed(raw)
                if not text:
                    continue
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                yield "token", text
            tail = think.flush()
            if tail:
                yield "token", tail
            usage = usage_of(agg) if agg is not None else {"input_tokens": 0, "output_tokens": 0}
            yield "final", {"decision": attempt, "usage": usage, "ms": (time.perf_counter() - t0) * 1000, "ttft_ms": ttft}
            return
        except Exception as e:  # noqa: BLE001
            if ttft is not None or i == 1 or not provider_available(_other(attempt.provider)):
                raise
            attempt = _fallback(attempt, e)
            if on_fallback:
                on_fallback(attempt)

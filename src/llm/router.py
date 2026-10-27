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
    _cooldown.clear()


def model_name(provider: str) -> str:
    return config.GEMINI_MODEL if provider == "gemini" else config.NVIDIA_MODEL


def provider_available(provider: str) -> bool:
    if provider in _test_models or any(k.startswith(provider + ":") for k in _test_models):
        return True
    return bool(config.GOOGLE_API_KEY if provider == "gemini" else config.NVIDIA_API_KEY)


def get_chat(provider: str, model: str | None = None):
    model = model or model_name(provider)
    if f"{provider}:{model}" in _test_models:          # tests can fake one specific model
        return _test_models[f"{provider}:{model}"]
    if provider in _test_models:
        return _test_models[provider]
    key = (provider, model)
    if key in _cache:
        return _cache[key]
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.2,
            timeout=config.LLM_TIMEOUT_S,
            max_retries=0,            # quota errors are handled by the model chain below, not by waiting
        )
    else:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            api_key=config.NVIDIA_API_KEY,
            base_url=config.NVIDIA_BASE_URL,
            temperature=0.3,
            timeout=config.LLM_TIMEOUT_S,
            max_retries=1,
            stream_usage=True,
        )
    _cache[key] = llm
    return llm


# ---- Fallback chain ------------------------------------------------------------------------------
# Quota / overload errors (429, 503) on one Gemini model -> try the next Gemini model first (each model
# has its own free-tier quota), so the provider choice made by decide() is preserved when possible.
# Any other error, or all Gemini models exhausted -> the other provider. Exhausted models are skipped
# for COOLDOWN_S so later calls don't pay for the same 429 again.
COOLDOWN_S = 60.0
_cooldown: dict[str, float] = {}


def _is_capacity_error(err: Exception) -> bool:
    t = f"{type(err).__name__} {err}".lower()
    return any(k in t for k in ("429", "resource_exhausted", "ratelimit", "rate limit", "quota", "503", "unavailable",
                                "high demand", "overloaded"))


def _models_for(provider: str) -> list[str]:
    if provider == "gemini":
        return list(dict.fromkeys([config.GEMINI_MODEL, *config.GEMINI_FALLBACK_MODELS]))
    return [config.NVIDIA_MODEL]


def _chain(decision: "RouteDecision") -> list[tuple[str, str]]:
    first, other = decision.provider, _other(decision.provider)
    chain = [(first, m) for m in _models_for(first)]
    if provider_available(other):
        chain += [(other, m) for m in _models_for(other)]
    now = time.time()
    live = [pm for pm in chain if _cooldown.get(pm[1], 0) <= now]
    return live or chain[:1]


def _retry_decision(d: "RouteDecision", provider: str, model: str, err: Exception) -> "RouteDecision":
    why = f"{d.model} failed ({type(err).__name__}: {str(err)[:120]})"
    if provider == d.provider:
        return RouteDecision(d.role, provider, model, d.dimension,
                             f"{d.reason} [{why}; retried on {model}, same provider]", d.signals)
    return RouteDecision(d.role, provider, model, "fallback", f"{why}; retried on {provider} ({model}).", d.signals)


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
    chain = _chain(decision)
    attempt = decision if (decision.provider, decision.model) == chain[0] else \
        RouteDecision(decision.role, chain[0][0], chain[0][1], decision.dimension, decision.reason, decision.signals)
    for i, (provider, model) in enumerate(chain):
        llm = get_chat(provider, model)
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
        except Exception as e:  # noqa: BLE001 - any provider error moves down the chain
            if _is_capacity_error(e):
                _cooldown[model] = time.time() + COOLDOWN_S
            if i == len(chain) - 1:
                raise
            nxt = chain[i + 1]
            if nxt[0] == provider and not _is_capacity_error(e):
                # not a quota problem: another model of the same provider won't help, jump providers
                rest = [pm for pm in chain[i + 1:] if pm[0] != provider]
                if not rest:
                    raise
                nxt = rest[0]
                chain[i + 1:] = rest
            attempt = _retry_decision(attempt, nxt[0], nxt[1], e)
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
    chain = _chain(decision)
    attempt = decision if (decision.provider, decision.model) == chain[0] else \
        RouteDecision(decision.role, chain[0][0], chain[0][1], decision.dimension, decision.reason, decision.signals)
    for i, (provider, model) in enumerate(chain):
        llm = get_chat(provider, model)
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
            if _is_capacity_error(e):
                _cooldown[model] = time.time() + COOLDOWN_S
            if ttft is not None or i == len(chain) - 1:
                raise
            nxt = chain[i + 1]
            if nxt[0] == provider and not _is_capacity_error(e):
                rest = [pm for pm in chain[i + 1:] if pm[0] != provider]
                if not rest:
                    raise
                nxt = rest[0]
                chain[i + 1:] = rest
            attempt = _retry_decision(attempt, nxt[0], nxt[1], e)
            if on_fallback:
                on_fallback(attempt)

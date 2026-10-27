"""End-to-end orchestration tests (offline). Each maps to a rubric line in docs/architecture.md."""
import asyncio

import pytest

from src.agents.tools import parse_web_results
from src.data import db
from src.graph.graph import GRAPH
from src.llm import router
from src.llm.router import _ThinkFilter


def run(query, **extra):
    state = {"session_id": "t", "query": query, "history": extra.pop("history", []), "completed": [], "failed": [],
             "notes": [], "citations": [], "steps": 0, "plan": None, **extra}

    async def go():
        events = []
        async for chunk in GRAPH.astream(state, stream_mode="custom"):
            events.append(chunk)
        return events
    return asyncio.run(go())


def kinds(events, ev="trace"):
    return [e["data"]["kind"] for e in events if e["event"] == ev]


def answer(events):
    return "".join(e["data"]["text"] for e in events if e["event"] == "token")


def test_research_path_streams_with_citations_and_trace():
    ev = run("How does RAG combine retrieval and generation?")
    k = kinds(ev)
    assert k[:2] == ["llm_route", "plan"]
    assert "rag_retrieval" in k and "agent_route" in k and "respond_done" in k
    cites = [e["data"] for e in ev if e["event"] == "citation"]
    assert cites and cites[0]["kind"] == "corpus" and "#c" in cites[0]["id"] and cites[0]["url"].startswith("https://arxiv.org/abs/")
    assert len([e for e in ev if e["event"] == "token"]) > 3          # streamed in pieces, not one blob
    assert "[1]" in answer(ev)


def test_mcp_tool_shared_by_research_and_summary(offline_env):
    for q in ("What is the latest news on RAG?", "Summarize the ReAct paper and its follow-up work"):
        ev = run(q)
        assert "worker_error" not in kinds(ev), [e for e in ev if e["data"].get("kind") == "worker_error"]
    agents_using_mcp = {c["query"] for c in offline_env["mcp"].calls}
    assert len(offline_env["mcp"].calls) == 2 and len(agents_using_mcp) == 2


def test_mcp_trace_names_agent():
    ev = run("What is the latest news on RAG?")
    mcp = [e["data"] for e in ev if e["event"] == "trace" and e["data"]["kind"] == "mcp_tool_call"]
    assert mcp and mcp[0]["node"] == "research" and mcp[0]["server"] == "tavily"
    ev2 = run("Summarize the ReAct paper")
    mcp2 = [e["data"] for e in ev2 if e["event"] == "trace" and e["data"]["kind"] == "mcp_tool_call"]
    assert mcp2 and mcp2[0]["node"] == "summary"


def test_data_agent_recovers_from_sql_error():
    ev = run("How many papers are there? BADSQL")
    sql = [e["data"] for e in ev if e["event"] == "trace" and e["data"]["kind"] == "sql_query"]
    assert len(sql) == 2 and sql[0]["error"] and not sql[1]["error"]


def test_worker_failure_does_not_crash_graph():
    ev = run("Compare RAG and ReAct", force_fail="research")
    k = kinds(ev)
    assert "worker_error" in k
    routes = [e["data"] for e in ev if e["event"] == "trace" and e["data"]["kind"] == "agent_route"]
    assert routes[-1]["next"] == "respond"
    assert any("failed" in r["reason"] for r in routes)
    assert answer(ev)                                                  # still answers


def test_routing_varies_by_query():
    simple = run("How does RAG work?")
    complex_ = run("Compare RAG and ReAct")
    prov = lambda ev: [e["data"]["provider"] for e in ev if e["event"] == "trace"
                       and e["data"]["kind"] == "llm_route" and e["data"]["node"] == "respond"][0]
    assert prov(simple) == "gemini" and prov(complex_) == "nvidia"


def test_provider_fallback_is_traced(offline_env):
    offline_env["gemini"].fail = True
    ev = run("How does RAG work?")
    routes = [e["data"] for e in ev if e["event"] == "trace" and e["data"]["kind"] == "llm_route"]
    assert any(r["dimension"] == "fallback" and r["provider"] == "nvidia" for r in routes)
    assert "nvidia" in answer(ev)


def test_both_providers_down_still_returns_message(offline_env):
    offline_env["gemini"].fail = True
    offline_env["nvidia"].fail = True
    ev = run("How does RAG work?")
    assert "error" in [e["event"] for e in ev]
    assert "couldn't generate" in answer(ev)


def test_greeting_skips_workers():
    ev = run("hi")
    routes = [e["data"] for e in ev if e["event"] == "trace" and e["data"]["kind"] == "agent_route"]
    assert routes[0]["next"] == "respond"


def test_router_decisions_are_pure_and_documented():
    assert router.decide("supervisor").provider == "gemini"
    assert router.decide("respond", complexity="complex").provider == "nvidia"
    assert router.decide("respond", n_sources=2).dimension == "capability"
    assert router.decide("respond", context_chars=9000).provider == "nvidia"
    assert router.decide("respond", context_chars=100).provider == "gemini"


def test_sql_tool_is_read_only():
    with pytest.raises(db.SQLError):
        db.run_sql("DELETE FROM papers")
    with pytest.raises(db.SQLError):
        db.run_sql("SELECT 1; DROP TABLE papers")
    assert db.run_sql("SELECT COUNT(*) AS n FROM papers")[0]["n"] == 3


def test_think_filter_across_chunks():
    f = _ThinkFilter()
    out = "".join(f.feed(x) for x in ["Hel", "lo <thi", "nk>", "secret", "</thi", "nk> world", " <think>x</think>!"])
    assert out == "Hello  world !"
    g = _ThinkFilter()
    assert "".join(g.feed(x) for x in ["a < b", " and 3<4"]) == "a < b and 3<4"


def test_parse_web_results_formats():
    j = parse_web_results('{"results":[{"title":"T","url":"https://a.b","content":"c"}]}', "research")
    assert j[0]["url"] == "https://a.b"
    t = parse_web_results("Title: X\nURL: https://x.y\nContent: hello\n\nTitle: Y\nURL: https://y.z\nContent: bye", "summary")
    assert [r["url"] for r in t] == ["https://x.y", "https://y.z"]


def test_gemini_quota_rotates_to_another_gemini_model(offline_env):
    """429 on the primary Gemini model -> next Gemini model (same provider), not NVIDIA."""
    from src import config
    from tests.fakes import FakeChat, default_script

    class Quota(FakeChat):
        def _check(self):
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded for gemini-3.5-flash-lite")

    backup = FakeChat(name_tag="gemini-backup", script=default_script("gemini-backup"), calls=[])
    router.set_test_models({"gemini": Quota(name_tag="g", script=default_script("g"), calls=[]),
                            f"gemini:{config.GEMINI_FALLBACK_MODELS[0]}": backup,
                            "nvidia": offline_env["nvidia"]})
    ev = run("How does RAG work?")
    routes = [e["data"] for e in ev if e["event"] == "trace" and e["data"]["kind"] == "llm_route"]
    assert any(r["provider"] == "gemini" and r["model"] == config.GEMINI_FALLBACK_MODELS[0] and "same provider" in r["reason"]
               for r in routes)
    assert "gemini-backup" in answer(ev)
    assert not any(r["dimension"] == "fallback" for r in routes)        # never had to leave Gemini


def test_all_gemini_models_exhausted_falls_back_to_nvidia(offline_env):
    from tests.fakes import FakeChat, default_script

    class Quota(FakeChat):
        def _check(self):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    router.set_test_models({"gemini": Quota(name_tag="g", script=default_script("g"), calls=[]),
                            "nvidia": offline_env["nvidia"]})
    ev = run("How does RAG work?")
    routes = [e["data"] for e in ev if e["event"] == "trace" and e["data"]["kind"] == "llm_route"]
    assert any(r["dimension"] == "fallback" and r["provider"] == "nvidia" for r in routes)
    assert "nvidia" in answer(ev)

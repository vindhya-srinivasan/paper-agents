"""SSE endpoint and durable-history behaviour through the real FastAPI app (offline fakes)."""
import json

from fastapi.testclient import TestClient


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def client():
    import main
    return TestClient(main.app)


def test_chat_streams_sse_and_saves_history():
    with client() as c:
        r = c.post("/api/chat", json={"query": "How does RAG work?", "session_id": "s1"})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        ev = parse_sse(r.text)
        names = [e for e, _ in ev]
        assert names[0] == "trace" and names[-1] == "done"
        assert "token" in names and "citation" in names
        done = ev[-1][1]
        assert done["timings_ms"]["ttft"] is not None and done["timings_ms"]["total"] >= done["timings_ms"]["ttft"]
        assert done["usage"]["input_tokens"] > 0 and "respond" in done["providers"]

        h = c.get("/api/history", params={"session_id": "s1"}).json()
        assert [t["role"] for t in h["turns"]] == ["user", "assistant"]

        r2 = c.post("/api/chat", json={"query": "And the second one?", "session_id": "s1"})
        loaded = [d for e, d in parse_sse(r2.text) if e == "trace" and d["kind"] == "history_loaded"][0]
        assert loaded["turns"] == 2

        c.post("/api/reset", json={"session_id": "s1"})
        assert c.get("/api/history", params={"session_id": "s1"}).json()["turns"] == []


def test_sessions_are_isolated():
    with client() as c:
        c.post("/api/chat", json={"query": "How does RAG work?", "session_id": "a"})
        assert c.get("/api/history", params={"session_id": "b"}).json()["turns"] == []


def test_validation_and_static():
    with client() as c:
        assert c.post("/api/chat", json={"query": "", "session_id": "x"}).status_code == 422
        assert c.post("/api/chat", json={"query": "q", "session_id": "../etc"}).status_code == 422
        assert c.post("/api/chat", json={"query": "q", "session_id": "x", "debug_fail_agent": "boom"}).status_code == 400
        assert c.get("/").status_code == 200
        assert c.get("/healthz").json()["status"] == "ok"


def test_forced_failure_via_api():
    with client() as c:
        ev = parse_sse(c.post("/api/chat", json={"query": "How does RAG work?", "session_id": "f",
                                                 "debug_fail_agent": "research"}).text)
        assert any(e == "trace" and d["kind"] == "worker_error" for e, d in ev)
        assert ev[-1][0] == "done"

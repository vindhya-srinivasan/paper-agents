// Vanilla JS client: POSTs to /api/chat and consumes the text/event-stream with fetch + ReadableStream.
// SSE event names: token, trace, citation, error, done.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const messages = $("messages"), form = $("form"), queryEl = $("query"), sendBtn = $("send");
  const panels = { routing: $("panel-routing"), trace: $("panel-trace"), sources: $("panel-sources") };
  const failAgent = new URLSearchParams(location.search).get("fail"); // test hook: ?fail=research
  let sessionId = loadSession();
  let turnNo = 0, busy = false;

  function loadSession() {
    let id = null;
    try { id = localStorage.getItem("pa_session"); } catch (_) {}
    if (!id) { id = "s-" + Math.random().toString(36).slice(2, 12); saveSession(id); }
    return id;
  }
  function saveSession(id) { try { localStorage.setItem("pa_session", id); } catch (_) {} }

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const clock = (t) => new Date(t * 1000).toLocaleTimeString([], { minute: "2-digit", second: "2-digit" });

  // Minimal, safe markdown: escape first, then paragraphs, bullets, **bold**, `code`, [n] citation links.
  function render(text) {
    const blocks = esc(text).split(/\n{2,}/).map((b) => {
      const lines = b.split("\n");
      if (lines.every((l) => /^\s*([-*]|\d+\.)\s+/.test(l))) {
        return "<ul>" + lines.map((l) => "<li>" + l.replace(/^\s*([-*]|\d+\.)\s+/, "") + "</li>").join("") + "</ul>";
      }
      return "<p>" + lines.join("<br>") + "</p>";
    }).join("");
    return blocks
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[(\d{1,2})\]/g, '<a class="cite" href="#src-$1" data-n="$1">[$1]</a>');
  }

  function addMsg(role, html) {
    $("empty")?.remove();
    const el = document.createElement("div");
    el.className = "msg " + role;
    el.innerHTML = html;
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
    return el;
  }

  function showTab(name) {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
    Object.entries(panels).forEach(([k, p]) => p.classList.toggle("active", k === name));
  }
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
  messages.addEventListener("click", (e) => {
    const a = e.target.closest(".cite");
    if (!a) return;
    e.preventDefault();
    showTab("sources");
    const card = $("src-" + a.dataset.n);
    if (card) { card.scrollIntoView({ block: "nearest" }); card.classList.add("hl"); setTimeout(() => card.classList.remove("hl"), 1500); }
  });

  // ---------- inspector panels ----------
  function providerBadge(d) {
    return `<span class="badge ${esc(d.provider)}">${esc(d.provider)}</span><span class="badge dim">${esc(d.model)}</span>` +
      `<span class="badge ${esc(d.dimension)}">${esc(d.dimension)}</span>`;
  }

  function newTurn(q) {
    turnNo += 1;
    panels.routing.querySelector(".hint")?.remove();
    const el = document.createElement("div");
    el.className = "turn";
    el.innerHTML = `<h3>Turn ${turnNo}</h3><div class="q">${esc(q)}</div><div class="plan"></div><div class="routes"></div>`;
    panels.routing.prepend(el);
    panels.trace.innerHTML = "";
    panels.sources.innerHTML = '<p class="hint">Waiting for sources…</p>';
    $("traceCount").textContent = "0"; $("srcCount").textContent = "0";
    return el;
  }

  function onTrace(d, turnEl) {
    const count = $("traceCount"); count.textContent = String(+count.textContent + 1);
    if (d.kind === "plan") {
      turnEl.querySelector(".plan").innerHTML = "<strong>Plan:</strong> " +
        (d.plan.length ? d.plan.map((a) => `<span class="agent">${esc(a)}</span>`).join("→ ") : "<em>answer directly</em>") +
        ` <span class="badge dim">${esc(d.complexity)}</span><div class="d">${esc(d.reason)}</div>`;
    } else if (d.kind === "llm_route") {
      const r = document.createElement("div");
      r.className = "route";
      r.innerHTML = `<div class="role">${esc(d.node)}</div><div>${providerBadge(d)}<div>${esc(d.reason)}</div></div>`;
      turnEl.querySelector(".routes").appendChild(r);
    } else if (d.kind === "agent_route" && d.next !== "respond") {
      const r = document.createElement("div");
      r.className = "route";
      r.innerHTML = `<div class="role">supervisor</div><div>→ <span class="agent">${esc(d.next)}</span> ${esc(d.reason)}</div>`;
      turnEl.querySelector(".routes").appendChild(r);
    } else if (d.kind === "worker_error") {
      const r = document.createElement("div");
      r.className = "route";
      r.innerHTML = `<div class="role">${esc(d.node)}</div><div><span class="badge fallback">failed</span>${esc(d.error)}</div>`;
      turnEl.querySelector(".routes").appendChild(r);
    }
    panels.trace.appendChild(traceRow(d));
    panels.trace.scrollTop = panels.trace.scrollHeight;
  }

  function summarize(d) {
    switch (d.kind) {
      case "request": return `instance ${d.instance_id} · rev ${d.revision} · store: ${d.state_store}`;
      case "history_loaded": return `${d.turns} prior turns loaded from the session store`;
      case "plan": return `plan [${d.plan.join(", ")}] · ${d.complexity} · ${d.reason}`;
      case "agent_route": return `next → ${d.next} · ${d.reason}`;
      case "llm_route": return `${d.node}: ${d.provider} (${d.model}) · ${d.dimension} · ${d.reason}`;
      case "agent_start": return `${d.node} agent started`;
      case "agent_end": return `${d.node} agent finished in ${d.ms} ms · ${d.citations} sources`;
      case "rag_retrieval": return `${d.node} → search_corpus("${d.query}") k=${d.k} · ${d.ms} ms · ${(d.chunk_ids || []).join(", ")}`;
      case "paper_read": return `${d.node} → read_paper("${d.paper}") · ${d.arxiv_id || "not found"} · ${d.ms} ms`;
      case "mcp_tool_call": return `${d.node} → MCP ${d.server}/${d.mcp_tool || "search"}("${d.query}") · ${d.ms} ms` + (d.error ? ` · ERROR ${d.error}` : ` · ${(d.urls || []).length} results`);
      case "sql_query": return `${d.node} → run_sql: ${d.sql}` + (d.error ? ` · ERROR ${d.error}` : ` · ${d.rows} rows`);
      case "worker_error": return `${d.node} FAILED: ${d.error} (graph continues)`;
      case "respond_done": return `answer streamed by ${d.provider} · TTFT ${d.ttft_ms} ms · ${d.ms} ms`;
      default: return d.message || "";
    }
  }

  function traceRow(d) {
    const el = document.createElement("div");
    el.className = "ev";
    const { t, kind, ...rest } = d;
    el.innerHTML = `<div class="t">${t ? clock(t) : ""}</div><div><span class="k ${esc(kind)}">${esc(kind)}</span>` +
      `<span class="d">${esc(summarize(d))}</span><details><summary>state</summary><pre>${esc(JSON.stringify(rest, null, 2))}</pre></details></div>`;
    return el;
  }

  function onCitation(c) {
    if (panels.sources.querySelector(".hint")) panels.sources.innerHTML = "";
    const count = $("srcCount"); count.textContent = String(+count.textContent + 1);
    const el = document.createElement("div");
    el.className = "src"; el.id = "src-" + c.n;
    const idLine = c.kind === "corpus" ? `arXiv:${esc(c.arxiv_id)} · chunk ${esc(c.id)} · ${esc(c.year)} · score ${esc(c.score)} · via ${esc(c.agent)}`
                                       : `web · ${esc(new URL(c.url).hostname)} · via ${esc(c.agent)}`;
    el.innerHTML = `<div><span class="n">[${c.n}]</span><a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.title)}</a></div>` +
      `<div class="id">${idLine}</div><div class="snip">${esc(c.snippet)}</div>`;
    panels.sources.appendChild(el);
  }

  // ---------- SSE over fetch ----------
  async function ask(q) {
    busy = true; sendBtn.disabled = true;
    addMsg("user", esc(q));
    const bot = addMsg("assistant", '<div class="body cursor"></div><div class="meta"></div>');
    const body = bot.querySelector(".body"), meta = bot.querySelector(".meta");
    const turnEl = newTurn(q);
    let text = "";
    try {
      const resp = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, session_id: sessionId, ...(failAgent ? { debug_fail_agent: failAgent } : {}) }),
      });
      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
      const reader = resp.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, i); buf = buf.slice(i + 2);
          let ev = "message", data = "";
          for (const line of frame.split("\n")) {
            if (line.startsWith("event:")) ev = line.slice(6).trim();
            else if (line.startsWith("data:")) data += line.slice(5).trim();
          }
          if (!data) continue;
          const d = JSON.parse(data);
          if (ev === "token") { text += d.text; body.innerHTML = render(text); messages.scrollTop = messages.scrollHeight; }
          else if (ev === "trace") onTrace(d, turnEl);
          else if (ev === "citation") onCitation(d);
          else if (ev === "error") { bot.insertAdjacentHTML("beforeend", `<div class="error">${esc(d.message)}</div>`); onTrace({ kind: "error", ...d }, turnEl); }
          else if (ev === "done") {
            const tm = d.timings_ms;
            const prov = [...new Set(Object.values(d.providers || {}).map((p) => p.split(":")[0]))].join(" + ");
            meta.innerHTML = `<span>TTFT ${Math.round(tm.ttft || 0)} ms</span><span>total ${Math.round(tm.total)} ms</span>` +
              `<span>retrieval ${Math.round(tm.retrieval)} ms</span><span>LLM ${Math.round(tm.llm)} ms</span>` +
              `<span>${d.usage.input_tokens}+${d.usage.output_tokens} tok</span><span>$${d.cost_usd.toFixed(5)}</span><span>${esc(prov)}</span>`;
            if (panels.sources.querySelector(".hint")) panels.sources.innerHTML = '<p class="hint">No sources for this turn.</p>';
          }
        }
      }
    } catch (err) {
      bot.insertAdjacentHTML("beforeend", `<div class="error">Request failed: ${esc(err.message)}. If the service was idle it may be cold-starting; try again.</div>`);
    } finally {
      body.classList.remove("cursor");
      if (!text) body.innerHTML = "<p><em>No answer.</em></p>";
      busy = false; sendBtn.disabled = false; queryEl.focus();
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = queryEl.value.trim();
    if (!q || busy) return;
    queryEl.value = "";
    ask(q);
  });
  queryEl.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });
  document.querySelectorAll(".ex").forEach((b) => b.addEventListener("click", () => { if (!busy) ask(b.textContent); }));

  $("reset").addEventListener("click", async () => {
    if (busy) return;
    try { await fetch("/api/reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: sessionId }) }); } catch (_) {}
    sessionId = "s-" + Math.random().toString(36).slice(2, 12); saveSession(sessionId);
    location.reload();
  });

  // Restore this session's history from the durable store (shows it survives restarts).
  async function getJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url} returned HTTP ${r.status}`);
    return r.json();
  }

  // Health and history are loaded independently so one failure never hides the other.
  async function init() {
    let h = {};
    try {
      h = await getJSON("/health");
      $("status").innerHTML = `instance ${esc(h.instance_id)} · ${esc(h.revision)}<br>session ${esc(sessionId)}`;
    } catch (e) {
      $("status").innerHTML = `status unavailable<br>session ${esc(sessionId)}`;
    }
    try {
      const hist = await getJSON("/api/history?session_id=" + encodeURIComponent(sessionId));
      for (const t of hist.turns) addMsg(t.role, t.role === "user" ? esc(t.content) : `<div class="body">${render(t.content)}</div>`);
      if (hist.turns.length) {
        panels.routing.innerHTML = `<p class="hint">Restored ${hist.turns.length / 2} earlier turn(s) of session ${esc(sessionId)} ` +
          `from the durable store (${esc(h.state_store || "server")}), served by instance ${esc(hist.instance_id)}.</p>`;
      }
    } catch (e) {
      panels.routing.innerHTML = `<p class="hint">Could not load earlier turns: ${esc(e.message)}</p>`;
    }
  }
  init();
})();

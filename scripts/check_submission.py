"""Pre-submission audit. Run it on the EXACT zip you will upload:

    python scripts/check_submission.py submission.zip
    bash scripts/make_zip.sh    (builds submission.zip from git-tracked files and runs this audit on it)

Checks packaging (no secrets/artifacts/junk, ignore files at root), pins, required docs and their
required sections, unfilled placeholders, and PROCESS_LOG length. Exit code 1 if anything fails.
"""
from __future__ import annotations

import py_compile
import re
import sys
import tempfile
import zipfile
from pathlib import Path

FORBIDDEN = [
    (r"(^|/)\.env$", ".env file"),
    (r"__pycache__/|\.pyc$", "Python bytecode"),
    (r"(^|/)\.DS_Store$", ".DS_Store"),
    (r"(^|/)(\.?venv|env)/", "virtualenv"),
    (r"(^|/)\.pytest_cache/", "pytest cache"),
    (r"(^|/)chroma_index/", "generated vector index"),
    (r"papers\.db$", "generated SQLite DB"),
    (r"\.pdf$", "PDF (raw corpus downloads)"),
    (r"(^|/)node_modules/", "node_modules"),
    (r"(^|/)\.git/", ".git directory"),
    (r"(^|/)\.idea/|(^|/)\.vscode/", "editor folder"),
    (r"(^|/)__MACOSX/", "macOS zip metadata"),
]
SECRETS = [
    (r"AIza[0-9A-Za-z_\-]{35}", "Google API key"),
    (r"nvapi-[A-Za-z0-9_\-]{30,}", "NVIDIA API key"),
    (r"tvly-(?:dev-)?[A-Za-z0-9]{20,}", "Tavily API key"),
    (r"sk-[A-Za-z0-9_\-]{30,}", "OpenAI-style key"),
    (r"-----BEGIN (RSA )?PRIVATE KEY-----", "private key"),
]
REQUIRED = ["main.py", "Dockerfile", "deploy.sh", "requirements.txt", ".env.example", "README.md", "PROCESS_LOG.md",
            "docs/architecture.md", "docs/evaluation.md", "static/index.html", "static/app.js", ".gitignore",
            ".dockerignore", ".gcloudignore", "data/papers.json", "src/eval/questions.json"]

results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((bool(ok), msg))


def main(target: str) -> int:
    p = Path(target)
    if p.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            z.extractall(tmp)
        tops = {n.split("/")[0] for n in names if "/" in n}
        root = tmp / next(iter(tops)) if len(tops) == 1 and not (tmp / "main.py").exists() else tmp
        check(p.stat().st_size < 60_000_000, f"zip size {p.stat().st_size / 1e6:.1f} MB (< 60 MB)")
    else:
        root = p
    files = [f for f in root.rglob("*") if f.is_file()]
    rel = [str(f.relative_to(root)).replace("\\", "/") for f in files]

    # Packaging
    for pat, label in FORBIDDEN:
        bad = [r for r in rel if re.search(pat, r)]
        check(not bad, f"no {label}" + (f": {bad[:3]}" if bad else ""))
    for req in REQUIRED:
        check(req in rel, f"has {req} (at repo root level)" if "/" not in req else f"has {req}")
    corpus = [r for r in rel if r.startswith("data/corpus/") and r.endswith(".txt")]
    check(len(corpus) >= 40, f"corpus text files: {len(corpus)} (>= 40)")

    # Secrets in any text file
    leaks = []
    for f, r in zip(files, rel):
        if f.suffix in {".png", ".jpg", ".zip"} or r.startswith("data/corpus/"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pat, label in SECRETS:
            if re.search(pat, text):
                leaks.append(f"{label} in {r}")
    check(not leaks, "no API keys / secrets in files" + (f": {leaks}" if leaks else ""))

    # Pins
    reqs = [line.strip() for line in (root / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")]
    loose = [line for line in reqs if "==" not in line or re.search(r"[<>~!]=?", line)]
    check(not loose, f"requirements.txt: all {len(reqs)} lines pinned with ==" + (f"; loose: {loose}" if loose else ""))

    env = (root / ".env.example").read_text()
    for key in ("GOOGLE_API_KEY", "GEMINI_MODEL", "NVIDIA_API_KEY", "NVIDIA_MODEL", "TAVILY_API_KEY", "GCP_PROJECT_ID"):
        check(re.search(rf"^{key}=", env, re.M), f".env.example has {key}")
    check("<DATE YOU RAN IT>" not in env, ".env.example: model verification date filled in")

    # Placeholders anywhere in docs/code
    holders = []
    for f, r in zip(files, rel):
        if f.suffix in {".md", ".py", ".html", ".js", ".example", ".sh"} and r != "scripts/check_submission.py":
            t = f.read_text(encoding="utf-8", errors="ignore")
            if "⟪FILL" in t or "<CLOUD RUN URL>" in t or re.search(r"\bTODO\b", t):
                holders.append(r)
    check(not holders, "no unfilled ⟪FILL⟫ / TODO / <CLOUD RUN URL> placeholders" + (f": {holders}" if holders else ""))

    # PROCESS_LOG
    pl = (root / "PROCESS_LOG.md").read_text()
    for h in ("AI tools used", "Development narrative", "Verification narrative", "What I learned"):
        check(re.search(rf"^#+\s*{h}", pl, re.M | re.I), f"PROCESS_LOG has section '{h}'")
    body = "\n".join(line for line in pl.splitlines() if not line.lstrip().startswith("#"))
    words = len(re.findall(r"\b[\w'’-]+\b", body))
    check(200 <= words <= 300, f"PROCESS_LOG body is {words} words (must be 200-300)")

    # architecture.md required content
    arch = (root / "docs/architecture.md").read_text().lower()
    for needle, what in [("```mermaid", "Mermaid diagram"), ("state schema", "state schema"),
                         ("tools", "tool access per agent"), ("routing", "LLM routing table"),
                         ("chunk", "chunking strategy"), ("k = 5", "retrieval k"),
                         ("prompt template", "augmentation prompt template"), ("durable", "durable-state design"),
                         ("mcp", "MCP tool-sharing rationale"), ("model / provider", "model/provider per agent")]:
        check(needle in arch, f"architecture.md covers {what}")

    # evaluation.md required content
    ev = (root / "docs/evaluation.md").read_text()
    for m in ("Faithfulness", "Answer relevancy", "Context precision", "Context recall"):
        row = re.search(rf"\|\s*{m}\s*\|[^\n]*\|\s*\**([0-9]*\.[0-9]+)\**\s*\|", ev)
        check(row, f"evaluation.md reports a number for {m}")
    evl = ev.lower()
    for needle, what in [("time to first token", "TTFT"), ("cold", "cold-start latency"), ("warm", "warm latency"),
                         ("retrieval", "retrieval latency split"), ("cost per query", "cost per query"),
                         ("provider comparison", "provider comparison")]:
        check(needle in evl, f"evaluation.md covers {what}")

    # README required content
    rd = (root / "README.md").read_text().lower()
    for needle, what in [("local development", "local run instructions"), ("deploy", "deploy instructions"),
                         ("provider requirements", "provider requirements"), ("per-query", "per-query cost estimate"),
                         ("monitor your usage", "instruction to monitor API usage"), ("https://", "live URL / links")]:
        check(needle in rd, f"README covers {what}")
    check(re.search(r"https://[a-z0-9-]+\.[a-z0-9-]+\.run\.app|https://[a-z0-9-]+-[a-z0-9]+\.a\.run\.app", rd),
          "README contains the live *.run.app URL")

    # Dockerfile shape
    df = (root / "Dockerfile").read_text()
    check("python:3.13-slim AS build" in df and df.count("FROM ") >= 2, "Dockerfile: multi-stage python:3.13-slim")
    check("USER appuser" in df, "Dockerfile: non-root user")
    check("${PORT" in df, "Dockerfile: binds Cloud Run $PORT")

    # Code compiles
    bad = []
    for f, r in zip(files, rel):
        if f.suffix == ".py":
            try:
                py_compile.compile(str(f), doraise=True, cfile=str(Path(tempfile.mkdtemp()) / "x.pyc"))
            except py_compile.PyCompileError as e:
                bad.append(f"{r}: {e.msg[:80]}")
    check(not bad, "all .py files compile" + (f": {bad}" if bad else ""))

    for ok, msg in results:
        print(("PASS  " if ok else "FAIL  ") + msg)
    failed = sum(not ok for ok, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} checks passed" + ("" if not failed else " - fix the FAIL lines before submitting"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))

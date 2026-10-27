"""Download the RAG corpus: ~50 arXiv papers on LLM agents and retrieval-augmented generation.

Run once on your machine (needs internet):   python scripts/fetch_corpus.py
Writes:
  data/corpus/<arxiv_id>.txt   extracted full text (references section removed)
  data/papers.json             metadata used for citations and the Data agent's SQLite table
Each ID is checked against an expected title keyword so a wrong ID can never slip into the corpus.
Metadata comes from each paper's arxiv.org/abs page (the export API rate-limits bulk queries with HTTP 429).
Safe to re-run: finished papers are skipped.
"""
from __future__ import annotations

import json
import re
import sys
import time
import ssl
import urllib.error
import urllib.request
from html import unescape
from pathlib import Path

import certifi

# certifi's CA bundle: python.org macOS builds ship without system certificates.
_SSL = ssl.create_default_context(cafile=certifi.where())

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
RAW = ROOT / "data" / "raw_pdfs"

# (arxiv_id, keyword that must appear in the title, topic tag)
PAPERS = [
    # Retrieval-augmented generation
    ("2005.11401", "Retrieval-Augmented Generation", "rag"),
    ("2002.08909", "REALM", "rag"),
    ("2004.04906", "Dense Passage Retrieval", "retrieval"),
    ("2007.01282", "Passage Retrieval", "rag"),
    ("2112.04426", "Retrieving from Trillions", "rag"),
    ("2208.03299", "Atlas", "rag"),
    ("2212.10496", "Zero-Shot Dense Retrieval", "retrieval"),
    ("2307.03172", "Lost in the Middle", "rag"),
    ("2309.15217", "RAGAS", "evaluation"),
    ("2312.10997", "Retrieval-Augmented Generation", "rag"),
    ("2310.11511", "Self-RAG", "rag"),
    ("2401.15884", "Corrective Retrieval", "rag"),
    ("2404.16130", "Local to Global", "rag"),
    ("2305.06983", "Active Retrieval", "rag"),
    ("2212.10509", "Interleaving Retrieval", "rag"),
    ("2210.03350", "Compositionality Gap", "reasoning"),
    ("2112.09332", "WebGPT", "agents"),
    # Reasoning and prompting
    ("2201.11903", "Chain-of-Thought", "reasoning"),
    ("2203.11171", "Self-Consistency", "reasoning"),
    ("2305.10601", "Tree of Thoughts", "reasoning"),
    ("2305.04091", "Plan-and-Solve", "reasoning"),
    ("2310.04406", "Tree Search", "agents"),
    # Tool use and agents
    ("2210.03629", "ReAct", "agents"),
    ("2302.04761", "Toolformer", "tools"),
    ("2205.00445", "MRKL", "tools"),
    ("2305.15334", "Gorilla", "tools"),
    ("2307.16789", "ToolLLM", "tools"),
    ("2303.17580", "HuggingGPT", "agents"),
    ("2303.11366", "Reflexion", "agents"),
    ("2305.16291", "Voyager", "agents"),
    ("2304.03442", "Generative Agents", "agents"),
    ("2310.08560", "MemGPT", "agents"),
    # Multi-agent systems
    ("2308.08155", "AutoGen", "multi-agent"),
    ("2308.00352", "MetaGPT", "multi-agent"),
    ("2303.17760", "CAMEL", "multi-agent"),
    ("2305.14325", "Multiagent Debate", "multi-agent"),
    # Surveys and benchmarks
    ("2308.11432", "Autonomous Agents", "survey"),
    ("2309.07864", "Large Language Model based Agents", "survey"),
    ("2308.03688", "AgentBench", "benchmark"),
    ("2310.06770", "SWE-bench", "benchmark"),
    ("2311.12983", "GAIA", "benchmark"),
    # Foundation models and alignment
    ("1706.03762", "Attention Is All You Need", "foundations"),
    ("2005.14165", "Few-Shot Learners", "foundations"),
    ("2203.02155", "Instructions with Human Feedback", "alignment"),
    ("2212.08073", "Constitutional AI", "alignment"),
    ("2305.18290", "Direct Preference Optimization", "alignment"),
    ("2106.09685", "LoRA", "foundations"),
    ("2210.11416", "Scaling Instruction-Finetuned", "foundations"),
    ("2302.13971", "LLaMA", "foundations"),
    ("2307.09288", "Llama 2", "foundations"),
]

REFS = re.compile(r"\n\s*(References|REFERENCES|Bibliography)\s*\n")


def clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(\w)", r"\1", text)          # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    m = REFS.search(text, pos=len(text) // 3)         # drop the bibliography (noise for retrieval)
    return (text[: m.start()] if m else text).strip()


UA = {"User-Agent": "csci599-a2-corpus-fetch/1.1 (course project; polite, 1 request / 3 s)"}


def http_get(url: str, attempts: int = 4) -> bytes:
    """GET with backoff on 429/5xx. arXiv rate-limits bursts, so waits grow 10s -> 30s -> 60s."""
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120, context=_SSL) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == attempts - 1:
                raise
            wait = int(e.headers.get("Retry-After") or (10, 30, 60, 90)[i])
            print(f"    HTTP {e.code} from arXiv, waiting {wait}s then retrying...", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError):
            if i == attempts - 1:
                raise
            time.sleep(10)
    raise RuntimeError("unreachable")


def _meta(html: str, name: str) -> list[str]:
    """Values of <meta name=... content=...> tags, whatever the attribute order or quote style."""
    out = []
    for tag in re.findall(r"<meta\b[^>]*>", html, re.I):
        attrs = {k: v1 or v2 for k, v1, v2 in re.findall(r"""(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)')""", tag)}
        if attrs.get("name") == name and "content" in attrs:
            out.append(unescape(attrs["content"]))
    return out


def parse_abs_page(html: str, aid: str) -> dict:
    """Metadata from the arxiv.org/abs page's citation <meta> tags (no export API needed)."""
    title = (_meta(html, "citation_title") or [""])[0].strip()
    authors = []
    for a in _meta(html, "citation_author"):
        last, _, first = a.partition(", ")
        authors.append(f"{first} {last}".strip() if first else a.strip())
    date = (_meta(html, "citation_date") or ["1900/01/01"])[0].replace("/", "-")
    abstract = " ".join((_meta(html, "citation_abstract") or [""])[0].split())
    cat = re.search(r'class="primary-subject">[^<]*\(([^)]+)\)', html)
    return {"arxiv_id": aid, "title": title, "authors": authors, "first_author": authors[0] if authors else "",
            "year": int(date[:4]), "published": date[:10], "primary_category": cat.group(1) if cat else "",
            "abstract": abstract, "url": f"https://arxiv.org/abs/{aid}"}


def main() -> int:
    CORPUS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    meta_cache = RAW / "meta"
    meta_cache.mkdir(exist_ok=True)
    meta, skipped = [], []
    for n, (aid, keyword, topic) in enumerate(PAPERS, 1):
        try:
            cache = meta_cache / f"{aid}.json"          # resume: never re-fetch what we already have
            if cache.exists():
                m = json.loads(cache.read_text())
            else:
                m = parse_abs_page(http_get(f"https://arxiv.org/abs/{aid}").decode("utf-8", "replace"), aid)
                time.sleep(3)
            if keyword.lower() not in m["title"].lower():
                skipped.append(f"{aid}: title '{m['title']}' does not contain '{keyword}'")
                continue
            cache.write_text(json.dumps(m))
            txt_path = CORPUS / f"{aid}.txt"
            if not txt_path.exists():
                pdf_path = RAW / f"{aid}.pdf"
                if not pdf_path.exists():
                    data = http_get(f"https://arxiv.org/pdf/{aid}")
                    if not data.startswith(b"%PDF"):
                        raise ValueError("arXiv did not return a PDF")
                    pdf_path.write_bytes(data)
                    time.sleep(3)
                pages = [(p.extract_text() or "") for p in PdfReader(str(pdf_path)).pages]
                txt_path.write_text(clean("\n".join(pages)), encoding="utf-8")
        except Exception as e:  # noqa: BLE001 - one bad paper must not stop the rest
            skipped.append(f"{aid}: {type(e).__name__}: {str(e)[:120]}")
            print(f"FAIL {n:2d}/{len(PAPERS)} {aid}  {skipped[-1]}", flush=True)
            continue
        meta.append({**m, "topic": topic, "chars": len(txt_path.read_text(encoding="utf-8"))})
        print(f"ok   {n:2d}/{len(PAPERS)} {aid}  {m['title'][:65]}", flush=True)

    meta.sort(key=lambda x: x["arxiv_id"])
    (ROOT / "data" / "papers.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n{len(meta)} papers written to data/corpus/, metadata in data/papers.json")
    for s in skipped:
        print("SKIPPED", s)
    if skipped:
        print("Re-run the script to retry skipped papers; finished ones are not downloaded again.")
    return 0 if len(meta) >= 40 else 1


if __name__ == "__main__":
    sys.exit(main())

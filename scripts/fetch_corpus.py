"""Download the RAG corpus: ~50 arXiv papers on LLM agents and retrieval-augmented generation.

Run once on your machine (needs internet):   python scripts/fetch_corpus.py
Writes:
  data/corpus/<arxiv_id>.txt   extracted full text (references section removed)
  data/papers.json             metadata used for citations and the Data agent's SQLite table
Each ID is checked against an expected title keyword so a wrong ID can never slip into the corpus.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import arxiv
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


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "csci599-a2-corpus-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    if not data.startswith(b"%PDF"):
        raise ValueError(f"{url} did not return a PDF")
    dest.write_bytes(data)


def main() -> int:
    CORPUS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=5)
    ids = [p[0] for p in PAPERS]
    expect = {p[0]: (p[1], p[2]) for p in PAPERS}
    meta, skipped = [], []
    for r in client.results(arxiv.Search(id_list=ids, max_results=len(ids))):
        aid = re.sub(r"v\d+$", "", r.get_short_id())
        keyword, topic = expect[aid]
        if keyword.lower() not in r.title.lower():
            skipped.append(f"{aid}: title '{r.title}' does not contain '{keyword}'")
            continue
        txt_path = CORPUS / f"{aid}.txt"
        if not txt_path.exists():
            pdf_path = RAW / f"{aid}.pdf"
            try:
                if not pdf_path.exists():
                    download(f"https://arxiv.org/pdf/{aid}", pdf_path)
                    time.sleep(3)  # arXiv asks for at most one request every 3 seconds
                pages = [(p.extract_text() or "") for p in PdfReader(str(pdf_path)).pages]
                txt_path.write_text(clean("\n".join(pages)), encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                skipped.append(f"{aid}: download/extract failed: {e}")
                continue
        meta.append({
            "arxiv_id": aid,
            "title": r.title.strip(),
            "authors": [a.name for a in r.authors],
            "first_author": r.authors[0].name if r.authors else "",
            "year": r.published.year,
            "published": r.published.date().isoformat(),
            "primary_category": r.primary_category,
            "topic": topic,
            "abstract": " ".join(r.summary.split()),
            "url": f"https://arxiv.org/abs/{aid}",
            "chars": len(txt_path.read_text(encoding="utf-8")),
        })
        print(f"ok  {aid}  {r.title[:70]}")
    meta.sort(key=lambda m: m["arxiv_id"])
    (ROOT / "data" / "papers.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n{len(meta)} papers written to data/corpus/, metadata in data/papers.json")
    for s in skipped:
        print("SKIPPED", s)
    missing = set(ids) - {m["arxiv_id"] for m in meta} - {s.split(":")[0] for s in skipped}
    for m in sorted(missing):
        print("MISSING (not returned by arXiv)", m)
    return 0 if len(meta) >= 40 else 1


if __name__ == "__main__":
    sys.exit(main())

"""Structured paper metadata in SQLite, queried by the Data agent through a read-only SQL tool."""
from __future__ import annotations

import sqlite3
from collections import Counter

from src import config

SCHEMA = """
CREATE TABLE papers (
  arxiv_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  first_author TEXT,
  n_authors INTEGER,
  year INTEGER,
  published TEXT,          -- ISO date of first arXiv version
  primary_category TEXT,   -- e.g. cs.CL, cs.AI, cs.LG
  topic TEXT,              -- rag | retrieval | reasoning | agents | tools | multi-agent | survey | benchmark | foundations | alignment | evaluation
  n_chunks INTEGER,        -- number of indexed chunks (proxy for paper length)
  chars INTEGER,           -- characters of extracted text
  url TEXT
);
CREATE TABLE authors (
  arxiv_id TEXT REFERENCES papers(arxiv_id),
  position INTEGER,        -- 1 = first author
  name TEXT
);
"""

SCHEMA_FOR_PROMPT = SCHEMA.strip()


def build_db(papers: list[dict], chunks: list[dict]) -> None:
    n_chunks = Counter(c["metadata"]["arxiv_id"] for c in chunks)
    config.PAPERS_DB.unlink(missing_ok=True)
    con = sqlite3.connect(config.PAPERS_DB)
    con.executescript(SCHEMA)
    con.executemany(
        "INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [(p["arxiv_id"], p["title"], p["first_author"], len(p["authors"]), p["year"], p["published"],
          p["primary_category"], p["topic"], n_chunks.get(p["arxiv_id"], 0), p["chars"], p["url"]) for p in papers],
    )
    con.executemany(
        "INSERT INTO authors VALUES (?,?,?)",
        [(p["arxiv_id"], i + 1, a) for p in papers for i, a in enumerate(p["authors"])],
    )
    con.commit()
    con.close()


class SQLError(ValueError):
    pass


def run_sql(query: str, max_rows: int = 50) -> list[dict]:
    """Execute ONE read-only SELECT. The connection is opened read-only, so writes fail even if the
    keyword check were bypassed."""
    q = query.strip().rstrip(";")
    if ";" in q or not q.lower().lstrip("( ").startswith(("select", "with")):
        raise SQLError("Only a single SELECT (or WITH ... SELECT) statement is allowed.")
    con = sqlite3.connect(f"file:{config.PAPERS_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(q).fetchmany(max_rows)
    except sqlite3.Error as e:
        raise SQLError(str(e)) from e
    finally:
        con.close()
    return [dict(r) for r in rows]


def find_papers(title_query: str, limit: int = 5) -> list[dict]:
    """Case-insensitive title/ID match, used by the Summary agent to resolve 'the ReAct paper'."""
    con = sqlite3.connect(f"file:{config.PAPERS_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        like = f"%{title_query.strip()}%"
        rows = con.execute(
            "SELECT arxiv_id, title, first_author, year, url FROM papers "
            "WHERE title LIKE ? OR arxiv_id = ? ORDER BY year LIMIT ?",
            (like, title_query.strip(), limit),
        ).fetchall()
    finally:
        con.close()
    return [dict(r) for r in rows]

"""query: SQL predicate over flattened card columns + optional FTS/semantic search."""

from __future__ import annotations

import json
import re
import sqlite3

from ..paths import corpus_dir

# Column names exposed to --where. Only these may appear in the predicate.
CARD_COLUMNS = {
    "family",
    "backbone",
    "action_head",
    "action_space",
    "chunk_size",
    "control_hz",
    "open_weights",
    "open_code",
    "open_data",
    "data_hours",
    "data_episodes",
    "data_source",
}


def _validate_where(where: str):
    if re.search(r"[;]", where) or not re.fullmatch(r"[\w\s()<>=!.,'\"+*/%-]*", where):
        raise ValueError(f"unsafe --where predicate: {where!r}")
    stripped = re.sub(r"'[^']*'|\"[^\"]*\"", "", where)  # ignore string literals
    used = set(re.findall(r"[a-z_]+", stripped.lower())) - {
        "and",
        "or",
        "not",
        "null",
        "is",
        "in",
        "like",
        "between",
        "true",
        "false",
    }
    unknown = used - CARD_COLUMNS
    if unknown:
        raise ValueError(f"unknown columns {sorted(unknown)}; allowed: {sorted(CARD_COLUMNS)}")


def _fts(conn: sqlite3.Connection, text: str, limit: int) -> list[str]:
    q = " ".join(re.findall(r"\w+", text))
    try:
        rows = conn.execute(
            "SELECT arxiv_id FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT ?",
            (q, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r[0] for r in rows]


def _semantic(conn: sqlite3.Connection, text: str, limit: int) -> list[str]:
    try:
        conn.execute("SELECT 1 FROM chunks LIMIT 1")
    except sqlite3.OperationalError:
        return []  # no depth=full papers yet
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    v = next(iter(model.embed([text])))
    try:
        rows = conn.execute(
            """
            SELECT m.arxiv_id FROM chunks c
            JOIN chunk_meta m ON m.rowid = c.rowid
            WHERE c.embedding MATCH ? AND k = ?
            ORDER BY distance LIMIT ?
            """,
            (v.tobytes(), limit * 10, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r[0] for r in rows]


def query(
    text: str = "",
    where: str | None = None,
    *,
    limit: int = 20,
    corpus=None,
) -> list[dict]:
    corpus = corpus or corpus_dir()
    conn = sqlite3.connect(corpus / "index.db")
    conn.row_factory = sqlite3.Row

    ids = None
    if text:
        fts_ids = _fts(conn, text, limit)
        if fts_ids:
            ids = fts_ids
        else:
            ids = _semantic(conn, text, limit)
        if not ids:
            return []

    sql = """
        SELECT p.arxiv_id, p.short_name, p.title, p.section, p.depth,
               c.family, c.backbone, c.action_head, c.action_space,
               c.chunk_size, c.control_hz, c.embodiment,
               c.data_hours, c.data_source, c.eval_sim, c.eval_real,
               c.open_weights, c.open_code, c.open_data, c.compute, c.limits
        FROM papers p LEFT JOIN cards c USING (arxiv_id)
    """
    params: list = []
    clauses = []
    if where:
        _validate_where(where)
        clauses.append(f"({where})")
    if ids is not None:
        clauses.append(f"p.arxiv_id IN ({','.join('?' * len(ids))})")
        params = list(ids)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY p.published DESC LIMIT ?"
    params.append(limit)

    out = []
    for row in conn.execute(sql, params):
        d = dict(row)
        d["embodiment"] = json.loads(d["embodiment"] or "[]")
        d["eval_sim"] = json.loads(d["eval_sim"] or "{}")
        d["eval_real"] = json.loads(d["eval_real"] or "{}")
        d["limits"] = json.loads(d["limits"] or "[]")
        d["open_weights"] = None if d["open_weights"] is None else bool(d["open_weights"])
        d["open_data"] = None if d["open_data"] is None else bool(d["open_data"])
        out.append(d)
    conn.close()
    return out

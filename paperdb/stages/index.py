"""index: rebuild index.db from files on disk. SQLite is derived, never authoritative.

- FTS5 over title + abstract + notes
- flattened card columns for query --where predicates
- sqlite-vec over paper.md chunks (depth=full only), fastembed bge-small CPU
"""

from __future__ import annotations

import json
import sqlite3

from ..paths import corpus_dir, load_resolved, papers_dir

DB_NAME = "index.db"
EMBED_DIM = 384  # BAAI/bge-small-en-v1.5


def _connect(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _schema(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            short_name TEXT, title TEXT, abstract TEXT, section TEXT,
            authors TEXT, published TEXT, categories TEXT,
            depth TEXT, starred INTEGER,
            has_pdf INTEGER, has_card INTEGER
        );
        CREATE TABLE IF NOT EXISTS cards (
            arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id),
            family TEXT, backbone TEXT, action_head TEXT, action_space TEXT,
            chunk_size INTEGER, control_hz REAL,
            embodiment TEXT, data_hours REAL, data_episodes INTEGER, data_source TEXT,
            eval_sim TEXT, eval_real TEXT,
            open_weights INTEGER, open_code TEXT, open_data INTEGER,
            compute TEXT, limits TEXT
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
            arxiv_id UNINDEXED, title, abstract, notes
        );
        """
    )
    # sqlite-vec is optional until the first depth=full paper exists.
    try:
        conn.enable_load_extension(True)
        import sqlite_vec

        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING vec0(embedding float[{EMBED_DIM}])"
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS chunk_meta (
                rowid INTEGER PRIMARY KEY, arxiv_id TEXT, pos INTEGER, text TEXT)"""
        )
    except (ImportError, AttributeError, sqlite3.OperationalError):
        pass  # vec search unavailable; FTS + card queries still work
    conn.commit()


def index(corpus=None) -> dict:
    corpus = corpus or corpus_dir()
    db_path = corpus / DB_NAME
    if db_path.exists():
        db_path.unlink()  # derived index: always rebuild from files
    conn = _connect(db_path)
    _schema(conn)

    records = load_resolved(corpus)
    n_cards = 0
    for r in records:
        d = papers_dir(corpus) / r["arxiv_id"]
        meta = {}
        if (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text())
        card = None
        cp = d / "card.json"
        if cp.exists():
            try:
                cj = json.loads(cp.read_text())
                card = cj.get("card") if cj.get("status") == "ok" else None
            except json.JSONDecodeError:
                pass
        notes = ""
        np = d / "notes.md"
        if np.exists():
            notes = np.read_text()
        conn.execute(
            "INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                r["arxiv_id"],
                r.get("short_name", ""),
                meta.get("title", ""),
                meta.get("abstract", ""),
                r.get("section", ""),
                json.dumps(meta.get("authors", [])),
                meta.get("published", ""),
                json.dumps(meta.get("categories", [])),
                r.get("depth", "card"),
                1 if r.get("starred") else 0,
                1 if (d / "paper.pdf").exists() else 0,
                1 if card else 0,
            ),
        )
        conn.execute(
            "INSERT INTO fts(arxiv_id, title, abstract, notes) VALUES (?,?,?,?)",
            (r["arxiv_id"], meta.get("title", ""), meta.get("abstract", ""), notes),
        )
        if card:
            n_cards += 1
            o = card.get("open") or {}
            data = card.get("data") or {}
            ev = card.get("eval") or {}
            conn.execute(
                "INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    r["arxiv_id"],
                    card.get("family"),
                    card.get("backbone"),
                    card.get("action_head"),
                    card.get("action_space"),
                    card.get("chunk_size"),
                    card.get("control_hz"),
                    json.dumps(card.get("embodiment", [])),
                    data.get("hours"),
                    data.get("episodes"),
                    data.get("source"),
                    json.dumps(ev.get("sim", {})),
                    json.dumps(ev.get("real", {})),
                    None if o.get("weights") is None else int(o["weights"]),
                    o.get("code"),
                    None if o.get("data") is None else int(o["data"]),
                    card.get("compute"),
                    json.dumps(card.get("limits", [])),
                ),
            )

    n_chunks = _index_chunks(conn, corpus)
    conn.commit()
    conn.close()
    return {"papers": len(records), "cards": n_cards, "chunks": n_chunks, "db": str(db_path)}


def _index_chunks(conn: sqlite3.Connection, corpus) -> int:
    """Embed paper.md chunks for depth=full papers. Empty at card seed."""
    try:
        conn.execute("SELECT 1 FROM chunks LIMIT 1")
    except sqlite3.OperationalError:
        return 0
    docs = []
    for r in load_resolved(corpus):
        if r.get("depth") != "full":
            continue
        md = papers_dir(corpus) / r["arxiv_id"] / "paper.md"
        if not md.exists():
            continue
        for i, chunk in enumerate(_chunk_text(md.read_text())):
            docs.append((r["arxiv_id"], i, chunk))
    if not docs:
        return 0
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vecs = list(model.embed([t for _, _, t in docs]))
    for (aid, pos, text), v in zip(docs, vecs):
        cur = conn.execute("INSERT INTO chunks(embedding) VALUES (?)", [v.tobytes()])
        conn.execute(
            "INSERT INTO chunk_meta(rowid, arxiv_id, pos, text) VALUES (?,?,?,?)",
            (cur.lastrowid, aid, pos, text),
        )
    conn.commit()
    return len(docs)


def _chunk_text(text: str, size: int = 1500, overlap: int = 200) -> list[str]:
    out = []
    step = size - overlap
    i = 0
    while i < len(text):
        part = text[i : i + size]
        if part.strip():
            out.append(part)
        i += step
    return out

"""meta: arXiv API metadata -> papers/<id>/meta.json. Idempotent, keyed by ID."""

from __future__ import annotations

import asyncio
import json

from ..arxiv import fetch_meta
from ..paths import load_resolved, paper_dir


def meta(corpus, limit: int | None = None) -> dict:
    records = load_resolved(corpus)
    if limit:
        records = records[:limit]
    ids = [r["arxiv_id"] for r in records]

    missing = [i for i in ids if not (paper_dir(i, corpus) / "meta.json").exists()]
    fetched: dict[str, dict] = {}
    if missing:
        fetched = asyncio.run(fetch_meta(missing))

    for i in missing:
        if i not in fetched:
            continue
        d = paper_dir(i, corpus)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(fetched[i], indent=2, ensure_ascii=False) + "\n")

    got = sum(1 for i in ids if (paper_dir(i, corpus) / "meta.json").exists())
    return {
        "total": len(ids),
        "requested": len(missing),
        "fetched": len([i for i in missing if i in fetched]),
        "missing_from_api": sorted(set(missing) - set(fetched)),
        "have_meta": got,
    }

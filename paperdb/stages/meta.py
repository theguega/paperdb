"""meta: arXiv API metadata -> papers/<id>/meta.json. Idempotent, keyed by ID."""

from __future__ import annotations

import asyncio
import json
import re

from ..arxiv import fetch_meta
from ..paths import load_resolved, paper_dir

ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")


def meta(corpus, limit: int | None = None) -> dict:
    records = load_resolved(corpus)
    if limit:
        records = records[:limit]
    ids = [r["arxiv_id"] for r in records]

    api_ids = [i for i in ids if ARXIV_ID_RE.match(i)]
    local = [r for r in records if not ARXIV_ID_RE.match(r["arxiv_id"])]

    missing = [i for i in api_ids if not (paper_dir(i, corpus) / "meta.json").exists()]
    fetched: dict[str, dict] = {}
    if missing:
        fetched = asyncio.run(fetch_meta(missing))

    for i in missing:
        if i not in fetched:
            continue
        d = paper_dir(i, corpus)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(fetched[i], indent=2, ensure_ascii=False) + "\n")

    # Non-arXiv papers (slug keys): minimal meta.json straight from manual.yaml.
    written_local = 0
    for r in local:
        d = paper_dir(r["arxiv_id"], corpus)
        mp = d / "meta.json"
        if mp.exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        mp.write_text(
            json.dumps(
                {
                    "arxiv_id": r["arxiv_id"],
                    "title": r["title"] or r["short_name"],
                    "abstract": "",
                    "authors": [],
                    "published": "",
                    "updated": "",
                    "categories": [],
                    "primary_category": "",
                    "pdf_url": r.get("pdf_url", ""),
                    "abs_url": r.get("url", ""),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        written_local += 1

    got = sum(1 for i in ids if (paper_dir(i, corpus) / "meta.json").exists())
    return {
        "total": len(ids),
        "fetched": len([i for i in missing if i in fetched]) + written_local,
        "missing_from_api": sorted(set(missing) - set(fetched)),
        "have_meta": got,
    }


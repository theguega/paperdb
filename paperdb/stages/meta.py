"""meta: paper metadata -> papers/<id>/meta.json. Idempotent, keyed by ID.

The arXiv API is the primary source: it batches 100 IDs per request, so a whole
corpus costs a handful of calls. The Hugging Face Hub is used for what arXiv
doesn't have - ai_summary, ai_keywords, linked models/datasets - and as a
fallback for IDs the arXiv API doesn't return. Hub calls are one-per-paper and
rate-limited, so enrichment only runs for papers being written fresh unless
--enrich asks for a backfill.
"""

from __future__ import annotations

import asyncio
import json
import re

from .. import hf
from ..arxiv import fetch_meta
from ..paths import load_resolved, paper_dir

ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")


def _write(d, rec: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")


def meta(corpus, limit: int | None = None, enrich: bool = False) -> dict:
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

    written, from_hub = 0, 0
    for i in missing:
        rec = fetched.get(i)
        if rec is None:
            rec = hf.meta_from_info(i)  # arXiv API missed it; try the Hub
            if rec is None:
                continue
            from_hub += 1
        else:
            rec.update(hf.enrichment(i))
        _write(paper_dir(i, corpus), rec)
        written += 1

    # Backfill Hub extras onto meta.json files written before enrichment existed.
    enriched = 0
    if enrich:
        for i in api_ids:
            mp = paper_dir(i, corpus) / "meta.json"
            if not mp.exists():
                continue
            rec = json.loads(mp.read_text())
            if any(k in rec for k in hf.ENRICH_KEYS):
                continue
            extra = hf.enrichment(i)
            if extra:
                rec.update(extra)
                _write(paper_dir(i, corpus), rec)
                enriched += 1

    # Non-arXiv papers (slug keys): minimal meta.json straight from manual.yaml.
    written_local = 0
    for r in local:
        d = paper_dir(r["arxiv_id"], corpus)
        mp = d / "meta.json"
        if mp.exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        _write(
            d,
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
        )
        written_local += 1

    got = sum(1 for i in ids if (paper_dir(i, corpus) / "meta.json").exists())
    return {
        "total": len(ids),
        "fetched": written + written_local,
        "from_hub": from_hub,
        "enriched": enriched,
        "missing_from_api": sorted(set(missing) - set(fetched)),
        "have_meta": got,
    }

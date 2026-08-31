"""fetch: download paper.pdf into papers/<id>/. Idempotent, keyed by ID."""

from __future__ import annotations

import asyncio
import json

import httpx

from ..paths import load_resolved, paper_dir, papers_dir

_DELAY_S = 1.0  # be polite to arXiv between PDF downloads


async def _download(
    client: httpx.AsyncClient, arxiv_id: str, url: str, dest, sem: asyncio.Semaphore
) -> str:
    async with sem:
        r = await client.get(url)
        await asyncio.sleep(_DELAY_S)
    if r.status_code != 200:
        return f"http {r.status_code}"
    body = r.content
    if not body.startswith(b"%PDF"):
        return "not a pdf (likely rate-limited)"
    dest.write_bytes(body)
    return "ok"


def fetch(corpus, limit: int | None = None) -> dict:
    records = load_resolved(corpus)
    if limit:
        records = records[:limit]
    todo: list[tuple[str, str, object]] = []  # (id, url, dest)
    skipped = 0
    for r in records:
        d = paper_dir(r["arxiv_id"], corpus)
        dest = d / "paper.pdf"
        if dest.exists() and dest.stat().st_size > 1000:
            skipped += 1
            continue
        mp = d / "meta.json"
        url = f"https://arxiv.org/pdf/{r['arxiv_id']}"
        if mp.exists():
            meta_url = json.loads(mp.read_text()).get("pdf_url", "")
            if meta_url:
                url = meta_url
        todo.append((r["arxiv_id"], url, dest))
    papers_dir(corpus).mkdir(parents=True, exist_ok=True)

    errors: dict[str, str] = {}

    async def run():
        sem = asyncio.Semaphore(2)
        async with httpx.AsyncClient(
            timeout=120.0, follow_redirects=True, headers={"User-Agent": "paperdb/0.1"}
        ) as client:
            results = await asyncio.gather(
                *[_download(client, i, u, d, sem) for i, u, d in todo], return_exceptions=True
            )
        for (i, _, _), res in zip(todo, results):
            if isinstance(res, Exception):
                errors[i] = str(res)[:200]
            elif res != "ok":
                errors[i] = res

    if todo:
        asyncio.run(run())

    have = sum(
        1
        for r in records
        if (paper_dir(r["arxiv_id"], corpus) / "paper.pdf").exists()
        and (paper_dir(r["arxiv_id"], corpus) / "paper.pdf").stat().st_size > 1000
    )
    return {
        "total": len(records),
        "downloaded": len(todo) - len(errors),
        "skipped_existing": skipped,
        "errors": errors,
        "have_pdf": have,
    }

"""fetch: download paper.pdf into papers/<id>/. Idempotent, keyed by ID."""

from __future__ import annotations

import asyncio

import httpx

from ..paths import load_resolved, paper_dir, papers_dir

_DELAY_S = 1.0  # be polite to arXiv between PDF downloads


async def _download(client: httpx.AsyncClient, arxiv_id: str, dest, sem: asyncio.Semaphore) -> str:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
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
    todo: list[tuple[str, object]] = []
    skipped = 0
    for r in records:
        dest = paper_dir(r["arxiv_id"], corpus) / "paper.pdf"
        if dest.exists() and dest.stat().st_size > 1000:
            skipped += 1
        else:
            todo.append((r["arxiv_id"], dest))
    papers_dir(corpus).mkdir(parents=True, exist_ok=True)

    errors: dict[str, str] = {}

    async def run():
        sem = asyncio.Semaphore(2)
        async with httpx.AsyncClient(
            timeout=120.0, follow_redirects=True, headers={"User-Agent": "paperdb/0.1"}
        ) as client:
            results = await asyncio.gather(
                *[_download(client, i, d, sem) for i, d in todo], return_exceptions=True
            )
        for (i, _), res in zip(todo, results):
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

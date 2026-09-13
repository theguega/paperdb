"""fetch: download paper.pdf into papers/<id>/. Idempotent, keyed by ID.

With the `hf` parse backend most papers never need a PDF, so fetch skips any
paper that already has paper.md. Use --all to pre-download regardless.
"""

from __future__ import annotations

import asyncio
import json
import re

import httpx

from ..paths import load_resolved, paper_dir, papers_dir

_DELAY_S = 1.0  # be polite to arXiv between PDF downloads


ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")


async def _get(client: httpx.AsyncClient, url: str, dest, sem: asyncio.Semaphore) -> str:
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


async def _download(
    client: httpx.AsyncClient, arxiv_id: str, url: str, dest, sem: asyncio.Semaphore
) -> str:
    """Try each candidate URL in turn; report the first attempt's error if all fail."""
    first = ""
    for candidate in _pdf_urls(arxiv_id, url):
        res = await _get(client, candidate, dest, sem)
        if res == "ok":
            return "ok"
        first = first or res
    return first


def _pdf_urls(arxiv_id: str, url: str) -> list[str]:
    """Candidate PDF URLs, most specific first.

    meta.json records the latest version, but arXiv sometimes 404s that version's
    PDF while an earlier one is still served (a v2 that ships only source, say),
    so fall back to the version-less URL and then v1.
    """
    urls = [url]
    if ARXIV_ID_RE.match(arxiv_id):
        for u in (f"https://arxiv.org/pdf/{arxiv_id}", f"https://arxiv.org/pdf/{arxiv_id}v1"):
            if u not in urls:
                urls.append(u)
    return urls


def _pdf_url(arxiv_id: str, corpus) -> str:
    mp = paper_dir(arxiv_id, corpus) / "meta.json"
    if mp.exists():
        url = json.loads(mp.read_text()).get("pdf_url", "")
        if url:
            return url
    return f"https://arxiv.org/pdf/{arxiv_id}"


def fetch_one(arxiv_id: str, corpus) -> str | None:
    """Download a single paper.pdf. Returns None on success, else the reason.

    Used by the parse stage when the Hub can't serve a paper and the PDF
    fallback needs a file that isn't on disk yet.
    """
    d = paper_dir(arxiv_id, corpus)
    d.mkdir(parents=True, exist_ok=True)
    dest = d / "paper.pdf"
    if dest.exists() and dest.stat().st_size > 1000:
        return None

    async def run() -> str:
        async with httpx.AsyncClient(
            timeout=120.0, follow_redirects=True, headers={"User-Agent": "paperdb/0.1"}
        ) as client:
            return await _download(
                client, arxiv_id, _pdf_url(arxiv_id, corpus), dest, asyncio.Semaphore(1)
            )

    try:
        res = asyncio.run(run())
    except Exception as e:  # noqa: BLE001 - report, don't crash the batch
        return str(e)[:200]
    return None if res == "ok" else res


def fetch(corpus, limit: int | None = None, all_papers: bool = False) -> dict:
    records = load_resolved(corpus)
    if limit:
        records = records[:limit]
    todo: list[tuple[str, str, object]] = []  # (id, url, dest)
    skipped, skipped_have_md = 0, 0
    for r in records:
        d = paper_dir(r["arxiv_id"], corpus)
        dest = d / "paper.pdf"
        if dest.exists() and dest.stat().st_size > 1000:
            skipped += 1
            continue
        if not all_papers and (d / "paper.md").exists():
            skipped_have_md += 1  # text already came from the Hub; no PDF needed
            continue
        todo.append((r["arxiv_id"], _pdf_url(r["arxiv_id"], corpus), dest))
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
        "skipped_have_md": skipped_have_md,
        "errors": errors,
        "have_pdf": have,
    }

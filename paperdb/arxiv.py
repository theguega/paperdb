"""arXiv API client: batched id_list queries, 3s pause between requests."""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET

import httpx

API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
BATCH = 100


def parse_feed(xml_text: str) -> dict[str, dict]:
    """Atom feed -> {arxiv_id: meta}. IDs are version-stripped."""
    out: dict[str, dict] = {}
    root = ET.fromstring(xml_text)
    for e in root.iter(f"{ATOM}entry"):
        raw_id = e.findtext(f"{ATOM}id") or ""
        arxiv_id = raw_id.rsplit("/", 1)[-1]
        arxiv_id = arxiv_id.split("v")[0] if "v" in arxiv_id.split("/")[-1] else arxiv_id
        if "." not in arxiv_id:
            continue  # skip non-canonical entries
        out[arxiv_id] = {
            "arxiv_id": arxiv_id,
            "title": " ".join((e.findtext(f"{ATOM}title") or "").split()),
            "abstract": " ".join((e.findtext(f"{ATOM}summary") or "").split()),
            "authors": [a.findtext(f"{ATOM}name") or "" for a in e.findall(f"{ATOM}author")],
            "published": e.findtext(f"{ATOM}published") or "",
            "updated": e.findtext(f"{ATOM}updated") or "",
            "categories": sorted(
                {c.get("term") for c in e.findall(f"{ATOM}category") if c.get("term")}
            ),
            "primary_category": next(
                (c.get("term") for c in e.findall(f"{ARXIV}primary_category") if c.get("term")),
                "",
            ),
            "pdf_url": next(
                (
                    l.get("href")
                    for l in e.findall(f"{ATOM}link")
                    if l.get("type") == "application/pdf"
                ),
                f"https://arxiv.org/pdf/{arxiv_id}",
            ),
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        }
    return out


async def fetch_meta(ids: list[str], *, pause_s: float = 3.0) -> dict[str, dict]:
    """Fetch metadata for all ids. Batches of 100, one request per batch."""
    out: dict[str, dict] = {}
    async with httpx.AsyncClient(
        timeout=60.0, follow_redirects=True, headers={"User-Agent": "paperdb/0.1"}
    ) as client:
        for i in range(0, len(ids), BATCH):
            batch = ids[i : i + BATCH]
            r = await client.get(
                API.replace("http://", "https://"),
                params={"id_list": ",".join(batch), "max_results": str(len(batch))},
            )
            r.raise_for_status()
            out.update(parse_feed(r.text))
            if i + BATCH < len(ids):
                await asyncio.sleep(pause_s)
    return out

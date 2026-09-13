"""Hugging Face CLI wrapper: paper markdown and metadata without touching a PDF.

`hf papers read` returns arXiv's HTML rendering as markdown, so for papers the
Hub serves we skip the PDF entirely. Coverage is partial (no HTML for most
pre-2022 papers, and the Hub does not index every arXiv ID), so every call here
returns None rather than raising - callers fall back to the PDF path.

The Hub rate-limits aggressively: concurrent reads fail with an opaque
"Request ID" error that looks nothing like a 429, so calls are serialised with
a delay and retried with backoff.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time

# A successful read is a whole paper. Some IDs exit 0 with a few hundred bytes
# of an SVG filename as the title - real failures that would silently produce
# empty cards, so anything under this floor counts as a miss.
MIN_MARKDOWN_BYTES = 5000

_DELAY_S = 2.5  # between consecutive hf calls; below this the Hub throttles
_RETRIES = 3
_TIMEOUT_S = 120

_last_call = 0.0


def available() -> bool:
    """True if the hf CLI is on PATH."""
    return shutil.which("hf") is not None


def _run(args: list[str]) -> tuple[int, str]:
    global _last_call
    wait = _DELAY_S - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    try:
        p = subprocess.run(
            ["hf", *args], capture_output=True, text=True, timeout=_TIMEOUT_S, check=False
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        _last_call = time.monotonic()
        return 1, str(e)[:200]
    _last_call = time.monotonic()
    return p.returncode, (p.stdout if p.returncode == 0 else p.stderr or p.stdout)


def _is_throttled(out: str) -> bool:
    """Hub throttling surfaces as a bare Request ID, not a status code."""
    return "Request ID" in out or "429" in out


def _call(args: list[str]) -> str | None:
    """Run hf with backoff on throttling. None on a genuine miss."""
    for attempt in range(_RETRIES):
        rc, out = _run(args)
        if rc == 0:
            return out
        if not _is_throttled(out):
            return None  # "not found on the Hub", 401, etc - a real miss
        time.sleep(_DELAY_S * (attempt + 2))
    return None


def read_paper(arxiv_id: str) -> str | None:
    """Paper markdown via `hf papers read`, or None if the Hub can't serve it."""
    if not available():
        return None
    out = _call(["papers", "read", arxiv_id])
    if out is None or len(out) < MIN_MARKDOWN_BYTES:
        return None
    return out


def paper_info(arxiv_id: str) -> dict | None:
    """Hub metadata via `hf papers info`, or None if unavailable."""
    if not available():
        return None
    out = _call(["papers", "info", arxiv_id, "--json"])
    if out is None:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


# Fields the Hub adds on top of the arXiv API. Kept as an explicit list so
# meta.json stays a predictable shape.
ENRICH_KEYS = ("ai_summary", "ai_keywords", "upvotes", "linked_models", "linked_datasets")


def enrichment_from(info: dict) -> dict:
    """Hub-only extras from an already-fetched info dict."""
    out = {k: info[k] for k in ENRICH_KEYS if info.get(k)}
    if out:
        out["hf_url"] = f"https://huggingface.co/papers/{info.get('id', '')}"
    return out


def enrichment(arxiv_id: str) -> dict:
    """Hub-only extras for meta.json. Empty dict when unavailable."""
    info = paper_info(arxiv_id)
    return enrichment_from(info) if info else {}


def meta_from_info(arxiv_id: str) -> dict | None:
    """Build a meta.json record from the Hub alone, for IDs the arXiv API misses.

    The Hub has no categories/primary_category, so those stay empty rather than
    being guessed.
    """
    info = paper_info(arxiv_id)
    if not info:
        return None
    published = info.get("published_at", "") or ""
    rec = {
        "arxiv_id": arxiv_id,
        "title": " ".join((info.get("title") or "").split()),
        "abstract": " ".join((info.get("summary") or "").split()),
        "authors": [a.get("name", "") for a in info.get("authors", []) if a.get("name")],
        "published": published,
        "updated": info.get("submitted_at", "") or published,
        "categories": [],
        "primary_category": "",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
    }
    rec.update(enrichment_from(info))
    return rec

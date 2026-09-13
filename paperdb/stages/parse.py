"""parse: paper text -> paper.md. Idempotent, keyed by ID.

Two routes to markdown. The `hf` backend asks the Hugging Face Hub for arXiv's
HTML rendering, which needs no PDF at all; the PDF backends (pymupdf4llm,
docling) convert a local paper.pdf. With backend = "hf" the PDF route is the
fallback, and parse downloads the PDF on demand, so `parse` alone is enough to
get text for a paper.

Files on disk stay the source of truth and any RAG framework (LangChain, ADK)
can read paper.md directly.
"""

from __future__ import annotations

import time

from ..paths import load_resolved, paper_dir, papers_dir


class NoText(Exception):
    """Backend could not produce markdown for this paper."""


def _convert_pymupdf4llm(pdf_path, out_path) -> None:
    import pymupdf4llm

    md = pymupdf4llm.to_markdown(str(pdf_path), show_progress=False)
    out_path.write_text(md, encoding="utf-8")


def _convert_docling(pdf_path, out_path) -> None:
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(str(pdf_path))
    out_path.write_text(result.document.export_to_markdown(), encoding="utf-8")


_PDF_BACKENDS = {"pymupdf4llm": _convert_pymupdf4llm, "docling": _convert_docling}
_BACKENDS = {"hf", *_PDF_BACKENDS}


def _via_hf(arxiv_id: str, out_path) -> None:
    from .. import hf

    md = hf.read_paper(arxiv_id)
    if md is None:
        raise NoText(f"hub has no markdown for {arxiv_id}")
    out_path.write_text(md, encoding="utf-8")


def _via_pdf(arxiv_id: str, d, out_path, backend: str, corpus) -> None:
    """Convert paper.pdf, downloading it first if it isn't there yet."""
    from .fetch import fetch_one

    pdf = d / "paper.pdf"
    if not pdf.exists():  # fetch validates the %PDF magic, so any file here is real
        err = fetch_one(arxiv_id, corpus)
        if err:
            raise NoText(f"no pdf ({err})")
    _PDF_BACKENDS[backend](pdf, out_path)


def parse(
    corpus,
    arxiv_id: str | None = None,
    force: bool = False,
    backend: str | None = None,
) -> dict:
    from ..config import load_config

    cfg = load_config().get("parse", {})
    backend = backend or cfg.get("backend", "hf")
    if backend not in _BACKENDS:
        raise ValueError(f"unknown parse backend {backend!r}; have {sorted(_BACKENDS)}")
    fallback = cfg.get("fallback_backend", "pymupdf4llm")
    if fallback not in _PDF_BACKENDS:
        raise ValueError(f"unknown fallback backend {fallback!r}; have {sorted(_PDF_BACKENDS)}")

    records = load_resolved(corpus)
    if arxiv_id:
        records = [r for r in records if r["arxiv_id"] == arxiv_id]
    todo = []
    for r in records:
        d = paper_dir(r["arxiv_id"], corpus)
        md = d / "paper.md"
        if force or not md.exists():
            todo.append((r["arxiv_id"], d, md))

    papers_dir(corpus).mkdir(parents=True, exist_ok=True)
    done, via_hf, via_pdf, errors = 0, 0, 0, {}
    t0 = time.time()
    for i, (aid, d, md) in enumerate(todo):
        d.mkdir(parents=True, exist_ok=True)
        try:
            if backend == "hf":
                try:
                    _via_hf(aid, md)
                    via_hf += 1
                except NoText:
                    _via_pdf(aid, d, md, fallback, corpus)
                    via_pdf += 1
            else:
                _via_pdf(aid, d, md, backend, corpus)
                via_pdf += 1
            done += 1
        except Exception as e:  # noqa: BLE001 - one bad paper must not stop the batch
            errors[aid] = str(e)[:200]
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    have = sum(1 for r in records if (paper_dir(r["arxiv_id"], corpus) / "paper.md").exists())
    return {
        "backend": backend,
        "fallback_backend": fallback if backend == "hf" else None,
        "total": len(records),
        "converted": done,
        "via_hf": via_hf,
        "via_pdf": via_pdf,
        "skipped_existing": len(records) - len(todo),
        "errors": errors,
        "have_md": have,
        "seconds": round(time.time() - t0),
    }

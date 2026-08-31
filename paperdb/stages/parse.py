"""parse: PDF -> paper.md. Local text extraction, idempotent, keyed by ID.

Backends are a config knob ([parse] backend); files on disk stay the source
of truth and any RAG framework (LangChain, ADK) can read paper.md directly.
"""

from __future__ import annotations

import time

from ..paths import load_resolved, paper_dir, papers_dir


def _convert_pymupdf4llm(pdf_path, out_path) -> None:
    import pymupdf4llm

    md = pymupdf4llm.to_markdown(str(pdf_path), show_progress=False)
    out_path.write_text(md, encoding="utf-8")


def _convert_docling(pdf_path, out_path) -> None:
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(str(pdf_path))
    out_path.write_text(result.document.export_to_markdown(), encoding="utf-8")


_BACKENDS = {"pymupdf4llm": _convert_pymupdf4llm, "docling": _convert_docling}


def parse(
    corpus, arxiv_id: str | None = None, force: bool = False, backend: str | None = None
) -> dict:
    from ..config import load_config

    cfg = load_config()
    backend = backend or cfg.get("parse", {}).get("backend", "pymupdf4llm")
    if backend not in _BACKENDS:
        raise ValueError(f"unknown parse backend {backend!r}; have {sorted(_BACKENDS)}")
    convert = _BACKENDS[backend]

    records = load_resolved(corpus)
    if arxiv_id:
        records = [r for r in records if r["arxiv_id"] == arxiv_id]
    todo = []
    for r in records:
        d = paper_dir(r["arxiv_id"], corpus)
        pdf, md = d / "paper.pdf", d / "paper.md"
        if pdf.exists() and (force or not md.exists()):
            todo.append((r["arxiv_id"], pdf, md))

    papers_dir(corpus).mkdir(parents=True, exist_ok=True)
    done, errors = 0, {}
    t0 = time.time()
    for i, (aid, pdf, md) in enumerate(todo):
        try:
            convert(pdf, md)
            done += 1
        except Exception as e:  # noqa: BLE001 - one bad PDF must not stop the batch
            errors[aid] = str(e)[:200]
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    have = sum(1 for r in records if (paper_dir(r["arxiv_id"], corpus) / "paper.md").exists())
    return {
        "backend": backend,
        "total": len(records),
        "converted": done,
        "skipped_existing": len(records) - len(todo),
        "errors": errors,
        "have_md": have,
        "seconds": round(time.time() - t0),
    }

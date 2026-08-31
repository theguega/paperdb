"""Shared paths. corpus/ layout is fixed; sources/*.yaml are the seed of truth."""

from __future__ import annotations

from pathlib import Path


def corpus_dir() -> Path:
    from .config import load_config

    return Path(load_config()["paths"]["corpus"])


def papers_dir(corpus: Path | None = None) -> Path:
    return (corpus or corpus_dir()) / "papers"


def paper_dir(arxiv_id: str, corpus: Path | None = None) -> Path:
    return papers_dir(corpus) / arxiv_id


def sources_dir(corpus: Path | None = None) -> Path:
    return (corpus or corpus_dir()) / "sources"


def load_resolved(corpus: Path | None = None) -> list[dict]:
    import yaml

    p = sources_dir(corpus) / "resolved.yaml"
    return yaml.safe_load(p.read_text()) or []

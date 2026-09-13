"""hf wrapper: size floor, throttle retry, and parse-stage routing."""

from __future__ import annotations

import json

import pytest

from paperdb import hf


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(hf.time, "sleep", lambda *_: None)
    monkeypatch.setattr(hf, "available", lambda: True)


def _fake_run(calls, *results):
    it = iter(results)

    def run(args):
        calls.append(args)
        return next(it)

    return run


def test_read_rejects_degenerate_output(monkeypatch):
    """Some IDs exit 0 with an SVG filename as the title - that is a miss."""
    monkeypatch.setattr(hf, "_run", _fake_run([], (0, "Title: si_sim.svg\n")))
    assert hf.read_paper("2608.21204") is None


def test_read_accepts_full_paper(monkeypatch):
    body = "# Paper\n" + "x" * hf.MIN_MARKDOWN_BYTES
    monkeypatch.setattr(hf, "_run", _fake_run([], (0, body)))
    assert hf.read_paper("2604.20246") == body


def test_not_found_is_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(
        hf, "_run", _fake_run(calls, (1, "Error: Paper '1811.04551' not found on the Hub."))
    )
    assert hf.read_paper("1811.04551") is None
    assert len(calls) == 1


def test_throttling_is_retried(monkeypatch):
    calls = []
    body = "x" * (hf.MIN_MARKDOWN_BYTES + 1)
    monkeypatch.setattr(
        hf,
        "_run",
        _fake_run(calls, (1, "Error: (Request ID: Root=1-abc)"), (0, body)),
    )
    assert hf.read_paper("2601.02427") == body
    assert len(calls) == 2


def test_meta_from_info_shape(monkeypatch):
    info = {
        "id": "2604.20246",
        "title": "Cortex 2.0",
        "summary": "  world  models  ",
        "authors": [{"name": "A. Aida"}, {"name": ""}],
        "published_at": "2026-04-22T00:00:00+00:00",
        "ai_keywords": ["world-model-based planning"],
        "upvotes": 6,
    }
    monkeypatch.setattr(hf, "_run", _fake_run([], (0, json.dumps(info))))
    rec = hf.meta_from_info("2604.20246")
    assert rec["abstract"] == "world models"
    assert rec["authors"] == ["A. Aida"]
    assert rec["categories"] == []  # the Hub has none; never guessed
    assert rec["ai_keywords"] == ["world-model-based planning"]
    assert rec["hf_url"].endswith("2604.20246")


# --- parse-stage routing --------------------------------------------------------


def _corpus(tmp_path, arxiv_id="2604.20246"):
    import yaml

    sources = tmp_path / "sources"
    sources.mkdir(parents=True)
    (sources / "resolved.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "arxiv_id": arxiv_id,
                    "short_name": "T",
                    "section": "s",
                    "title": "t",
                    "starred": False,
                    "depth": "card",
                    "source": "manual",
                }
            ],
            sort_keys=False,
        )
    )
    return tmp_path


def test_parse_prefers_hf_and_skips_pdf(tmp_path, monkeypatch):
    from paperdb.stages import parse as parse_mod

    body = "# Cortex 2.0\n" + "x" * hf.MIN_MARKDOWN_BYTES
    monkeypatch.setattr(parse_mod, "_via_pdf", lambda *a: pytest.fail("should not touch a PDF"))
    monkeypatch.setattr(hf, "read_paper", lambda _id: body)

    r = parse_mod.parse(_corpus(tmp_path), backend="hf")
    assert r["via_hf"] == 1 and r["via_pdf"] == 0
    assert (tmp_path / "papers" / "2604.20246" / "paper.md").read_text() == body


def test_parse_falls_back_to_pdf_when_hub_misses(tmp_path, monkeypatch):
    from paperdb.stages import parse as parse_mod

    monkeypatch.setattr(hf, "read_paper", lambda _id: None)

    def fake_pdf(arxiv_id, d, out_path, backend, corpus):
        out_path.write_text("from pdf")

    monkeypatch.setattr(parse_mod, "_via_pdf", fake_pdf)
    r = parse_mod.parse(_corpus(tmp_path, "1811.04551"), backend="hf")
    assert r["via_hf"] == 0 and r["via_pdf"] == 1
    assert (tmp_path / "papers" / "1811.04551" / "paper.md").read_text() == "from pdf"


def test_unknown_backend_rejected(tmp_path):
    from paperdb.stages.parse import parse

    with pytest.raises(ValueError, match="unknown parse backend"):
        parse(_corpus(tmp_path), backend="nope")


# --- fetch: arXiv version fallback ----------------------------------------------


def test_pdf_url_candidates_cover_stale_versions():
    """meta.json may name a version whose PDF 404s; v1 is the last resort."""
    from paperdb.stages.fetch import _pdf_urls

    urls = _pdf_urls("2506.23944", "https://arxiv.org/pdf/2506.23944v2")
    assert urls == [
        "https://arxiv.org/pdf/2506.23944v2",
        "https://arxiv.org/pdf/2506.23944",
        "https://arxiv.org/pdf/2506.23944v1",
    ]


def test_pdf_url_candidates_no_duplicates_for_bare_url():
    from paperdb.stages.fetch import _pdf_urls

    urls = _pdf_urls("2506.23944", "https://arxiv.org/pdf/2506.23944")
    assert urls.count("https://arxiv.org/pdf/2506.23944") == 1


def test_pdf_url_candidates_leave_non_arxiv_alone():
    """Slug-keyed papers have one hand-written pdf_url and no version scheme."""
    from paperdb.stages.fetch import _pdf_urls

    assert _pdf_urls("pinocchio", "https://example.org/paper.pdf") == [
        "https://example.org/paper.pdf"
    ]

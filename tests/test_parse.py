"""parse stage: idempotence + content, using a tiny generated PDF."""

from __future__ import annotations

import json

import yaml


def _make_corpus(tmp_path):
    import pymupdf

    sources = tmp_path / "sources"
    sources.mkdir(parents=True)
    (sources / "resolved.yaml").write_text(
        yaml.safe_dump(
            [{"arxiv_id": "1111.11111", "short_name": "T", "section": "s",
              "title": "t", "starred": False, "depth": "card", "source": "manual"}],
            sort_keys=False,
        )
    )
    d = tmp_path / "papers" / "1111.11111"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"title": "t", "abstract": "", "authors": []}))
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Flow matching control at 50 Hz for robot arms.")
    doc.save(str(d / "paper.pdf"))
    return tmp_path


def test_parse_converts_and_is_idempotent(tmp_path):
    from paperdb.stages.parse import parse

    _make_corpus(tmp_path)
    r = parse(tmp_path)
    assert r["converted"] == 1 and r["have_md"] == 1
    md = (tmp_path / "papers" / "1111.11111" / "paper.md").read_text()
    assert "Flow matching" in md
    r2 = parse(tmp_path)  # skip existing
    assert r2["converted"] == 0 and r2["skipped_existing"] == 1
    r3 = parse(tmp_path, force=True)
    assert r3["converted"] == 1

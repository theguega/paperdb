"""index + query against a synthetic corpus (no network, no agent)."""

from __future__ import annotations

import json

import yaml

from paperdb.stages.index import index
from paperdb.stages.query import query


def _make_corpus(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir(parents=True)
    records = [
        {
            "arxiv_id": "1111.11111",
            "short_name": "FastBot",
            "section": "VLA Models",
            "title": "t",
            "starred": True,
            "depth": "card",
            "source": "manual",
        },
        {
            "arxiv_id": "2222.22222",
            "short_name": "SlowBot",
            "section": "Policies",
            "title": "t",
            "starred": False,
            "depth": "card",
            "source": "manual",
        },
    ]
    (sources / "resolved.yaml").write_text(yaml.safe_dump(records, sort_keys=False))
    metas = {
        "1111.11111": {
            "title": "Flow Matching for Fast Robot Control",
            "abstract": "we use flow matching action heads at 50 Hz",
            "authors": ["A"],
            "published": "2024-01-01",
            "categories": ["cs.RO"],
        },
        "2222.22222": {
            "title": "Slow Grasping",
            "abstract": "diffusion policies",
            "authors": [],
            "published": "2023-01-01",
            "categories": [],
        },
    }
    cards = {
        "1111.11111": {
            "family": "vla",
            "backbone": "PaliGemma-3B",
            "control_hz": 50.0,
            "open": {"weights": True, "code": None, "data": False},
        },
        "2222.22222": {
            "family": "diffusion-policy",
            "control_hz": 10.0,
            "open": {"weights": False, "code": None, "data": None},
        },
    }
    for aid, m in metas.items():
        d = tmp_path / "papers" / aid
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps(m))
        (d / "card.json").write_text(json.dumps({"status": "ok", "card": cards[aid]}))
    return tmp_path


def test_index_rebuild_and_counts(tmp_path):
    _make_corpus(tmp_path)
    r = index(tmp_path)
    assert r["papers"] == 2 and r["cards"] == 2 and r["chunks"] == 0
    assert (tmp_path / "index.db").exists()


def test_query_where_predicates(tmp_path):
    _make_corpus(tmp_path)
    index(tmp_path)
    rows = query(where="control_hz >= 30", corpus=tmp_path)
    assert [r["arxiv_id"] for r in rows] == ["1111.11111"]
    rows = query(where="open_weights", corpus=tmp_path)
    assert [r["arxiv_id"] for r in rows] == ["1111.11111"]
    rows = query(where="family = 'diffusion-policy'", corpus=tmp_path)
    assert [r["arxiv_id"] for r in rows] == ["2222.22222"]
    rows = query(where="control_hz >= 30 AND open_weights", corpus=tmp_path)
    assert len(rows) == 1


def test_query_rejects_unsafe_where(tmp_path):
    _make_corpus(tmp_path)
    index(tmp_path)
    for bad in ["1=1; DROP TABLE papers", "secret_column > 3", "title LIKE '%x%'"]:
        try:
            query(where=bad, corpus=tmp_path)
            assert False, f"should have rejected {bad}"
        except ValueError:
            pass


def test_query_fts_text(tmp_path):
    _make_corpus(tmp_path)
    index(tmp_path)
    rows = query("flow matching", corpus=tmp_path)
    assert [r["arxiv_id"] for r in rows] == ["1111.11111"]
    # text AND where combined
    rows = query("grasping", where="control_hz < 30", corpus=tmp_path)
    assert [r["arxiv_id"] for r in rows] == ["2222.22222"]


def test_index_is_derived_and_rebuildable(tmp_path):
    _make_corpus(tmp_path)
    r1 = index(tmp_path)
    r2 = index(tmp_path)  # delete + rebuild must be identical
    assert r1["cards"] == r2["cards"]

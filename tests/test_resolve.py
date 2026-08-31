"""Golden test: parse the committed awesome-vla-wam README snapshot."""

from __future__ import annotations

from pathlib import Path

import yaml

from paperdb.stages.resolve import load_manual, parse_readme

FIXTURE = Path(__file__).parent / "fixtures" / "awesome-readme.md"


def _parse():
    return parse_readme(FIXTURE.read_text())


def test_golden_counts():
    papers, quarantined = _parse()
    ids = {p.arxiv_id for p in papers}
    assert len(papers) == len(ids), "arxiv ids must be unique"
    assert len(papers) > 200
    assert len(quarantined) > 0


def test_fields_and_starred():
    papers, _ = _parse()
    pi = {p.arxiv_id: p for p in papers}
    genie = pi["2402.15391"]
    assert genie.short_name == "Genie"
    assert genie.section == "World Models"
    assert genie.starred is True
    assert genie.title.startswith("Genie: Generative Interactive Environments")
    foca = pi["2606.20867"]
    assert foca.starred is False
    assert foca.short_name == "FOCA-VLA"
    assert "Future-Oriented Conditioning" in foca.title


def test_no_arxiv_link_quarantined():
    papers, quarantined = _parse()
    assert all(p.arxiv_id for p in papers)
    q = [x for x in quarantined if "Gemini Robotics 2" in x.raw]
    assert len(q) == 2  # listed in two sections, quarantined from both
    assert all(x.reason == "no arxiv link" for x in q)
    assert "VLA Models" in {x.section for x in q}


def test_malformed_ids_are_caught():
    # Valid-looking id with an impossible month -> quarantined as malformed.
    text = (
        FIXTURE.read_text()
        + "\n- **Bad**, Bad paper. [![arXiv](x)](https://arxiv.org/abs/2513.12345)\n"
        + "\n- **Bad2**, Bad2 paper. [![arXiv](x)](https://arxiv.org/abs/22607.08639)\n"
    )
    papers, quarantined = parse_readme(text)
    ids = {p.arxiv_id for p in papers}
    assert "2513.12345" not in ids and "22607.08639" not in ids
    reasons = {q.reason for q in quarantined}
    assert any(r.startswith("malformed") for r in reasons)  # 2513: month 13
    # 22607: the long digit run is rejected outright, line lands in quarantine too.
    assert any("Bad2" in q.raw for q in quarantined)


def test_pdf_and_html_links_resolve():
    papers, _ = _parse()
    pi = {p.arxiv_id: p for p in papers}
    assert "2607.08639" in pi, "versioned abs link (LingBot-VA 2) must resolve"


def test_duplicate_names_allowed():
    papers, _ = _parse()
    names = [p.short_name for p in papers]
    dupes = {n for n in names if names.count(n) > 1}
    assert dupes, "snapshot is expected to reuse short names; ids stay unique"


def test_manual_merge_wins(tmp_path):

    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "manual.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "arxiv_id": "2402.15391",
                    "short_name": "Genie-2",
                    "title": "Overridden title",
                    "section": "World Models",
                    "starred": True,
                    "depth": "full",
                },
                {"arxiv_id": "1111.11111", "short_name": "Only-Manual", "title": "t"},
            ]
        )
    )
    manual = load_manual(sources)
    assert {m.arxiv_id for m in manual} == {"2402.15391", "1111.11111"}
    assert manual[0].short_name == "Genie-2"
    assert manual[0].depth == "full"

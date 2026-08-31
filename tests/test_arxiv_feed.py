"""Feed parsing test against a recorded arXiv Atom response."""

from __future__ import annotations

from paperdb.arxiv import parse_feed

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2307.15818v3</id>
    <updated>2024-01-01T00:00:00Z</updated>
    <published>2023-07-28T17:59:31Z</published>
    <title>RT-2: Vision-Language-Action Models Transfer Web Knowledge</title>
    <summary>  Abstract with
      newlines and   extra spaces. </summary>
    <author><name>Anthony Brohan</name></author>
    <author><name>Someone Else</name></author>
    <category term="cs.RO"/>
    <category term="cs.LG"/>
    <arxiv:primary_category term="cs.RO"/>
    <link href="http://arxiv.org/abs/2307.15818v3" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2307.15818v3" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2402.15391v1</id>
    <updated>2024-02-01T00:00:00Z</updated>
    <published>2024-02-01T00:00:00Z</published>
    <title>Genie</title>
    <summary>S.</summary>
    <author><name>A</name></author>
    <category term="cs.AI"/>
    <arxiv:primary_category term="cs.AI"/>
  </entry>
</feed>
"""


def test_parse_feed_strips_versions_and_extracts_fields():
    m = parse_feed(FEED)
    assert set(m) == {"2307.15818", "2402.15391"}
    rt2 = m["2307.15818"]
    assert rt2["title"] == "RT-2: Vision-Language-Action Models Transfer Web Knowledge"
    assert rt2["abstract"] == "Abstract with newlines and extra spaces."
    assert rt2["authors"] == ["Anthony Brohan", "Someone Else"]
    assert rt2["categories"] == ["cs.LG", "cs.RO"]
    assert rt2["primary_category"] == "cs.RO"
    assert rt2["pdf_url"] == "http://arxiv.org/pdf/2307.15818v3"


def test_parse_feed_synthesizes_pdf_url_when_missing():
    m = parse_feed(FEED)
    assert m["2402.15391"]["pdf_url"] == "https://arxiv.org/pdf/2402.15391"

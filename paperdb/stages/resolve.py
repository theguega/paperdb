"""resolve: awesome-vla-wam README -> (arxiv_id, section, short_name, title, starred).

Files on disk are the source of truth; outputs are idempotent and keyed by
arXiv ID. Malformed lines go to quarantine.yaml with their raw text.
manual.yaml wins on conflict.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

REPO_URL = "https://github.com/DravenALG/awesome-vla-wam"

# Spec regex; html added - same ID namespace, losing the paper is worse than
# the deviation. (?<!\d) guards against malformed longer IDs like 22607.08639.
ARXIV_RE = re.compile(r"(?<!\d)arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})")
NAME_RE = re.compile(r"\*\*(.+?)\*\*")


@dataclass
class Paper:
    arxiv_id: str
    section: str
    short_name: str
    title: str
    starred: bool
    depth: str = "card"
    source: str = "awesome"
    url: str = ""
    pdf_url: str = ""
    raw: str = field(default="", repr=False)


@dataclass
class Quarantined:
    section: str
    reason: str
    raw: str


def sync_repo(cache_dir: Path) -> Path:
    """Clone or git-pull the awesome list into cache_dir. Returns README path."""
    repo = cache_dir / "awesome-vla-wam"
    if (repo / ".git").exists():
        subprocess.run(
            ["git", "-C", str(repo), "pull", "--ff-only"], check=True, capture_output=True
        )
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(repo)], check=True, capture_output=True
        )
    return repo / "README.md"


def _title_after_name(line: str, name_end: int) -> str:
    rest = line[name_end:].strip().lstrip(":, ").strip()
    # Title runs to the first badge link, if any.
    cut = rest.find(" [![")
    if cut >= 0:
        rest = rest[:cut]
    return rest.strip().rstrip(".").strip()


def parse_readme(text: str) -> tuple[list[Paper], list[Quarantined]]:
    papers: dict[str, Paper] = {}
    quarantined: list[Quarantined] = []
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        if not line.startswith("- "):
            continue
        m = ARXIV_RE.search(line)
        if not m:
            if NAME_RE.search(line):  # a real entry that just lacks an arXiv link
                quarantined.append(Quarantined(section, "no arxiv link", raw))
            continue
        arxiv_id = m.group(1)
        if not (1 <= int(arxiv_id[2:4]) <= 12):
            quarantined.append(Quarantined(section, f"malformed arxiv id {arxiv_id!r}", raw))
            continue
        nm = NAME_RE.search(line)
        if not nm:
            quarantined.append(Quarantined(section, "no bold name", raw))
            continue
        title = _title_after_name(line, nm.end())
        p = Paper(
            arxiv_id=arxiv_id,
            section=section,
            short_name=nm.group(1).strip(),
            title=title,
            starred="⭐" in line,
            raw=raw,
        )
        if arxiv_id in papers:
            # Same paper listed twice; first listing wins, names are not unique.
            continue
        papers[arxiv_id] = p
    return list(papers.values()), quarantined


def _write_if_changed(path: Path, text: str) -> bool:
    if path.exists() and path.read_text() == text:
        return False
    path.write_text(text)
    return True


def _dump_yaml(path: Path, obj) -> bool:
    return _write_if_changed(path, yaml.safe_dump(obj, sort_keys=False, allow_unicode=True))


def load_manual(sources_dir: Path) -> list[Paper]:
    """Manual entries. arxiv_id is the key; papers without one get a slug key
    (lowercased short_name) plus an explicit pdf_url - meta skips the arXiv
    API for those and fetch downloads from pdf_url directly."""
    p = sources_dir / "manual.yaml"
    if not p.exists():
        return []
    out = []
    for e in yaml.safe_load(p.read_text()) or []:
        name = e.get("short_name") or str(e.get("arxiv_id", ""))
        arxiv_id = str(e.get("arxiv_id") or "")
        if not arxiv_id:
            arxiv_id = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        out.append(
            Paper(
                arxiv_id=arxiv_id,
                section=e.get("section", "manual"),
                short_name=name,
                title=e.get("title", ""),
                starred=bool(e.get("starred", False)),
                depth=e.get("depth", "card"),
                source="manual",
                url=e.get("url", ""),
                pdf_url=e.get("pdf_url", ""),
            )
        )
    return out


def resolve(corpus: Path, skip_pull: bool = False) -> dict:
    sources = corpus / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    readme = (
        Path(sync_repo(corpus / ".cache"))
        if not skip_pull
        else corpus / ".cache" / "awesome-vla-wam" / "README.md"
    )
    papers, quarantined = parse_readme(readme.read_text())

    _dump_yaml(
        sources / "awesome-vla-wam.yaml",
        [{k: v for k, v in asdict(p).items() if k != "raw"} for p in papers],
    )
    _dump_yaml(
        sources / "quarantine.yaml",
        [{"section": q.section, "reason": q.reason, "raw": q.raw} for q in quarantined],
    )

    manual = load_manual(sources)
    merged: dict[str, Paper] = {p.arxiv_id: p for p in papers}
    for m in manual:
        merged[m.arxiv_id] = m  # manual wins
    _dump_yaml(
        sources / "resolved.yaml",
        [{k: v for k, v in asdict(p).items() if k != "raw"} for p in merged.values()],
    )

    by_section: dict[str, int] = {}
    for p in papers:
        by_section[p.section] = by_section.get(p.section, 0) + 1
    return {
        "resolved": len(merged),
        "awesome": len(papers),
        "manual": len(manual),
        "quarantined": len(quarantined),
        "by_section": by_section,
        "quarantine": [{"reason": q.reason, "raw": q.raw[:120]} for q in quarantined],
    }

"""paperdb CLI. One command per stage, each takes --json."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import load_config, write_default_config
from .stages import extract as extract_stage
from .stages import fetch as fetch_stage
from .stages import index as index_stage
from .stages import meta as meta_stage
from .stages import parse as parse_stage
from .stages import query as query_stage
from .stages import resolve as resolve_stage
from .stages.doctor import doctor as doctor_stage

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Local robotics-paper corpus.")


@app.callback()
def _main():
    """paperdb: local robotics-paper corpus."""


@app.command("doctor")
def doctor(as_json: bool = typer.Option(False, "--json", help="Machine-readable output")):
    """Probe every configured agent CLI: PATH, auth, envelope."""
    doctor_stage(json_output=as_json)


@app.command("resolve")
def resolve(as_json: bool = typer.Option(False, "--json", help="Machine-readable output")):
    """Sync awesome-vla-wam, parse entries, merge manual.yaml, write sources/."""
    write_default_config()
    r = resolve_stage.resolve(_corpus())
    if as_json:
        print(json.dumps(r, indent=2))
    else:
        print(f"resolved: {r['resolved']}  (awesome {r['awesome']} + manual {r['manual']})")
        print(f"quarantined: {r['quarantined']}")
        for s, n in sorted(r["by_section"].items()):
            print(f"  {s:28} {n}")


@app.command("meta")
def meta_cmd(
    limit: int = typer.Option(None, "--limit", help="Only first N papers"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
):
    """Fetch arXiv API metadata for resolved papers (3s between requests)."""
    r = meta_stage.meta(_corpus(), limit)
    _report(r, as_json)


@app.command("fetch")
def fetch_cmd(
    limit: int = typer.Option(None, "--limit", help="Only first N papers"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
):
    """Download paper.pdf for papers with meta (bounded concurrency)."""
    r = fetch_stage.fetch(_corpus(), limit)
    _report(r, as_json)


def _report(r: dict, as_json: bool):
    if as_json:
        print(json.dumps(r, indent=2))
    else:
        errors = r.pop("errors", None)
        extra = r.pop("missing_from_api", None)
        print(json.dumps(r, indent=2))
        if extra:
            print("missing_from_api:", extra)
        if errors:
            print("errors:", json.dumps(errors, indent=2))


@app.command("extract")
def extract_cmd(
    limit: int = typer.Option(None, "--limit", help="Only first N papers without cards"),
    adapter: str = typer.Option(None, "--adapter", help="Override [agent].adapter for this run"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
):
    """Extract card.json via the configured agent CLI (batched abstracts)."""
    r = extract_stage.extract(_corpus(), limit, adapter)
    _report(r, as_json)


@app.command("parse")
def parse_cmd(
    id: str = typer.Option(None, "--id", help="Parse a single paper"),
    all_papers: bool = typer.Option(False, "--all", help="Parse every fetched PDF"),
    force: bool = typer.Option(False, "--force", help="Re-parse even if paper.md exists"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
):
    """Convert paper.pdf -> paper.md with the configured [parse] backend."""
    if not (id or all_papers):
        print("give --id <arxiv_id> or --all")
        raise typer.Exit(1)
    r = parse_stage.parse(_corpus(), id, force)
    _report(r, as_json)


@app.command("index")
def index_cmd(as_json: bool = typer.Option(False, "--json", help="Machine-readable output")):
    """Rebuild corpus/index.db from files (FTS5 + flattened cards)."""
    r = index_stage.index(_corpus())
    _report(r, as_json)


@app.command("query")
def query_cmd(
    text: str = typer.Argument("", help="Free-text search (FTS, falls back to semantic)"),
    where: str = typer.Option(None, "--where", help="SQL predicate over card columns"),
    limit: int = typer.Option(20, "--limit"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
):
    """Query the corpus: e.g. paperdb query \"flow matching\" --where \"control_hz >= 30\""""
    rows = query_stage.query(text, where, limit=limit, corpus=_corpus())
    if as_json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            print(f"{r['arxiv_id']:12} {str(r['short_name'])[:24]:24} {r['title'][:70]}")


def _corpus() -> Path:
    return Path(load_config()["paths"]["corpus"])


if __name__ == "__main__":
    app()

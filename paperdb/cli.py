"""paperdb CLI. One command per stage, each takes --json."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import load_config, write_default_config
from .stages import fetch as fetch_stage
from .stages import meta as meta_stage
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


def _corpus() -> Path:
    return Path(load_config()["paths"]["corpus"])


if __name__ == "__main__":
    app()

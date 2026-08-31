"""paperdb CLI. One command per stage, each takes --json."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import load_config, write_default_config
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


def _corpus() -> Path:
    return Path(load_config()["paths"]["corpus"])


if __name__ == "__main__":
    app()

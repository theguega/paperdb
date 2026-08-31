"""paperdb CLI. One command per stage, each takes --json."""

from __future__ import annotations

import typer

from .stages.doctor import doctor as doctor_stage

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Local robotics-paper corpus.")


@app.callback()
def _main():
    """paperdb: local robotics-paper corpus."""


@app.command("doctor")
def doctor(json: bool = typer.Option(False, "--json", help="Machine-readable output")):
    """Probe every configured agent CLI: PATH, auth, envelope."""
    doctor_stage(json_output=json)


if __name__ == "__main__":
    app()

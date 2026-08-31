"""Configuration: one paperdb.toml, sane defaults if absent. No API keys, ever."""

from __future__ import annotations

import tomllib
from pathlib import Path

DEFAULT_TOML = """\
[agent]
adapter = "claude"        # claude | cursor | cline
concurrency = 2
timeout_s = 300           # mandatory: cursor print-mode can hang indefinitely

[agent.adapters.claude]
argv = ["claude", "-p", "--output-format", "json",
        "--allowedTools", "Read", "--max-turns", "3"]
prompt = "stdin"
envelope = "json"
text_path = "result"

[agent.adapters.cursor]
argv = ["agent", "-p", "--output-format", "json", "--trust", "--mode", "ask"]
prompt = "arg"
envelope = "json"
text_path = "result"

[agent.adapters.cline]
argv = ["cline", "--json"]
prompt = "arg"           # cline 3.x rejects piped stdin in --json mode
envelope = "ndjson"
text_path = "text"

[paths]
corpus = "corpus"

[parse]
backend = "pymupdf4llm"   # pymupdf4llm | docling (drop-in backends, local, no keys)

"""


def config_path() -> Path:
    return Path("paperdb.toml")


def load_config() -> dict:
    """Read paperdb.toml from cwd; fall back to defaults if missing."""
    p = config_path()
    if not p.exists():
        return tomllib.loads(DEFAULT_TOML)
    return tomllib.loads(p.read_text())


def write_default_config(path: Path | None = None) -> Path:
    p = path or config_path()
    if not p.exists():
        p.write_text(DEFAULT_TOML)
    return p

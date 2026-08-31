"""Adapter layer: shell out to whichever agent CLI the user is logged into.

No API keys, no SDKs. One subprocess per invocation. Three knobs per CLI:
argv template, prompt placement (stdin vs arg), output envelope shape.
Every invocation runs in a scratch cwd, is read-only by default (wired into
the argv presets in paperdb.toml), and is wrapped in a mandatory timeout
because cursor's print mode can hang indefinitely.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config

_PROMPT_PLACEMENT = {"stdin", "arg"}
_ENVELOPES = {"json", "ndjson"}
_LIMIT_HINTS = (
    "rate limit",
    "usage limit",
    "rate_limit",
    "429",
    "credit balance",
    "overloaded",
    "capacity",
)


class AgentError(RuntimeError):
    """Failure kinds: limit (retryable), timeout (retryable), failure, envelope."""

    def __init__(self, kind: str, message: str, exit_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.exit_code = exit_code


@dataclass(frozen=True)
class Adapter:
    name: str
    argv: list[str]
    prompt: str  # "stdin" | "arg"
    envelope: str  # "json" | "ndjson"
    text_path: str

    @classmethod
    def from_config(cls, name: str, cfg: dict) -> Adapter:
        a = cls(
            name=name,
            argv=[str(x) for x in cfg["argv"]],
            prompt=cfg.get("prompt", "stdin"),
            envelope=cfg.get("envelope", "json"),
            text_path=cfg.get("text_path", "result"),
        )
        if a.prompt not in _PROMPT_PLACEMENT:
            raise AgentError("config", f"adapter {name!r}: prompt must be stdin|arg")
        if a.envelope not in _ENVELOPES:
            raise AgentError("config", f"adapter {name!r}: envelope must be json|ndjson")
        return a


def get_adapter(name: str | None = None, config: dict | None = None) -> Adapter:
    cfg = config or load_config()
    name = name or cfg["agent"]["adapter"]
    try:
        block = cfg["agent"]["adapters"][name]
    except KeyError as e:
        raise AgentError("config", f"no [agent.adapters.{name}] block in paperdb.toml") from e
    return Adapter.from_config(name, block)


def strip_fences(text: str) -> str:
    """Models fence JSON even when told not to."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _dig(obj: Any, path: str) -> str | None:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur if isinstance(cur, str) else None


def extract_text(adapter: Adapter, stdout: str) -> str:
    """Pull the payload text out of a CLI's envelope. Raises on unparseable json."""
    if adapter.envelope == "json":
        s = stdout.strip()
        # Some CLIs emit log lines before the JSON object; start at the first '{'.
        if not s.startswith("{"):
            i = s.find("{")
            if i < 0:
                raise AgentError("envelope", "no JSON object on stdout")
            s = s[i:]
        obj = json.loads(s)
        text = _dig(obj, adapter.text_path)
        if text is None:
            raise AgentError("envelope", f"missing text_path {adapter.text_path!r} in envelope")
        return strip_fences(text)
    # ndjson: ignore unparseable lines, drop partials, concatenate text fields.
    parts: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("partial") is True:
            continue
        text = _dig(obj, adapter.text_path)
        if text:
            parts.append(text)
    return "".join(parts)


def _classify(code: int, stderr: str) -> AgentError:
    low = stderr.lower()
    if any(h in low for h in _LIMIT_HINTS):
        return AgentError("limit", stderr.strip()[:500], code)
    return AgentError("failure", stderr.strip()[:500] or f"exit code {code}", code)


async def invoke(
    adapter: Adapter, prompt: str, *, timeout_s: float, cwd: Path | None = None
) -> tuple[int, str, str]:
    """Run the CLI once. Returns (exit_code, stdout, stderr). Timeout is mandatory."""
    argv = list(adapter.argv)
    stdin_data: bytes | None = None
    if adapter.prompt == "arg":
        argv.append(prompt)
    else:
        stdin_data = prompt.encode()

    scratch = cwd or Path(tempfile.mkdtemp(prefix="paperdb-agent-"))
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=scratch,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin_data), timeout=timeout_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise AgentError("timeout", f"{adapter.name}: timed out after {timeout_s}s") from None
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


def _with_files(prompt: str, files: list[str] | tuple[str, ...]) -> str:
    if not files:
        return prompt
    paths = "\n".join(str(Path(f).resolve()) for f in files)
    return (
        f"{prompt}\n\n"
        "Read these files with your file-read tool (absolute paths; never paste "
        f"file contents into a pipe or argv):\n{paths}"
    )


async def run_async(
    prompt: str,
    files: list[str] | tuple[str, ...] = (),
    *,
    adapter: Adapter | None = None,
    timeout_s: float | None = None,
) -> str:
    """One read-only agent invocation; returns the extracted text."""
    a = adapter or get_adapter()
    cfg = load_config()
    t = timeout_s if timeout_s is not None else float(cfg["agent"]["timeout_s"])
    code, out, err = await invoke(a, _with_files(prompt, files), timeout_s=t)
    if code != 0:
        raise _classify(code, err)
    text = extract_text(a, out)
    if not text.strip():
        raise AgentError("envelope", f"{a.name}: empty payload")
    return text


def run(
    prompt: str,
    files: list[str] | tuple[str, ...] = (),
    *,
    adapter: Adapter | None = None,
    timeout_s: float | None = None,
) -> str:
    return asyncio.run(run_async(prompt, files, adapter=adapter, timeout_s=timeout_s))


def probe(adapter: Adapter, *, timeout_s: float = 60) -> dict:
    """Doctor check: on PATH? authenticates? envelope parses?"""
    want = '{"ok": true}'
    if shutil.which(adapter.argv[0]) is None:
        return {"adapter": adapter.name, "status": "not-on-path", "detail": adapter.argv[0]}
    try:
        code, out, err = asyncio.run(
            invoke(adapter, "Reply with exactly " + want, timeout_s=timeout_s)
        )
    except AgentError as e:
        return {"adapter": adapter.name, "status": e.kind, "detail": str(e)}
    if code != 0:
        low = err.lower()
        if any(h in low for h in ("log in", "login", "auth", "sign in", "unauthorized")):
            return {"adapter": adapter.name, "status": "auth", "detail": err.strip()[:300]}
        e = _classify(code, err)
        return {"adapter": adapter.name, "status": e.kind, "detail": str(e)}
    try:
        text = extract_text(adapter, out)
    except AgentError as e:
        return {"adapter": adapter.name, "status": "envelope", "detail": str(e)}
    if want.replace(" ", "") not in text.replace(" ", "").replace("'", '"'):
        return {
            "adapter": adapter.name,
            "status": "envelope",
            "detail": f"payload: {text[:200]!r}",
        }
    return {"adapter": adapter.name, "status": "ok", "detail": text[:100]}

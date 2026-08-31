"""Envelope tests: one recorded-stdout fixture per adapter, plus fences and ndjson partials."""

from __future__ import annotations

import json
import sys

import pytest

from paperdb.agent import Adapter, AgentError, extract_text, run, strip_fences

CLAUDE = Adapter("claude", ["claude"], "stdin", "json", "result")
CURSOR = Adapter("cursor", ["agent"], "arg", "json", "result")
CLINE = Adapter("cline", ["cline"], "stdin", "ndjson", "text")


# --- json envelope (claude / cursor): recorded stdout shapes -------------------


def test_claude_json_envelope():
    stdout = json.dumps(
        {"type": "result", "subtype": "success", "result": '{"family": "vla"}', "is_error": False}
    )
    assert extract_text(CLAUDE, stdout) == '{"family": "vla"}'


def test_cursor_json_envelope_with_log_preamble():
    stdout = 'some log line\n{"result":"{\\"ok\\":true}"}\n'
    assert extract_text(CURSOR, stdout) == '{"ok":true}'


def test_json_envelope_missing_text_path():
    with pytest.raises(AgentError) as e:
        extract_text(CLAUDE, json.dumps({"type": "result"}))
    assert e.value.kind == "envelope"


# --- ndjson envelope (cline): partials and junk lines --------------------------


def test_cline_ndjson_drops_partials_and_junk():
    stream = "\n".join(
        [
            json.dumps({"partial": True, "text": "thinking out lo"}),
            json.dumps({"partial": True, "text": "ud"}),
            "not json at all",
            "",
            json.dumps({"partial": False, "text": '{"a": 1}'}),
            json.dumps({"text": '{"b": 2}'}),
        ]
    )
    assert extract_text(CLINE, stream) == '{"a": 1}{"b": 2}'


def test_ndjson_all_partials_gives_empty():
    stream = json.dumps({"partial": True, "text": "x"})
    assert extract_text(CLINE, stream) == ""


# --- fence stripping ------------------------------------------------------------


def test_strip_fences():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fences('{"a": 1}') == '{"a": 1}'


def test_extract_strips_fenced_payload():
    stdout = json.dumps({"result": '```json\n{"ok": true}\n```'})
    assert extract_text(CLAUDE, stdout) == '{"ok": true}'


# --- end-to-end subprocess through a fake CLI (no network, no real agent) ------

FAKE_STDIN_JSON = (
    'import json,sys\np = sys.stdin.read()\nprint(json.dumps({"result": p.strip()}))\n'
)

FAKE_NDJSON = (
    "import json,sys\n"
    "p = sys.stdin.read()\n"
    'print(json.dumps({"partial": True, "text": "IGNORE"}))\n'
    'print(json.dumps({"text": p.strip()}))\n'
)


def _fake_adapter(argv_script: str, **kw) -> Adapter:
    return Adapter("fake", [sys.executable, "-c", argv_script], "stdin", "json", "result", **kw)


def test_run_stdin_roundtrip():
    a = _fake_adapter(FAKE_STDIN_JSON)
    assert run("hello paperdb", adapter=a, timeout_s=30) == "hello paperdb"


def test_run_strips_fenced_output():
    script = (
        "import json,sys\n"
        "p = sys.stdin.read()\n"
        'print(json.dumps({"result": "```json\\n" + p.strip() + "\\n```"}))\n'
    )
    a = _fake_adapter(script)
    assert run("body", adapter=a, timeout_s=30) == "body"


def test_run_files_get_absolute_paths_in_prompt(tmp_path):
    a = _fake_adapter(FAKE_STDIN_JSON)
    f = tmp_path / "paper.md"
    f.write_text("x")
    out = run("extract", files=[f], adapter=a, timeout_s=30)
    assert str(f.resolve()) in out


def test_run_nonzero_exit_is_failure():
    a = Adapter("bad", [sys.executable, "-c", "import sys; sys.exit(3)"], "stdin", "json", "result")
    with pytest.raises(AgentError) as e:
        run("x", adapter=a, timeout_s=30)
    assert e.value.kind == "failure"


def test_run_timeout_is_retryable():
    script = "import time; time.sleep(30)"
    a = Adapter("slow", [sys.executable, "-c", script], "stdin", "json", "result")
    with pytest.raises(AgentError) as e:
        run("x", adapter=a, timeout_s=1)
    assert e.value.kind == "timeout"

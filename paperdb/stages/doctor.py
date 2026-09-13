"""doctor: probe the parse backend and every configured adapter before doing anything else."""

from __future__ import annotations

from .. import hf
from ..agent import get_adapter, probe
from ..config import load_config


def doctor(json_output: bool = False) -> list[dict]:
    cfg = load_config()
    names = list(cfg["agent"]["adapters"])
    timeout = min(float(cfg["agent"]["timeout_s"]), 120)
    results = [probe(get_adapter(n, cfg), timeout_s=timeout) for n in names]
    active = cfg["agent"]["adapter"]

    backend = cfg.get("parse", {}).get("backend", "hf")
    parse_status = {
        "backend": backend,
        "hf_cli": "ok" if hf.available() else "not-on-path",
    }

    if json_output:
        import json

        print(json.dumps({"active": active, "parse": parse_status, "adapters": results}, indent=2))
    else:
        print(f"paperdb doctor - active adapter: {active}\n")
        for r in results:
            mark = {"ok": "ok", "not-on-path": "!!", "auth": "!!"}.get(r["status"], "!!")
            print(f"[{mark}] {r['adapter']:8} {r['status']:12} {r['detail']}")
        mark = "ok" if parse_status["hf_cli"] == "ok" else "!!"
        detail = (
            "hf papers read"
            if parse_status["hf_cli"] == "ok"
            else "install: curl -LsSf https://hf.co/cli/install.sh | bash"
        )
        print(f"\n[{mark}] {'hf':8} {parse_status['hf_cli']:12} {detail}")
        if backend == "hf" and parse_status["hf_cli"] != "ok":
            print(
                '     [parse] backend = "hf" but the CLI is missing; every paper '
                "will fall back to PDF conversion."
            )
        print("\nRun doctor again after installing/logging into any CLI.")
    return results

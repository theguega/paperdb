"""doctor: probe every configured adapter before doing anything else."""

from __future__ import annotations

from ..agent import get_adapter, probe
from ..config import load_config


def doctor(json_output: bool = False) -> list[dict]:
    cfg = load_config()
    names = list(cfg["agent"]["adapters"])
    timeout = min(float(cfg["agent"]["timeout_s"]), 120)
    results = [probe(get_adapter(n, cfg), timeout_s=timeout) for n in names]
    active = cfg["agent"]["adapter"]
    if json_output:
        import json

        print(json.dumps({"active": active, "adapters": results}, indent=2))
    else:
        print(f"paperdb doctor - active adapter: {active}\n")
        for r in results:
            mark = {"ok": "ok", "not-on-path": "!!", "auth": "!!"}.get(r["status"], "!!")
            print(f"[{mark}] {r['adapter']:8} {r['status']:12} {r['detail']}")
        print("\nRun doctor again after installing/logging into any CLI.")
    return results

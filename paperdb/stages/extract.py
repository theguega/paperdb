"""extract: agent-run structured extraction into papers/<id>/card.json.

Card depth batches abstracts (~20 per invocation); full depth does one
invocation per paper with paper.md (comes with the promote flow).
Prompts are fully self-contained - no follow-up turn exists.
"""

from __future__ import annotations

import asyncio
import json
import re

from pydantic import ValidationError

from ..agent import AgentError, run_async
from ..paths import load_resolved, paper_dir, papers_dir
from ..schemas import Card

BATCH_SIZE = 20

PROMPT_TMPL = """You are extracting structured metadata from robotics papers for a research corpus.

Below are {n} paper records. For EACH, produce exactly one JSON object.

Output rules (strict):
- Output ONLY a JSON array of objects. No prose, no markdown fences, no explanation.
- Every object must include "arxiv_id" copied exactly from the record.
- Emit null for anything the record does not state explicitly. NEVER guess or infer a number.
- Never ask questions; the record below is all the information that exists.

Per-object schema:
- "family": one of "vla","wam","diffusion-policy","rl","world-model","dataset","benchmark"
- "backbone": string name of the vision-language backbone, or null
- "action_head": one of "flow-matching","diffusion","fast-tokens","ar-bins","latent","mlp", or null
- "action_space": one of "joint","ee-delta","ee-abs","latent", or null
- "chunk_size": action chunk size (integer) if stated, else null
- "control_hz": control frequency in Hz (number) if stated, else null
- "embodiment": list of robot embodiments (e.g. ["Franka","ALOHA"]); [] if none stated
- "data": {{"hours": number|null, "episodes": integer|null, "source": "in-house"|"OXE"|"human-video"|"mixed"|null}}
- "eval": {{"sim": {{benchmark: number}}, "real": {{benchmark: number}}}} - success rates as given; {{}} if none
- "open": {{"weights": true|false|null, "code": url-string|null, "data": true|false|null}} - whether weights/code/datasets are released
- "compute": training compute if stated (e.g. "64 A100s"), else null
- "limits": list of failure modes or limitations the paper itself admits; [] if none stated

Records:

{records}"""


def _records_text(papers: list[dict]) -> str:
    out = []
    for p in papers:
        abstract = p.get("abstract") or "(no abstract available)"
        out.append(f"[{p['arxiv_id']}] {p.get('title', '')}\n{abstract}")
    return "\n\n".join(out)


def _parse_array(text: str) -> list[dict]:
    """Parse a JSON array out of the model payload; tolerate stray prose."""
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"no JSON array in payload ({len(text)} chars)")


def _save_raw(corpus, name: str, text: str):
    d = papers_dir(corpus).parent / ".cache" / "agent-raw"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.txt").write_text(text)


def _write_cards(corpus, results: dict[str, dict], raw: str, errors: dict[str, str]):
    """results: id -> validated card dict; errors: id -> validation message."""
    for arxiv_id, card in results.items():
        d = paper_dir(arxiv_id, corpus)
        d.mkdir(parents=True, exist_ok=True)
        (d / "card.json").write_text(
            json.dumps({"status": "ok", "card": card}, indent=2, ensure_ascii=False) + "\n"
        )
    for aid, err in errors.items():
        d = paper_dir(aid, corpus)
        d.mkdir(parents=True, exist_ok=True)
        (d / "card.json").write_text(
            json.dumps(
                {"status": "failed", "raw": raw[:2000], "error": err},
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )


async def _invoke_batch(prompt: str, adapter, timeout_s: float) -> tuple[str, list[dict]]:
    """One agent invocation -> (raw payload, parsed items). Timeout is retryable once."""
    last_err: AgentError | None = None
    for _ in range(2):
        try:
            raw = await run_async(prompt, adapter=adapter, timeout_s=timeout_s)
            return raw, _parse_array(raw)
        except AgentError as e:
            if e.kind == "timeout":
                last_err = e
                continue
            raise
    raise last_err  # type: ignore[misc]


def _validate(items: list[dict]) -> tuple[dict[str, dict], dict[str, str]]:
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for it in items:
        if not isinstance(it, dict) or not it.get("arxiv_id"):
            continue
        aid = str(it["arxiv_id"])
        try:
            results[aid] = Card.model_validate(
                {k: v for k, v in it.items() if k != "arxiv_id"}
            ).model_dump()
        except ValidationError as e:
            errors[aid] = str(e.errors()[-1])[:300]
    return results, errors


async def _run_batches(
    corpus, batches: list[list[dict]], adapter, timeout_s: float, raw_prefix: str
) -> dict:
    ok, failed, stopped = 0, 0, None
    for bi, papers in enumerate(batches):
        prompt = PROMPT_TMPL.format(n=len(papers), records=_records_text(papers))
        try:
            raw, items = await _invoke_batch(prompt, adapter, timeout_s)
        except AgentError as e:
            if e.kind == "limit":
                stopped = f"rate limit at batch {bi}; re-run to resume"
                break
            # hard failure: mark this batch failed, keep going with the rest
            for p in papers:
                _write_cards(corpus, {}, "", {p["arxiv_id"]: str(e)})
                failed += 1
            continue
        _save_raw(corpus, f"{raw_prefix}-b{bi:03d}", raw)
        results, errors = _validate(items)

        if errors:  # one retry with the validation errors appended
            retry = (
                prompt + "\n\nYour previous reply failed schema validation for these records. "
                "Re-emit ONLY the full JSON array, fixing these errors:\n"
                + "\n".join(f"- {aid}: {msg}" for aid, msg in errors.items())
            )
            try:
                raw, items = await _invoke_batch(retry, adapter, timeout_s)
                _save_raw(corpus, f"{raw_prefix}-b{bi:03d}-retry", raw)
                results, errors = _validate(items)
            except AgentError as e:
                results, errors = {}, {p["arxiv_id"]: f"retry failed: {e}" for p in papers}

        _write_cards(corpus, results, raw, errors)
        ok += len(results)
        failed += len(errors)
    return {"ok": ok, "failed": failed, "stopped": stopped}


def _has_valid_card(corpus, arxiv_id: str) -> bool:
    cp = paper_dir(arxiv_id, corpus) / "card.json"
    if not cp.exists():
        return False
    try:
        return json.loads(cp.read_text()).get("status") == "ok"
    except (json.JSONDecodeError, OSError):
        return False


def _todo(corpus, limit: int | None) -> list[dict]:
    todo = []
    for r in load_resolved(corpus):
        if _has_valid_card(corpus, r["arxiv_id"]):
            continue  # resume: skip anything with a valid card.json
        mp = paper_dir(r["arxiv_id"], corpus) / "meta.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        todo.append(
            {
                "arxiv_id": r["arxiv_id"],
                "title": m.get("title", ""),
                "abstract": m.get("abstract", ""),
            }
        )
    return todo[:limit] if limit else todo


def extract(corpus, limit: int | None = None, adapter_name: str | None = None) -> dict:
    from ..agent import get_adapter
    from ..config import load_config

    cfg = load_config()
    adapter = get_adapter(adapter_name, cfg)
    todo = _todo(corpus, limit)
    batches = [todo[i : i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    result = asyncio.run(
        _run_batches(corpus, batches, adapter, float(cfg["agent"]["timeout_s"]), "card")
    )
    result["remaining"] = sum(
        1 for r in load_resolved(corpus) if not _has_valid_card(corpus, r["arxiv_id"])
    )
    return result

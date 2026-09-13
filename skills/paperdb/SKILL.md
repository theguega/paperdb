---
name: paperdb
description: "Work with the local robotics-paper corpus (paperdb). Use when the agent needs to (1) search or filter papers by topic, family (vla/wam/world-model/diffusion-policy/rl/benchmark/dataset/control/library), open-weights, control_hz, data_hours, or any card field; (2) add a paper to the corpus (pull its markdown, extract a card, and index it); or (3) run the paperdb CLI (resolve/meta/parse/extract/index/query/fetch). Triggers include mentions of 'paperdb', 'add a paper to the corpus', 'search the papers for <topic>', '.claude/skills/paperdb', or the corpus/ dir."
---

# paperdb: local robotics-paper corpus

`paperdb` is a CLI over a local corpus of robotics papers (VLA, world-action models, control). Run every command from the repo root with the venv active (`source .venv/bin/activate`, or use `.venv/bin/paperdb ...`).

## Search / query

| Goal | Command |
|------|---------|
| Free-text search (FTS over title+abstract+notes; falls back to semantic search over full-text chunks on paraphrases) | `paperdb query "flow matching"` |
| Structured filter on card fields | `paperdb query "" --where "family = 'vla'"` |
| Combined text + filter | `paperdb query "flow" --where "open_weights = 1 and control_hz >= 30"` |
| Machine-readable rows (for agents) | `paperdb query "<text>" --where "..." --json` |
| More/fewer results | append `--limit N` |

- Filterable card columns: `family, backbone, action_head, action_space, chunk_size, control_hz, open_weights, open_code, open_data, data_hours, data_episodes, data_source`.
- Common `family` values: `vla, wam, world-model, diffusion-policy, rl, benchmark, dataset, control, library`.
- Each result is a row with `arxiv_id`; the full text body lives at `corpus/papers/<arxiv_id>/paper.md` - read it when you need to answer from actual paper content.
- `index.db` is rebuilt from files; if searches look stale, rerun `paperdb index`.

## Add a paper

End-to-end flow (arXiv paper):

1. **Add to `corpus/sources/manual.yaml`** - append, with required keys for the entry type:
   - arXiv paper: `arxiv_id` (e.g. `'2410.11758'`), `short_name`, `title`.
   - Non-arXiv (slug key): `short_name`, `title`, `pdf_url` (no `arxiv_id`).
2. `paperdb resolve` - re-parses sources and regenerates `corpus/sources/resolved.yaml` (manual wins on conflict).
3. `paperdb meta` - writes `papers/<id>/meta.json` (arXiv API + Hugging Face Hub extras).
4. `paperdb parse --id <arxiv_id>` - writes `papers/<id>/paper.md`.
5. `paperdb index` - rebuilds `corpus/index.db` so the new paper is searchable.
6. Verify with `paperdb query "<short_name>"`.

There is **no separate download step**: `parse` pulls markdown from the Hugging Face Hub (`hf papers read`), which needs no PDF and takes about a second. Only if the Hub can't serve that paper does it download the PDF itself and convert it. Check the result - `via_hf: 1` means no PDF was touched, `via_pdf: 1` means it fell back.

Optional extras: `paperdb extract` uses the configured agent CLI (claude/cursor/cline) to build a structured `card.json`; run `paperdb doctor` first to confirm an agent CLI is on PATH. Set `depth: full` in manual.yaml to get deeper chunk-level semantic indexing.

## Paper text: Hub vs PDF

- Default backend is `hf`; `[parse] fallback_backend` (pymupdf4llm) handles the rest. Requires the `hf` CLI on PATH (`curl -LsSf https://hf.co/cli/install.sh | bash`).
- The Hub covers ~3 in 4 of this corpus. It has no HTML for most pre-2022 papers and doesn't index every arXiv ID.
- **The Hub flattens tables into space-separated text.** If you need numbers from a results table (`control_hz`, success rates, data hours) and `paper.md` looks ambiguous, re-parse that one paper from its PDF: `paperdb parse --id <id> --backend pymupdf4llm --force`.
- `paperdb fetch` is now only for bulk PDF pre-download; it skips any paper that already has `paper.md` (use `--all` to override).
- Don't run Hub-backed parses in parallel - the Hub throttles hard and returns an opaque "Request ID" error. The CLI already serialises them.

## Notes

- Corpus layout is fixed: `corpus/sources/*.yaml` are the seed of truth, `papers/<id>/` holds `meta.json`/`card.json`/`paper.md` (plus `paper.pdf` only when the Hub couldn't serve it), and `corpus/index.db` is derived (never edit by hand; always `paperdb index`).
- Don't hand-edit `resolved.yaml` or `index.db` - edit `manual.yaml`, then `paperdb resolve`.

## Refresh the curated list

`paperdb resolve` git-pulls the upstream awesome-vla-wam list into `corpus/.cache/` and re-derives `resolved.yaml`, so it is both the fetch and the update. After it reports a higher `resolved` count, ingest the new papers: `paperdb meta`, `paperdb parse --all`, `paperdb extract`, `paperdb index`.

Entries upstream that carry an arXiv link are kept even when they lack a `**Bold**` name (the title becomes the short_name). What lands in `corpus/sources/quarantine.yaml` is entries with **no arXiv link at all** - tooling (MuJoCo, ROS, PyBullet), websites, and dataset pages. Those have no paper to ingest; add one to `manual.yaml` with a `pdf_url` if you want it in the corpus.
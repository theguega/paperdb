---
name: paperdb
description: "Work with the local robotics-paper corpus (paperdb). Use when the agent needs to (1) search or filter papers by topic, family (vla/wam/world-model/diffusion-policy/rl/benchmark/dataset/control/library), open-weights, control_hz, data_hours, or any card field; (2) add a paper to the corpus (fetch its PDF, convert it to markdown, and index it); or (3) run the paperdb CLI (resolve/meta/fetch/extract/parse/index/query). Triggers include mentions of 'paperdb', 'add a paper to the corpus', 'search the papers for <topic>', '.claude/skills/paperdb', or the corpus/ dir."
---

# paperdb: local robotics-paper corpus

`paperdb` is a CLI over a local corpus of robotics papers (VLA, world-action models, control). Run every command from the repo root with the venv active (`source .venv/bin/activate`, or use `.venv/bin/paperdb ...`).

## Search / query

| Goal | Command |
|------|---------|
| Free-text search (FTS over title+abstract+notes) | `paperdb query "flow matching"` |
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
3. `paperdb meta` - fetches arXiv API metadata into `papers/<id>/meta.json`.
4. `paperdb fetch` - downloads `papers/<id>/paper.pdf`.
5. `paperdb parse --all` (or `--id <arxiv_id>`) - converts the PDF to `papers/<id>/paper.md`.
6. `paperdb index` - rebuilds `corpus/index.db` so the new paper is searchable.
7. Verify with `paperdb query "<short_name>"`.

Optional extras: `paperdb extract` uses the configured agent CLI (claude/cursor/cline) to build a structured `card.json`; run `paperdb doctor` first to confirm an agent CLI is on PATH. Set `depth: full` in manual.yaml to get deeper chunk-level semantic indexing.

## Notes

- Corpus layout is fixed: `corpus/sources/*.yaml` are the seed of truth, `papers/<id>/` holds `meta.json`/`card.json`/`paper.pdf`/`paper.md`, and `corpus/index.db` is derived (never edit by hand; always `paperdb index`).
- Don't hand-edit `resolved.yaml` or `index.db` - edit `manual.yaml`, then `paperdb resolve`.
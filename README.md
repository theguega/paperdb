# paperdb

Local corpus of robotics papers (VLA, world-action models, control) that an agent can query.

## Pipeline

One command per stage, each takes `--json`:

1. `resolve` - syncs awesome-vla-wam, parses entries, merges `manual.yaml`
2. `meta` - fetches arXiv API metadata
3. `fetch` - downloads `paper.pdf`
4. `extract` - builds structured `card.json` via an agent CLI (claude/cursor/cline)
5. `parse` - converts `paper.pdf` -> `paper.md`
6. `index` - rebuilds `corpus/index.db` (FTS5 + flattened cards + sqlite-vec chunks)
7. `query` - free-text and/or structured search

## Usage

```bash
paperdb resolve
paperdb meta
paperdb fetch
paperdb extract
paperdb parse --all
paperdb index
paperdb query "flow matching" --where "family = 'vla'"
```

`corpus/sources/*.yaml` are the seed of truth; `papers/<id>/` holds the PDF/markdown/cards; `corpus/index.db` is derived and rebuilt from files. Large artifacts (PDFs, `.venv`, the index) are gitignored.
# paperdb

Local corpus of robotics papers (VLA, world-action models, control) that an agent can query.

## Pipeline

One command per stage, each takes `--json`:

1. `resolve` - git-pulls awesome-vla-wam, parses entries, merges `manual.yaml`
2. `meta` - fetches arXiv API metadata, enriched with Hugging Face Hub extras
3. `parse` - builds `paper.md` (Hub markdown, else PDF conversion)
4. `extract` - builds structured `card.json` via an agent CLI (claude/cursor/cline)
5. `index` - rebuilds `corpus/index.db` (FTS5 + flattened cards + sqlite-vec chunks)
6. `query` - free-text and/or structured search

`fetch` still exists for bulk PDF pre-download, but the pipeline no longer needs
it: `parse` downloads a PDF on demand when it has to fall back.

## Usage

```bash
paperdb resolve
paperdb meta
paperdb parse --all
paperdb extract
paperdb index
paperdb query "flow matching" --where "family = 'vla'"
```

## Where paper text comes from

`parse` has two routes, chosen by `[parse] backend` in `paperdb.toml`:

- **`hf`** (default) - `hf papers read <id>` returns arXiv's HTML rendering as
  markdown. No PDF, about a second per paper. Requires the Hugging Face CLI:
  `curl -LsSf https://hf.co/cli/install.sh | bash`.
- **`pymupdf4llm`** / **`docling`** - convert a local `paper.pdf`.

The Hub serves roughly 3 in 4 of this corpus. It has no HTML for most pre-2022
papers and doesn't index every arXiv ID, so anything it misses falls through to
`fallback_backend`, which downloads the PDF on demand. Set `backend =
"pymupdf4llm"` to skip the Hub entirely and work fully offline from PDFs.

Two Hub quirks the wrapper handles, both of which would otherwise corrupt the
corpus silently: a handful of IDs exit 0 with a few hundred bytes of an SVG
filename instead of a paper (rejected by a size floor), and concurrent requests
are throttled with an opaque "Request ID" error rather than a 429 (calls are
serialised and retried). Note that the Hub flattens tables into space-separated
text, where `pymupdf4llm` keeps pipe-table structure - if a paper's numbers
matter more than its prose, `parse --id <id> --backend pymupdf4llm --force`
re-does that one from the PDF.

Re-running `resolve` is how the curated list is refreshed: it pulls upstream
and re-derives `resolved.yaml`. Entries with an arXiv link are kept even without
a `**Bold**` name; `quarantine.yaml` collects only entries with no arXiv link
(tooling, websites, dataset pages), which have no paper to ingest.

`corpus/sources/*.yaml` are the seed of truth; `papers/<id>/` holds the
markdown/cards (and a PDF only when one was needed); `corpus/index.db` is
derived and rebuilt from files. Large artifacts (PDFs, `.venv`, the index) are
gitignored.
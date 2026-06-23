# pr-review-agent

An autonomous GitHub PR review agent: given a pull request, it retrieves
relevant code context, generates a small number of high-signal review
comments, and posts them. Ships with an eval harness that measures agent
quality against historical human reviews.

No agent frameworks — the agent loop is plain Python over raw HTTP/SDK calls.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| 1 | Scaffold, GitHub ingestion, diff parsing, `pra fetch` | done |
| 2 | Context retrieval (AST-based symbol resolution) | done |
| 3 | Agent loop (triage → review, confidence gate, dedup) | done |
| 4 | Posting + dry-run + idempotency | done |
| 5 | Eval harness (dataset, run, judge, report) | done |
| + | RAG retrieval add-on (hybrid vector+BM25, retrieval eval) | in progress |

## Install

Requires Python 3.11 or 3.12 (both tested in CI; `.python-version` pins
3.11). Create the venv with an explicit interpreter rather than `python3`,
whose default version varies by machine:

```sh
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

The RAG retrieval path is an optional extra (local embeddings + LanceDB);
the AST path needs none of it:

```sh
.venv/bin/pip install -e ".[rag]"
```

## Configuration

Copy `config.toml.example` to `config.toml` and adjust. Secrets come from
environment variables only (`GITHUB_TOKEN`, `GEMINI_API_KEY`,
`ANTHROPIC_API_KEY`); they are never read from the config file or logged.

## Usage

```sh
# Sanity check: fetch a PR and print a parsed diff summary
pra fetch owner/repo 123

# Check out the PR head and print the symbol context the agent would retrieve
pra context owner/repo 123

# Review a PR — dry run by default: prints comments, posts nothing
pra review owner/repo 123

# Actually post the review (single PR review, idempotent per head SHA)
pra review owner/repo 123 --post

# Summarize model spend per run (cache hits, tokens, cost, estimator drift)
pra cost report

# Eval harness: collect human-reviewed PRs, run the agent, judge, report
pra eval build-dataset owner/repo --since 2026-01-01 --out dataset/
pra eval run dataset/ --backend gemini --max-cases 3   # agent over each case
pra eval judge dataset/ --backend gemini --delay 5     # Claude Code judge
pra eval report dataset/

# RAG path (needs the rag extra): build the per-commit index, choose a retriever
pra index owner/repo --ref <sha>
pra review owner/repo 123 --retriever hybrid

# Retrieval benchmark: AST vs RAG vs hybrid over a gold set
pra retrieval-eval gold.jsonl --out retrieval.md
```

Model backends read keys from the environment: `GEMINI_API_KEY` for the
agent loop and the Gemini-Pro ceiling, `ANTHROPIC_API_KEY` for the
`anthropic` backend. The default eval judge backend is `claude-code`, which
shells out to the `claude` CLI on a subscription (no API key, recorded at
$0). Ollama needs no key. Every model call goes through a SQLite cache —
re-running an identical prompt is free and logged.

`GITHUB_TOKEN` is optional for public repos but strongly recommended
(unauthenticated requests are limited to 60/hour).

## Retrieval: AST baseline + RAG

Review quality depends on the context handed to the model. The agent builds
that context with one of three interchangeable retrievers, selected per run
with `--retriever {ast,rag,hybrid}` (default `ast`); each returns the same
`PRContext`, so the review engine is agnostic to the choice.

- **ast** (baseline) — for each changed file, parse it, find which imported
  names the added lines reference, resolve those imports to files in the repo,
  and extract just the referenced function/class definitions. Deterministic and
  precise for direct references; no model, no index.
- **rag** — AST-aware chunking of the repo (functions, methods, class
  skeletons), local code embeddings (`jina-embeddings-v2-base-code` via
  sentence-transformers), and a LanceDB table carrying both a dense vector and a
  BM25 full-text index. A query built from the changed lines runs vector and
  full-text search; the two rankings are fused with **Reciprocal Rank Fusion**.
- **hybrid** — union of the AST and RAG results, de-duplicated and re-ranked
  under one token budget.

Indexes are cached on disk keyed by repo + commit SHA. Embeddings and search
run locally; no paid retrieval APIs.

### Retrieval benchmark

`pra retrieval-eval` scores each retriever against a gold set — for a set of
PRs, the repo symbols that *should* be retrieved for each changed file — and
reports **recall@k**, **MRR**, and **nDCG@k**. Gold is retriever-independent
(the symbols the changed lines actually reference and that are defined in the
repo), so it does not bake in any one retriever's behavior.

```sh
pra retrieval-eval gold.jsonl --retrievers ast,rag,hybrid --out retrieval.md
```

Each gold line pins one PR's expected symbols (ids are `module_path::name`):

```json
{"repo": "owner/repo", "number": 123, "sha": "<head-sha>",
 "queries": [{"file": "pkg/a.py", "relevant": ["pkg/util.py::helper"]}]}
```

| Retriever | Queries | Recall@5 | Recall@10 | MRR | nDCG@10 |
|-----------|--------:|---------:|----------:|----:|--------:|
| ast       | – | – | – | – | – |
| rag       | – | – | – | – | – |
| hybrid    | – | – | – | – | – |

<!-- Numbers populated by running the command above against the gold set. -->

## Development

```sh
ruff check .          # lint
ruff format --check . # formatting
mypy                  # strict type check
pytest                # tests (offline; fixtures only, no live API calls)
```

Design decisions with trade-offs are recorded in [DESIGN.md](DESIGN.md).

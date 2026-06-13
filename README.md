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

## Install

Requires Python 3.11 or 3.12 (both tested in CI; `.python-version` pins
3.11). Create the venv with an explicit interpreter rather than `python3`,
whose default version varies by machine:

```sh
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
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
```

Model backends read keys from the environment: `GEMINI_API_KEY` for the
agent loop and the Gemini-Pro ceiling, `ANTHROPIC_API_KEY` for the
`anthropic` backend. The default eval judge backend is `claude-code`, which
shells out to the `claude` CLI on a subscription (no API key, recorded at
$0). Ollama needs no key. Every model call goes through a SQLite cache —
re-running an identical prompt is free and logged.

`GITHUB_TOKEN` is optional for public repos but strongly recommended
(unauthenticated requests are limited to 60/hour).

## Development

```sh
ruff check .          # lint
ruff format --check . # formatting
mypy                  # strict type check
pytest                # tests (offline; fixtures only, no live API calls)
```

Design decisions with trade-offs are recorded in [DESIGN.md](DESIGN.md).

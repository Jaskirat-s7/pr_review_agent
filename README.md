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
| 2 | Context retrieval (AST-based symbol resolution) | planned |
| 3 | Agent loop (triage → review, confidence gate, dedup) | planned |
| 4 | Posting + dry-run + idempotency | planned |
| 5 | Eval harness (dataset, judge, report) | planned |

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
```

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

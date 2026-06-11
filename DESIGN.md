# Design notes

Decisions with trade-offs, one short note each. Newest at the bottom.

## Phase 1

### httpx + hand-rolled GitHub client (vs PyGithub)
Per spec. We only need four endpoints; a thin client gives us full control
over rate-limit headers, Accept negotiation (JSON vs raw diff), and retries.
Tests use `httpx.MockTransport` with recorded fixtures instead of adding a
mocking library (respx/vcr) — zero extra deps, fully offline.

### Rate limiting: pre-flight sleep with a hard cap
After every response we record `X-RateLimit-Remaining/Reset`. Before the next
request, if remaining is below `min_rate_limit_remaining` we sleep until the
window resets. 403/429 responses with `Retry-After` or an exhausted window are
retried after the indicated delay. Any required sleep longer than
`max_sleep_seconds` (default 120s) raises `RateLimitError` instead — for an
agent that will run in CI/cron, failing loudly beats silently hanging for up
to an hour. `sleep` and `now` are injectable, so backoff is unit-tested
without real waiting.

### Diff parsing: count-driven, strict
Hunk bodies are consumed by exactly `old_count + new_count` accounting lines
from the `@@` header, so a `-`-prefixed body line can never be confused with a
`--- a/...` file header. Anything unexpected raises `DiffParseError` rather
than being skipped — review comments need exact line anchors, so a silently
wrong parse is worse than a crash. Path resolution priority:
`---`/`+++` markers > `rename from/to` > the `diff --git` line (which is
ambiguous for paths containing spaces). `\ No newline at end of file` markers
are validated but not stored: they never consume a line number, and we don't
reconstruct diffs.

### Config: strict TOML parsing
Unknown keys and wrong value types in `config.toml` raise `ConfigError`
instead of being ignored, so typos in a deployed config fail at startup, not
as a silently-default behavior. Secrets are env-only by construction: the
config loader has no code path that reads keys from disk.

### Dev interpreter pinned to 3.11
The local default `python3` (3.14) has a broken `ensurepip`; the project venv
uses Homebrew `python3.11`, which is also the project floor. `.python-version`
pins 3.11 so pyenv/uv users get the same interpreter, and CI tests 3.11 and
3.12 explicitly — a future change to the machine's `python3` default cannot
silently change the dev interpreter.

## Phase 2

### PR head checkout: pull ref + SHA verification, env-only auth
`git fetch --depth 1 origin refs/pull/<n>/head` into a temp dir gives exactly
the tree the PR diff describes, without history. The fetched commit is
verified against the head SHA from the API; a mismatch (force-push between
API call and fetch) raises rather than reviewing one tree with line anchors
from another. The token reaches git as an `Authorization` header via
`GIT_CONFIG_*` environment variables — never in argv (visible in `ps`) or in
the on-disk remote URL. Workspace tests use a local `file://` origin with a
hand-made `refs/pull/7/head`, so they exercise the real git path offline.

### Reference detection: added lines only, maximal attribute chains
Only added-line references drive retrieval: deleted lines reference the old
tree and pure context lines aren't under review. Attribute chains count once
at full length (`pkg.mod.f`, never also `pkg.mod`), and usage is matched
through the file's import bindings (aliases, dotted imports, relative
imports resolved against the containing package). Stdlib modules
(`sys.stdlib_module_names`) are silently skipped; anything else that can't be
resolved to a repo file is reported as unresolved rather than dropped.

### What counts as a reference (verified, pinned by tests)
Annotation position counts: `def handle(req: Request) -> Response:` on an
added line references `Request`/`Response`, because annotations are ordinary
Load-context expression nodes in the AST. `from __future__ import
annotations` changes nothing here — PEP 563 defers runtime *evaluation*, but
the parsed AST still contains real `Name`/`Attribute` nodes. Decorators count
on their own lines (`@router.get(...)` references `router.get`). **Known
gap:** explicitly quoted string annotations (`def f(x: "Request")`) parse as
`Constant` strings and are not re-parsed into expressions; rare in 3.11+
code, where PEP 563 makes quoting unnecessary. Pinned by a test so a future
fix (or regression) is visible.

### Budget: ~4 chars/token estimate, most-referenced first
No tokenizer dependency — the budget is a guardrail, not an invoice, and a
chars/4 estimate is backend-neutral across Gemini/Anthropic/Ollama. Ranking
is reference-count desc, then size asc (more evidence of relevance wins;
ties prefer cheaper symbols). Re-exports are followed exactly one hop —
both explicit (`pkg/__init__.py` doing `from .impl import X`) and star form
(`from .impl import *`, scanning each starred module for the definition).
The first live smoke test (httpx) showed both public-API patterns are
ubiquitous; one hop covers them without risking cycles. Module-level constants stay out of scope:
only top-level function/class definitions are retrieved, per spec.

## Decisions approved for later phases (reviewed 2026-06-11)

These were agreed before Phase 2 started; implement phases against them.

1. **Eval pre-review state (Phase 5).** Reconstruct the diff at the commit
   the first human review targeted (via review comments'
   `original_commit_id`), falling back to the final merged diff when the old
   head is unfetchable. Every dataset case records `reconstructed: true/false`
   in the JSONL so headline metrics can exclude or disclose contaminated
   (final-diff) cases.
2. **Idempotency marker (Phase 4).** HTML comment marker in the review body
   with a deterministic run-key of `repo + PR number + head SHA`: re-running
   on an unchanged PR is a no-op, while a new push permits a fresh review.
   Only markers in comments authored by the bot token's own identity count —
   a copy of the marker string in someone else's comment must not suppress a
   review.
3. **Non-Python hunks (Phase 3).** Reviewed, but segmented: every generated
   comment is tagged with whether retrieved context was available, and
   no-context comments face a higher confidence threshold (hallucination risk
   is higher from the hunk alone). Phase 5 reports recall/false-positive
   metrics split by context vs. no-context.
4. **Model pricing (Phase 3+).** Per-model $/token prices live in
   `config.toml` with a `prices_as_of = "YYYY-MM"` field; cost reports state
   the pricing vintage they were computed with.
5. **Token-estimator validation (Phase 3).** Every model call logs the
   chars/4 *estimate* alongside the API-reported actual input tokens, both
   stored in the SQLite call log, so the divisor can be validated or
   corrected with data instead of asserted ("ran ~N% hot/cold" must be a
   measured claim).

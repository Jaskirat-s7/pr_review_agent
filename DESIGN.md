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

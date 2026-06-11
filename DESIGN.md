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

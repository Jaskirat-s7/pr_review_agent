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

## Phase 3

### ModelClient protocol: one call shape, no sampling knobs
`complete(system, messages, max_tokens, purpose)` → text + API-reported token
counts. Deliberately no temperature parameter: current Anthropic models
(Opus 4.7+) reject sampling parameters with HTTP 400, and reproducibility
for repeated runs comes from the SQLite cache, not from sampling settings.
Gemini pins temperature 0 internally; Ollama likewise. ``purpose`` is a
telemetry label recorded in the ledger ("triage"/"review"/"judge").

### Cache + ledger: one SQLite file, two tables
Cache key is exactly SHA256(model + system + messages) per spec, serialized
canonically (sorted JSON). Hits cost $0, skip the backend entirely, and are
still written to the ledger. Every call row stores API-reported input/output
tokens *and* the chars/4 estimate (decision #5) so `pra cost report` shows
measured estimator drift per run instead of an asserted heuristic. Unknown
models cost $0 with a loud warning rather than failing the run; prices carry
a `prices_as_of` vintage (decision #4).

### Confidence thresholds are unvalidated priors, by design
0.6 (with context) / 0.8 (without) live in `[review]`, not in code. The
*values* are arbitrary priors; the *ordering* is the deliberate part
(no-context comments face a higher bar, decision #3). They exist so Phase 5
has a knob to sweep: the judge's match/false-positive labels per confidence
band are exactly the data needed to tune them, and the eval report should
make that sweep cheap. Until then, no precision/recall claim is attached to
these numbers.

### Malformed model output: no silent drops, no naive retry
Three failure layers, all visible: (1) repair-lite parsing tolerates code
fences and prose-wrapped JSON; (2) a response that still isn't valid JSON
of the right shape fails the whole hunk and increments
`triage_failures`/`review_failures`; (3) an unusable element inside a valid
array increments `dropped_malformed_item`. In eval, a silent drop is
indistinguishable from "the agent found nothing", which corrupts recall —
so every drop has a counter, and Phase 5 can exclude or flag PRs with
nonzero failure counts instead of scoring them as misses. There is
deliberately no retry: an identical re-prompt would hit the response cache
and replay the same malformed text. If Phase 5 shows a material
parse-failure rate, the fix is a cache-bypassing retry with a stricter
re-prompt — recorded as the known next step, not implemented speculatively.

### Engine: fail-closed triage, anchor validation, advisory severity
Triage parse failures skip the hunk (counted, never silently reviewed at
full price). Review comments must anchor to a line number that actually has
a new-file line in the hunk — context lines are allowed as anchors so
deletion-only hunks ("removed the null check") remain commentable; invented
line numbers are dropped and counted. Unknown severities are coerced to
"minor" instead of dropping the finding (severity is advisory; the comment
text is the value). Confidence is gated per decision #3: no-context comments
face the higher threshold, and every comment carries `has_context` for
Phase 5's split metrics.

### Lint dedup: best-effort, exact location match
ruff + mypy run on the PR head workspace (`--ignore-missing-imports`, since
the OSS repo's dependencies aren't installed); agent comments matching a
linter finding's exact (file, line) are dropped. Missing executables degrade
to "no findings" with a warning — a review run shouldn't die because a
deploy environment lacks mypy. Tools are resolved from PATH or the running
interpreter's bin directory (virtualenv installs).

### Prompts: versioned package resources, not inline strings
Four templates (triage/review × system/user) under
`src/pr_review_agent/prompts/`, loaded via importlib.resources, rendered
with string.Template (JSON braces stay literal). Static instructions sit in
the system prompt and volatile hunk content in the user message, so backend
prompt caches get a stable prefix. Phase 5's judge prompts join the same
directory.

## Phase 4

### Dry-run is the default; posting is an explicit `--post`
The spec's `--dry-run` flag is inverted into an explicit opt-in: an agent
that posts to a real OSS repo should require a deliberate flag to do so,
and `--post` reads less ambiguously than `--no-dry-run`.

### Idempotency: deterministic run key, bot-author check (decision #2)
The review body carries `<!-- pr-review-agent:repo#N@head_sha -->`. Before
posting, the bot's own identity is resolved via `GET /user` and only
markers in reviews *it* authored count — a copied marker string in someone
else's comment cannot suppress a review. Same head → no-op; new push → new
key → fresh review. Empty results are not posted at all (no marker, no
noise); re-runs on the same head re-evaluate from the response cache, so
the repeat costs $0.

### POSTs never blind-retry on 5xx
A lost 502 on `create_review` may still have posted server-side, and a
double review is worse than a failed run. The client now retries 5xx only
for idempotent requests; rate-limit rejections (429 / 403-limit) retry for
everything since rejection means not-executed. Comment anchors post with
`side: "RIGHT"` to match the engine's new-file line numbers.

## Phase 5 (dataset builder, judge, report; `eval run` deferred)

### Pre-review reconstruction (decision #1 implemented)
A case's diff is reconstructed at the commit the *first* substantive
reviewer saw — the earliest review comment's `original_commit_id`, diffed
via the compare API. When that commit is gone (force-push then GC), the
case falls back to the final merged diff with `reconstructed: false`.
Every case records the flag; the report splits recall by it and labels the
fallback subset as contaminated (the merged diff may already incorporate
fixes the human comments prompted). "Substantive" = top-level (not a
reply), not bot-authored, ≥10 characters — replies are discussion, not
independent findings.

### Judge: per-comment questions, error is a verdict
One judge call per human comment (match/partial/miss + matched agent
index) and one per unmatched agent comment (plausible-extra vs
false-positive), all through the Anthropic backend with purpose="judge" —
cached and cost-tracked like every other call. Unparseable judge output is
verdict "error", never coerced to miss; out-of-range matched indices are
discarded but the verdict survives. Agent runs with zero comments are
scored miss without spending judge calls. The 20% CSV sample is seeded
(default 42) so a re-export reproduces the same rows.

### Judge backend: Claude Code subprocess, $0 ledger, batch delays
The judge defaults to a `ClaudeCodeClient` that shells out to `claude -p
--output-format json --max-turns 1` (system via `--append-system-prompt`,
user turns via stdin). This draws on a Claude Code subscription instead of
metered API credits — cross-family from the Gemini system under test, which
keeps the judge independent. Usage flows through the same SQLite ledger, but
cost is recorded as $0: the ledger tracks *API* spend and these calls don't
incur any. The model label is the fixed string `claude-code` (mapped to $0
pricing) so the plan's underlying model can change without reshaping the
ledger. Verified live against the real CLI — the JSON envelope
(`is_error`/`result`/`usage.{input,output}_tokens`) matches the parser, and
input_tokens is the fresh-input count (cache tokens excluded), consistent
with the Anthropic backend. Because a 50-PR batch can exceed a single
Claude Code 5-hour usage window, `pra eval judge --delay N` spaces cases out
and a re-run resumes (judgments rewritten per backend; cached calls replay
free).

### Ceiling baseline: Gemini 2.5 Pro, not Opus API
The ceiling column compares the Flash system-under-test against a
frontier-of-the-same-family model. Originally Opus via the Anthropic API;
switched to Gemini 2.5 Pro on the same free tier so the whole eval has
~$0 marginal cost while preserving the "chose the cheap model deliberately,
measured the gap" narrative. The ceiling run label is configurable
(`models.ceiling_backend`, default `gemini-pro`); the report marks that
column "(ceiling)". Cost-per-PR columns remain API-list-equivalent (the
portable "what it costs at scale" number, and it keeps the Phase 3 cost
machinery honest); a standing report note states actual marginal spend was
~$0 — both numbers, neither overwritten.

### Report: recall = match + 0.5·partial, errors excluded
The 0.5 partial weight is stated in the report legend, not buried in code.
Judge errors are excluded from denominators and get their own row — an
unjudged comment must never silently count as a miss. The headline table
grows an "(ceiling)" column when anthropic-backend judgments exist;
contamination and context splits (decisions #1 and #3) are standing
sections. `pra eval run` is the remaining Phase 5 piece, deliberately
deferred: schemas for run results are already fixed, so the runner slots
in without reshaping the judge or report.

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

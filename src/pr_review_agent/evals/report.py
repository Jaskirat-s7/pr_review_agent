"""Markdown eval report aggregated from judgments, runs, and cases.

Scoring: recall = (match + 0.5 * partial) / (match + partial + miss).
Judge errors are excluded from denominators and reported in their own row —
an unjudged comment must never silently count as a miss. The headline table
gains a "(ceiling)" column when an anthropic-backend run was judged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pr_review_agent.evals.schema import (
    CASES_FILE,
    CaseJudgment,
    EvalCase,
    RunResult,
    judgments_path,
    load_cases,
    load_judgments,
    load_runs,
    runs_path,
)

DEFAULT_CEILING_BACKEND = "gemini-pro"


@dataclass
class _RecallCounts:
    match: int = 0
    partial: int = 0
    miss: int = 0
    error: int = 0

    def add(self, verdict: str) -> None:
        if verdict == "match":
            self.match += 1
        elif verdict == "partial":
            self.partial += 1
        elif verdict == "miss":
            self.miss += 1
        else:
            self.error += 1

    @property
    def recall(self) -> float | None:
        denominator = self.match + self.partial + self.miss
        if denominator == 0:
            return None
        return (self.match + 0.5 * self.partial) / denominator


@dataclass
class BackendMetrics:
    backend: str
    cases: int = 0
    failure_cases: int = 0  # cases whose run had triage/review failures
    overflow_cases: int = 0  # cases skipped for context overflow (not judged)
    overall: _RecallCounts = field(default_factory=_RecallCounts)
    reconstructed: _RecallCounts = field(default_factory=_RecallCounts)
    fallback: _RecallCounts = field(default_factory=_RecallCounts)
    plausible: int = 0
    false_positive: int = 0
    extra_error: int = 0
    cost_total: float = 0.0
    # agent-comment outcomes split by has_context (approved decision #3)
    matched_ctx: int = 0
    matched_no_ctx: int = 0
    plausible_ctx: int = 0
    plausible_no_ctx: int = 0
    fp_ctx: int = 0
    fp_no_ctx: int = 0


def compute_metrics(
    backend: str,
    cases: list[EvalCase],
    runs: list[RunResult],
    judgments: list[CaseJudgment],
) -> BackendMetrics:
    cases_by_key = {(c.repo, c.number): c for c in cases}
    runs_by_key = {(r.repo, r.number): r for r in runs}
    metrics = BackendMetrics(backend=backend)
    metrics.overflow_cases = sum(1 for r in runs if r.status != "ok")
    for judgment in judgments:
        key = (judgment.repo, judgment.number)
        case = cases_by_key.get(key)
        run = runs_by_key.get(key)
        if case is None or run is None:
            continue
        metrics.cases += 1
        metrics.cost_total += run.cost_usd
        if run.failures > 0:
            metrics.failure_cases += 1
        subset = metrics.reconstructed if case.reconstructed else metrics.fallback
        for hj in judgment.human_judgments:
            metrics.overall.add(hj.verdict)
            subset.add(hj.verdict)
            if hj.verdict in ("match", "partial") and hj.matched_agent_index is not None:
                if _has_context(run, hj.matched_agent_index):
                    metrics.matched_ctx += 1
                else:
                    metrics.matched_no_ctx += 1
        for ej in judgment.extra_judgments:
            with_context = _has_context(run, ej.agent_index)
            if ej.verdict == "plausible-extra":
                metrics.plausible += 1
                metrics.plausible_ctx += with_context
                metrics.plausible_no_ctx += not with_context
            elif ej.verdict == "false-positive":
                metrics.false_positive += 1
                metrics.fp_ctx += with_context
                metrics.fp_no_ctx += not with_context
            else:
                metrics.extra_error += 1
    return metrics


def generate_report(dataset_dir: Path, *, ceiling_backend: str = DEFAULT_CEILING_BACKEND) -> str:
    cases = load_cases(dataset_dir / CASES_FILE)
    backends = (
        sorted(path.stem for path in (dataset_dir / "judgments").glob("*.jsonl"))
        if (dataset_dir / "judgments").is_dir()
        else []
    )
    reconstructed_count = sum(1 for c in cases if c.reconstructed)

    lines = [
        "# pr-review-agent eval report",
        "",
        f"Dataset: {len(cases)} case(s) — {reconstructed_count} reconstructed pre-review "
        f"state(s), {len(cases) - reconstructed_count} final-diff fallback(s).",
        "",
        "Scoring: recall = (match + 0.5·partial) / (match + partial + miss). "
        "Judge errors are excluded from denominators and reported separately.",
        "",
    ]
    if not backends:
        lines.append("No judgments found. Run `pra eval judge` first.")
        return "\n".join(lines) + "\n"

    all_metrics = []
    for backend in backends:
        judgments = load_judgments(judgments_path(dataset_dir, backend))
        run_file = runs_path(dataset_dir, backend)
        runs = load_runs(run_file) if run_file.is_file() else []
        all_metrics.append(compute_metrics(backend, cases, runs, judgments))

    lines.extend(_headline_table(all_metrics, ceiling_backend))
    lines.extend(_contamination_table(all_metrics, ceiling_backend))
    lines.extend(_context_tables(all_metrics, ceiling_backend))
    lines.extend(
        [
            "## Cost",
            "",
            "Cost-per-PR columns are API-list-equivalent for the system under "
            "test — what each run would cost at metered API rates. Actual "
            "marginal spend for this eval was ~$0: the free-tier runs (Gemini, "
            "Cerebras) and the Claude Code judge (a subscription) are recorded "
            "at list-equivalent or $0 — the ledger tracks API spend, which "
            "these calls do not incur.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _column_label(backend: str, ceiling_backend: str) -> str:
    return f"{backend} (ceiling)" if backend == ceiling_backend else backend


def _fmt_recall(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _headline_table(all_metrics: list[BackendMetrics], ceiling_backend: str) -> list[str]:
    header = (
        "| metric | "
        + " | ".join(_column_label(m.backend, ceiling_backend) for m in all_metrics)
        + " |"
    )
    divider = "|---|" + "---:|" * len(all_metrics)
    rows = [
        ("cases judged", [str(m.cases) for m in all_metrics]),
        ("recall vs human comments", [_fmt_recall(m.overall.recall) for m in all_metrics]),
        (
            "match / partial / miss",
            [f"{m.overall.match}/{m.overall.partial}/{m.overall.miss}" for m in all_metrics],
        ),
        ("judge errors", [str(m.overall.error + m.extra_error) for m in all_metrics]),
        (
            "false positives per PR",
            [f"{m.false_positive / m.cases:.2f}" if m.cases else "n/a" for m in all_metrics],
        ),
        (
            "plausible extras per PR",
            [f"{m.plausible / m.cases:.2f}" if m.cases else "n/a" for m in all_metrics],
        ),
        ("cases with model failures", [str(m.failure_cases) for m in all_metrics]),
        (
            "cases skipped (context overflow)",
            [str(m.overflow_cases) for m in all_metrics],
        ),
        (
            "cost per PR (USD)",
            [f"${m.cost_total / m.cases:.4f}" if m.cases else "n/a" for m in all_metrics],
        ),
    ]
    out = ["## Headline", "", header, divider]
    out.extend(f"| {name} | " + " | ".join(values) + " |" for name, values in rows)
    out.append("")
    return out


def _contamination_table(all_metrics: list[BackendMetrics], ceiling_backend: str) -> list[str]:
    header = (
        "| subset | "
        + " | ".join(_column_label(m.backend, ceiling_backend) for m in all_metrics)
        + " |"
    )
    divider = "|---|" + "---:|" * len(all_metrics)
    return [
        "## Recall by dataset contamination",
        "",
        "Reconstructed cases score the agent on the code the first reviewer "
        "actually saw; fallback cases use the merged diff and may already "
        "incorporate fixes the human comments prompted.",
        "",
        header,
        divider,
        "| reconstructed (clean) | "
        + " | ".join(_fmt_recall(m.reconstructed.recall) for m in all_metrics)
        + " |",
        "| final-diff fallback | "
        + " | ".join(_fmt_recall(m.fallback.recall) for m in all_metrics)
        + " |",
        "",
    ]


def _context_tables(all_metrics: list[BackendMetrics], ceiling_backend: str) -> list[str]:
    out = ["## Agent-comment outcomes by retrieved context", ""]
    for m in all_metrics:
        out.extend(
            [
                f"### {_column_label(m.backend, ceiling_backend)}",
                "",
                "| outcome | with context | without context |",
                "|---|---:|---:|",
                f"| matched a human comment | {m.matched_ctx} | {m.matched_no_ctx} |",
                f"| plausible extra | {m.plausible_ctx} | {m.plausible_no_ctx} |",
                f"| false positive | {m.fp_ctx} | {m.fp_no_ctx} |",
                "",
            ]
        )
    return out


def _has_context(run: RunResult, agent_index: int) -> bool:
    if 0 <= agent_index < len(run.comments):
        return run.comments[agent_index].has_context
    return False

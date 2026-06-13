"""Command-line interface (``pra``)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from pr_review_agent.config import AppConfig, ConfigError, github_token, load_config
from pr_review_agent.context.models import PRContext
from pr_review_agent.context.retriever import ContextRetriever
from pr_review_agent.diff.models import FileDiff, FileStatus
from pr_review_agent.diff.parser import DiffParseError, parse_diff
from pr_review_agent.evals.dataset import build_dataset
from pr_review_agent.evals.judge import EvalJudge, export_sample
from pr_review_agent.evals.report import generate_report
from pr_review_agent.evals.schema import (
    CASES_FILE,
    EvalDataError,
    judgments_path,
    load_cases,
    load_runs,
    runs_path,
    sample_path,
    write_jsonl,
)
from pr_review_agent.github.client import GitHubClient, GitHubError
from pr_review_agent.github.models import PullRequest
from pr_review_agent.models.base import ModelError
from pr_review_agent.models.factory import build_model_client
from pr_review_agent.models.store import CachingModelClient, CallStore, RunSummary
from pr_review_agent.review.engine import ReviewEngine
from pr_review_agent.review.lint import RuffMypyRunner
from pr_review_agent.review.models import ReviewResult
from pr_review_agent.review.post import (
    find_existing_review,
    review_body,
    run_key,
    to_draft_comments,
)
from pr_review_agent.workspace import WorkspaceError, pr_head_workspace

app = typer.Typer(
    name="pra",
    help="Autonomous GitHub PR review agent.",
    no_args_is_help=True,
)
cost_app = typer.Typer(help="Model spend reporting.", no_args_is_help=True)
app.add_typer(cost_app, name="cost")
eval_app = typer.Typer(
    help="Eval harness: dataset of human-reviewed PRs, judge, report.",
    no_args_is_help=True,
)
app.add_typer(eval_app, name="eval")
console = Console()
err_console = Console(stderr=True)


@app.callback()
def main() -> None:
    """pr-review-agent command line."""


@app.command()
def fetch(
    repo: Annotated[str, typer.Argument(help="Repository in owner/name form.")],
    number: Annotated[int, typer.Argument(min=1, help="Pull request number.")],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.toml (default: ./config.toml)."),
    ] = None,
) -> None:
    """Fetch a pull request and print a parsed diff summary."""
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    token = github_token()
    if token is None:
        err_console.print(
            "[yellow]GITHUB_TOKEN not set; using unauthenticated API (60 requests/hour).[/yellow]"
        )

    try:
        with GitHubClient(token, config=config.github) as client:
            pr = client.get_pr(repo, number)
            diff_text = client.get_pr_diff(repo, number)
    except (GitHubError, ValueError) as exc:
        err_console.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    try:
        files = parse_diff(diff_text)
    except DiffParseError as exc:
        err_console.print(f"[red]diff parse error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    _print_summary(pr, files)


@app.command()
def context(
    repo: Annotated[str, typer.Argument(help="Repository in owner/name form.")],
    number: Annotated[int, typer.Argument(min=1, help="Pull request number.")],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.toml (default: ./config.toml)."),
    ] = None,
) -> None:
    """Check out the PR head and print the retrieved symbol context."""
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    token = github_token()
    if token is None:
        err_console.print(
            "[yellow]GITHUB_TOKEN not set; using unauthenticated API (60 requests/hour).[/yellow]"
        )

    try:
        with GitHubClient(token, config=config.github) as client:
            pr = client.get_pr(repo, number)
            diff_text = client.get_pr_diff(repo, number)
        files = parse_diff(diff_text)
        clone_url = f"{config.github.clone_base_url.rstrip('/')}/{repo}.git"
        with pr_head_workspace(clone_url, number, pr.head_sha, token=token) as workdir:
            retriever = ContextRetriever(workdir, token_budget=config.context.token_budget)
            pr_context = retriever.retrieve(files)
    except (GitHubError, WorkspaceError, DiffParseError, ValueError) as exc:
        err_console.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    _print_context(pr, pr_context, config)


@app.command()
def review(
    repo: Annotated[str, typer.Argument(help="Repository in owner/name form.")],
    number: Annotated[int, typer.Argument(min=1, help="Pull request number.")],
    post: Annotated[
        bool,
        typer.Option("--post", help="Actually post the review. Default is a dry run."),
    ] = False,
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="Override the [models] backend (gemini|anthropic|ollama)."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.toml (default: ./config.toml)."),
    ] = None,
) -> None:
    """Review a pull request. Dry-run by default: prints comments, posts nothing."""
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    token = github_token()
    if post and token is None:
        err_console.print("[red]error:[/red] --post requires GITHUB_TOKEN to be set.")
        raise typer.Exit(code=1)
    if token is None:
        err_console.print(
            "[yellow]GITHUB_TOKEN not set; using unauthenticated API (60 requests/hour).[/yellow]"
        )

    try:
        with GitHubClient(token, config=config.github) as client:
            pr = client.get_pr(repo, number)
            key = run_key(repo, number, pr.head_sha)
            if post:
                bot_login = client.get_authenticated_user()
                existing = find_existing_review(client.list_reviews(repo, number), key, bot_login)
                if existing is not None:
                    console.print(
                        f"Review #{existing.review_id} already posted for {escape(key)}; "
                        "nothing to do."
                    )
                    return
            diff_text = client.get_pr_diff(repo, number)
            files = parse_diff(diff_text)
            result, summary = _run_engine(config, repo, number, pr, files, token, backend)
            if not post:
                _print_review(pr, result, summary, posted=False)
                return
            if not result.comments:
                console.print(
                    "No comments above the bar; not posting an empty review. "
                    "(A re-run on this head will re-evaluate from cache.)"
                )
                return
            review_id = client.create_review(
                repo,
                number,
                commit_id=pr.head_sha,
                body=review_body(key, result.comments),
                comments=to_draft_comments(result.comments),
            )
            _print_review(pr, result, summary, posted=True, review_id=review_id)
    except (
        GitHubError,
        WorkspaceError,
        DiffParseError,
        ModelError,
        ConfigError,
        ValueError,
    ) as exc:
        err_console.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc


def _run_engine(
    config: AppConfig,
    repo: str,
    number: int,
    pr: PullRequest,
    files: list[FileDiff],
    token: str | None,
    backend: str | None,
) -> tuple[ReviewResult, RunSummary | None]:
    clone_url = f"{config.github.clone_base_url.rstrip('/')}/{repo}.git"
    ledger_run_id = (
        f"{repo}#{number}@{pr.head_sha[:8]} {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )
    model_client = build_model_client(backend or config.models.backend, config.models)
    with pr_head_workspace(clone_url, number, pr.head_sha, token=token) as workdir:
        retriever = ContextRetriever(workdir, token_budget=config.context.token_budget)
        pr_context = retriever.retrieve(files)
        with CallStore(Path(config.models.db_path)) as store:
            caching = CachingModelClient(
                model_client, store, run_id=ledger_run_id, pricing=config.models.pricing
            )
            engine = ReviewEngine(caching, config=config.review, lint_runner=RuffMypyRunner())
            result = engine.review(files, pr_context, repo_root=workdir)
            summary = next((s for s in store.run_summaries() if s.run_id == ledger_run_id), None)
    return result, summary


def _print_review(
    pr: PullRequest,
    result: ReviewResult,
    summary: RunSummary | None,
    *,
    posted: bool,
    review_id: int | None = None,
) -> None:
    mode = f"posted review #{review_id}" if posted else "dry run — nothing posted"
    console.print(f"[bold]PR #{pr.number}: {escape(pr.title)}[/bold] — {mode}\n")
    if result.comments:
        table = Table(show_edge=False, pad_edge=False)
        table.add_column("location")
        table.add_column("sev")
        table.add_column("conf", justify="right")
        table.add_column("ctx")
        table.add_column("comment", overflow="fold")
        for comment in result.comments:
            table.add_row(
                escape(f"{comment.file_path}:{comment.line}"),
                comment.severity.value,
                f"{comment.confidence:.2f}",
                "yes" if comment.has_context else "no",
                escape(comment.body),
            )
        console.print(table)
    else:
        console.print("No comments above the bar.")
    stats = result.stats
    console.print(
        f"\n{stats.hunks_total} hunk(s) triaged, {stats.hunks_flagged} flagged, "
        f"{stats.drafts_generated} draft(s); dropped: "
        f"{stats.dropped_low_confidence} low-confidence, "
        f"{stats.dropped_invalid_line} bad-anchor, "
        f"{stats.dropped_malformed_item} malformed, "
        f"{stats.dropped_lint_duplicate} lint-duplicate, "
        f"{stats.dropped_over_cap} over-cap; "
        f"{stats.triage_failures + stats.review_failures} model failure(s)"
    )
    if summary is not None:
        console.print(
            f"{summary.calls} model call(s), {summary.cache_hits} cache hit(s), "
            f"~${summary.cost_usd:.4f}"
        )


@eval_app.command("build-dataset")
def eval_build_dataset(
    repo: Annotated[str, typer.Argument(help="Repository in owner/name form.")],
    since: Annotated[str, typer.Option("--since", help="Earliest merge date, YYYY-MM-DD.")],
    out: Annotated[Path, typer.Option("--out", help="Dataset output directory.")],
    min_comments: Annotated[
        int,
        typer.Option("--min-comments", help="Minimum substantive human review comments."),
    ] = 2,
    max_cases: Annotated[
        int, typer.Option("--max-cases", help="Stop after collecting this many cases.")
    ] = 25,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.toml (default: ./config.toml)."),
    ] = None,
) -> None:
    """Collect merged, substantively reviewed PRs into JSONL eval cases."""
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    token = github_token()
    if token is None:
        err_console.print(
            "[yellow]GITHUB_TOKEN not set; using unauthenticated API (60 requests/hour).[/yellow]"
        )
    try:
        with GitHubClient(token, config=config.github) as client:
            stats = build_dataset(
                client,
                repo,
                since=since,
                min_comments=min_comments,
                out_dir=out,
                max_cases=max_cases,
            )
    except (GitHubError, ValueError) as exc:
        err_console.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"Selected {stats.selected} case(s) from {stats.scanned} scanned PR(s): "
        f"{stats.reconstructed} reconstructed, {stats.fallback_final_diff} final-diff "
        f"fallback(s); skipped {stats.skipped_few_comments} with too few substantive "
        f"comments, {stats.skipped_unmerged} unmerged/out-of-window."
    )
    console.print(f"Wrote {escape(str(out / CASES_FILE))}")


@eval_app.command("judge")
def eval_judge(
    dataset_dir: Annotated[Path, typer.Argument(help="Dataset directory.")],
    backend: Annotated[
        str,
        typer.Option("--backend", help="Which backend's run results to judge."),
    ] = "gemini",
    judge_backend: Annotated[
        str | None,
        typer.Option(
            "--judge-backend", help="Judge model backend (default: models.judge_backend)."
        ),
    ] = None,
    delay: Annotated[
        float,
        typer.Option("--delay", help="Seconds between cases (spread across usage windows)."),
    ] = 0.0,
    sample_fraction: Annotated[
        float,
        typer.Option("--sample-fraction", help="Fraction of judgments exported to CSV."),
    ] = 0.2,
    seed: Annotated[
        int, typer.Option("--seed", help="Sampling seed (reproducible CSV sample).")
    ] = 42,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.toml (default: ./config.toml)."),
    ] = None,
) -> None:
    """Judge a backend's run results against the human comments.

    The judge runs on models.judge_backend (Claude Code by default, drawing
    on a subscription). A 50-PR batch can exceed a single Claude Code 5-hour
    usage window — use --delay to spread it, and re-run to resume (judgments
    are rewritten per backend, cached calls replay for free).
    """
    try:
        config = load_config(config_path)
        cases = load_cases(dataset_dir / CASES_FILE)
        run_file = runs_path(dataset_dir, backend)
        if not run_file.is_file():
            raise EvalDataError(
                f"no run results at {run_file}; execute the eval run step for "
                f"backend {backend!r} first"
            )
        runs = load_runs(run_file)
        judge_name = judge_backend or config.models.judge_backend
        judge_model = build_model_client(judge_name, config.models)
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        with CallStore(Path(config.models.db_path)) as store:
            caching = CachingModelClient(
                judge_model,
                store,
                run_id=f"judge:{backend} via {judge_name} {timestamp}",
                pricing=config.models.pricing,
            )
            judgments = EvalJudge(caching, delay_seconds=delay).judge_all(cases, runs)
    except (ConfigError, EvalDataError, ModelError) as exc:
        err_console.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    count = write_jsonl(judgments_path(dataset_dir, backend), judgments)
    sampled = export_sample(
        judgments,
        cases,
        runs,
        sample_path(dataset_dir, backend),
        fraction=sample_fraction,
        seed=seed,
    )
    console.print(
        f"Judged {count} case(s) for backend {escape(backend)}; "
        f"exported {sampled} judgment(s) to "
        f"{escape(str(sample_path(dataset_dir, backend)))} for manual validation."
    )


@eval_app.command("report")
def eval_report(
    dataset_dir: Annotated[Path, typer.Argument(help="Dataset directory.")],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.toml (default: ./config.toml)."),
    ] = None,
) -> None:
    """Render the markdown eval report and write it to <dataset>/report.md."""
    try:
        config = load_config(config_path)
        markdown = generate_report(dataset_dir, ceiling_backend=config.models.ceiling_backend)
    except (ConfigError, EvalDataError) as exc:
        err_console.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    report_file = dataset_dir / "report.md"
    report_file.write_text(markdown, encoding="utf-8")
    typer.echo(markdown)
    err_console.print(f"[dim]written to {escape(str(report_file))}[/dim]")


@cost_app.command("report")
def cost_report(
    db: Annotated[
        Path | None,
        typer.Option("--db", help="Path to the call database (default: models.db_path)."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.toml (default: ./config.toml)."),
    ] = None,
) -> None:
    """Summarize model spend per run, including cache hits and estimate drift."""
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    db_path = db if db is not None else Path(config.models.db_path)
    if not db_path.is_file():
        err_console.print(f"[red]error:[/red] no call database at {escape(str(db_path))}")
        raise typer.Exit(code=1)
    with CallStore(db_path) as store:
        summaries = store.run_summaries()
    if not summaries:
        console.print("No model calls recorded yet.")
        return

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("run", overflow="fold")
    table.add_column("started")
    table.add_column("calls", justify="right")
    table.add_column("hits", justify="right")
    table.add_column("in tok", justify="right")
    table.add_column("out tok", justify="right")
    table.add_column("cost USD", justify="right")
    table.add_column("est drift", justify="right")
    for summary in summaries:
        drift = summary.estimate_drift
        table.add_row(
            escape(summary.run_id),
            summary.started_at,
            str(summary.calls),
            str(summary.cache_hits),
            str(summary.input_tokens),
            str(summary.output_tokens),
            f"{summary.cost_usd:.4f}",
            "n/a" if drift is None else f"{drift:+.1%}",
        )
    console.print(table)
    total_cost = sum(s.cost_usd for s in summaries)
    total_calls = sum(s.calls for s in summaries)
    total_hits = sum(s.cache_hits for s in summaries)
    console.print(
        f"\n{len(summaries)} run(s), {total_calls} call(s) ({total_hits} cache hit(s)), "
        f"total [bold]${total_cost:.4f}[/bold] "
        f"(prices as of {escape(config.models.prices_as_of)})"
    )


def _print_context(pr: PullRequest, pr_context: PRContext, config: AppConfig) -> None:
    console.print(f"[bold]PR #{pr.number}: {escape(pr.title)}[/bold] — retrieved context\n")
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("changed file", overflow="fold")
    table.add_column("symbol")
    table.add_column("kind")
    table.add_column("defined in", overflow="fold")
    table.add_column("lines", justify="right")
    table.add_column("~tokens", justify="right")
    table.add_column("refs", justify="right")
    for file_context in pr_context.files:
        for symbol in file_context.symbols:
            table.add_row(
                escape(file_context.file_path),
                escape(symbol.name),
                symbol.kind.value,
                escape(f"{symbol.module_path}:{symbol.lineno}"),
                str(symbol.end_lineno - symbol.lineno + 1),
                str(symbol.est_tokens),
                str(symbol.reference_count),
            )
    console.print(table)
    for file_context in pr_context.files:
        if file_context.unresolved:
            unresolved = ", ".join(file_context.unresolved)
            console.print(
                f"[dim]{escape(file_context.file_path)}: not in repo (skipped): "
                f"{escape(unresolved)}[/dim]"
            )
    console.print(
        f"\n~{pr_context.total_tokens} tokens of {config.context.token_budget} budget; "
        f"{pr_context.dropped_symbols} symbol(s) dropped by budget"
    )


def _print_summary(pr: PullRequest, files: list[FileDiff]) -> None:
    flags = ""
    if pr.draft:
        flags += " [dim]\\[draft][/dim]"
    if pr.merged:
        flags += " [magenta](merged)[/magenta]"
    console.print(f"[bold]PR #{pr.number}: {escape(pr.title)}[/bold]{flags}")
    console.print(
        f"{escape(pr.author)} wants to merge "
        f"[cyan]{escape(pr.head_ref)}[/cyan] ({pr.head_sha[:8]}) "
        f"into [cyan]{escape(pr.base_ref)}[/cyan]"
    )
    console.print()

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("status")
    table.add_column("file", overflow="fold")
    table.add_column("+", justify="right", style="green")
    table.add_column("-", justify="right", style="red")
    table.add_column("hunks", justify="right")
    for file_diff in files:
        if file_diff.status in (FileStatus.RENAMED, FileStatus.COPIED):
            name = f"{file_diff.old_path} → {file_diff.new_path}"
        else:
            name = file_diff.path
        if file_diff.is_binary:
            name += " (binary)"
        table.add_row(
            file_diff.status.value,
            escape(name),
            str(file_diff.additions),
            str(file_diff.deletions),
            str(len(file_diff.hunks)),
        )
    console.print(table)

    total_additions = sum(f.additions for f in files)
    total_deletions = sum(f.deletions for f in files)
    console.print(
        f"\n{len(files)} file(s) changed, "
        f"[green]{total_additions} insertion(s)(+)[/green], "
        f"[red]{total_deletions} deletion(s)(-)[/red]"
    )

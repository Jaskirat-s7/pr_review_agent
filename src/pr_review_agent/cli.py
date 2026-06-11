"""Command-line interface (``pra``)."""

from __future__ import annotations

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
from pr_review_agent.github.client import GitHubClient, GitHubError
from pr_review_agent.github.models import PullRequest
from pr_review_agent.models.store import CallStore
from pr_review_agent.workspace import WorkspaceError, pr_head_workspace

app = typer.Typer(
    name="pra",
    help="Autonomous GitHub PR review agent.",
    no_args_is_help=True,
)
cost_app = typer.Typer(help="Model spend reporting.", no_args_is_help=True)
app.add_typer(cost_app, name="cost")
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

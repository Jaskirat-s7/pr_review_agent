"""Command-line interface (``pra``)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from pr_review_agent.config import ConfigError, github_token, load_config
from pr_review_agent.diff.models import FileDiff, FileStatus
from pr_review_agent.diff.parser import DiffParseError, parse_diff
from pr_review_agent.github.client import GitHubClient, GitHubError
from pr_review_agent.github.models import PullRequest

app = typer.Typer(
    name="pra",
    help="Autonomous GitHub PR review agent.",
    no_args_is_help=True,
)
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

"""Tests for `pra cost report`."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from pr_review_agent import cli
from pr_review_agent.models.store import CallStore

runner = CliRunner()


def seed(path: Path) -> None:
    with CallStore(path) as store:
        store.record_call(
            run_id="octo/widgets#7@abc123",
            purpose="triage",
            model="gemini-2.5-flash",
            cache_key="k1",
            cache_hit=False,
            input_tokens=1000,
            output_tokens=50,
            est_input_tokens=900,
            cost_usd=0.000425,
        )
        store.record_call(
            run_id="octo/widgets#7@abc123",
            purpose="review",
            model="gemini-2.5-flash",
            cache_key="k2",
            cache_hit=True,
            input_tokens=1000,
            output_tokens=50,
            est_input_tokens=900,
            cost_usd=0.0,
        )


def test_cost_report_summarizes_runs(tmp_path: Path) -> None:
    db = tmp_path / "calls.sqlite3"
    seed(db)
    # Wide terminal so the rich table doesn't wrap cell contents.
    result = runner.invoke(cli.app, ["cost", "report", "--db", str(db)], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "octo/widgets#7@abc123" in result.output
    assert "0.0004" in result.output
    assert "1 cache hit" in result.output
    assert "est drift" in result.output


def test_cost_report_missing_db_fails(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["cost", "report", "--db", str(tmp_path / "nope.sqlite3")])
    assert result.exit_code == 1

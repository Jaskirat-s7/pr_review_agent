"""Tests for the SQLite cache + cost ledger wrapping every model call."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from pr_review_agent.config import ModelPricing
from pr_review_agent.models.base import ModelMessage, ModelResponse
from pr_review_agent.models.store import CachingModelClient, CallStore, cache_key, compute_cost

PRICING = {"fake-model": ModelPricing(input_per_mtok=1.0, output_per_mtok=2.0)}


class FakeClient:
    """Counts real completions; returns fixed usage numbers."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(
        self,
        system: str,
        messages: Sequence[ModelMessage],
        *,
        max_tokens: int = 1024,
        purpose: str = "",
    ) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            text=f"reply-{self.calls}", model="fake-model", input_tokens=100, output_tokens=20
        )


def make(tmp_path: Path) -> tuple[CachingModelClient, FakeClient, CallStore]:
    store = CallStore(tmp_path / "calls.sqlite3")
    inner = FakeClient()
    client = CachingModelClient(inner, store, run_id="run-1", pricing=PRICING)
    return client, inner, store


def test_identical_call_hits_cache_and_is_free(tmp_path: Path) -> None:
    client, inner, store = make(tmp_path)
    messages = [ModelMessage("user", "hello")]
    first = client.complete("sys", messages, purpose="triage")
    second = client.complete("sys", messages, purpose="triage")
    assert inner.calls == 1  # second call never reached the backend
    assert not first.cached
    assert second.cached
    assert second.text == first.text

    (summary,) = store.run_summaries()
    assert summary.calls == 2
    assert summary.cache_hits == 1
    expected_miss_cost = (100 * 1.0 + 20 * 2.0) / 1_000_000
    assert summary.cost_usd == pytest.approx(expected_miss_cost)  # hit costs $0
    store.close()


def test_different_prompt_misses_cache(tmp_path: Path) -> None:
    client, inner, store = make(tmp_path)
    client.complete("sys", [ModelMessage("user", "one")])
    client.complete("sys", [ModelMessage("user", "two")])
    client.complete("other-system", [ModelMessage("user", "one")])
    assert inner.calls == 3
    store.close()


def test_cache_persists_across_store_instances(tmp_path: Path) -> None:
    path = tmp_path / "calls.sqlite3"
    store = CallStore(path)
    inner = FakeClient()
    client = CachingModelClient(inner, store, run_id="run-1", pricing=PRICING)
    client.complete("sys", [ModelMessage("user", "hello")])
    store.close()

    store2 = CallStore(path)
    inner2 = FakeClient()
    client2 = CachingModelClient(inner2, store2, run_id="run-2", pricing=PRICING)
    response = client2.complete("sys", [ModelMessage("user", "hello")])
    assert response.cached
    assert inner2.calls == 0
    store2.close()


def test_estimate_drift_is_recorded_for_misses(tmp_path: Path) -> None:
    client, _, store = make(tmp_path)
    client.complete("sys", [ModelMessage("user", "x" * 400)])
    (summary,) = store.run_summaries()
    assert summary.input_tokens_misses == 100
    assert summary.est_input_tokens_misses > 0
    assert summary.estimate_drift is not None
    store.close()


def test_unknown_model_pricing_costs_zero(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        cost = compute_cost("mystery-model", 1000, 1000, PRICING)
    assert cost == 0.0
    assert "no pricing configured" in caplog.text


def test_cache_key_is_order_sensitive_and_deterministic() -> None:
    messages = [ModelMessage("user", "a"), ModelMessage("assistant", "b")]
    key1 = cache_key("m", "sys", messages)
    key2 = cache_key("m", "sys", list(messages))
    assert key1 == key2
    assert key1 != cache_key("m", "sys", list(reversed(messages)))
    assert key1 != cache_key("other-model", "sys", messages)

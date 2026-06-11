"""SQLite-backed response cache and call ledger.

Every model call goes through :class:`CachingModelClient`:

- Cache key = SHA256 over (model, system prompt, messages), per spec. Hits
  are free ($0, no API call) and still logged to the ledger.
- Every call records API-reported input/output tokens, the chars/4 input
  estimate (so the estimator can be validated against reality), and the
  computed cost in USD.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from pr_review_agent.config import ModelPricing
from pr_review_agent.estimate import estimate_tokens
from pr_review_agent.models.base import ModelClient, ModelMessage, ModelResponse

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    response_text TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    run_id TEXT NOT NULL,
    purpose TEXT NOT NULL,
    model TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    cache_hit INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    est_input_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Aggregated spend for one run."""

    run_id: str
    started_at: str
    calls: int
    cache_hits: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    est_input_tokens_misses: int
    input_tokens_misses: int

    @property
    def estimate_drift(self) -> float | None:
        """Relative error of the chars/4 estimate vs API-reported input
        tokens, over cache misses only (hits replay stored counts)."""
        if self.input_tokens_misses <= 0:
            return None
        return (self.est_input_tokens_misses - self.input_tokens_misses) / (
            self.input_tokens_misses
        )


class CallStore:
    """SQLite store holding the response cache and the call ledger."""

    def __init__(self, path: Path) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def lookup(self, cache_key: str) -> ModelResponse | None:
        row = self._conn.execute(
            "SELECT model, response_text, input_tokens, output_tokens"
            " FROM cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        model, text, input_tokens, output_tokens = row
        return ModelResponse(
            text=str(text),
            model=str(model),
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            cached=True,
        )

    def save(self, cache_key: str, response: ModelResponse) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache"
            " (cache_key, model, response_text, input_tokens, output_tokens, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                cache_key,
                response.model,
                response.text,
                response.input_tokens,
                response.output_tokens,
                _now(),
            ),
        )
        self._conn.commit()

    def record_call(
        self,
        *,
        run_id: str,
        purpose: str,
        model: str,
        cache_key: str,
        cache_hit: bool,
        input_tokens: int,
        output_tokens: int,
        est_input_tokens: int,
        cost_usd: float,
    ) -> None:
        self._conn.execute(
            "INSERT INTO calls (created_at, run_id, purpose, model, cache_key, cache_hit,"
            " input_tokens, output_tokens, est_input_tokens, cost_usd)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _now(),
                run_id,
                purpose,
                model,
                cache_key,
                int(cache_hit),
                input_tokens,
                output_tokens,
                est_input_tokens,
                cost_usd,
            ),
        )
        self._conn.commit()

    def run_summaries(self) -> list[RunSummary]:
        rows = self._conn.execute(
            "SELECT run_id, MIN(created_at), COUNT(*), SUM(cache_hit),"
            " SUM(input_tokens), SUM(output_tokens), SUM(cost_usd),"
            " SUM(CASE WHEN cache_hit = 0 THEN est_input_tokens ELSE 0 END),"
            " SUM(CASE WHEN cache_hit = 0 THEN input_tokens ELSE 0 END)"
            " FROM calls GROUP BY run_id ORDER BY MIN(created_at)"
        ).fetchall()
        return [
            RunSummary(
                run_id=str(row[0]),
                started_at=str(row[1]),
                calls=int(row[2]),
                cache_hits=int(row[3]),
                input_tokens=int(row[4]),
                output_tokens=int(row[5]),
                cost_usd=float(row[6]),
                est_input_tokens_misses=int(row[7]),
                input_tokens_misses=int(row[8]),
            )
            for row in rows
        ]


class CachingModelClient:
    """Wraps any ModelClient with the cache and the cost ledger."""

    def __init__(
        self,
        inner: ModelClient,
        store: CallStore,
        *,
        run_id: str,
        pricing: Mapping[str, ModelPricing],
    ) -> None:
        self._inner = inner
        self._store = store
        self._run_id = run_id
        self._pricing = pricing

    @property
    def model(self) -> str:
        return self._inner.model

    def complete(
        self,
        system: str,
        messages: Sequence[ModelMessage],
        *,
        max_tokens: int = 1024,
        purpose: str = "",
    ) -> ModelResponse:
        key = cache_key(self._inner.model, system, messages)
        est_input = estimate_tokens(system + "".join(m.content for m in messages))
        cached = self._store.lookup(key)
        if cached is not None:
            logger.info(
                "cache hit: %s call to %s (key %s)", purpose or "model", cached.model, key[:12]
            )
            self._store.record_call(
                run_id=self._run_id,
                purpose=purpose,
                model=cached.model,
                cache_key=key,
                cache_hit=True,
                input_tokens=cached.input_tokens,
                output_tokens=cached.output_tokens,
                est_input_tokens=est_input,
                cost_usd=0.0,
            )
            return cached
        response = self._inner.complete(system, messages, max_tokens=max_tokens, purpose=purpose)
        cost = compute_cost(
            response.model, response.input_tokens, response.output_tokens, self._pricing
        )
        self._store.save(key, response)
        self._store.record_call(
            run_id=self._run_id,
            purpose=purpose,
            model=response.model,
            cache_key=key,
            cache_hit=False,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            est_input_tokens=est_input,
            cost_usd=cost,
        )
        return response


def cache_key(model: str, system: str, messages: Sequence[ModelMessage]) -> str:
    """SHA256 over (model, system prompt, messages), serialized canonically."""
    payload = json.dumps(
        {
            "model": model,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: Mapping[str, ModelPricing],
) -> float:
    entry = pricing.get(model)
    if entry is None:
        logger.warning("no pricing configured for model %r; recording $0 cost", model)
        return 0.0
    return (input_tokens * entry.input_per_mtok + output_tokens * entry.output_per_mtok) / 1_000_000


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")

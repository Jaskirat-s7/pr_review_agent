"""Thin GitHub REST client over httpx.

Rate-limit aware: every response's ``X-RateLimit-*`` headers are recorded,
and the client sleeps before the next request once the remaining budget drops
below a configured threshold. Transient failures (429/5xx/secondary rate
limits) are retried with backoff. ``sleep`` and ``now`` are injectable so all
of this is unit-testable without real waiting.

The auth token is sent only as a request header and is never logged.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import TracebackType
from typing import Any, Self, cast

import httpx

from pr_review_agent.config import GitHubConfig
from pr_review_agent.github.models import (
    PRFile,
    PullRequest,
    Review,
    ReviewComment,
    ReviewCommentDraft,
)

logger = logging.getLogger(__name__)

_ACCEPT_JSON = "application/vnd.github+json"
_ACCEPT_DIFF = "application/vnd.github.v3.diff"
_API_VERSION = "2022-11-28"
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_SERVER_ERROR_BACKOFF_CAP = 30.0


class GitHubError(Exception):
    """Base class for GitHub client errors."""


class GitHubAPIError(GitHubError):
    """An API request failed with a non-retryable (or retry-exhausted) status."""

    def __init__(self, message: str, *, status_code: int, url: str) -> None:
        super().__init__(f"{message} (HTTP {status_code} for {url})")
        self.status_code = status_code
        self.url = url


class RateLimitError(GitHubError):
    """The rate limit is exhausted and the reset is too far away to wait for."""


class GitHubClient:
    """Minimal GitHub REST client for pull-request ingestion."""

    def __init__(
        self,
        token: str | None = None,
        *,
        config: GitHubConfig | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._config = config or GitHubConfig()
        self._sleep = sleep
        self._now = now
        headers = {
            "Accept": _ACCEPT_JSON,
            "X-GitHub-Api-Version": _API_VERSION,
            "User-Agent": "pr-review-agent",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(
            base_url=self._config.base_url,
            headers=headers,
            timeout=self._config.timeout_seconds,
            transport=transport,
            follow_redirects=True,
        )
        self._rate_remaining: int | None = None
        self._rate_reset: float | None = None

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
        self._client.close()

    # -- public API ---------------------------------------------------------

    def get_pr(self, repo: str, number: int) -> PullRequest:
        """Fetch pull request metadata."""
        response = self._request("GET", f"/repos/{_validate_repo(repo)}/pulls/{number}")
        return PullRequest.from_api(_json_object(response))

    def get_pr_diff(self, repo: str, number: int) -> str:
        """Fetch the full unified diff for a pull request."""
        response = self._request(
            "GET",
            f"/repos/{_validate_repo(repo)}/pulls/{number}",
            accept=_ACCEPT_DIFF,
        )
        return response.text

    def get_pr_files(self, repo: str, number: int) -> list[PRFile]:
        """Fetch the changed files (with patch hunks) for a pull request."""
        path = f"/repos/{_validate_repo(repo)}/pulls/{number}/files"
        return [PRFile.from_api(item) for item in self._paginate(path)]

    def list_review_comments(self, repo: str, number: int) -> list[ReviewComment]:
        """Fetch all inline review comments on a pull request."""
        path = f"/repos/{_validate_repo(repo)}/pulls/{number}/comments"
        return [ReviewComment.from_api(item) for item in self._paginate(path)]

    def list_reviews(self, repo: str, number: int) -> list[Review]:
        """Fetch all reviews on a pull request (for idempotency checks)."""
        path = f"/repos/{_validate_repo(repo)}/pulls/{number}/reviews"
        return [Review.from_api(item) for item in self._paginate(path)]

    def list_pull_requests(self, repo: str, *, state: str = "closed") -> Iterator[PullRequest]:
        """Iterate PRs, most recently updated first (lazy; stop early)."""
        path = f"/repos/{_validate_repo(repo)}/pulls"
        params = {"state": state, "sort": "updated", "direction": "desc"}
        for item in self._paginate(path, extra_params=params):
            yield PullRequest.from_api(item)

    def compare_diff(self, repo: str, base: str, head: str) -> str:
        """Unified diff between two commits (pre-review state reconstruction)."""
        response = self._request(
            "GET",
            f"/repos/{_validate_repo(repo)}/compare/{base}...{head}",
            accept=_ACCEPT_DIFF,
        )
        return response.text

    def resolve_commit_sha(self, repo: str, ref: str = "HEAD") -> str:
        """Resolve a ref (branch, tag, or SHA) to a full commit SHA.

        ``HEAD`` resolves the repository's default-branch head.
        """
        response = self._request("GET", f"/repos/{_validate_repo(repo)}/commits/{ref}")
        sha = _json_object(response).get("sha")
        if not isinstance(sha, str) or not sha:
            raise GitHubAPIError(
                f"could not resolve {ref!r} to a commit",
                status_code=response.status_code,
                url=str(response.request.url),
            )
        return sha

    def get_authenticated_user(self) -> str:
        """Return the login of the token's identity (the bot account)."""
        response = self._request("GET", "/user")
        login = _json_object(response).get("login")
        if not isinstance(login, str) or not login:
            raise GitHubAPIError(
                "could not determine the authenticated user",
                status_code=response.status_code,
                url=str(response.request.url),
            )
        return login

    def create_review(
        self,
        repo: str,
        number: int,
        *,
        commit_id: str,
        body: str,
        comments: Sequence[ReviewCommentDraft],
    ) -> int:
        """Post a single PR review with inline comments; returns the review id.

        Not retried on 5xx: a create may have succeeded server-side even when
        the response was lost, and a double-posted review is worse than a
        failed run.
        """
        payload = {
            "commit_id": commit_id,
            "body": body,
            "event": "COMMENT",
            "comments": [
                {"path": c.path, "line": c.line, "side": "RIGHT", "body": c.body} for c in comments
            ],
        }
        response = self._request(
            "POST",
            f"/repos/{_validate_repo(repo)}/pulls/{number}/reviews",
            json=payload,
            idempotent=False,
        )
        review_id = _json_object(response).get("id")
        return review_id if isinstance(review_id, int) else 0

    # -- internals ----------------------------------------------------------

    def _paginate(
        self, path: str, extra_params: Mapping[str, str] | None = None
    ) -> Iterator[dict[str, Any]]:
        url: str | None = path
        params: Mapping[str, str] | None = {"per_page": "100", **(extra_params or {})}
        while url is not None:
            response = self._request("GET", url, params=params)
            payload = response.json()
            if not isinstance(payload, list):
                raise GitHubAPIError(
                    "expected a JSON array",
                    status_code=response.status_code,
                    url=str(response.request.url),
                )
            for item in payload:
                if not isinstance(item, dict):
                    raise GitHubAPIError(
                        "expected JSON objects in array",
                        status_code=response.status_code,
                        url=str(response.request.url),
                    )
                yield cast("dict[str, Any]", item)
            next_link = response.links.get("next")
            url = next_link.get("url") if next_link else None
            params = None  # the "next" URL already carries its query string

    def _request(
        self,
        method: str,
        url: str,
        *,
        accept: str = _ACCEPT_JSON,
        params: Mapping[str, str] | None = None,
        json: Any | None = None,
        idempotent: bool = True,
    ) -> httpx.Response:
        self._wait_for_rate_limit_budget()
        for attempt in range(self._config.max_retries + 1):
            response = self._client.request(
                method, url, params=params, json=json, headers={"Accept": accept}
            )
            self._record_rate_limit(response)
            if response.is_success:
                return response
            if attempt < self._config.max_retries and self._is_retryable(response, idempotent):
                delay = self._retry_delay(response, attempt)
                logger.warning(
                    "GitHub request failed with HTTP %d; retrying in %.1fs (attempt %d/%d)",
                    response.status_code,
                    delay,
                    attempt + 1,
                    self._config.max_retries,
                )
                self._sleep(delay)
                continue
            raise GitHubAPIError(
                _error_message(response),
                status_code=response.status_code,
                url=str(response.request.url),
            )
        raise AssertionError("unreachable")  # pragma: no cover

    def _wait_for_rate_limit_budget(self) -> None:
        remaining = self._rate_remaining
        if remaining is None or remaining >= self._config.min_rate_limit_remaining:
            return
        if self._rate_reset is None:
            return
        delay = self._rate_reset - self._now() + 1.0
        if delay <= 0:
            self._rate_remaining = None  # window already reset
            return
        if delay > self._config.max_sleep_seconds:
            if remaining > 0:
                logger.warning(
                    "rate limit low (%d remaining) but reset is %.0fs away; proceeding",
                    remaining,
                    delay,
                )
                return
            raise RateLimitError(
                f"rate limit exhausted; resets in {delay:.0f}s, "
                f"more than max_sleep_seconds={self._config.max_sleep_seconds:.0f}"
            )
        logger.info("rate limit low (%d remaining); sleeping %.1fs until reset", remaining, delay)
        self._sleep(delay)
        self._rate_remaining = None

    def _record_rate_limit(self, response: httpx.Response) -> None:
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")
        if remaining is not None:
            with contextlib.suppress(ValueError):
                self._rate_remaining = int(remaining)
        if reset is not None:
            with contextlib.suppress(ValueError):
                self._rate_reset = float(reset)

    def _is_retryable(self, response: httpx.Response, idempotent: bool) -> bool:
        # Rate-limit rejections (429 / 403) are always safe to retry: the
        # request was refused, not executed. 5xx is ambiguous — the request
        # may have been applied — so only idempotent requests retry on it.
        if response.status_code == 429:
            return True
        if response.status_code == 403 and self._looks_rate_limited(response):
            return True
        return idempotent and response.status_code in _RETRYABLE_STATUS

    @staticmethod
    def _looks_rate_limited(response: httpx.Response) -> bool:
        if "Retry-After" in response.headers:
            return True
        return bool(response.headers.get("X-RateLimit-Remaining") == "0")

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 1.0
        elif response.status_code in (403, 429) and self._rate_reset is not None:
            delay = self._rate_reset - self._now() + 1.0
        else:
            delay = min(2.0**attempt, _SERVER_ERROR_BACKOFF_CAP)
        if delay > self._config.max_sleep_seconds:
            raise RateLimitError(
                f"rate limit retry requires waiting {delay:.0f}s, "
                f"more than max_sleep_seconds={self._config.max_sleep_seconds:.0f}"
            )
        return max(delay, 1.0)


def _validate_repo(repo: str) -> str:
    owner, sep, name = repo.partition("/")
    if not sep or not owner or not name or "/" in name:
        raise ValueError(f"repo must be in 'owner/name' form, got {repo!r}")
    return repo


def _json_object(response: httpx.Response) -> dict[str, Any]:
    payload = response.json()
    if not isinstance(payload, dict):
        raise GitHubAPIError(
            "expected a JSON object",
            status_code=response.status_code,
            url=str(response.request.url),
        )
    return cast("dict[str, Any]", payload)


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, str) and message:
            return message
    return response.text[:200] or "request failed"

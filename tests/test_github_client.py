"""Tests for the GitHub REST client, using httpx.MockTransport (no live calls)."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from conftest import load_fixture
from pr_review_agent.config import GitHubConfig
from pr_review_agent.github.client import (
    GitHubAPIError,
    GitHubClient,
    RateLimitError,
)

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(
    handler: Handler,
    *,
    config: GitHubConfig | None = None,
    token: str | None = "test-token",
    now: float = 1_000.0,
) -> tuple[GitHubClient, list[float]]:
    sleeps: list[float] = []
    client = GitHubClient(
        token,
        config=config or GitHubConfig(),
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        now=lambda: now,
    )
    return client, sleeps


def test_get_pr_parses_fields_and_sends_auth() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/octo/widgets/pulls/123"
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.headers["Accept"] == "application/vnd.github+json"
        return httpx.Response(200, text=load_fixture("api", "pr.json"))

    client, _ = make_client(handler)
    with client:
        pr = client.get_pr("octo/widgets", 123)
    assert pr.number == 123
    assert pr.title == "Add retry logic to fetcher"
    assert pr.author == "octocat"
    assert pr.base_ref == "main"
    assert pr.head_ref == "feature/retry"
    assert pr.head_sha.startswith("bbb222")
    assert not pr.merged
    assert pr.changed_files == 2


def test_no_auth_header_without_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return httpx.Response(200, text=load_fixture("api", "pr.json"))

    client, _ = make_client(handler, token=None)
    with client:
        client.get_pr("octo/widgets", 123)


def test_get_pr_diff_uses_diff_accept_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Accept"] == "application/vnd.github.v3.diff"
        return httpx.Response(200, text="diff --git a/f b/f\n")

    client, _ = make_client(handler)
    with client:
        diff = client.get_pr_diff("octo/widgets", 123)
    assert diff.startswith("diff --git")


def test_get_pr_files_follows_pagination() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["per_page"] == "100"
        if request.url.params.get("page") == "2":
            return httpx.Response(200, text=load_fixture("api", "files_page2.json"))
        next_url = "https://api.github.com/repos/octo/widgets/pulls/123/files?per_page=100&page=2"
        return httpx.Response(
            200,
            text=load_fixture("api", "files_page1.json"),
            headers={"Link": f'<{next_url}>; rel="next"'},
        )

    client, _ = make_client(handler)
    with client:
        files = client.get_pr_files("octo/widgets", 123)
    assert [f.filename for f in files] == ["src/app.py", "utils/util_helpers.py"]
    assert files[0].patch is not None and files[0].patch.startswith("@@ -1,2 +1,3 @@")
    assert files[1].previous_filename == "utils/helpers.py"


def test_list_review_comments() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/octo/widgets/pulls/123/comments"
        return httpx.Response(200, text=load_fixture("api", "comments.json"))

    client, _ = make_client(handler)
    with client:
        comments = client.list_review_comments("octo/widgets", 123)
    assert [c.comment_id for c in comments] == [9001, 9002]
    assert comments[0].line == 12
    assert comments[0].author == "alice"
    assert comments[1].line is None
    assert comments[1].in_reply_to_id == 9001


def test_sleeps_until_reset_when_rate_limit_budget_is_low() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        headers = {"X-RateLimit-Remaining": "5", "X-RateLimit-Reset": "1030"}
        return httpx.Response(200, text=load_fixture("api", "pr.json"), headers=headers)

    config = GitHubConfig(min_rate_limit_remaining=10)
    client, sleeps = make_client(handler, config=config, now=1_000.0)
    with client:
        client.get_pr("octo/widgets", 123)
        assert sleeps == []  # first call: no budget info yet
        client.get_pr("octo/widgets", 123)
    assert calls == 2
    assert sleeps == [31.0]  # reset(1030) - now(1000) + 1


def test_retries_403_rate_limit_until_reset() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1010"}
            return httpx.Response(403, json={"message": "rate limited"}, headers=headers)
        return httpx.Response(200, text=load_fixture("api", "pr.json"))

    client, sleeps = make_client(handler, now=1_000.0)
    with client:
        pr = client.get_pr("octo/widgets", 123)
    assert pr.number == 123
    assert sleeps == [11.0]


def test_rate_limit_reset_beyond_cap_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "2000"}
        return httpx.Response(403, json={"message": "rate limited"}, headers=headers)

    config = GitHubConfig(max_sleep_seconds=120.0)
    client, sleeps = make_client(handler, config=config, now=1_000.0)
    with client, pytest.raises(RateLimitError, match="max_sleep_seconds"):
        client.get_pr("octo/widgets", 123)
    assert sleeps == []


def test_retries_server_errors_with_backoff() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, text=load_fixture("api", "pr.json"))

    client, sleeps = make_client(handler)
    with client:
        pr = client.get_pr("octo/widgets", 123)
    assert pr.number == 123
    assert sleeps == [1.0, 2.0]


def test_non_retryable_error_raises_with_message_and_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    client, sleeps = make_client(handler)
    with client, pytest.raises(GitHubAPIError, match="Not Found") as exc_info:
        client.get_pr("octo/widgets", 999)
    assert exc_info.value.status_code == 404
    assert sleeps == []


def test_retry_after_header_is_honored() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(403, text="slow down", headers={"Retry-After": "7"})
        return httpx.Response(200, text=load_fixture("api", "pr.json"))

    client, sleeps = make_client(handler)
    with client:
        client.get_pr("octo/widgets", 123)
    assert sleeps == [7.0]


@pytest.mark.parametrize("bad_repo", ["plainname", "owner/", "/repo", "a/b/c", ""])
def test_invalid_repo_is_rejected(bad_repo: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be sent")

    client, _ = make_client(handler)
    with client, pytest.raises(ValueError, match="owner/name"):
        client.get_pr(bad_repo, 1)

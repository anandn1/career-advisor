import base64

import pytest
import requests

from app.core.config import get_settings
from app.services import github


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


@pytest.mark.parametrize("username", ["octocat", "a", "a-b-c", "A1", "x" * 39])
def test_valid_usernames_are_accepted(username):
    assert github.validate_username(username) == username


@pytest.mark.parametrize(
    "username",
    [
        "",
        "-start",
        "end-",
        "dou--ble",
        "x" * 40,
        "../repos/victim/private",
        "a/b",
        "a?x=1",
        "a#b",
        "a b",
        "ünï",
    ],
)
def test_invalid_usernames_are_rejected(username):
    with pytest.raises(github.InvalidUsernameError):
        github.validate_username(username)


@pytest.mark.parametrize(
    "fetch",
    [
        github.fetch_profile,
        github.fetch_repos,
        github.fetch_profile_readme,
        lambda name: github.fetch_profile_and_repos(name, 5),
    ],
)
def test_invalid_username_makes_no_request(fetch, monkeypatch):
    monkeypatch.setattr(github.requests, "get", lambda *a, **k: pytest.fail("request was made"))
    with pytest.raises(github.InvalidUsernameError):
        fetch("../repos/victim/private/contents/secret.txt?x=")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("octocat", "octocat"),
        ("  octocat  ", "octocat"),
        ("github.com/octocat", "octocat"),
        ("https://github.com/octocat", "octocat"),
        ("https://github.com/octocat/", "octocat"),
        ("https://www.github.com/octocat?tab=repositories", "octocat"),
    ],
)
def test_parse_username_accepts_names_and_profile_urls(text, expected):
    assert github.parse_username(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "https://gitlab.com/octocat",
        "https://github.com/octocat/repo",
        "https://github.com/",
        "https://evil.example/github.com/octocat",
        "octocat?tab=x",
    ],
)
def test_parse_username_rejects_other_urls(text):
    with pytest.raises(github.InvalidUsernameError):
        github.parse_username(text)


def _capture(monkeypatch):
    seen = []

    def fake_get(url, headers=None, params=None, timeout=None):
        seen.append({"url": url, "headers": headers})
        return FakeResponse(responses.get(url, {}))

    responses = {}
    monkeypatch.setattr(github.requests, "get", fake_get)
    return seen, responses


def test_profile_request_targets_the_users_endpoint(monkeypatch):
    seen, _ = _capture(monkeypatch)
    github.fetch_profile("octocat")
    assert [r["url"] for r in seen] == ["https://api.github.com/users/octocat"]


def test_token_is_sent_when_configured(monkeypatch):
    monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "secret-token")
    get_settings.cache_clear()
    seen, _ = _capture(monkeypatch)
    github.fetch_profile("octocat")
    assert seen[0]["headers"]["Authorization"] == "Bearer secret-token"


def test_no_authorization_header_without_token(monkeypatch):
    seen, _ = _capture(monkeypatch)
    github.fetch_profile("octocat")
    assert "Authorization" not in seen[0]["headers"]


def test_languages_url_outside_the_api_is_not_requested(monkeypatch):
    seen, _ = _capture(monkeypatch)
    repos = [{"languages_url": "https://evil.example/steal"}]
    assert github.fetch_repo_languages(repos) == []
    assert seen == []


def test_repo_languages_are_merged_and_sorted(monkeypatch):
    _, responses = _capture(monkeypatch)
    responses["https://api.github.com/repos/o/a/languages"] = {"Python": 10, "Go": 5}
    responses["https://api.github.com/repos/o/b/languages"] = {"Python": 1, "Rust": 2}
    repos = [
        {"languages_url": "https://api.github.com/repos/o/a/languages"},
        {"languages_url": "https://api.github.com/repos/o/b/languages"},
        {"languages_url": ""},
    ]
    assert github.fetch_repo_languages(repos) == ["Go", "Python", "Rust"]


@pytest.mark.parametrize("payload", [{"message": "Not Found"}, "oops", None, 42])
def test_non_list_repos_payload_gives_no_repos(payload, monkeypatch):
    _, responses = _capture(monkeypatch)
    responses["https://api.github.com/users/octocat/repos"] = payload
    assert github.fetch_repos("octocat") == []


def test_repo_fields_are_trimmed_and_defaulted(monkeypatch):
    _, responses = _capture(monkeypatch)
    responses["https://api.github.com/users/octocat/repos"] = [
        {"name": "hello", "description": None, "language": None, "stargazers_count": 3},
        "not-a-repo",
    ]
    assert github.fetch_repos("octocat") == [
        {
            "name": "hello",
            "description": "",
            "language": "",
            "stars": 3,
            "html_url": "",
            "languages_url": "",
        }
    ]


def test_profile_readme_is_decoded(monkeypatch):
    _, responses = _capture(monkeypatch)
    url = "https://api.github.com/repos/octocat/octocat/contents/README.md"
    responses[url] = {"content": base64.b64encode("Hi, I use Rust ✓".encode()).decode()}
    assert github.fetch_profile_readme("octocat") == "Hi, I use Rust ✓"


@pytest.mark.parametrize("payload", [{"content": "!!not base64!!"}, {}, [], None])
def test_unusable_readme_payload_gives_none(payload, monkeypatch):
    _, responses = _capture(monkeypatch)
    responses["https://api.github.com/repos/octocat/octocat/contents/README.md"] = payload
    assert github.fetch_profile_readme("octocat") is None


def test_request_errors_give_empty_profile(monkeypatch):
    def failing_get(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(github.requests, "get", failing_get)
    assert github.fetch_profile("octocat") == {}


class ErrorResponse:
    def __init__(self, error):
        self.error = error

    def raise_for_status(self):
        if isinstance(self.error, requests.HTTPError):
            raise self.error

    def json(self):
        raise self.error


@pytest.mark.parametrize(
    "error", [requests.HTTPError("404 Not Found"), requests.JSONDecodeError("bad", "", 0)]
)
def test_http_and_json_errors_give_empty_results(error, monkeypatch):
    monkeypatch.setattr(github.requests, "get", lambda *a, **k: ErrorResponse(error))
    assert (github.fetch_profile("octocat"), github.fetch_repos("octocat")) == ({}, [])


def test_profile_and_repos_forward_the_repo_limit(monkeypatch):
    seen = []

    def fake_get(url, headers=None, params=None, timeout=None):
        seen.append(params)
        return FakeResponse([] if url.endswith("/repos") else {"login": "octocat"})

    monkeypatch.setattr(github.requests, "get", fake_get)
    result = github.fetch_profile_and_repos("octocat", 6)
    assert result == {"profile": {"login": "octocat"}, "repos": []}
    assert {"sort": "updated", "per_page": 6} in seen


def test_null_repo_fields_become_defaults(monkeypatch):
    _, responses = _capture(monkeypatch)
    responses["https://api.github.com/users/octocat/repos"] = [
        {"name": None, "stargazers_count": None, "html_url": None, "languages_url": None}
    ]
    repo = github.fetch_repos("octocat")[0]
    assert (repo["name"], repo["stars"], repo["html_url"], repo["languages_url"]) == ("", 0, "", "")


@pytest.mark.parametrize(("limit", "sent"), [(-5, 1), (0, 1), (6, 6), (500, github.MAX_REPOS)])
def test_repo_limit_is_kept_within_githubs_range(limit, sent, monkeypatch):
    seen = []
    monkeypatch.setattr(
        github.requests,
        "get",
        lambda url, headers=None, params=None, timeout=None: (
            seen.append(params) or FakeResponse([])
        ),
    )
    github.fetch_repos("octocat", limit)
    assert seen[0]["per_page"] == sent

"""Read-only GitHub REST API client for public profile data.

Network and payload problems are logged and turned into empty results, but an
invalid username raises InvalidUsernameError before any request is made: the
username becomes part of a URL that carries our access token.
"""

import base64
import binascii
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests

from app.core.config import get_settings

logger = logging.getLogger(__name__)

API_ROOT = "https://api.github.com"
REQUEST_TIMEOUT_SECONDS = 10
MAX_REPOS = 100
# GitHub usernames: 1-39 letters, digits or single inner hyphens.
USERNAME_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}")
GITHUB_HOSTS = {"github.com", "www.github.com"}


class InvalidUsernameError(ValueError):
    pass


def validate_username(username: str) -> str:
    if not isinstance(username, str) or not USERNAME_PATTERN.fullmatch(username):
        raise InvalidUsernameError(f"Not a valid GitHub username: {username!r}")
    return username


def parse_username(text: str) -> str:
    """Accept 'octocat', 'github.com/octocat' or 'https://github.com/octocat/?tab=x'."""
    value = text.strip()
    if "/" in value:
        url = urlparse(value if "://" in value else f"https://{value}")
        if url.hostname not in GITHUB_HOSTS:
            raise InvalidUsernameError(f"Not a GitHub profile URL: {text!r}")
        segments = [segment for segment in url.path.split("/") if segment]
        value = segments[0] if len(segments) == 1 else ""
    return validate_username(value)


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = get_settings().github_access_token
    if token:
        headers["Authorization"] = f"Bearer {token.get_secret_value()}"
    return headers


def _get_json(url: str, params: dict | None = None):
    """GET a GitHub API URL; returns None (and logs) on any failure."""
    if not url.startswith(f"{API_ROOT}/"):
        logger.warning("Refusing non-GitHub-API URL %s", url)
        return None
    try:
        response = requests.get(
            url, headers=_headers(), params=params, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        logger.warning("GitHub request failed for %s: %s", url, error)
        return None


def fetch_profile(username: str) -> dict:
    payload = _get_json(f"{API_ROOT}/users/{validate_username(username)}")
    return payload if isinstance(payload, dict) else {}


def fetch_repos(username: str, limit: int = MAX_REPOS) -> list[dict]:
    """Most recently updated public repos, trimmed to the fields we use."""
    raw = _get_json(
        f"{API_ROOT}/users/{validate_username(username)}/repos",
        params={"sort": "updated", "per_page": limit},
    )
    if not isinstance(raw, list):
        return []
    return [
        {
            "name": repo.get("name", ""),
            "description": repo.get("description") or "",
            "language": repo.get("language") or "",
            "stars": repo.get("stargazers_count", 0),
            "html_url": repo.get("html_url", ""),
            "languages_url": repo.get("languages_url", ""),
        }
        for repo in raw
        if isinstance(repo, dict)
    ]


def fetch_repo_languages(repos: list[dict]) -> list[str]:
    """Union of all languages GitHub detects across the given repos."""
    languages: set[str] = set()
    for repo in repos:
        if repo.get("languages_url"):
            payload = _get_json(repo["languages_url"])
            if isinstance(payload, dict):
                languages.update(payload)
    return sorted(languages)


def fetch_profile_readme(username: str) -> str | None:
    """Text of the user's profile README (the `<username>/<username>` repo), if any."""
    name = validate_username(username)
    payload = _get_json(f"{API_ROOT}/repos/{name}/{name}/contents/README.md")
    if not isinstance(payload, dict) or "content" not in payload:
        return None
    try:
        return base64.b64decode(payload["content"]).decode("utf-8", errors="replace")
    except (binascii.Error, TypeError, ValueError):
        logger.warning("Profile README for %s is not valid base64", name)
        return None


def fetch_profile_and_repos(username: str, repo_limit: int) -> dict:
    validate_username(username)
    with ThreadPoolExecutor(max_workers=2) as pool:
        profile = pool.submit(fetch_profile, username)
        repos = pool.submit(fetch_repos, username, repo_limit)
        return {"profile": profile.result(), "repos": repos.result()}

"""Read-only GitHub REST API client for public profile data."""

import base64
import logging
from concurrent.futures import ThreadPoolExecutor

import requests

from app.core.config import get_settings

logger = logging.getLogger(__name__)

API_ROOT = "https://api.github.com"
REQUEST_TIMEOUT_SECONDS = 10
MAX_REPOS = 100


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = get_settings().github_access_token
    if token:
        headers["Authorization"] = f"Bearer {token.get_secret_value()}"
    return headers


def _get_json(url: str, params: dict | None = None):
    """GET a GitHub API URL; returns None (and logs) on any failure."""
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
    return _get_json(f"{API_ROOT}/users/{username}") or {}


def fetch_repos(username: str, limit: int = MAX_REPOS) -> list[dict]:
    """Most recently updated public repos, trimmed to the fields we use."""
    raw = _get_json(
        f"{API_ROOT}/users/{username}/repos",
        params={"sort": "updated", "per_page": limit},
    )
    return [
        {
            "name": repo.get("name", ""),
            "description": repo.get("description") or "",
            "language": repo.get("language") or "",
            "stars": repo.get("stargazers_count", 0),
            "html_url": repo.get("html_url", ""),
            "languages_url": repo.get("languages_url", ""),
        }
        for repo in raw or []
    ]


def fetch_repo_languages(repos: list[dict]) -> list[str]:
    """Union of all languages GitHub detects across the given repos."""
    languages: set[str] = set()
    for repo in repos:
        if repo.get("languages_url"):
            languages.update(_get_json(repo["languages_url"]) or {})
    return sorted(languages)


def fetch_profile_readme(username: str) -> str | None:
    """Text of the user's profile README (the `<username>/<username>` repo), if any."""
    payload = _get_json(f"{API_ROOT}/repos/{username}/{username}/contents/README.md")
    if not payload or "content" not in payload:
        return None
    return base64.b64decode(payload["content"]).decode("utf-8", errors="replace")


def fetch_profile_and_repos(username: str, repo_limit: int) -> dict:
    with ThreadPoolExecutor(max_workers=2) as pool:
        profile = pool.submit(fetch_profile, username)
        repos = pool.submit(fetch_repos, username, repo_limit)
        return {"profile": profile.result(), "repos": repos.result()}

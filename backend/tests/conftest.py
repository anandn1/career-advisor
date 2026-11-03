"""Shared fixtures: every test runs with fake settings, a temp data dir and fresh caches.

No test reads the real .env, touches backend/data, opens Postgres or calls an LLM.
"""

import pytest
from langchain_core.runnables import RunnableLambda
from sqlalchemy import create_engine, event

from app.agents import curriculum_architect, interview_coach, job_market_analyst
from app.core import checkpointer, config, resources
from app.db import session as db_session
from app.db.models import Base

CACHED_FACTORIES = (
    config.get_settings,
    resources.get_llm,
    resources.get_resume_llm,
    resources.get_embeddings,
    resources.get_chroma,
    resources.get_web_search,
    checkpointer.get_checkpointer,
    job_market_analyst.get_job_market_graph,
    curriculum_architect.get_curriculum_graph,
    curriculum_architect.get_course_catalog,
    interview_coach.get_interview_graph,
    db_session.get_engine,
    db_session._session_factory,
)
OPTIONAL_KEYS = (
    "COHERE_API_KEY",
    "TAVILY_API_KEY",
    "GITHUB_ACCESS_TOKEN",
    "HF_TOKEN",
    "CORS_ORIGINS",
    "DB_URL",
)


def _clear_caches() -> None:
    for factory in CACHED_FACTORIES:
        factory.cache_clear()


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Fake required settings, no optional keys, no .env files, data under tmp_path."""
    monkeypatch.setitem(config.Settings.model_config, "env_file", None)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    for key in OPTIONAL_KEYS:
        monkeypatch.delenv(key, raising=False)
    _clear_caches()
    yield
    _clear_caches()


class FakeReply:
    def __init__(self, content: str, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeLLM:
    """Returns scripted replies in order (the last one repeats) and records prompts."""

    def __init__(self, *replies, structured=None):
        self.replies = [r if isinstance(r, FakeReply) else FakeReply(r) for r in replies]
        self.structured = structured
        self.prompts = []

    def invoke(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        index = min(len(self.prompts), len(self.replies)) - 1
        return self.replies[index]

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return RunnableLambda(lambda _: self.structured)


@pytest.fixture
def sqlite_engine(tmp_path, monkeypatch):
    """App tables in a temporary SQLite file, used by session_scope() and the API."""
    engine = create_engine(f"sqlite:///{tmp_path / 'app.db'}")
    # Enforce foreign keys (and ON DELETE CASCADE) like Postgres does in production.
    event.listen(
        engine, "connect", lambda connection, _: connection.execute("PRAGMA foreign_keys=ON")
    )
    Base.metadata.create_all(engine)
    monkeypatch.setattr(db_session, "get_engine", lambda: engine)
    db_session._session_factory.cache_clear()
    return engine


@pytest.fixture
def fake_llm():
    return FakeLLM

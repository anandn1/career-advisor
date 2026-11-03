import pytest
from pydantic import ValidationError

from app.core import checkpointer
from app.core.config import Settings


def settings(**values):
    return Settings(_env_file=None, **values)


@pytest.mark.parametrize(
    ("url", "normalized"),
    [
        ("postgres://u:p@h/db", "postgresql://u:p@h/db"),
        ("postgresql://u:p@h/db", "postgresql://u:p@h/db"),
    ],
)
def test_database_url_scheme_is_normalized(url, normalized, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", url)
    assert settings().database_url == normalized


@pytest.mark.parametrize(
    ("url", "sqlalchemy_url"),
    [
        ("postgres://u:p%40ss@h/db", "postgresql+psycopg://u:p%40ss@h/db"),
        ("postgresql://h/db", "postgresql+psycopg://h/db"),
        ("postgresql+psycopg://h/db", "postgresql+psycopg://h/db"),
        ("sqlite:///x.db", "sqlite:///x.db"),
    ],
)
def test_sqlalchemy_url_uses_psycopg_3(url, sqlalchemy_url, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", url)
    assert settings().sqlalchemy_url == sqlalchemy_url


def test_db_url_is_accepted_as_an_alias(monkeypatch):
    monkeypatch.delenv("DATABASE_URL")
    monkeypatch.setenv("DB_URL", "postgresql://alias/db")
    assert settings().database_url == "postgresql://alias/db"


def test_missing_required_settings_fail_fast(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(ValidationError, match="groq_api_key"):
        settings()


@pytest.mark.parametrize(
    ("raw", "origins"),
    [
        ("http://a,http://b", ["http://a", "http://b"]),
        (" http://a , ,http://b ", ["http://a", "http://b"]),
        ('["http://a", "http://b"]', ["http://a", "http://b"]),
        ("http://only", ["http://only"]),
    ],
)
def test_cors_origins_accept_csv_or_json(raw, origins, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", raw)
    assert settings().cors_origins == origins


def test_cors_origins_default_to_local_frontend():
    assert settings().cors_origins == ["http://localhost:3000"]


def test_empty_optional_keys_count_as_unset(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "")
    assert settings().cohere_api_key is None


def test_data_paths_live_under_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    s = settings()
    assert {
        s.market_chroma_dir.parent,
        s.courses_chroma_dir.parent,
        s.interview_chroma_dir.parent,
        s.output_dir.parent,
    } == {tmp_path}


def test_secrets_are_not_shown_in_repr():
    assert "test-groq-key" not in repr(settings())


def test_checkpointer_pool_is_closed_when_setup_fails(monkeypatch):
    closed = []

    class Pool:
        def __init__(self, **kwargs):
            pass

        def close(self):
            closed.append(True)

    class Saver:
        def __init__(self, pool):
            pass

        def setup(self):
            raise ConnectionError("database down")

    monkeypatch.setattr(checkpointer, "ConnectionPool", Pool)
    monkeypatch.setattr(checkpointer, "PostgresSaver", Saver)
    with pytest.raises(ConnectionError):
        checkpointer.get_checkpointer()
    assert closed == [True]


def test_database_password_is_not_shown_in_repr(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:hunter2@h/db")
    assert "hunter2" not in repr(settings())


def test_each_store_has_its_own_directory():
    s = settings()
    directories = [s.market_chroma_dir, s.courses_chroma_dir, s.interview_chroma_dir, s.output_dir]
    assert len(set(directories)) == len(directories)

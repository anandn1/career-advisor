"""Checks that need real Postgres (JSONB, migrations). Skipped unless TEST_DATABASE_URL is set.

Point TEST_DATABASE_URL at a disposable database: these tests drop and recreate its schema.
"""

import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, make_url, text

from app.core.config import BACKEND_DIR, get_settings
from app.db import career_sessions
from app.db import session as db_session
from app.db.models import RunKind, RunStatus

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set"),
]


@pytest.fixture
def migrated_database(monkeypatch):
    url = make_url(os.environ["TEST_DATABASE_URL"])
    if not (url.database or "").endswith("_test"):
        pytest.fail("TEST_DATABASE_URL must name a disposable database ending in '_test'")
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    get_settings.cache_clear()
    engine = create_engine(get_settings().sqlalchemy_url)
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    command.upgrade(Config(str(BACKEND_DIR / "alembic.ini")), "head")
    monkeypatch.setattr(db_session, "get_engine", lambda: engine)
    db_session._session_factory.cache_clear()
    yield engine
    engine.dispose()


def test_migrations_create_the_app_tables(migrated_database):
    tables = set(inspect(migrated_database).get_table_names())
    expected = {"sessions", "agent_runs", "resume_files", "resume_templates", "tags"}
    assert expected <= tables


def test_migrations_can_be_rolled_back(migrated_database):
    command.downgrade(Config(str(BACKEND_DIR / "alembic.ini")), "base")
    assert "agent_runs" not in inspect(migrated_database).get_table_names()


def test_json_columns_round_trip_on_postgres(migrated_database):
    with db_session.session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        career_session.skills = ["Go", "C#"]
        run = career_sessions.create_run(
            db,
            career_session,
            RunKind.JOB_MARKET,
            {"nested": {"list": [1, "ü"]}},
            status=RunStatus.RUNNING,
        )
        run_id = run.id
    with db_session.session_scope() as db:
        stored = career_sessions.get_run(db, run_id)
        assert (stored.input, stored.session.skills) == (
            {"nested": {"list": [1, "ü"]}},
            ["Go", "C#"],
        )


def test_deleting_a_session_cascades_in_postgres(migrated_database):
    with db_session.session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        run_id = career_sessions.create_run(db, career_session, RunKind.JOB_MARKET, {}).id
        db.delete(career_session)
    with db_session.session_scope() as db:
        assert career_sessions.get_run(db, run_id) is None


def test_models_and_migrations_match(migrated_database):
    command.check(Config(str(BACKEND_DIR / "alembic.ini")))


def test_migrations_upgrade_again_after_rollback(migrated_database):
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    expected = {"sessions", "agent_runs", "resume_files", "resume_templates", "tags"}
    assert expected <= set(inspect(migrated_database).get_table_names())


def test_enum_columns_round_trip_on_postgres(migrated_database):
    with db_session.session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        run_id = career_sessions.create_run(db, career_session, RunKind.RESUME_REVIEW, {}).id
    with db_session.session_scope() as db:
        stored = career_sessions.get_run(db, run_id)
        assert (stored.kind, stored.status) == (RunKind.RESUME_REVIEW, RunStatus.QUEUED)


def test_duplicate_unfinished_runs_are_rejected_by_postgres(migrated_database):
    from sqlalchemy.exc import IntegrityError

    with db_session.session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        career_sessions.create_run(db, career_session, RunKind.JOB_MARKET, {})
        session_id = career_session.id
    with pytest.raises(IntegrityError), db_session.session_scope() as db:
        career_session = career_sessions.get_session(db, session_id)
        career_sessions.create_run(db, career_session, RunKind.JOB_MARKET, {})


def test_expiry_bulk_delete_cascades_in_postgres(migrated_database):
    from datetime import timedelta

    from app.db.models import ResumeFile, utcnow

    with db_session.session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        career_session.last_seen_at = utcnow() - timedelta(days=40)
        run = career_sessions.create_run(db, career_session, RunKind.RESUME, {})
        db.add(ResumeFile(run_id=run.id, latex="x"))
        run_id = run.id
    with db_session.session_scope() as db:
        career_sessions.delete_expired_sessions(db, timedelta(days=30))
    with db_session.session_scope() as db:
        assert db.get(ResumeFile, run_id) is None


def test_unique_index_exempts_interviews_and_finished_runs(migrated_database):
    with db_session.session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        for _ in range(2):
            career_sessions.create_run(
                db, career_session, RunKind.INTERVIEW, {}, status=RunStatus.RUNNING
            )
        career_sessions.create_run(
            db, career_session, RunKind.JOB_MARKET, {}, status=RunStatus.SUCCEEDED
        )
        career_sessions.create_run(db, career_session, RunKind.JOB_MARKET, {})
        session_id = career_session.id
    with db_session.session_scope() as db:
        assert len(career_sessions.list_runs(db, session_id)) == 4


def test_result_that_postgres_rejects_still_ends_the_run(migrated_database):
    from concurrent.futures import Executor, Future

    from app.services import runner

    class Inline(Executor):
        def submit(self, fn, /, *args, **kwargs):
            future = Future()
            future.set_result(fn(*args, **kwargs))
            return future

    with db_session.session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        run_id = career_sessions.create_run(db, career_session, RunKind.JOB_MARKET, {}).id
    handlers = {RunKind.JOB_MARKET: lambda context: {"text": "a\x00b"}}  # JSONB rejects NUL
    runner.RunExecutor(handlers, Inline()).execute(run_id)
    with db_session.session_scope() as db:
        stored = career_sessions.get_run(db, run_id)
        assert (stored.status, stored.error) == (RunStatus.FAILED, runner.SAVE_FAILED_MESSAGE)

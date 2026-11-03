from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db import seed_templates
from app.db.models import Base, ResumeTemplate, Tag
from app.db.templates import find_template_for_role, get_or_create_tag, template_exists


@pytest.fixture
def make_session():
    """Sessions on a fresh in-memory SQLite database with the app's tables."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def session(make_session):
    with make_session() as db:
        yield db


def add_template(session, name, *tags):
    template = ResumeTemplate(
        name=name, latex_src=f"% {name}", tags=[get_or_create_tag(session, t) for t in tags]
    )
    session.add(template)
    session.flush()
    return template


@pytest.fixture
def catalog(session):
    add_template(session, "a-engineering", "Engineering")
    add_template(session, "b-software", "Software Development", "Technology")
    add_template(session, "c-design", "Design & User Experience")
    add_template(session, "d-startup", "Startup")
    return session


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("UX Designer", "c-design"),
        ("Frontend Developer", "b-software"),
        ("Software Developer", "b-software"),
        ("Mechanical Engineer", "a-engineering"),
        ("Art Director", "a-engineering"),
        ("", "a-engineering"),
        ("100%_ role", "a-engineering"),
    ],
)
def test_template_matches_the_role(catalog, role, expected):
    assert find_template_for_role(catalog, role).name == expected


def test_no_templates_gives_none(session):
    assert find_template_for_role(session, "Software Engineer") is None


def test_tie_goes_to_the_first_template_by_name(session):
    add_template(session, "z-tech", "Technology")
    add_template(session, "m-tech", "Technology")
    assert find_template_for_role(session, "Technology Lead").name == "m-tech"


def test_existing_tag_is_reused(session):
    first = get_or_create_tag(session, "Startup")
    session.flush()
    assert get_or_create_tag(session, "Startup") is first


def test_template_exists_by_name(catalog):
    assert template_exists(catalog, "c-design") and not template_exists(catalog, "missing")


# --- Seeding ---


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def invoke(self, messages):
        return type("Reply", (), {"content": self.reply})()


@pytest.mark.parametrize(
    ("reply", "tags"),
    [
        ("Technology, Startup", ["Technology", "Startup"]),
        ("Technology, Technology", ["Technology"]),
        ("technology, STARTUP ", ["Technology", "Startup"]),
        ("Technology, Underwater Basket Weaving", ["Technology"]),
        ("", []),
    ],
)
def test_llm_tags_are_normalized(reply, tags, monkeypatch):
    monkeypatch.setattr(seed_templates, "get_llm", lambda: FakeLLM(reply))
    assert seed_templates.tag_template("\\documentclass{article}") == tags


@pytest.fixture
def seed_into(make_session, monkeypatch, tmp_path):
    """Run seed_templates against SQLite with the given LLM reply; return a session."""

    @contextmanager
    def scope():
        with make_session() as db, db.begin():
            yield db

    monkeypatch.setattr(seed_templates, "session_scope", scope)

    def run(reply, *names):
        for name in names:
            (tmp_path / f"{name}.tex").write_text(f"\\documentclass{{article}} % {name}")
        monkeypatch.setattr(seed_templates, "get_llm", lambda: FakeLLM(reply))
        return seed_templates.seed_templates(tmp_path), make_session()

    return run


def test_seeding_adds_each_template(seed_into):
    added, db = seed_into("Technology", "one", "two")
    assert added == 2 and sorted(db.scalars(select(ResumeTemplate.name))) == [
        "one",
        "two",
    ]


def test_duplicate_llm_tags_do_not_break_seeding(seed_into):
    added, db = seed_into("Technology, Technology", "one")
    assert added == 1 and [t.name for t in db.scalars(select(Tag))] == ["Technology"]


def test_seeding_twice_adds_nothing_new(seed_into):
    seed_into("Technology", "one")
    added, _ = seed_into("Technology", "one")
    assert added == 0


def test_tags_are_shared_between_templates(seed_into):
    _, db = seed_into("Technology", "one", "two")
    assert len(list(db.scalars(select(Tag)))) == 1

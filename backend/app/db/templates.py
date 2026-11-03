"""Queries for resume templates and their tags."""

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ResumeTemplate, Tag

MIN_STEM_LENGTH = 3
_SUFFIXES = ("ment", "ing", "ers", "er", "s")
_WORD = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    """Crude stem so 'designer'/'design' and 'engineer'/'engineering' compare equal."""
    stripped = True
    while stripped:
        stripped = False
        for suffix in _SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM_LENGTH:
                word, stripped = word[: -len(suffix)], True
                break
    return word


def _stems(text: str) -> set[str]:
    return {_stem(word) for word in _WORD.findall(text.lower()) if len(word) >= MIN_STEM_LENGTH}


def list_templates(session: Session) -> list[ResumeTemplate]:
    return list(session.scalars(select(ResumeTemplate).order_by(ResumeTemplate.name)))


def get_template(session: Session, template_id: str) -> ResumeTemplate | None:
    return session.get(ResumeTemplate, template_id)


def find_template_for_role(session: Session, target_role: str) -> ResumeTemplate | None:
    """The template whose tags share the most words with the role; first by name on a tie.

    Falls back to the first template by name when nothing matches; None if there are none.
    """
    templates = list_templates(session)
    role_stems = _stems(target_role)

    def overlap(template: ResumeTemplate) -> int:
        return len(role_stems & set().union(*(_stems(tag.name) for tag in template.tags)))

    best = max(templates, key=overlap, default=None)  # max keeps the first of equals
    if best is not None and overlap(best) == 0:
        return templates[0]
    return best


def get_or_create_tag(session: Session, name: str) -> Tag:
    tag = session.scalars(select(Tag).where(Tag.name == name)).first()
    if tag is None:
        tag = Tag(name=name)
        session.add(tag)
    return tag


def template_exists(session: Session, name: str) -> bool:
    query = select(ResumeTemplate.id).where(ResumeTemplate.name == name)
    return session.scalars(query).first() is not None

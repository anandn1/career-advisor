"""Queries for resume templates and their tags."""

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.db.models import ResumeTemplate, Tag

MIN_MATCH_WORD_LENGTH = 3


def list_templates(session: Session) -> list[ResumeTemplate]:
    return list(session.scalars(select(ResumeTemplate).order_by(ResumeTemplate.name)))


def get_template(session: Session, template_id: str) -> ResumeTemplate | None:
    return session.get(ResumeTemplate, template_id)


def find_template_for_role(session: Session, target_role: str) -> ResumeTemplate | None:
    """A template tagged with a word of the role (e.g. 'Software'), else any template."""
    words = [word.lower() for word in target_role.split() if len(word) >= MIN_MATCH_WORD_LENGTH]
    by_name = select(ResumeTemplate).order_by(ResumeTemplate.name).limit(1)
    if words:
        tag_matches = [func.lower(Tag.name).contains(word, autoescape=True) for word in words]
        match = session.scalars(by_name.join(ResumeTemplate.tags).where(or_(*tag_matches))).first()
        if match:
            return match
    return session.scalars(by_name).first()


def get_or_create_tag(session: Session, name: str) -> Tag:
    tag = session.scalars(select(Tag).where(Tag.name == name)).first()
    if tag is None:
        tag = Tag(name=name)
        session.add(tag)
    return tag


def template_exists(session: Session, name: str) -> bool:
    query = select(ResumeTemplate.id).where(ResumeTemplate.name == name)
    return session.scalars(query).first() is not None

"""Skill names: how postings' skills are matched to the user's and to each other."""

import re

_SKILL_WORD = re.compile(r"[a-z0-9+#]+")  # keeps "c++" and "c#" apart from "c"

# Postings list these among their skills, but they are not gaps a course closes.
SOFT_SKILLS = {
    "communication", "teamwork", "collaboration", "problem solving", "scrum", "agile",
    "agile methodologies", "kanban", "critical thinking", "project management",
    "user experience", "user interface", "leadership", "presentation", "time management",
}  # fmt: skip
# Words that only qualify a skill, so "microservices architecture" is "microservices".
QUALIFIER_WORDS = {
    "architecture", "basics", "concepts", "development", "framework", "frameworks",
    "fundamentals", "language", "languages", "principles", "programming", "skills",
}  # fmt: skip


def normalize_skill(skill: str) -> str:
    return re.sub(r"[\s-]+", " ", skill.lower()).strip(" .,;")


def _words(skill: str) -> tuple[str, ...]:
    return tuple(_SKILL_WORD.findall(normalize_skill(skill)))


def _plain(words: tuple[str, ...]) -> tuple[str, ...]:
    """The words without qualifiers: "microservices architecture" is "microservices"."""
    return tuple(word for word in words if word not in QUALIFIER_WORDS) or words


_SOFT_SKILL_WORDS = frozenset(_words(skill) for skill in SOFT_SKILLS)


def is_soft_skill(skill: str) -> bool:
    words = _words(skill)
    return bool(words) and (words in _SOFT_SKILL_WORDS or _plain(words) in _SOFT_SKILL_WORDS)


def same_skill(a: str, b: str) -> bool:
    """The same words, or one is the other plus qualifiers ("Microservices Architecture" and
    "microservices"); two different qualifiers ("software architecture" and "software
    development") are different skills."""
    words_a, words_b = _words(a), _words(b)
    if not words_a or not words_b:
        return False
    return words_a == words_b or _plain(words_a) == words_b or _plain(words_b) == words_a


def _spaced(skill: str) -> str:
    """Lower case with spaces tidied, keeping hyphens and "&" ("go-to-market", "r&d")."""
    return re.sub(r"\s+", " ", skill.lower()).strip(" .,;")


def has_skill(skill: str, user_skills: list[str]) -> bool:
    """The user names the same skill, or a longer one that starts with it as a separate
    word: "AWS Lambda" covers "aws"; "Objective-C", "Go-to-market" and "R&D" do not cover
    "c", "go" or "r"."""
    if not _words(skill):
        return False
    prefix = _spaced(skill) + " "
    return any(same_skill(skill, have) or _spaced(have).startswith(prefix) for have in user_skills)


def group_skills(skills: list[str]) -> dict[str, str]:
    """Each skill (normalized) and the name of its group: the shortest spelling of the same
    skill in the list. Blank or punctuation-only names are left out."""
    names = sorted({normalize_skill(s) for s in skills if _words(s)}, key=lambda s: (len(s), s))
    groups: dict[str, str] = {}
    for name in names:
        groups[name] = next(
            (g for g in dict.fromkeys(groups.values()) if same_skill(name, g)), name
        )
    return groups

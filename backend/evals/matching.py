"""Matching model output against expected labels that allow several spellings."""

import re
from dataclasses import dataclass

# Punctuation that is part of a name (C++, C#, Next.js, CI/CD, GD&T) is kept.
_SEPARATORS = re.compile(r"[\s_\-]+")


_MIN_PLURAL_LENGTH = 4  # "aws", "css" and "gcs" are not plurals


def _singular(word: str) -> str:
    plural = len(word) >= _MIN_PLURAL_LENGTH and word.endswith("s") and not word.endswith("ss")
    return word[:-1] if plural else word


def normalise(text: str) -> str:
    """Lower case, one space between words, plurals folded ('APIs' == 'API').

    Both sides of a comparison go through this, so folding 'Redis' to 'redi' is harmless.
    """
    words = _SEPARATORS.sub(" ", text.lower()).strip(" .,;:'\"()").split(" ")
    return " ".join(_singular(word) for word in words)


_PARENTHETICAL = re.compile(r"\s*\([^)]*\)")


def _groups_hit(item: str, groups: list[list[str]]) -> list[int]:
    """Groups whose spelling the item is, ignoring a note in brackets ('Kubernetes (EKS)').

    Only whole-item matches count. Matching words inside an item would let one long item
    ('Go, Python and Redis' on one line, or a pasted paragraph) claim many skills, while the
    app shows it as a single skill; it would also read 'R&D' as R or 'X-ray' as Ray.
    """
    text = normalise(_PARENTHETICAL.sub("", item))
    return [i for i, group in enumerate(groups) if text in {normalise(a) for a in group}]


@dataclass
class ListMatch:
    found: list[str]  # expected groups covered (first spelling of each)
    missed: list[str]  # expected groups not covered
    matched_items: list[str]  # returned items that name an expected group
    acceptable: list[str]  # returned items that are fair but not expected
    extras: list[str]  # returned items that fit nothing

    @property
    def recall(self) -> float:
        expected = len(self.found) + len(self.missed)
        return len(self.found) / expected if expected else 1.0

    @property
    def precision(self) -> float:
        returned = len(self.matched_items) + len(self.acceptable) + len(self.extras)
        return (returned - len(self.extras)) / returned if returned else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def as_dict(self) -> dict:
        return {
            "recall": round(self.recall, 3),
            "precision": round(self.precision, 3),
            "missed": self.missed,
            "extras": self.extras,
        }


def match_list(
    items: list[str], expected: list[list[str]], acceptable: tuple[str, ...] | list[str] = ()
) -> ListMatch:
    """Which expected groups the items cover, and which items fit nothing."""
    hit: set[int] = set()
    matched_items, acceptable_items, extras = [], [], []
    acceptable_groups = [[alias] for alias in acceptable]
    for item in items:
        groups = _groups_hit(item, expected)
        if groups:
            hit.update(groups)
            matched_items.append(item)
        elif _groups_hit(item, acceptable_groups):
            acceptable_items.append(item)
        else:
            extras.append(item)
    return ListMatch(
        found=[expected[i][0] for i in sorted(hit)],
        missed=[group[0] for i, group in enumerate(expected) if i not in hit],
        matched_items=matched_items,
        acceptable=acceptable_items,
        extras=extras,
    )

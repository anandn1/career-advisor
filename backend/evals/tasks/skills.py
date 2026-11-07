"""Skill extraction from resumes and GitHub profiles, and template tagging."""

from unittest.mock import patch

from app.agents import skills
from app.db.seed_templates import TEMPLATES_DIR, tag_template
from evals.core import CaseResult, Outcome, TaskContext, load_data
from evals.matching import match_list

# The same bar for every case (a recall threshold would allow a miss on 16 skills but not 8).
MAX_EXTRAS_TO_PASS = 1


class FallbackUsedError(RuntimeError):
    """The agent gave up on the LLM's answer and used its keyword matcher instead."""


def _no_fallback(_text: str) -> list[str]:
    raise FallbackUsedError("the LLM returned no skills, so the agent used its keyword matcher")


def _extract(case: dict) -> list[str]:
    """The LLM's skills exactly as the app gets them, with the keyword fallback switched off.

    Resumes go through the app's own entry point. GitHub cases skip the GitHub API calls:
    their text is already in the shape extract_skills_from_github builds.
    """
    if case["source"] == "resume":
        with patch.object(skills, "match_known_skills", _no_fallback):
            return skills.extract_skills_from_text(case["text"])
    found = skills._ask_llm_for_skills(skills.GITHUB_SYSTEM_PROMPT, case["text"])
    if not found:
        raise FallbackUsedError("the LLM returned no skills")
    return found


def skill_extraction(ctx: TaskContext) -> list[CaseResult]:
    def run(case: dict) -> Outcome:
        returned = _extract(case)
        match = match_list(returned, case["expected"], case["acceptable"])
        passed = not match.missed and len(match.extras) <= MAX_EXTRAS_TO_PASS
        return Outcome(match.f1, passed, {**match.as_dict(), "returned": returned})

    return [
        ctx.run_case("skill_extraction", case["id"], lambda case=case: run(case))
        for case in load_data("skills")["cases"]
    ]


def score_tags(case: dict, tags: list[str]) -> Outcome:
    """Every required group has a chosen tag, no forbidden tag, and at least one tag."""
    missing = [group for group in case["required"] if not set(group) & set(tags)]
    wrong = [tag for tag in tags if tag in case["forbidden"]]
    checks = [not missing, not wrong, bool(tags)]
    details = {"tags": tags, "missing": missing, "forbidden_chosen": wrong}
    return Outcome(sum(checks) / len(checks), all(checks), details)


def template_tagging(ctx: TaskContext) -> list[CaseResult]:
    def run(case: dict) -> Outcome:
        source = case.get("latex") or (TEMPLATES_DIR / f"{case['template']}.tex").read_text(
            encoding="utf-8"
        )
        return score_tags(case, tag_template(source))

    return [
        ctx.run_case("template_tagging", case["template"], lambda case=case: run(case))
        for case in load_data("tagging")["cases"]
    ]


TASKS = {"skill_extraction": skill_extraction, "template_tagging": template_tagging}

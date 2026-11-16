"""Curriculum architect: planner, course parsing, critic, and full runs on the catalog."""

import json
from unittest.mock import patch

from app.agents import curriculum_architect as curriculum
from evals.core import CaseResult, Outcome, TaskContext, load_data
from evals.matching import match_list


def _cases(section: str) -> list[dict]:
    return load_data("curriculum")[section]


def planner(ctx: TaskContext) -> list[CaseResult]:
    def run(case: dict) -> Outcome:
        state = {
            "skill_gap_analysis": case["gaps"],
            "resume_inventory": json.dumps(case["inventory"], indent=2),
        }
        output = curriculum.planner_node(state)
        match = match_list(output["skills_to_find"], case["expected"])
        level = output["user_profile"].skill_level
        level_ok = level in case["skill_levels"]
        details = {**match.as_dict(), "skill_level": level, "returned": output["skills_to_find"]}
        score = match.recall * (1.0 if level_ok else 0.5)
        return Outcome(score, match.recall == 1.0 and not match.extras and level_ok, details)

    return [
        ctx.run_case("curriculum_planner", case["id"], lambda case=case: run(case))
        for case in _cases("planner")
    ]


def _same_page(url: str) -> str:
    """'https://x.com/a/' and 'https://x.com/a' are the same course page."""
    return url.strip().rstrip("/")


def course_parsing(ctx: TaskContext) -> list[CaseResult]:
    def run(case: dict) -> Outcome:
        state = {"current_skill": case["skill"], "raw_search_results": case["results"]}
        # A stand-in catalog keeps these courses out of the real one. (Patching only
        # _store_courses would also stop the real catalog seeding if built here first.)
        with patch.object(curriculum, "get_course_catalog"):
            courses = curriculum.process_and_store_courses(state)["course_catalog"]
        if not courses:
            raise RuntimeError("no courses extracted (structured output failed or was empty)")
        urls = [_same_page(course.url) for course in courses]
        given = {_same_page(result["url"]) for result in case["results"]}
        missed = [url for url in case["required"] if _same_page(url) not in urls]
        non_courses = [url for url in case["forbidden"] if _same_page(url) in urls]
        invented = [url for url in urls if url not in given]
        checks = [not missed, not non_courses, not invented]
        details = {
            "urls": urls,
            "missed": missed,
            "non_courses": non_courses,
            "invented": invented,
            "optional_kept": [url for url in case["allowed"] if _same_page(url) in urls],
        }
        return Outcome(sum(checks) / len(checks), all(checks), details)

    return [
        ctx.run_case("course_parsing", case["id"], lambda case=case: run(case))
        for case in _cases("course_parsing")
    ]


def critic(ctx: TaskContext) -> list[CaseResult]:
    def run(case: dict) -> Outcome:
        path = curriculum.PersonalizedLearningPath.model_validate(case["path"])
        critique = curriculum.critic_node({"draft_path": path})["structured_critique"]
        passed = critique.is_approved == case["expected_approved"]
        details = {
            "expected_approved": case["expected_approved"],
            "approved": critique.is_approved,
            "revisions_needed": critique.revisions_needed,
        }
        return Outcome(float(passed), passed, details)

    return [
        ctx.run_case("curriculum_critic", case["id"], lambda case=case: run(case))
        for case in _cases("critic")
    ]


def _catalog_urls() -> set[str]:
    return {_same_page(course.url) for course in curriculum._load_seed_courses()}


def end_to_end(ctx: TaskContext) -> list[CaseResult]:
    """The whole graph on catalog-covered gaps, with web search off so runs are comparable."""
    catalog_urls = _catalog_urls()

    def run(case: dict) -> Outcome:
        inputs = curriculum.learning_path_inputs(
            case["gaps"], case["user_skills"], case["target_role"]
        )
        with patch.object(curriculum, "get_web_search", return_value=None):
            state = curriculum.build_learning_path(*inputs)
        path = state.get("draft_path")
        if path is None:
            raise RuntimeError("the architect never submitted a valid learning path")
        covered = match_list(
            [r.skill_name for r in path.recommendations], [[g] for g in case["gaps"]]
        )
        invented = [
            r.course_url
            for r in path.recommendations
            if _same_page(r.course_url) not in catalog_urls
        ]
        approved = bool(
            state.get("structured_critique") and state["structured_critique"].is_approved
        )
        checks = [not covered.missed, not invented, approved]
        details = {
            "uncovered_gaps": covered.missed,
            "invented_urls": invented,
            "critic_approved": approved,
            "revisions": state.get("revision_count", 0),
            "recommendations": len(path.recommendations),
        }
        return Outcome(sum(checks) / len(checks), all(checks), details)

    return [
        ctx.run_case("curriculum_end_to_end", case["id"], lambda case=case: run(case))
        for case in _cases("end_to_end")
    ]


TASKS = {
    "curriculum_planner": planner,
    "course_parsing": course_parsing,
    "curriculum_critic": critic,
    "curriculum_end_to_end": end_to_end,
}

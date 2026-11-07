"""The floor for each classification task: the best score a constant answer gets.

A model's result means little until it beats this. Computed from the labelled data alone,
with the same scoring as the tasks; no model is called.
"""

from itertools import combinations
from statistics import mean

from app.db.seed_templates import PREDEFINED_TAGS
from evals.core import load_data
from evals.tasks.interview import GRADE_ORDER, score_grade
from evals.tasks.skills import score_tags

MAX_CONSTANT_TAGS = 2  # constant answers of one or two tags


def _best(task: str, candidates: dict[str, list[tuple[float, bool]]]) -> dict:
    """The constant answer with the highest mean score, with its pass count."""
    answer, outcomes = max(candidates.items(), key=lambda item: mean(s for s, _ in item[1]))
    return {
        "task": task,
        "constant_answer": answer,
        "score": round(mean(s for s, _ in outcomes), 2),
        "passed": sum(p for _, p in outcomes),
        "cases": len(outcomes),
    }


def tagging_floor() -> dict:
    cases = load_data("tagging")["cases"]
    candidates = {}
    for size in range(1, MAX_CONSTANT_TAGS + 1):
        for tags in combinations(PREDEFINED_TAGS, size):
            outcomes = [score_tags(case, list(tags)) for case in cases]
            candidates[", ".join(tags)] = [(o.score, o.passed) for o in outcomes]
    return _best("template_tagging", candidates)


def grading_floor() -> dict:
    cases = load_data("interview")["grading"]
    candidates = {}
    for grade in GRADE_ORDER:
        outcomes = [score_grade(case["grade"], grade) for case in cases]
        candidates[grade] = [(o.score, o.passed) for o in outcomes]
    return _best("answer_grading", candidates)


def _label_floor(task: str, labels: list, answers: dict[str, object]) -> dict:
    candidates = {
        name: [(float(answer == label), answer == label) for label in labels]
        for name, answer in answers.items()
    }
    return _best(task, candidates)


def decision_floor() -> dict:
    labels = [case["expected"] for case in load_data("interview")["follow_up_decision"]]
    answers = {"always FOLLOW_UP": "FOLLOW_UP", "always NEW_QUESTION": "NEW_QUESTION"}
    return _label_floor("follow_up_decision", labels, answers)


def critic_floor() -> dict:
    labels = [case["expected_approved"] for case in load_data("curriculum")["critic"]]
    return _label_floor(
        "curriculum_critic", labels, {"always approve": True, "always reject": False}
    )


def floors() -> list[dict]:
    return [tagging_floor(), grading_floor(), decision_floor(), critic_floor()]


def floors_table() -> str:
    lines = [
        "| Task | Best constant answer | Score | Passed |",
        "| :--- | :--- | ---: | ---: |",
    ]
    for floor in floors():
        lines.append(
            f"| {floor['task']} | {floor['constant_answer']} | {floor['score']:.2f} "
            f"| {floor['passed']}/{floor['cases']} |"
        )
    return "\n".join(lines)

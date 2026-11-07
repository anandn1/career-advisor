"""Interview coach: grading answers, deciding on follow-ups, writing them, summarising."""

import re
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from app.agents import interview_coach as coach
from evals.core import CaseResult, JudgeRequest, Outcome, TaskContext, load_data

GRADE_ORDER = ["bad", "moderate", "good", "excellent"]
NEAR_MISS_SCORE = 0.5

FOLLOW_UP_CRITERIA = [
    "Asks exactly one question",
    "Probes the specific gap, vagueness or mistake in the candidate's answer",
    "Concise and natural, with no praise, filler or preamble",
]
SUMMARY_CRITERIA = [
    "Faithful to the report: mentions no result, strength or weakness that is not in it",
    "Covers how the candidate did on each topic in the report",
    "Concise and useful to the candidate, with a clear takeaway",
]


def _grading_cases() -> dict[str, dict]:
    return {case["id"]: case for case in load_data("interview")["grading"]}


def _state_after_answer(case: dict, grade: str) -> dict:
    """Interview state right after the candidate answered `case` and it was graded."""
    entry = {
        "question_id": case["id"],
        "question": case["question"],
        "answer": case["answer"],
        "evaluation": grade,
        "skill": "general",
        "feedback": "",
    }
    return {
        "messages": [
            AIMessage(content=case["question"], name=coach.INTERVIEWER),
            HumanMessage(content=case["answer"]),
        ],
        "evaluation_report": [entry],
        "current_question_id": case["id"],
        "follow_up_count": 0,
    }


_app_parse_grade = coach._parse_grade


class UnreadableGradeError(RuntimeError):
    """The reply had no grade the app can read, so the app would use its default grade."""


def _strict_parse_grade(text: str) -> str:
    """The app's grade parser, but failing where the app quietly falls back to 'moderate'."""
    value = coach._parse_field("GRADE", text, "").lower()
    # Whole words, as the app reads them ("badly" is no grade), and exactly one of them: an
    # echoed "[bad/moderate/good/excellent]" would otherwise read as "excellent".
    named = [grade for grade in coach.GRADES if re.search(rf"\b{grade}\b", value)]
    if len(named) != 1:
        raise UnreadableGradeError(f"no single readable grade in: {text[:200]!r}")
    return _app_parse_grade(text)


def score_grade(expected: str, got: str) -> Outcome:
    """Full marks for the exact grade, half for one step off, nothing further away."""
    distance = abs(GRADE_ORDER.index(got) - GRADE_ORDER.index(expected))
    score = {0: 1.0, 1: NEAR_MISS_SCORE}.get(distance, 0.0)
    return Outcome(score, distance == 0, {"expected": expected, "got": got})


def answer_grading(ctx: TaskContext) -> list[CaseResult]:
    def run(case: dict) -> Outcome:
        with patch.object(coach, "_parse_grade", _strict_parse_grade):
            grade, skill, feedback = coach._grade_answer(case["question"], case["answer"])
        if grade == coach.EVALUATION_ERROR:
            raise RuntimeError(feedback)
        outcome = score_grade(case["grade"], grade)
        return Outcome(outcome.score, outcome.passed, {**outcome.details, "feedback": feedback})

    return [
        ctx.run_case("answer_grading", case["id"], lambda case=case: run(case))
        for case in _grading_cases().values()
    ]


DECISIONS = ("FOLLOW_UP", "NEW_QUESTION")


class UnreadableDecisionError(RuntimeError):
    """The reply did not name one decision; the app would quietly move on."""


class _RecordingLlm:
    """Passes calls through to the model and keeps each reply's text."""

    def __init__(self, llm):
        self._llm = llm
        self.replies: list[str] = []

    def invoke(self, *args, **kwargs):
        reply = self._llm.invoke(*args, **kwargs)
        self.replies.append(str(reply.content))
        return reply


def _require_one_decision(replies: list[str]) -> None:
    """The router reads anything without FOLLOW_UP as NEW_QUESTION, so an empty or echoed
    reply ("FOLLOW_UP or NEW_QUESTION") would pass cases without deciding anything."""
    if len(replies) != 1:
        raise UnreadableDecisionError(f"expected one decision call, saw {len(replies)}")
    named = [decision for decision in DECISIONS if decision in replies[0].upper()]
    if len(named) != 1:
        raise UnreadableDecisionError(f"reply names no single decision: {replies[0][:200]!r}")


def follow_up_decision(ctx: TaskContext) -> list[CaseResult]:
    graded = _grading_cases()

    def run(case: dict) -> Outcome:
        source = {**graded.get(case["id"], {}), **case}
        recording = _RecordingLlm(coach.get_llm())
        with patch.object(coach, "get_llm", lambda: recording):
            route = coach.decide_action_router(_state_after_answer(source, case["grade"]))
        _require_one_decision(recording.replies)
        choice = "FOLLOW_UP" if route == "ask_follow_up" else "NEW_QUESTION"
        passed = choice == case["expected"]
        return Outcome(float(passed), passed, {"expected": case["expected"], "got": choice})

    return [
        ctx.run_case("follow_up_decision", case["id"], lambda case=case: run(case))
        for case in load_data("interview")["follow_up_decision"]
    ]


# The prompt forbids openers like "Good answer"; only clear cases are checked here.
_PRAISE_OPENER = re.compile(
    r"^\W*(great|good|nice|excellent|okay|ok|thanks|thank you|interesting|sure)\b", re.I
)


def _follow_up_checks(question: str) -> Outcome:
    """Objective part of the follow-up score: at most one question, no praise opener.

    'At most one' rather than 'exactly one' because 'Walk me through ...' is a fine
    follow-up without a question mark; the judge decides whether it asks one thing.
    """
    checks = {
        "at_most_one_question": question.count("?") <= 1,
        "no_praise_opener": not _PRAISE_OPENER.match(question),
    }
    passed = all(checks.values())
    return Outcome(sum(checks.values()) / len(checks), passed, checks)


def follow_up_request(case: dict, question: str) -> JudgeRequest:
    return JudgeRequest(
        "Write one probing follow-up question for a technical interview answer.",
        f"Question: {case['question']}\nAnswer: {case['answer']}\nGrade: {case['grade']}",
        question,
        FOLLOW_UP_CRITERIA,
    )


def summary_request(case: dict, summary: str) -> JudgeRequest:
    return JudgeRequest(
        "Summarise a candidate's mock-interview performance from the grader's report.",
        case["report"],
        summary,
        SUMMARY_CRITERIA,
    )


def follow_up_question(ctx: TaskContext) -> list[CaseResult]:
    graded = _grading_cases()

    def run(case: dict) -> Outcome:
        reply = coach.ask_follow_up_node(_state_after_answer(case, case["grade"]))
        question = str(reply["messages"][0].content)
        if question == "Let's move on.":
            raise RuntimeError("the LLM call failed")
        request = follow_up_request(case, question)
        objective = _follow_up_checks(question)
        return ctx.judged_outcome(request, {"question": question}, objective)

    return [
        ctx.run_case("follow_up_question", case_id, lambda case_id=case_id: run(graded[case_id]))
        for case_id in load_data("interview")["follow_up_questions"]
    ]


def _report_entries(report: str) -> list[dict]:
    """'- skill: grade | feedback' lines back into the report entries the node reads."""
    entries = []
    for line in report.splitlines():
        skill, _, rest = line.removeprefix("- ").partition(": ")
        grade, _, feedback = rest.partition(" | ")
        entries.append({"skill": skill, "evaluation": grade, "feedback": feedback})
    return entries


def interview_summary(ctx: TaskContext) -> list[CaseResult]:
    def run(case: dict) -> Outcome:
        entries = _report_entries(case["report"])
        reply = coach.synthesize_and_update_node({"evaluation_report": entries})
        summary = str(reply["messages"][0].content).removeprefix("Interview complete!").strip()
        if summary == f"Completed {len(entries)} questions.":
            raise RuntimeError("the LLM call failed")
        return ctx.judged_outcome(summary_request(case, summary), {"summary": summary})

    return [
        ctx.run_case("interview_summary", case["id"], lambda case=case: run(case))
        for case in load_data("interview")["summaries"]
    ]


TASKS = {
    "answer_grading": answer_grading,
    "follow_up_decision": follow_up_decision,
    "follow_up_question": follow_up_question,
    "interview_summary": interview_summary,
}

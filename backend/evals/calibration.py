"""Does the judge tell good outputs from deliberately bad ones? Run before trusting its scores."""

import logging
from dataclasses import asdict, dataclass

from evals.core import PASSING_JUDGE_SCORE, JudgeRequest, load_data
from evals.judge import Judge, JudgeUnavailableError
from evals.tasks import interview, resume

FAILING_JUDGE_SCORE = 0.5
logger = logging.getLogger(__name__)


def _source_cases() -> dict[str, dict]:
    interview_data = load_data("interview")
    resume_data = load_data("resume")
    return {
        "follow_up_question": {case["id"]: case for case in interview_data["grading"]},
        "interview_summary": {case["id"]: case for case in interview_data["summaries"]},
        "resume_generation": {case["id"]: case for case in resume_data["generation"]},
        "resume_review": {case["id"]: case for case in resume_data["review"]},
    }


REQUEST_BUILDERS = {
    "follow_up_question": interview.follow_up_request,
    "interview_summary": interview.summary_request,
    "resume_generation": resume.resume_request,
    "resume_review": resume.review_request,
}


@dataclass
class PairResult:
    pair_id: str
    task: str
    good_score: float | None  # None when the judge was unavailable for this pair
    bad_score: float | None

    @property
    def passed(self) -> bool:
        if self.good_score is None or self.bad_score is None:
            return False
        return self.good_score >= PASSING_JUDGE_SCORE and self.bad_score < FAILING_JUDGE_SCORE


def _request(pair: dict, output: str) -> JudgeRequest:
    case = _source_cases()[pair["task"]][pair["source_case"]]
    return REQUEST_BUILDERS[pair["task"]](case, output)


def calibrate(judge: Judge) -> list[PairResult]:
    results = []
    for pair in load_data("judge_calibration")["pairs"]:
        good, bad = (_score(judge, pair, pair[side]) for side in ("good", "bad"))
        results.append(PairResult(pair["id"], pair["task"], good, bad))
    return results


def _score(judge: Judge, pair: dict, output: str) -> float | None:
    """The judge's 0..1 score, or None if it is unavailable (the pair then fails)."""
    try:
        return round(judge.score(**asdict(_request(pair, output))).normalised, 3)
    except JudgeUnavailableError as error:
        logger.warning("Calibration pair %s not judged: %s", pair["id"], str(error)[:200])
        return None


def calibration_table(results: list[PairResult]) -> str:
    lines = [
        f"Judge passes a pair when good >= {PASSING_JUDGE_SCORE} and bad < {FAILING_JUDGE_SCORE}.",
        "",
        "| Pair | Task | Good | Bad | Passed |",
        "| :--- | :--- | ---: | ---: | :---: |",
    ]
    for r in results:
        mark = "yes" if r.passed else "NO"
        good, bad = (f"{x:.2f}" if x is not None else "n/a" for x in (r.good_score, r.bad_score))
        lines.append(f"| {r.pair_id} | {r.task} | {good} | {bad} | {mark} |")
    return "\n".join(lines)

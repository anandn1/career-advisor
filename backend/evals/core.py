"""What every eval task shares: loading cases, running one case, recording the result."""

import copy
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path

from evals.judge import CriterionScore, Judge, JudgeUnavailableError, Verdict
from evals.models import ModelSpec, Usage, UsageRecorder, is_rate_limit

DATA_DIR = Path(__file__).resolve().parent / "data"
logger = logging.getLogger(__name__)


@cache
def _read_data(name: str) -> dict:
    return json.loads((DATA_DIR / f"{name}.json").read_text(encoding="utf-8"))


def load_data(name: str) -> dict:
    """A labelled data file; a fresh copy each time, so no task can change another's data."""
    return copy.deepcopy(_read_data(name))


PASSING_JUDGE_SCORE = 0.75
MIN_CRITERION_SCORE = 3  # out of 5: one bad aspect, e.g. an invented fact, fails the case
RATE_LIMITED = "rate limited, not scored"
MAX_ERROR_CHARS = 500


@dataclass(frozen=True)
class JudgeRequest:
    """Everything the judge needs, saved with the result so it can be judged later."""

    task: str
    source: str
    output: str
    criteria: list[str]


@dataclass
class Outcome:
    """What a case produced: a 0..1 score, whether it passed, and details for the report."""

    score: float | None
    passed: bool
    details: dict = field(default_factory=dict)


@dataclass
class CaseResult:
    task: str
    case_id: str
    model: str
    score: float | None  # 0..1; a crash scores 0; None when the judge was unavailable
    passed: bool
    usage: Usage
    cost: float | None
    details: dict = field(default_factory=dict)
    error: str | None = None
    run: int = 1  # which repeat of the same model and case


@dataclass
class TaskContext:
    spec: ModelSpec
    recorder: UsageRecorder
    judge: Judge | None
    latex_image: str | None = None
    run: int = 1

    def run_case(self, task: str, case_id: str, run: Callable[[], Outcome]) -> CaseResult:
        """Run one case, charging it the tokens it used; any exception is a failed case."""
        before = self.recorder.total.snapshot()
        rate_limited = False
        try:
            outcome = run()
            error = None
        except Exception as exc:  # a model that crashes the agent fails the case, not the run
            logger.warning("%s/%s on %s failed: %s", task, case_id, self.spec.label, exc)
            outcome, error = Outcome(0.0, False), f"{type(exc).__name__}: {exc}"
            rate_limited = is_rate_limit(exc)
        usage = self.recorder.total.snapshot() - before
        raised_model_error = error is not None and not rate_limited  # keeps its own message
        if usage.failed_calls and not raised_model_error:
            # A real failure is evidence against the model, even if a quota error came too.
            # (The agents catch provider errors and fall back: a default grade, "Let's move
            # on", a new question, so a dead model could otherwise pass cases.)
            outcome = Outcome(0.0, False, outcome.details)
            error = f"{usage.failed_calls} LLM call(s) failed; the agent fell back"
        elif rate_limited or usage.rate_limited_calls:
            # Out of quota says nothing about the model: leave the case unscored, for good
            # (rejudge skips it too). A crash raised after a refused call lands here too: it is
            # most likely the agent choking on the missing reply, not the model's mistake.
            outcome = Outcome(None, False, outcome.details)
            reason = error or f"{usage.rate_limited_calls} call(s) refused"
            error = f"{RATE_LIMITED}: {reason}"
        elif outcome.score is None and self.run > 1:
            error = "not judged: only the first run of each model is judged"
        elif outcome.score is None:
            error = "not judged yet: run python -m evals --rejudge <results file>"
        return CaseResult(
            task=task,
            case_id=case_id,
            model=self.spec.label,
            score=outcome.score,
            passed=outcome.passed,
            usage=usage,
            cost=self.spec.cost(usage.input_tokens, usage.output_tokens),
            details=outcome.details,
            error=error[:MAX_ERROR_CHARS] if error else None,
            run=self.run,
        )

    def judged_outcome(
        self, request: JudgeRequest, details: dict, objective: Outcome | None = None
    ) -> Outcome:
        """Outcome from the judge plus any objective checks; unscored if the judge is down.

        The request and the objective result are kept in the details, so `--rejudge`
        can finish the case later with the same judge.
        """
        details = {
            **details,
            "judge_request": asdict(request),
            "objective": asdict(objective) if objective else None,
        }
        if self.judge is None:
            return Outcome(None, False, details)
        try:
            verdict = self.judge.score(**asdict(request))
        except JudgeUnavailableError as error:
            logger.warning("Not judged yet: %s", str(error)[:200])
            return Outcome(None, False, details)
        return apply_verdict(details, verdict)


def apply_verdict(details: dict, verdict: Verdict) -> Outcome:
    """Combine a judge verdict with the case's objective checks (averaged; both must pass).

    The judge passes a case when its mean is high enough and no criterion scores below
    MIN_CRITERION_SCORE, so a good average cannot hide one serious failure.
    """
    judge_score = round(verdict.normalised, 3)
    details = {
        **details,
        "judge": {s.criterion: [s.score, s.reason] for s in verdict.scores},
        "judge_score": judge_score,
    }
    objective = details.get("objective")
    judge_passed = judge_score >= PASSING_JUDGE_SCORE and all(
        s.score >= MIN_CRITERION_SCORE for s in verdict.scores
    )
    if objective is None:
        return Outcome(judge_score, judge_passed, details)
    score = (objective["score"] + judge_score) / 2
    return Outcome(score, objective["passed"] and judge_passed, details)


def rejudge(results: list[CaseResult], judge: Judge) -> int:
    """Judge the saved cases that have no score yet; returns how many were finished."""
    finished = 0
    for result in results:
        request = result.details.get("judge_request")
        rate_limited = bool(result.error and result.error.startswith(RATE_LIMITED))
        if result.score is not None or request is None or result.run > 1 or rate_limited:
            continue  # only first runs are judged, and a rate-limited output is a fallback
        try:
            verdict = judge.score(**request)
        except JudgeUnavailableError as error:
            logger.warning("%s/%s still not judged: %s", result.task, result.case_id, error)
            continue
        outcome = apply_verdict(result.details, verdict)
        result.score, result.passed, result.details = outcome.score, outcome.passed, outcome.details
        result.error = None
        finished += 1
    return finished


def rescore(results: list[CaseResult]) -> int:
    """Apply the current pass rule to judged cases from their saved verdicts.

    No judge calls are made. Cases with an error keep their result: a fallback after a
    failed call stays at 0 and a rate-limited case stays unscored, whatever the verdict.
    Returns how many cases changed between passed and failed.
    """
    changed = 0
    for result in results:
        saved = result.details.get("judge")
        if result.error or result.score is None or not saved:
            continue
        verdict = Verdict(
            scores=[
                CriterionScore(criterion=criterion, score=score, reason=reason)
                for criterion, (score, reason) in saved.items()
            ]
        )
        outcome = apply_verdict(result.details, verdict)
        changed += outcome.passed != result.passed
        result.score, result.passed, result.details = outcome.score, outcome.passed, outcome.details
    return changed

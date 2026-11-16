"""The model-eval harness (backend/evals) scores fairly: offline, with fake models and judges."""

import json
import subprocess
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.agents import curriculum_architect as curriculum
from app.agents import interview_coach, skills
from app.db.seed_templates import PREDEFINED_TAGS, TEMPLATES_DIR
from app.services import latex
from evals import baselines, report
from evals.calibration import PairResult, calibrate
from evals.core import (
    DECIDED_BY_APP,
    RATE_LIMITED,
    CaseResult,
    JudgeRequest,
    Outcome,
    TaskContext,
    apply_verdict,
    load_data,
    rejudge,
    rescore,
)
from evals.judge import CriterionScore, Judge, JudgeUnavailableError, Verdict
from evals.matching import match_list, normalise
from evals.models import ModelSpec, Usage, UsageRecorder
from evals.tasks import curriculum as curriculum_task
from evals.tasks import interview as interview_task
from evals.tasks import resume as resume_task
from evals.tasks import skills as skills_task

# --- Matching model output against labels ---


def test_one_item_holding_several_skills_claims_none_of_them():
    match = match_list(["Go\nPython\nRedis"], [["Go"], ["Python"], ["Redis"]])
    assert match.found == []
    assert match.extras == ["Go\nPython\nRedis"]


@pytest.mark.parametrize("item", ["R&D", "X-ray", "Go-to-market", "Shell Oil", "Not Python"])
def test_a_skill_name_inside_a_longer_item_is_not_a_match(item):
    groups = [["R"], ["Ray"], ["Go"], ["Shell"], ["Python"]]
    assert match_list([item], groups).found == []


def test_a_note_in_brackets_is_ignored():
    match = match_list(["Kubernetes (EKS)"], [["Kubernetes"]])
    assert match.found == ["Kubernetes"]


def test_any_listed_spelling_matches_its_group():
    match = match_list(["Golang", "postgres"], [["Go", "Golang"], ["PostgreSQL", "Postgres"]])
    assert match.found == ["Go", "PostgreSQL"]
    assert match.recall == 1.0


@pytest.mark.parametrize(
    ("returned", "label"),
    [
        ("Chrome Extension", "Chrome Extensions"),
        ("APIs", "API"),
        ("rest  APIs", "REST-API"),
    ],
)
def test_plural_case_and_separator_differences_still_match(returned, label):
    assert normalise(returned) == normalise(label)


@pytest.mark.parametrize(
    ("first", "second"),
    [("C++", "C"), ("C#", "C"), ("AWS", "AW"), ("CSS", "CS"), ("Next.js", "NextJS")],
)
def test_names_that_differ_in_punctuation_or_length_stay_different(first, second):
    assert normalise(first) != normalise(second)


def test_extras_lower_precision_but_acceptable_items_do_not():
    match = match_list(["Python", "Testing", "teamwork"], [["Python"]], acceptable=["Testing"])
    assert match.recall == 1.0
    assert match.precision == pytest.approx(2 / 3)
    assert match.extras == ["teamwork"]


def test_returning_nothing_scores_zero():
    match = match_list([], [["Python"]])
    assert (match.recall, match.precision, match.f1) == (0.0, 0.0, 0.0)


# --- Running a case ---


def _context(judge=None) -> TaskContext:
    return TaskContext(ModelSpec("openai/gpt-oss-20b"), UsageRecorder(), judge)


def test_a_case_whose_llm_call_failed_scores_zero_even_if_the_agent_fell_back():
    ctx = _context()

    def run():
        ctx.recorder.on_llm_error(RuntimeError("401"), run_id=uuid4())
        return Outcome(1.0, True)  # what the agent's fallback would have produced

    result = ctx.run_case("task", "case", run)
    assert (result.score, result.passed) == (0.0, False)
    assert "fell back" in result.error


def test_a_case_that_raises_scores_zero_and_keeps_the_error():
    def run():
        raise ValueError("bad tool call")

    result = _context().run_case("task", "case", run)
    assert (result.score, result.passed) == (0.0, False)
    assert result.error == "ValueError: bad tool call"


def test_tokens_and_cost_are_charged_to_the_case_that_used_them():
    ctx = _context()
    ctx.recorder.total.input_tokens = 500  # used by an earlier case

    def run():
        ctx.recorder.total.input_tokens += 1_000_000
        ctx.recorder.total.output_tokens += 1_000_000
        return Outcome(1.0, True)

    result = ctx.run_case("task", "case", run)
    assert result.usage.input_tokens == 1_000_000
    assert result.cost == pytest.approx(0.075 + 0.30)


def test_the_cost_of_a_model_without_a_known_price_is_unknown():
    assert ModelSpec("someone/new-model").cost(1000, 1000) is None


def test_a_model_spec_carries_its_reasoning_effort():
    spec = ModelSpec.parse("openai/gpt-oss-120b@low")
    assert (spec.name, spec.reasoning_effort, spec.label) == (
        "openai/gpt-oss-120b",
        "low",
        "openai/gpt-oss-120b@low",
    )


# --- Judging ---

REQUEST = JudgeRequest("task", "source", "output", ["first", "second"])


def _verdict(*scores: int) -> Verdict:
    return Verdict(
        scores=[CriterionScore(criterion=f"c{i}", score=s, reason="") for i, s in enumerate(scores)]
    )


class FakeJudge:
    models = ("fake-judge",)

    def __init__(self, verdict=None, error=None):
        self.verdict, self.error, self.requests = verdict, error, []

    def score(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.verdict


def test_judge_scores_are_scaled_from_one_to_five_onto_zero_to_one():
    assert _verdict(1, 1).normalised == 0.0
    assert _verdict(5, 5).normalised == 1.0
    assert _verdict(3, 4).normalised == pytest.approx(0.625)


def test_a_judged_case_averages_the_judge_with_the_objective_checks():
    outcome = apply_verdict({"objective": {"score": 0.5, "passed": False}}, _verdict(5, 5))
    assert outcome.score == pytest.approx(0.75)
    assert not outcome.passed  # an objective check failed


def test_a_judged_case_without_objective_checks_passes_on_the_judge_alone():
    assert apply_verdict({"objective": None}, _verdict(4, 4)).passed
    assert not apply_verdict({"objective": None}, _verdict(3, 4)).passed


def test_one_criterion_below_three_fails_a_case_with_a_good_average():
    outcome = apply_verdict({"objective": None}, _verdict(5, 5, 2))
    assert outcome.score == pytest.approx(0.75)  # the mean alone would pass
    assert not outcome.passed


def test_a_criterion_at_exactly_three_does_not_fail_the_case():
    assert apply_verdict({"objective": None}, _verdict(5, 5, 3)).passed  # mean 0.83


def test_a_low_criterion_fails_a_case_whose_objective_checks_passed():
    details = {"objective": {"score": 1.0, "passed": True}}
    assert not apply_verdict(details, _verdict(5, 5, 2)).passed


def _judged(scores: list[int], passed: bool, run: int = 1, objective=None) -> CaseResult:
    saved = {f"c{i}": [s, "why"] for i, s in enumerate(scores)}
    details = {"objective": objective, "judge": saved, "judge_score": 0.0}
    return CaseResult("t", "a", "m", 0.5, passed, Usage(), 0.0, details, run=run)


def test_rescore_fails_a_case_the_old_average_rule_passed():
    case = _judged([5, 5, 2], passed=True)
    assert rescore([case]) == 1
    assert not case.passed
    assert case.score == pytest.approx(0.75)


def test_rescore_leaves_a_case_that_still_passes_unchanged():
    case = _judged([5, 4, 3], passed=True)
    assert rescore([case]) == 0
    assert case.passed


def test_rescore_keeps_the_objective_checks_in_the_score_and_the_pass():
    case = _judged([5, 5, 5], passed=False, objective={"score": 0.5, "passed": False})
    assert rescore([case]) == 0
    assert case.score == pytest.approx(0.75)
    assert not case.passed


def test_rescore_keeps_the_judge_reasons():
    case = _judged([5, 2], passed=True)
    rescore([case])
    assert case.details["judge"] == {"c0": [5, "why"], "c1": [2, "why"]}


def test_rescore_skips_cases_without_a_saved_verdict():
    unjudged = CaseResult("t", "b", "m", None, False, Usage(), 0.0, {"judge": {}}, "not judged")
    objective_only = CaseResult("t", "c", "m", 1.0, True, Usage(), 0.0, {"recall": 1.0})
    assert rescore([unjudged, objective_only]) == 0
    assert (unjudged.score, unjudged.passed) == (None, False)
    assert (objective_only.score, objective_only.passed) == (1.0, True)


def test_rescore_keeps_a_case_whose_model_calls_failed_at_zero():
    details = apply_verdict({"objective": None}, _verdict(5, 5, 5)).details
    error = "1 LLM call(s) failed; the agent fell back"
    case = CaseResult("t", "a", "m", 0.0, False, Usage(failed_calls=1), 0.0, details, error)
    assert rescore([case]) == 0
    assert (case.score, case.passed) == (0.0, False)


def test_rescore_leaves_a_rate_limited_case_unscored_even_with_a_saved_verdict():
    details = apply_verdict({"objective": None}, _verdict(5, 5)).details
    case = CaseResult("t", "a", "m", None, False, Usage(), 0.0, details, f"{RATE_LIMITED}: x")
    assert rescore([case]) == 0
    assert (case.score, case.passed) == (None, False)


def test_rescore_skips_an_unscored_case_even_with_a_saved_verdict():
    details = apply_verdict({"objective": None}, _verdict(5, 5)).details
    case = CaseResult("t", "a", "m", None, False, Usage(), 0.0, details)
    assert rescore([case]) == 0
    assert (case.score, case.passed) == (None, False)


def test_rescore_counts_a_case_that_now_passes():
    case = _judged([5, 5, 3], passed=False)
    assert rescore([case]) == 1
    assert case.passed


def test_rescore_counts_only_the_cases_that_changed():
    flips, stays = _judged([5, 5, 2], passed=True), _judged([5, 5, 5], passed=True)
    skipped = CaseResult("t", "b", "m", None, False, Usage(), 0.0, {}, "not judged")
    assert rescore([flips, stays, skipped]) == 1


def test_rescore_recomputes_the_judge_score_and_keeps_other_details():
    case = _judged([5, 4, 3], passed=True)
    case.details["judge_request"] = {"task": "x"}
    rescore([case])
    assert case.score == pytest.approx(0.75)
    assert case.details["judge_score"] == pytest.approx(0.75)
    assert case.details["judge_request"] == {"task": "x"}


def test_the_rescore_command_saves_the_new_result_and_keeps_the_metadata(tmp_path, capsys):
    from evals.__main__ import _rescore

    path = tmp_path / "results.json"
    report.save([_judged([5, 5, 2], passed=True)], {"commit": "abc"}, path)
    assert _rescore(path) == 0
    results, metadata = report.load(path)
    assert (results[0].passed, results[0].score) == (False, pytest.approx(0.75))
    assert metadata == {"commit": "abc"}
    assert "1 cases changed" in capsys.readouterr().out


def test_the_rescore_option_runs_without_a_judge_key(tmp_path, monkeypatch):
    import evals.__main__ as cli

    path = tmp_path / "results.json"
    report.save([_judged([5, 5, 2], passed=True)], {}, path)
    monkeypatch.setattr(cli, "_isolate_data_dir", lambda: None)
    monkeypatch.setattr("sys.argv", ["evals", "--rescore", str(path)])
    assert cli.main() == 0
    assert not report.load(path)[0][0].passed


def test_rescoring_twice_changes_nothing_the_second_time():
    case = _judged([5, 5, 2], passed=True)
    rescore([case])
    assert rescore([case]) == 0
    assert not case.passed


def test_a_saved_score_outside_one_to_five_stops_the_rescore_before_saving(tmp_path):
    from evals.__main__ import _rescore

    path = tmp_path / "results.json"
    report.save([_judged([5, 7], passed=True)], {}, path)
    before = path.read_text()
    with pytest.raises(ValueError):
        _rescore(path)
    assert path.read_text() == before


def test_a_case_is_left_unscored_when_the_judge_is_down():
    ctx = _context(FakeJudge(error=JudgeUnavailableError("429")))
    result = ctx.run_case("task", "case", lambda: ctx.judged_outcome(REQUEST, {}))
    assert result.score is None
    assert result.details["judge_request"]["output"] == "output"
    assert "not judged yet" in result.error


def test_rejudging_finishes_only_the_unscored_cases():
    judged = CaseResult("t", "a", "m", 1.0, True, Usage(), 0.0, {"judge_request": {}})
    pending_details = {
        "judge_request": {"task": "t", "source": "s", "output": "o", "criteria": ["x"]}
    }
    pending = CaseResult("t", "b", "m", None, False, Usage(), 0.0, pending_details, "not judged")
    judge = FakeJudge(_verdict(5))

    assert rejudge([judged, pending], judge) == 1
    assert (pending.score, pending.passed, pending.error) == (1.0, True, None)
    assert len(judge.requests) == 1


class FakeGeminiClient:
    def __init__(self, verdict: Verdict):
        self.models = SimpleNamespace(
            generate_content=lambda **_: SimpleNamespace(parsed=verdict, text="")
        )


def test_a_verdict_that_skips_criteria_is_rejected():
    judge = Judge(FakeGeminiClient(_verdict(5)), ("gemini-test",))
    with pytest.raises(JudgeUnavailableError, match="expected 2"):
        judge.score(task="t", source="s", output="o", criteria=["first", "second"])


def test_calibration_marks_a_pair_failed_when_the_judge_is_down():
    results = calibrate(FakeJudge(error=JudgeUnavailableError("429")))
    assert results
    assert all(r.good_score is None and not r.passed for r in results)


@pytest.mark.parametrize(
    ("good", "bad", "passed"),
    [(0.75, 0.49, True), (0.74, 0.0, False), (1.0, 0.5, False)],
)
def test_a_calibration_pair_needs_good_high_and_bad_low(good, bad, passed):
    assert PairResult("p", "t", good, bad).passed is passed


# --- Task-specific checks ---


class FakeLlm:
    def __init__(self, reply: str):
        self.reply = reply

    def invoke(self, *_args, **_kwargs):
        return SimpleNamespace(content=self.reply)


def test_an_empty_skill_list_fails_instead_of_using_the_keyword_fallback(monkeypatch):
    monkeypatch.setattr(skills, "get_llm", lambda: FakeLlm(""))
    case = {"source": "resume", "text": "Python and Docker"}
    with pytest.raises(skills_task.FallbackUsedError):
        skills_task._extract(case)


def test_a_grading_reply_without_a_grade_line_fails_instead_of_defaulting(monkeypatch):
    monkeypatch.setattr(interview_coach, "get_llm", lambda: FakeLlm("SKILL: sql\nFEEDBACK: fine"))
    results = interview_task.answer_grading(_context())
    assert all(r.error.startswith("UnreadableGradeError") for r in results)


class DeadLlm:
    def invoke(self, *_args, **_kwargs):
        raise RuntimeError("404 model_not_found")


@pytest.mark.parametrize("llm", [DeadLlm(), FakeLlm(""), FakeLlm("FOLLOW_UP or NEW_QUESTION")])
def test_no_decision_from_the_model_passes_no_follow_up_case(monkeypatch, llm):
    # The router itself catches errors and reads anything without FOLLOW_UP as NEW_QUESTION.
    monkeypatch.setattr(interview_coach, "get_llm", lambda: llm)
    results = interview_task.follow_up_decision(_context())
    assert results
    assert not any(r.passed for r in results)


def _decision_results(monkeypatch, reply: str):
    """(results, whether the model decides it) for each follow-up decision case."""
    monkeypatch.setattr(interview_coach, "get_llm", lambda: FakeLlm(reply))
    results = interview_task.follow_up_decision(_context())
    cases = load_data("interview")["follow_up_decision"]
    return list(zip(results, map(interview_task.decided_by_the_model, cases), strict=True))


def test_a_clear_decision_is_scored_against_the_label(monkeypatch):
    results = _decision_results(monkeypatch, "NEW_QUESTION")
    cases = load_data("interview")["follow_up_decision"]
    scored = [result.passed for result, by_model in results if by_model]
    labels = [
        c["expected"] == "NEW_QUESTION" for c in cases if interview_task.decided_by_the_model(c)
    ]
    assert scored == labels and len(scored) == 12


def test_cases_the_app_decides_by_itself_are_not_scored(monkeypatch):
    by_app = [
        result for result, by_model in _decision_results(monkeypatch, "FOLLOW_UP") if not by_model
    ]
    assert [(r.score, r.passed, r.error) for r in by_app] == [(None, False, DECIDED_BY_APP)] * 4


def test_cases_the_app_decides_agree_with_their_labels(monkeypatch):
    by_app = [r for r, by_model in _decision_results(monkeypatch, "FOLLOW_UP") if not by_model]
    assert [(r.details["got"], r.details["expected"]) for r in by_app] == [
        ("NEW_QUESTION", "NEW_QUESTION")
    ] * 4


def test_the_model_decides_as_many_follow_ups_as_new_questions():
    cases = load_data("interview")["follow_up_decision"]
    labels = [c["expected"] for c in cases if interview_task.decided_by_the_model(c)]
    assert labels.count("FOLLOW_UP") == labels.count("NEW_QUESTION") == 6


def test_the_grade_alone_does_not_predict_every_decision():
    cases = load_data("interview")["follow_up_decision"]
    by_model = [c for c in cases if interview_task.decided_by_the_model(c)]
    assert any((c["grade"] == "moderate") != (c["expected"] == "FOLLOW_UP") for c in by_model)


def test_a_failed_decision_call_is_not_mistaken_for_one_the_app_made(monkeypatch):
    monkeypatch.setattr(interview_coach, "get_llm", lambda: DeadLlm())
    results = interview_task.follow_up_decision(_context())
    cases = load_data("interview")["follow_up_decision"]
    by_model = [
        r for r, c in zip(results, cases, strict=True) if interview_task.decided_by_the_model(c)
    ]
    assert [(r.score, r.error == DECIDED_BY_APP) for r in by_model] == [(0.0, False)] * 12


@pytest.mark.parametrize("reply", ["GRADE: badly", "GRADE: excellently done", "GRADE: goodish"])
def test_a_grade_the_app_cannot_read_as_a_word_fails(reply):
    with pytest.raises(interview_task.UnreadableGradeError):
        interview_task._strict_parse_grade(reply)


def test_a_readable_grade_passes_the_strict_parser():
    assert interview_task._strict_parse_grade("**GRADE:** Good\nSKILL: sql") == "good"


@pytest.mark.parametrize(
    ("expected", "got", "score"),
    [("good", "good", 1.0), ("good", "excellent", 0.5), ("bad", "good", 0.0)],
)
def test_grades_one_step_off_get_half_marks(expected, got, score):
    assert interview_task.score_grade(expected, got).score == score


@pytest.mark.parametrize(
    ("question", "passed"),
    [
        ("How would you avoid collisions?", True),
        ("Walk me through the eviction path.", True),
        ("Great answer! How would you shard it?", False),
        ("What about A? And B?", False),
    ],
)
def test_follow_up_questions_are_one_question_without_praise(question, passed):
    assert interview_task._follow_up_checks(question).passed is passed


def test_tags_must_cover_required_groups_and_avoid_forbidden_ones():
    case = {"required": [["Engineering"]], "forbidden": ["Startup"]}
    assert skills_task.score_tags(case, ["Engineering"]).passed
    assert not skills_task.score_tags(case, ["Engineering", "Startup"]).passed
    assert not skills_task.score_tags(case, ["Technology"]).passed
    assert not skills_task.score_tags({"required": [], "forbidden": []}, []).passed


@pytest.mark.parametrize(
    ("text", "invented"),
    [
        ("cut latency by 40%", []),
        ("cut latency by 40.0%", []),
        ("grew 25 percent", ["25"]),
        ("3x faster and 2× cheaper", ["2", "3"]),
        (r"a $3\times$ speed-up", ["3"]),
    ],
)
def test_metrics_not_in_the_profile_count_as_invented(text, invented):
    assert resume_task.invented_metrics(text, ["40"]) == invented


def test_escaped_latex_is_read_as_plain_text():
    source = r"\documentclass{article}\begin{document}R\&D, 40\%, a\_b\end{document}"
    assert "R&D, 40%, a_b" in resume_task._plain_text(source)


def test_compiling_without_docker_is_not_checked(monkeypatch):
    monkeypatch.setattr(latex, "is_available", lambda: False)

    def no_docker(*_args, **_kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(subprocess, "run", no_docker)
    assert resume_task._compile("source", "image") == "not checked"


def test_a_compile_that_times_out_is_a_failure(monkeypatch):
    monkeypatch.setattr(latex, "is_available", lambda: False)

    def too_slow(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("docker", 1)

    monkeypatch.setattr(subprocess, "run", too_slow)
    assert resume_task._compile("source", "image").startswith("failed")


def test_course_urls_match_with_or_without_a_trailing_slash():
    assert curriculum_task._same_page("https://x.dev/a/") == curriculum_task._same_page(
        "https://x.dev/a"
    )


# --- Floors and reports ---


def test_no_constant_answer_passes_more_than_half_the_cases():
    for floor in baselines.floors():
        assert floor["passed"] <= floor["cases"] / 2, floor


def test_the_summary_shows_the_spread_across_repeats():
    results = [
        CaseResult("t", "a", "m", score, True, Usage(), 0.0, run=run)
        for run, score in [(1, 1.0), (2, 0.5)]
    ]
    assert "0.75 [0.50-1.00]" in report.summary_table(results)


def test_unscored_cases_are_left_out_of_the_mean():
    results = [
        CaseResult("t", "a", "m", 1.0, True, Usage(), 0.0),
        CaseResult("t", "b", "m", None, False, Usage(), 0.0),
    ]
    assert "| 1.00 (1/1)" in report.summary_table(results)


def test_results_round_trip_through_a_file(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_git", lambda *_args: "abc1234")
    results = [CaseResult("t", "a", "m", 0.5, False, Usage(calls=2), 0.01, {"x": 1}, "e", 2)]
    metadata = report.run_metadata(["m"], ["t"], 2, ["gemini-test"])
    report.save(results, metadata, tmp_path / "r.json")

    loaded, loaded_metadata = report.load(tmp_path / "r.json")
    assert loaded == results
    assert loaded_metadata["judge_models"] == ["gemini-test"]
    assert set(loaded_metadata["data_sha256"]) >= {"skills.json", "tagging.json"}


# --- Scoring inside the tasks (fake models) ---


def _first(results: list[CaseResult], case_id: str) -> CaseResult:
    return next(r for r in results if r.case_id == case_id)


def test_skill_extraction_allows_one_extra_skill_but_not_two(monkeypatch):
    case = load_data("skills")["cases"][0]
    expected = ", ".join(group[0] for group in case["expected"])

    monkeypatch.setattr(skills, "get_llm", lambda: FakeLlm(f"{expected}, teamwork"))
    assert _first(skills_task.skill_extraction(_context()), case["id"]).passed

    monkeypatch.setattr(skills, "get_llm", lambda: FakeLlm(f"{expected}, teamwork, Rust"))
    assert not _first(skills_task.skill_extraction(_context()), case["id"]).passed


def test_an_empty_github_skill_list_fails(monkeypatch):
    monkeypatch.setattr(skills, "get_llm", lambda: FakeLlm(""))
    case = next(c for c in load_data("skills")["cases"] if c["source"] == "github")
    with pytest.raises(skills_task.FallbackUsedError):
        skills_task._extract(case)


def _courses(*urls: str) -> curriculum.NewCourseList:
    return curriculum.NewCourseList(
        courses=[
            curriculum.Course(id=str(i), title="t", difficulty=2, url=url, description="d")
            for i, url in enumerate(urls)
        ]
    )


def test_course_parsing_fails_on_a_url_that_was_not_in_the_search_results(monkeypatch, fake_llm):
    case = load_data("curriculum")["course_parsing"][0]
    found = _courses(*case["required"], "https://invented.example/course")
    monkeypatch.setattr(curriculum, "get_llm", lambda: fake_llm(structured=found))

    result = _first(curriculum_task.course_parsing(_context()), case["id"])
    assert not result.passed
    assert result.details["invented"] == ["https://invented.example/course"]


def test_course_parsing_passes_when_only_the_real_courses_are_kept(monkeypatch, fake_llm):
    case = load_data("curriculum")["course_parsing"][0]
    found = _courses(*(url + "/" for url in case["required"]))  # trailing slash: same page
    monkeypatch.setattr(curriculum, "get_llm", lambda: fake_llm(structured=found))
    assert _first(curriculum_task.course_parsing(_context()), case["id"]).passed


def _planner_output(skills_to_find: list[str], level: str) -> curriculum.PlannerOutput:
    profile = curriculum.UserProfile(interests=[], skill_level=level, past_courses=[0])
    return curriculum.PlannerOutput(user_profile=profile, skills_to_find=skills_to_find)


def test_planner_extras_fail_and_a_wrong_level_halves_the_score(monkeypatch, fake_llm):
    case = load_data("curriculum")["planner"][0]
    every_gap = [group[0] for group in case["expected"]]

    output = _planner_output([*every_gap, "Cobol"], "Intermediate")
    monkeypatch.setattr(curriculum, "get_llm", lambda: fake_llm(structured=output))
    with_extra = _first(curriculum_task.planner(_context()), case["id"])
    assert (with_extra.score, with_extra.passed) == (1.0, False)

    output = _planner_output(every_gap, "Expert")
    monkeypatch.setattr(curriculum, "get_llm", lambda: fake_llm(structured=output))
    wrong_level = _first(curriculum_task.planner(_context()), case["id"])
    assert (wrong_level.score, wrong_level.passed) == (0.5, False)


def test_a_critic_that_approves_everything_passes_only_the_approvable_paths(monkeypatch, fake_llm):
    approve_all = curriculum.Critique(is_approved=True, revisions_needed="")
    monkeypatch.setattr(curriculum, "get_llm", lambda: fake_llm(structured=approve_all))
    results = curriculum_task.critic(_context())
    expected = [c["expected_approved"] for c in load_data("curriculum")["critic"]]
    assert [r.passed for r in results] == expected


def _resume_result(body: str):
    from app.agents.resume_strategist import ContactInfo, ResumeDraft, ResumeResult

    draft = ResumeDraft(contact=ContactInfo(), summary="s")
    latex_source = f"\\documentclass{{article}}\\begin{{document}}{body}\\end{{document}}"
    return ResumeResult(draft=draft, latex=latex_source)


def _faithful_body(case: dict) -> str:
    return " ".join(fact.replace("%", "\\%") for fact in case["must_keep"])


def test_a_faithful_resume_passes_when_the_compile_cannot_be_checked(monkeypatch):
    case = load_data("resume")["generation"][0]
    body = _faithful_body(case)
    monkeypatch.setattr(resume_task, "generate_resume", lambda *_: _resume_result(body))
    monkeypatch.setattr(resume_task, "_compile", lambda *_: "not checked")
    results = resume_task.resume_generation(_context(FakeJudge(_verdict(5, 5, 5))))
    result = _first(results, case["id"])
    assert result.passed
    assert result.details["objective"]["score"] == 1.0  # "not checked" is not counted


def test_template_text_left_in_the_resume_fails_it(monkeypatch):
    case = load_data("resume")["generation"][0]
    body = _faithful_body(case) + " " + case["template_markers"][0]
    monkeypatch.setattr(resume_task, "generate_resume", lambda *_: _resume_result(body))
    monkeypatch.setattr(resume_task, "_compile", lambda *_: "ok")
    results = resume_task.resume_generation(_context(FakeJudge(_verdict(5, 5, 5))))
    result = _first(results, case["id"])
    assert not result.passed
    assert result.details["leaked_template_text"] == [case["template_markers"][0]]


def test_the_judge_reads_the_finished_resume_not_the_draft(monkeypatch):
    monkeypatch.setattr(resume_task, "generate_resume", lambda *_: _resume_result("FINAL TEXT"))
    monkeypatch.setattr(resume_task, "_compile", lambda *_: "ok")
    judge = FakeJudge(_verdict(5, 5, 5))
    resume_task.resume_generation(_context(judge))
    assert all("FINAL TEXT" in request["output"] for request in judge.requests)


def test_a_review_with_too_few_points_fails_even_if_the_judge_likes_it(monkeypatch):
    from app.agents.resume_reviewer import ResumeReview

    three = ["a", "b", "c"]
    short = ResumeReview(summary="s", strengths=["a", "b"], weaknesses=three, suggestions=three)
    monkeypatch.setattr(resume_task, "review_resume", lambda *_: short)
    results = resume_task.resume_review(_context(FakeJudge(_verdict(5, 5, 5, 5))))
    assert results
    assert not any(r.passed for r in results)


# --- Swapping and metering models ---


def test_use_model_replaces_every_agent_model():
    import importlib

    from evals.models import GENERAL_LLM_SITES, RESUME_LLM_SITE, use_model

    spec, recorder = ModelSpec("openai/gpt-oss-20b", "low"), UsageRecorder()
    with use_model(spec, recorder):
        for site in (*GENERAL_LLM_SITES, RESUME_LLM_SITE):
            module, name = site.rsplit(".", 1)
            llm = getattr(importlib.import_module(module), name)()
            assert (llm.model_name, llm.reasoning_effort) == ("openai/gpt-oss-20b", "low")
            assert recorder in llm.callbacks


def test_the_recorder_counts_calls_tokens_and_reasoning_from_a_real_chat_model():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    usage = {
        "input_tokens": 3,
        "output_tokens": 5,
        "total_tokens": 8,
        "output_token_details": {"reasoning": 2},
    }
    recorder = UsageRecorder()
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok", usage_metadata=usage)]))
    model.invoke("hi", config={"callbacks": [recorder]})
    total = recorder.total
    assert (total.calls, total.input_tokens, total.output_tokens) == (1, 3, 5)
    assert total.reasoning_tokens == 2
    assert total.failed_calls == 0


# --- Judging, continued ---


def test_a_low_judge_score_fails_even_if_the_objective_checks_pass():
    assert not apply_verdict({"objective": {"score": 1.0, "passed": True}}, _verdict(1, 1)).passed


def test_with_judging_deferred_a_case_is_left_unscored():
    ctx = _context(None)
    result = ctx.run_case("t", "c", lambda: ctx.judged_outcome(REQUEST, {}))
    assert result.score is None
    assert result.details["judge_request"]["criteria"] == ["first", "second"]


def test_rejudging_keeps_cases_the_judge_still_cannot_score():
    details = {"judge_request": {}}
    pending = CaseResult("t", "b", "m", None, False, Usage(), 0.0, details, "not judged")
    assert rejudge([pending], FakeJudge(error=JudgeUnavailableError("429"))) == 0
    assert pending.score is None


def test_repeat_runs_are_never_judged():
    ctx = TaskContext(ModelSpec("m"), UsageRecorder(), None, run=2)
    repeat = ctx.run_case("t", "c", lambda: ctx.judged_outcome(REQUEST, {}))
    assert "only the first run" in repeat.error

    judge = FakeJudge(_verdict(5, 5))
    assert rejudge([repeat], judge) == 0
    assert judge.requests == []


class ScriptedGeminiClient:
    """Answers generate_content with each scripted reply in turn."""

    def __init__(self, *replies):
        replies = iter(replies)
        self.models = SimpleNamespace(generate_content=lambda **_: next(replies))


def test_the_judge_falls_back_to_its_next_model():
    unparsed = SimpleNamespace(parsed=None, text="not json")
    client = ScriptedGeminiClient(unparsed, SimpleNamespace(parsed=_verdict(5, 4), text=""))
    verdict = Judge(client, ("first-model", "second-model")).score(
        task="t", source="s", output="o", criteria=["a", "b"]
    )
    assert [s.score for s in verdict.scores] == [5, 4]


def test_the_judge_is_unavailable_when_every_model_fails():
    client = ScriptedGeminiClient(SimpleNamespace(parsed=None, text=""))
    with pytest.raises(JudgeUnavailableError):
        Judge(client, ("only-model",)).score(task="t", source="s", output="o", criteria=["a"])


# --- Compiling resumes ---


def _no_local_tex(monkeypatch):
    monkeypatch.setattr(latex, "is_available", lambda: False)


def test_without_tex_or_an_image_the_compile_is_not_checked(monkeypatch):
    _no_local_tex(monkeypatch)
    assert resume_task._compile("source", None) == "not checked"


def test_a_latex_error_in_the_container_is_a_failure(monkeypatch):
    _no_local_tex(monkeypatch)

    def latex_error(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "docker", stderr="! Undefined control sequence")

    monkeypatch.setattr(subprocess, "run", latex_error)
    assert resume_task._compile("source", "image") == "failed: ! Undefined control sequence"


@pytest.mark.parametrize(
    ("returncode", "stderr"), [(125, "Unable to find image"), (1, "LatexUnavailableError: no TeX")]
)
def test_a_missing_image_or_tex_is_not_charged_to_the_model(monkeypatch, returncode, stderr):
    _no_local_tex(monkeypatch)

    def docker_problem(*_args, **_kwargs):
        raise subprocess.CalledProcessError(returncode, "docker", stderr=stderr)

    monkeypatch.setattr(subprocess, "run", docker_problem)
    assert resume_task._compile("source", "image") == "not checked"


# --- The labelled data hangs together ---


def test_every_case_reference_in_the_labelled_data_resolves():
    interview_data = load_data("interview")
    grading_ids = {case["id"] for case in interview_data["grading"]}
    assert set(interview_data["follow_up_questions"]) <= grading_ids
    for case in interview_data["follow_up_decision"]:
        assert case["id"] in grading_ids or {"question", "answer"} <= case.keys(), case["id"]

    for case in load_data("resume")["review"]:
        assert resume_task.resume_text(case["resume"])
    for case in load_data("resume")["generation"]:
        if case["template"]:
            assert (TEMPLATES_DIR / f"{case['template']}.tex").is_file()
    for case in load_data("tagging")["cases"]:
        assert "latex" in case or (TEMPLATES_DIR / f"{case['template']}.tex").is_file()


def test_every_tag_in_the_tagging_labels_is_one_the_app_offers():
    for case in load_data("tagging")["cases"]:
        named = {tag for group in case["required"] for tag in group} | set(case["forbidden"])
        assert named <= set(PREDEFINED_TAGS), case["template"]


def test_changing_loaded_data_does_not_change_the_next_load():
    load_data("skills")["cases"].clear()
    assert load_data("skills")["cases"]


def test_old_results_files_without_metadata_still_load(tmp_path):
    from dataclasses import asdict

    path = tmp_path / "old.json"
    path.write_text(json.dumps([asdict(CaseResult("t", "a", "m", 1.0, True, Usage(), None))]))
    results, metadata = report.load(path)
    assert (len(results), metadata) == (1, {})


# --- Each check fails a case on its own ---


def test_an_echoed_grade_template_fails_the_strict_parser():
    with pytest.raises(interview_task.UnreadableGradeError):
        interview_task._strict_parse_grade("GRADE: [bad/moderate/good/excellent]")


def test_no_follow_up_decision_case_has_a_grade_the_router_skips():
    # The router answers good and excellent grades itself, without asking the model.
    for case in load_data("interview")["follow_up_decision"]:
        assert case["grade"] in {"bad", "moderate"}, case["id"]


def test_a_judge_network_error_leaves_the_case_unscored():
    import httpx

    def offline(**_):
        raise httpx.ConnectError("no network")

    judge = Judge(SimpleNamespace(models=SimpleNamespace(generate_content=offline)), ("g",))
    ctx = _context(judge)
    result = ctx.run_case("t", "c", lambda: ctx.judged_outcome(REQUEST, {}))
    assert result.score is None


def test_rejudging_keeps_a_failed_objective_check():
    details = {
        "judge_request": {"task": "t", "source": "s", "output": "o", "criteria": ["x"]},
        "objective": {"score": 0.0, "passed": False, "details": {}},
    }
    result = CaseResult("t", "b", "m", None, False, Usage(), 0.0, details)
    rejudge([result], FakeJudge(_verdict(5)))
    assert (result.score, result.passed) == (0.5, False)


def test_calibration_scores_the_good_and_the_bad_output_separately():
    pairs = {pair["good"]: True for pair in load_data("judge_calibration")["pairs"]}

    class TellsThemApart:
        models = ("fake",)

        def score(self, output, **_):
            return _verdict(5, 5) if output in pairs else _verdict(1, 1)

    results = calibrate(TellsThemApart())
    assert all((r.good_score, r.bad_score, r.passed) == (1.0, 0.0, True) for r in results)


def test_a_verdict_with_extra_criteria_is_rejected():
    judge = Judge(FakeGeminiClient(_verdict(5, 5, 5)), ("gemini-test",))
    with pytest.raises(JudgeUnavailableError, match="expected 2"):
        judge.score(task="t", source="s", output="o", criteria=["first", "second"])


def test_a_verdict_that_names_a_criterion_twice_is_rejected():
    repeated = Verdict(
        scores=[
            CriterionScore(criterion="Faithful", score=2, reason="invented a team"),
            CriterionScore(criterion="faithful ", score=5, reason="fine"),
        ]
    )
    judge = Judge(FakeGeminiClient(repeated), ("gemini-test",))
    with pytest.raises(JudgeUnavailableError, match="twice"):
        judge.score(task="t", source="s", output="o", criteria=["first", "second"])


def test_input_and_output_tokens_have_their_own_prices():
    spec = ModelSpec("openai/gpt-oss-20b")
    assert spec.cost(1_000_000, 0) == pytest.approx(0.075)
    assert spec.cost(0, 1_000_000) == pytest.approx(0.30)


def test_the_resume_model_keeps_the_apps_output_budget_and_temperature():
    from app.agents import resume_strategist
    from app.core.resources import RESUME_MAX_TOKENS
    from evals.models import use_model

    with use_model(ModelSpec("openai/gpt-oss-120b"), UsageRecorder()):
        llm = resume_strategist.get_resume_llm()
    assert (llm.max_tokens, llm.temperature) == (RESUME_MAX_TOKENS, 0.1)


def test_repeats_must_be_at_least_one():
    import argparse

    from evals.__main__ import _positive_int

    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("0")
    assert _positive_int("3") == 3


def test_the_summary_shows_errors_and_cost():
    failed = CaseResult("t", "a", "m", 0.0, False, Usage(), 0.002, error="boom")
    assert "(0/1, 1 err) $2.00" in report.summary_table([failed])


def test_a_long_error_is_cut_short():
    def run():
        raise ValueError("x" * 2000)

    assert len(_context().run_case("t", "c", run).error) == 500


def test_a_local_latex_error_is_a_failure(monkeypatch):
    monkeypatch.setattr(latex, "is_available", lambda: True)

    def broken(_source):
        raise latex.LatexCompileError("exit 1", "! Missing $ inserted")

    monkeypatch.setattr(latex, "compile_pdf", broken)
    assert resume_task._compile("source", None) == "failed: ! Missing $ inserted"


def test_plain_text_leaves_out_the_preamble_and_comments():
    source = "\\usepackage{secret}\\begin{document}\n% Your Name\nShown \\textbf{text}\n"
    text = resume_task._plain_text(source)
    assert "secret" not in text
    assert "Your Name" not in text
    assert "Shown" in text


@pytest.mark.parametrize(
    "text", [r"grew 75~\%", r"grew \textbf{75}\%", r"grew 75\,\%", r"grew \textbf{\emph{75}}x"]
)
def test_metrics_written_with_latex_spacing_or_formatting_are_caught(text):
    assert resume_task.invented_metrics(resume_task._plain_text(text), ["40"]) == ["75"]


def test_thousands_separators_are_read_as_one_number():
    assert resume_task.invented_metrics("up 1,000%", []) == ["1000"]


def _generation_result(monkeypatch, body: str, compiled: str = "ok") -> CaseResult:
    case = load_data("resume")["generation"][0]
    monkeypatch.setattr(resume_task, "generate_resume", lambda *_: _resume_result(body))
    monkeypatch.setattr(resume_task, "_compile", lambda *_: compiled)
    results = resume_task.resume_generation(_context(FakeJudge(_verdict(5, 5, 5))))
    return _first(results, case["id"])


def test_an_invented_metric_fails_the_resume(monkeypatch):
    case = load_data("resume")["generation"][0]
    result = _generation_result(monkeypatch, _faithful_body(case) + " grew 75\\%")
    assert not result.passed
    assert result.details["invented_metrics"] == ["75"]


def test_a_dropped_fact_fails_the_resume(monkeypatch):
    case = load_data("resume")["generation"][0]
    kept = " ".join(fact.replace("%", "\\%") for fact in case["must_keep"][1:])
    result = _generation_result(monkeypatch, kept)
    assert not result.passed
    assert result.details["dropped_facts"] == [case["must_keep"][0]]


def test_a_resume_that_does_not_compile_fails(monkeypatch):
    case = load_data("resume")["generation"][0]
    result = _generation_result(monkeypatch, _faithful_body(case), compiled="failed: x")
    assert not result.passed
    assert result.details["objective"]["score"] < 1.0


def test_the_judge_sees_the_resume_body_without_the_preamble(monkeypatch):
    judge = FakeJudge(_verdict(5, 5, 5))
    monkeypatch.setattr(resume_task, "generate_resume", lambda *_: _resume_result("BODY"))
    monkeypatch.setattr(resume_task, "_compile", lambda *_: "ok")
    resume_task.resume_generation(_context(judge))
    assert judge.requests
    assert all(r["output"].startswith("BODY") for r in judge.requests)


def test_a_review_with_too_many_points_fails(monkeypatch):
    from app.agents.resume_reviewer import ResumeReview

    three, six = ["a", "b", "c"], list("abcdef")
    long = ResumeReview(summary="s", strengths=six, weaknesses=three, suggestions=three)
    monkeypatch.setattr(resume_task, "review_resume", lambda *_: long)
    results = resume_task.resume_review(_context(FakeJudge(_verdict(5, 5, 5, 5))))
    assert results
    assert all(r.error is None and not r.passed for r in results)


def test_course_parsing_fails_on_a_kept_non_course(monkeypatch, fake_llm):
    case = load_data("curriculum")["course_parsing"][1]
    found = _courses(*case["required"], case["forbidden"][0])
    monkeypatch.setattr(curriculum, "get_llm", lambda: fake_llm(structured=found))
    result = _first(curriculum_task.course_parsing(_context()), case["id"])
    assert not result.passed
    assert result.details["non_courses"] == [case["forbidden"][0]]


def test_course_parsing_fails_on_a_missed_course(monkeypatch, fake_llm):
    case = load_data("curriculum")["course_parsing"][1]
    found = _courses(*case["required"][1:])
    monkeypatch.setattr(curriculum, "get_llm", lambda: fake_llm(structured=found))
    result = _first(curriculum_task.course_parsing(_context()), case["id"])
    assert not result.passed
    assert result.details["missed"] == [case["required"][0]]


def test_a_missed_skill_fails_skill_extraction(monkeypatch):
    case = load_data("skills")["cases"][0]
    all_but_one = ", ".join(group[0] for group in case["expected"][1:])
    monkeypatch.setattr(skills, "get_llm", lambda: FakeLlm(all_but_one))
    result = _first(skills_task.skill_extraction(_context()), case["id"])
    assert not result.passed
    assert result.details["missed"] == [case["expected"][0][0]]


def test_a_missed_gap_fails_the_planner(monkeypatch, fake_llm):
    case = load_data("curriculum")["planner"][0]
    output = _planner_output([group[0] for group in case["expected"][1:]], "Intermediate")
    monkeypatch.setattr(curriculum, "get_llm", lambda: fake_llm(structured=output))
    result = _first(curriculum_task.planner(_context()), case["id"])
    assert not result.passed
    assert result.score < 1.0


def _learning_path_state(urls: list[str], skills_named: list[str], approved: bool) -> dict:
    recommendations = [
        curriculum.CourseRecommendation(
            skill_name=skill, course_title="t", course_url=url, description="d"
        )
        for skill, url in zip(skills_named, urls, strict=True)
    ]
    path = curriculum.PersonalizedLearningPath(
        user_summary="s", skill_gaps=skills_named, recommendations=recommendations
    )
    critique = curriculum.Critique(is_approved=approved, revisions_needed="")
    return {"draft_path": path, "structured_critique": critique, "revision_count": 1}


@pytest.mark.parametrize(
    ("change", "failed_check"),
    [("drop a gap", "uncovered_gaps"), ("invent a url", "invented_urls"), ("reject", None)],
)
def test_each_end_to_end_check_can_fail_a_run(monkeypatch, change, failed_check):
    case = load_data("curriculum")["end_to_end"][0]
    catalog = sorted(curriculum_task._catalog_urls())
    gaps, urls = list(case["gaps"]), catalog[: len(case["gaps"])]
    if change == "drop a gap":
        gaps, urls = gaps[1:], urls[1:]
    if change == "invent a url":
        urls[0] = "https://invented.example/course"
    state = _learning_path_state(urls, gaps, approved=change != "reject")
    monkeypatch.setattr(curriculum, "build_learning_path", lambda *_: state)

    result = _first(curriculum_task.end_to_end(_context()), case["id"])
    assert not result.passed
    if failed_check:
        assert result.details[failed_check]
    else:
        assert result.details["critic_approved"] is False


def test_a_complete_end_to_end_run_passes(monkeypatch):
    case = load_data("curriculum")["end_to_end"][0]
    urls = sorted(curriculum_task._catalog_urls())[: len(case["gaps"])]
    state = _learning_path_state(urls, list(case["gaps"]), approved=True)
    monkeypatch.setattr(curriculum, "build_learning_path", lambda *_: state)
    assert _first(curriculum_task.end_to_end(_context()), case["id"]).passed


def test_the_follow_up_fallback_text_is_an_error(monkeypatch):
    monkeypatch.setattr(interview_coach, "get_llm", lambda: DeadLlm())
    results = interview_task.follow_up_question(_context(FakeJudge(_verdict(5, 5, 5))))
    assert results
    assert all(r.error and not r.passed for r in results)


def test_the_summary_fallback_text_is_an_error(monkeypatch):
    monkeypatch.setattr(interview_coach, "get_llm", lambda: DeadLlm())
    results = interview_task.interview_summary(_context(FakeJudge(_verdict(5, 5, 5))))
    assert results
    assert all(r.error and not r.passed for r in results)


def test_the_summary_prompt_carries_the_report_lines_unchanged(monkeypatch, fake_llm):
    llm = fake_llm("A fine summary.")
    monkeypatch.setattr(interview_coach, "get_llm", lambda: llm)
    interview_task.interview_summary(_context(FakeJudge(_verdict(5, 5, 5))))
    for case, prompt in zip(load_data("interview")["summaries"], llm.prompts, strict=True):
        assert case["report"] in str(prompt)


@pytest.mark.parametrize(
    ("question", "score"), [("How would you shard it?", 1.0), ("Great! What? Why?", 0.0)]
)
def test_the_follow_up_objective_score_counts_the_checks_passed(question, score):
    assert interview_task._follow_up_checks(question).score == score


def test_a_clear_follow_up_decision_is_scored_too(monkeypatch):
    results = _decision_results(monkeypatch, "FOLLOW_UP")
    cases = load_data("interview")["follow_up_decision"]
    scored = [result.passed for result, by_model in results if by_model]
    labels = [c["expected"] == "FOLLOW_UP" for c in cases if interview_task.decided_by_the_model(c)]
    assert scored == labels


# --- Rate limits are the account's problem, not the model's ---


def _rate_limit_error():
    import groq
    import httpx

    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
    return groq.RateLimitError("Rate limit reached: tokens per day", response=response, body=None)


def test_a_rate_limited_call_is_counted_apart_from_failures():
    recorder = UsageRecorder()
    recorder.on_llm_error(_rate_limit_error(), run_id=uuid4())
    recorder.on_llm_error(RuntimeError("bad tool call"), run_id=uuid4())
    assert (recorder.total.rate_limited_calls, recorder.total.failed_calls) == (1, 1)


def test_a_quota_error_is_recognised_by_its_http_status_too():
    from evals.models import is_rate_limit

    class QuotaError(Exception):
        status_code = 429

    assert is_rate_limit(QuotaError())
    assert not is_rate_limit(RuntimeError("bad tool call"))


def test_a_case_that_hits_a_rate_limit_is_left_unscored():
    def run():
        raise _rate_limit_error()

    result = _context().run_case("task", "case", run)
    assert result.score is None
    reason = "RateLimitError: Rate limit reached: tokens per day"
    assert result.error == f"rate limited, not scored: {reason}"


def test_a_fallback_caused_by_a_rate_limit_is_left_unscored_even_if_it_passed():
    ctx = _context()

    def run():
        ctx.recorder.on_llm_error(_rate_limit_error(), run_id=uuid4())
        return Outcome(1.0, True, {"reply": "fallback"})  # what the fallback happened to score

    result = ctx.run_case("task", "case", run)
    assert (result.score, result.passed) == (None, False)
    assert result.error == "rate limited, not scored: 1 call(s) refused"
    assert result.details == {"reply": "fallback"}  # kept for reading the results file


def test_a_real_failure_counts_against_the_model_even_with_a_rate_limit_in_the_case():
    ctx = _context()

    def run():
        ctx.recorder.on_llm_error(RuntimeError("bad tool call"), run_id=uuid4())
        ctx.recorder.on_llm_error(_rate_limit_error(), run_id=uuid4())
        return Outcome(1.0, True)

    result = ctx.run_case("task", "case", run)
    assert (result.score, result.passed) == (0.0, False)
    assert "fell back" in result.error


def test_a_refused_fallback_on_a_repeat_run_is_labelled_rate_limited():
    # Repeat runs are unscored anyway; the label decides whether the summary counts the refusal.
    ctx = TaskContext(ModelSpec("m"), UsageRecorder(), None, run=2)

    def run():
        ctx.recorder.on_llm_error(_rate_limit_error(), run_id=uuid4())
        return ctx.judged_outcome(REQUEST, {})

    assert ctx.run_case("t", "c", run).error == "rate limited, not scored: 1 call(s) refused"


def test_rejudging_never_scores_a_rate_limited_case():
    ctx = _context(None)

    def run():
        ctx.recorder.on_llm_error(_rate_limit_error(), run_id=uuid4())
        return ctx.judged_outcome(REQUEST, {})  # the agent's fallback output, saved for judging

    result = ctx.run_case("task", "case", run)
    assert rejudge([result], FakeJudge(_verdict(5, 5))) == 0
    assert result.score is None
    assert result.error.startswith("rate limited, not scored")


def test_the_summary_counts_rate_limited_cases_apart_from_unjudged_ones():
    results = [
        CaseResult("t", "a", "m", 1.0, True, Usage(), 0.0),
        CaseResult("t", "b", "m", None, False, Usage(), 0.0, error="rate limited, not scored: x"),
        CaseResult("t", "c", "m", None, False, Usage(), 0.0, error="not judged yet"),
    ]
    assert "| 1.00 (1/1, 1 rate-limited)" in report.summary_table(results)


def _provider_error(error_class, status: int):
    import httpx

    response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com"))
    return error_class("tool_use_failed", response=response, body=None)


@pytest.mark.parametrize(
    ("error_name", "status"),
    [("BadRequestError", 400), ("AuthenticationError", 401), ("InternalServerError", 500)],
)
def test_other_provider_errors_are_failures_not_rate_limits(error_name, status):
    import groq

    recorder = UsageRecorder()
    recorder.on_llm_error(_provider_error(getattr(groq, error_name), status), run_id=uuid4())
    assert (recorder.total.failed_calls, recorder.total.rate_limited_calls) == (1, 0)


def test_a_raised_rate_limit_does_not_hide_an_earlier_failed_call():
    ctx = _context()

    def run():
        ctx.recorder.on_llm_error(RuntimeError("bad tool call"), run_id=uuid4())
        raise _rate_limit_error()

    result = ctx.run_case("t", "c", run)
    assert (result.score, result.passed) == (0.0, False)
    assert result.error == "1 LLM call(s) failed; the agent fell back"


def test_a_crash_after_a_refused_call_is_put_down_to_the_refusal():
    # The agent got no reply to work with, so the crash is not counted against the model.
    ctx = _context()

    def run():
        ctx.recorder.on_llm_error(_rate_limit_error(), run_id=uuid4())
        raise ValueError("no courses extracted")

    result = ctx.run_case("t", "c", run)
    assert result.score is None
    assert result.error == "rate limited, not scored: ValueError: no courses extracted"


def test_the_summary_counts_cases_the_app_decided_apart():
    results = [
        CaseResult("t", "a", "m", 1.0, True, Usage(), 0.0),
        CaseResult("t", "b", "m", None, False, Usage(), 0.0, error=DECIDED_BY_APP),
    ]
    assert "| 1.00 (1/1, 1 decided by the app)" in report.summary_table(results)


def test_cost_and_time_leave_out_cases_the_app_decided():
    ran = CaseResult("t", "a", "m", 1.0, True, Usage(seconds=10.0), 0.002)
    by_app = CaseResult("t", "b", "m", None, False, Usage(), 0.0, error=DECIDED_BY_APP)
    assert "$2.00 10.0s" in report.summary_table([ran, by_app])


def test_an_unscored_reason_from_the_task_is_kept():
    result = _context().run_case("t", "c", lambda: Outcome(None, False, {}, DECIDED_BY_APP))
    assert (result.score, result.error) == (None, DECIDED_BY_APP)


def test_cost_and_time_are_averaged_over_the_cases_that_ran():
    ran = CaseResult("t", "a", "m", 1.0, True, Usage(seconds=10.0), 0.002)
    refused = CaseResult("t", "b", "m", None, False, Usage(), 0.0, error="rate limited, not scored")
    assert "$2.00 10.0s" in report.summary_table([ran, refused])

import json

import pytest
from conftest import FakeReply
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents import curriculum_architect as ca


def course(course_id="c1", title="Course", topics=("Python",), description="", **extra):
    fields = {"difficulty": 2, "url": "https://example.com", "prereqs": []} | extra
    return ca.Course(
        id=course_id,
        title=title,
        topics=list(topics),
        description=description,
        **fields,
    )


def profile(interests=("python",), level="Intermediate", past=()):
    return ca.UserProfile(interests=list(interests), skill_level=level, past_courses=list(past))


class FakeStore:
    def __init__(self, courses=(), fail_on_add=False):
        self.documents = [ca._course_to_document(c) for c in courses]
        self.fail_on_add = fail_on_add
        self.added = []

    def similarity_search(self, query, k):
        return self.documents[:k]

    def add_documents(self, documents, ids):
        if self.fail_on_add:
            raise RuntimeError("store unavailable")
        self.added.extend(ids)


# --- Interest score ---


def test_interest_score_without_overlap_is_the_base_score():
    assert ca._interest_score({"python"}, {"excel"}) == ca.NO_INTEREST_OVERLAP_SCORE


def test_any_overlap_scores_above_no_overlap():
    interests = {str(i) for i in range(10)}
    assert ca._interest_score(interests, {"0"}) > ca._interest_score(interests, {"zz"})


def test_interest_score_rises_with_more_overlap():
    interests = {"a", "b", "c", "d"}
    assert ca._interest_score(interests, {"a", "b"}) > ca._interest_score(interests, {"a"})


def test_interest_score_is_capped_at_one():
    assert ca._interest_score({"a", "b"}, {"a", "b", "c"}) == 1.0


def test_no_interests_scores_the_base_score():
    assert ca._interest_score(set(), {"python"}) == ca.NO_INTEREST_OVERLAP_SCORE


# --- Keyword matching and catalog search ---


@pytest.mark.parametrize(
    ("text", "keyword", "expected"),
    [
        ("go basics", "go", True),
        ("learn google cloud", "go", False),
        ("c++ for beginners", "c++", True),
        ("node.js in depth", "node.js", True),
        ("pythonic code", "python", False),
    ],
)
def test_keywords_match_whole_words_only(text, keyword, expected):
    assert ca._mentions_word(text, keyword) is expected


def test_catalog_keeps_only_courses_that_mention_the_skill(monkeypatch):
    store = FakeStore([course("k8s", "Kubernetes Deep Dive"), course("xl", "Intro to Excel")])
    monkeypatch.setattr(ca, "get_course_catalog", lambda: store)
    result = ca.fetch_course_catalog({"current_skill": "Kubernetes"})
    assert [c.id for c in result["course_catalog"]] == ["k8s"]


def test_unrelated_catalog_courses_lead_to_a_web_search(monkeypatch):
    monkeypatch.setattr(
        ca, "get_course_catalog", lambda: FakeStore([course(title="Intro to Excel")])
    )
    state = {"current_skill": "Kubernetes"} | ca.fetch_course_catalog(
        {"current_skill": "Kubernetes"}
    )
    assert ca.route_after_catalog(state) == "web_search_for_courses"


def test_matching_catalog_courses_go_to_evaluation():
    assert ca.route_after_catalog({"course_catalog": [course()]}) == "start_evaluation"


def test_courses_survive_the_document_round_trip():
    original = course("c9", "T", topics=("Extract, Transform, Load", "B"), prereqs=[101, 202])
    assert ca._document_to_course(ca._course_to_document(original)) == original


# --- Web search results ---


def test_no_web_search_tool_gives_no_results(monkeypatch):
    monkeypatch.setattr(ca, "get_web_search", lambda: None)
    assert ca.web_search_for_courses({"current_skill": "Go"}) == {"raw_search_results": []}


def test_web_search_results_are_unwrapped(monkeypatch):
    class Search:
        def invoke(self, query):
            return {"results": [{"url": "u"}]}

    monkeypatch.setattr(ca, "get_web_search", lambda: Search())
    assert ca.web_search_for_courses({"current_skill": "Go"}) == {
        "raw_search_results": [{"url": "u"}]
    }


def test_web_search_failure_gives_no_results(monkeypatch):
    class Search:
        def invoke(self, query):
            raise RuntimeError("quota exceeded")

    monkeypatch.setattr(ca, "get_web_search", lambda: Search())
    assert ca.web_search_for_courses({"current_skill": "Go"}) == {"raw_search_results": []}


def _process(monkeypatch, fake_llm, courses, store):
    monkeypatch.setattr(
        ca, "get_llm", lambda: fake_llm(structured=ca.NewCourseList(courses=courses))
    )
    monkeypatch.setattr(ca, "get_course_catalog", lambda: store)
    return ca.process_and_store_courses({"current_skill": "Go", "raw_search_results": [{"x": 1}]})


def test_duplicate_course_ids_from_the_llm_are_dropped(monkeypatch, fake_llm):
    store = FakeStore()
    result = _process(monkeypatch, fake_llm, [course("dup"), course("dup"), course("other")], store)
    assert [c.id for c in result["course_catalog"]] == ["dup", "other"]


def test_deduplicated_courses_are_stored(monkeypatch, fake_llm):
    store = FakeStore()
    _process(monkeypatch, fake_llm, [course("dup"), course("dup")], store)
    assert store.added == ["dup"]


def test_store_failure_still_returns_the_courses(monkeypatch, fake_llm):
    result = _process(monkeypatch, fake_llm, [course("c1")], FakeStore(fail_on_add=True))
    assert [c.id for c in result["course_catalog"]] == ["c1"]


def test_no_search_results_skip_the_llm(monkeypatch):
    monkeypatch.setattr(ca, "get_llm", lambda: pytest.fail("LLM was called"))
    assert ca.process_and_store_courses({"raw_search_results": []}) == {"course_catalog": []}


# --- Parallel checks and scoring ---


def test_prerequisite_zero_means_none_required():
    state = {"user_profile": profile(), "course_catalog": [course(prereqs=[0])]}
    assert ca.check_prerequisites(state)["evaluation_logs"][0]["pass"] is True


def test_missing_prerequisite_fails_the_check():
    state = {
        "user_profile": profile(past=[101]),
        "course_catalog": [course(prereqs=[101, 202])],
    }
    assert ca.check_prerequisites(state)["evaluation_logs"][0]["pass"] is False


@pytest.mark.parametrize(
    ("level", "difficulty", "ok"),
    [
        ("Beginner", 2, True),
        ("Beginner", 3, False),
        ("Advanced", 5, True),
        ("Unknown", 4, True),
        ("Unknown", 5, False),
    ],
)
def test_workload_depends_on_skill_level(level, difficulty, ok):
    state = {
        "user_profile": profile(level=level),
        "course_catalog": [course(difficulty=difficulty)],
    }
    assert ca.estimate_workload(state)["evaluation_logs"][0]["load_ok"] is ok


def _logs(course_id, prereq=True, interest=0.8, workload=True):
    return [
        {"type": "prereq", "course_id": course_id, "pass": prereq},
        {"type": "interest", "course_id": course_id, "score": interest},
        {"type": "workload", "course_id": course_id, "load_ok": workload},
    ]


def test_eligible_course_scores_interest_out_of_ten():
    state = {
        "course_catalog": [course("c1")],
        "evaluation_logs": _logs("c1", interest=0.8),
    }
    assert ca.aggregate_scores(state)["evaluated_courses"][0]["score"] == 8.0


@pytest.mark.parametrize("failing", [{"prereq": False}, {"workload": False}])
def test_failed_check_scores_zero(failing):
    state = {
        "course_catalog": [course("c1")],
        "evaluation_logs": _logs("c1", **failing),
    }
    assert ca.aggregate_scores(state)["evaluated_courses"][0]["score"] == 0


def test_logs_for_unknown_courses_are_ignored():
    state = {
        "course_catalog": [course("c1")],
        "evaluation_logs": _logs("c1") + _logs("ghost"),
    }
    assert [e["course"].id for e in ca.aggregate_scores(state)["evaluated_courses"]] == ["c1"]


def _evaluated(*scores):
    return [{"course": course(f"c{i}", f"Course {i}"), "score": s} for i, s in enumerate(scores)]


def test_recommendations_keep_the_top_eligible_courses():
    state = {"current_skill": "Go", "evaluated_courses": _evaluated(2, 9, 0, 5, 7)}
    titles = [r["course_title"] for r in ca.recommend_enrollment(state)["all_found_courses"]]
    assert titles == ["Course 1", "Course 4", "Course 3"]


def test_alternatives_include_zero_scores():
    state = {"current_skill": "Go", "evaluated_courses": _evaluated(0, 0)}
    assert len(ca.suggest_alternatives(state)["all_found_courses"]) == 2


@pytest.mark.parametrize(
    ("scores", "route"),
    [
        ((0, 3), "recommend_enrollment"),
        ((0, 0), "suggest_alternatives"),
        ((), "suggest_alternatives"),
    ],
)
def test_scoring_routes_by_eligibility(scores, route):
    assert ca.route_after_scoring({"evaluated_courses": _evaluated(*scores)}) == route


# --- Planner and iteration ---


def test_planner_deduplicates_and_caps_skills(monkeypatch, fake_llm):
    skills = ["go", "go"] + [f"skill{i}" for i in range(ca.MAX_SKILLS + 5)]
    output = ca.PlannerOutput(user_profile=profile(), skills_to_find=skills)
    monkeypatch.setattr(ca, "get_llm", lambda: fake_llm(structured=output))
    result = ca.planner_node({"resume_inventory": "", "skill_gap_analysis": ""})
    assert result["skills_to_find"] == (["go"] + [f"skill{i}" for i in range(ca.MAX_SKILLS - 1)])


def test_iteration_takes_the_next_skill():
    result = ca.iteration_controller_node({"skills_to_find": ["go", "rust"]})
    assert (result["current_skill"], result["skills_to_find"]) == ("go", ["rust"])


def test_iteration_ends_when_skills_run_out():
    assert (
        ca.route_after_iteration(ca.iteration_controller_node({"skills_to_find": []}))
        == "architect"
    )


def test_overwrite_reducer_resets_on_an_empty_list():
    assert ca.list_overwrite_reducer([1, 2], []) == []


def test_overwrite_reducer_appends_otherwise():
    assert ca.list_overwrite_reducer([1], [2]) == [1, 2]


# --- Submission parsing ---

VALID_PATH = {
    "user_summary": "Summary.",
    "skill_gaps": ["go"],
    "recommendations": [
        {
            "skill_name": "go",
            "course_title": "Go",
            "course_url": "u",
            "description": "d",
        }
    ],
}


@pytest.mark.parametrize(
    "args",
    [
        {"path": VALID_PATH},
        VALID_PATH,
        json.dumps({"path": VALID_PATH}),
        {"path": json.dumps(VALID_PATH)},
    ],
)
def test_valid_submissions_are_parsed(args):
    assert ca._path_from_tool_args(args).skill_gaps == ["go"]


@pytest.mark.parametrize(
    "args",
    [
        '["not", "a", "dict"]',
        "42",
        '"just a string"',
        {"path": ["list"]},
        {"path": 7},
        {"path": {"user_summary": "missing fields"}},
        "not json",
    ],
)
def test_malformed_submissions_raise_value_error(args):
    with pytest.raises(ValueError):
        ca._path_from_tool_args(args)


def _tool_call(call_id, args):
    return {
        "name": "submit_learning_path",
        "args": args,
        "id": call_id,
        "type": "tool_call",
    }


def _submit(*calls):
    return ca.handle_submission_node({"messages": [AIMessage(content="", tool_calls=list(calls))]})


def test_valid_submission_is_accepted():
    result = _submit(_tool_call("c1", {"path": VALID_PATH}))
    assert result["submission_accepted"] and result["draft_path"].user_summary == "Summary."


def test_malformed_submission_is_rejected_not_raised():
    result = _submit(_tool_call("c1", {"path": ["not", "a", "dict"]}))
    assert result["submission_accepted"] is False and result["revision_count"] == 1


def test_every_tool_call_gets_a_reply():
    result = _submit(_tool_call("c1", {"path": VALID_PATH}), _tool_call("c2", {"path": VALID_PATH}))
    assert [m.tool_call_id for m in result["messages"]] == ["c1", "c2"]


def test_rejected_submission_replies_to_every_tool_call():
    result = _submit(_tool_call("c1", {"path": 7}), _tool_call("c2", {"path": 7}))
    assert (
        all(isinstance(m, ToolMessage) for m in result["messages"]) and len(result["messages"]) == 2
    )


def test_missing_tool_call_is_rejected():
    result = ca.handle_submission_node({"messages": [AIMessage(content="I think...")]})
    assert result["submission_accepted"] is False and result["messages"] == []


@pytest.mark.parametrize(
    ("state", "route"),
    [
        ({"submission_accepted": True, "revision_count": 0}, "critic"),
        ({"submission_accepted": False, "revision_count": 1}, "prepare_revision"),
        ({"submission_accepted": False, "revision_count": ca.MAX_REVISIONS}, ca.END),
    ],
)
def test_submission_routing(state, route):
    assert ca.route_after_submission(state) == route


def test_failed_resubmission_does_not_send_an_old_draft_to_the_critic():
    state = {"submission_accepted": False, "revision_count": 1, "draft_path": object()}
    assert ca.route_after_submission(state) == "prepare_revision"


@pytest.mark.parametrize(
    ("approved", "revisions", "route"),
    [(True, 1, ca.END), (False, 1, "prepare_revision"), (False, ca.MAX_REVISIONS, ca.END)],
)
def test_critique_routing(approved, revisions, route):
    state = {
        "structured_critique": ca.Critique(is_approved=approved, revisions_needed=""),
        "revision_count": revisions,
    }
    assert ca.route_after_critique(state) == route


# --- Architect node ---


def _architect_state(**extra):
    return {
        "user_profile": profile(),
        "all_found_courses": [],
        "revision_count": 0,
    } | extra


def test_architect_records_each_response_once(monkeypatch, fake_llm):
    no_tool, with_tool = FakeReply("thinking"), FakeReply("", [_tool_call("c1", {})])
    monkeypatch.setattr(ca, "get_llm", lambda: fake_llm(no_tool, with_tool))
    messages = ca.architect_node(_architect_state())["messages"]
    assert [m for m in messages if m in (no_tool, with_tool)] == [no_tool, with_tool]


def test_architect_gives_up_after_the_last_attempt(monkeypatch, fake_llm):
    llm = fake_llm("no tool call")
    monkeypatch.setattr(ca, "get_llm", lambda: llm)
    ca.architect_node(_architect_state())
    assert len(llm.prompts) == ca.MAX_ARCHITECT_ATTEMPTS


def test_architect_does_not_nudge_after_the_final_attempt(monkeypatch, fake_llm):
    monkeypatch.setattr(ca, "get_llm", lambda: fake_llm("no tool call"))
    last = ca.architect_node(_architect_state())["messages"][-1]
    assert not isinstance(last, HumanMessage)


def test_revision_pass_returns_only_new_messages(monkeypatch, fake_llm):
    history = [HumanMessage(content="earlier"), HumanMessage(content="feedback")]
    reply = FakeReply("", [_tool_call("c2", {})])
    monkeypatch.setattr(ca, "get_llm", lambda: fake_llm(reply))
    result = ca.architect_node(_architect_state(revision_count=1, messages=history))
    assert result["messages"] == [reply]


# --- Course catalog seeding ---


def test_seed_courses_have_unique_ids():
    ids = [c.id for c in ca._load_seed_courses()]
    assert len(ids) == len(set(ids))


def test_learning_path_skips_soft_skills(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        ca,
        "build_learning_path",
        lambda gaps, inventory: captured.update(gaps=gaps) or {},
    )
    ca.plan_learning_path(["docker", "communication", "Teamwork"], [], "Dev")
    assert captured["gaps"] == "docker"


def test_learning_path_keeps_soft_skills_when_they_are_all_there_is(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        ca,
        "build_learning_path",
        lambda gaps, inventory: captured.update(gaps=gaps) or {},
    )
    ca.plan_learning_path(["communication"], [], "Dev")
    assert captured["gaps"] == "communication"


def test_catalog_is_seeded_on_first_use(monkeypatch):
    class Store(FakeStore):
        def get(self, ids):
            return {"ids": []}

    store = Store()
    monkeypatch.setattr(ca, "get_chroma", lambda path: store)
    ca.get_course_catalog()
    assert sorted(store.added) == sorted(c.id for c in ca._load_seed_courses())


def test_existing_seed_courses_are_not_added_again(monkeypatch):
    seeded = [c.id for c in ca._load_seed_courses()]

    class Store(FakeStore):
        def get(self, ids):
            return {"ids": seeded}

    store = Store()
    monkeypatch.setattr(ca, "get_chroma", lambda path: store)
    ca.get_course_catalog()
    assert store.added == []


def test_documents_from_the_store_can_have_empty_lists():
    document = Document(
        page_content="x",
        metadata=course(prereqs=[], topics=()).model_dump() | {"prereqs": "", "topics": ""},
    )
    assert ca._document_to_course(document).topics == []


def test_older_comma_separated_metadata_still_loads():
    metadata = course().model_dump() | {"prereqs": "101,202", "topics": "Python, SQL"}
    restored = ca._document_to_course(Document(page_content="x", metadata=metadata))
    assert (restored.prereqs, restored.topics) == ([101, 202], ["Python", "SQL"])


@pytest.mark.parametrize(
    ("skill", "title", "expected"),
    [
        ("Machine Learning", "Learning Excel", False),
        ("Machine Learning", "Machine Learning A-Z", True),
        ("Security Information and Event Management", "Data and Society", False),
        (
            "Security Information and Event Management",
            "Security Information & Event Management",
            True,
        ),
        ("and", "Pandas and NumPy", True),
    ],
)
def test_multi_word_skills_need_every_keyword(skill, title, expected):
    keywords = ca._skill_keywords(skill)
    assert ca._course_mentions_all(course(title=title, topics=()), keywords) is expected


def test_planner_dedupes_skills_ignoring_case(monkeypatch, fake_llm):
    output = ca.PlannerOutput(user_profile=profile(), skills_to_find=["Go", "go", "Rust", "GO"])
    monkeypatch.setattr(ca, "get_llm", lambda: fake_llm(structured=output))
    result = ca.planner_node({"resume_inventory": "", "skill_gap_analysis": ""})
    assert result["skills_to_find"] == ["Go", "Rust"]


def test_llm_extraction_failure_gives_no_courses(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    class BrokenLLM:
        def with_structured_output(self, schema):
            return RunnableLambda(lambda _: (_ for _ in ()).throw(ValueError("bad json")))

    monkeypatch.setattr(ca, "get_llm", lambda: BrokenLLM())
    state = {"current_skill": "Go", "raw_search_results": [{"x": 1}]}
    assert ca.process_and_store_courses(state) == {"course_catalog": []}


class GraphLLM:
    """Fake LLM for a full graph run: planner, tool-calling architect and an approving critic."""

    def __init__(self, skills):
        from langchain_core.runnables import RunnableLambda

        self.RunnableLambda = RunnableLambda
        self.skills = skills

    def with_structured_output(self, schema):
        outputs = {
            ca.PlannerOutput: ca.PlannerOutput(user_profile=profile(), skills_to_find=self.skills),
            ca.Critique: ca.Critique(is_approved=True, revisions_needed=""),
            ca.NewCourseList: ca.NewCourseList(courses=[]),
        }
        return self.RunnableLambda(lambda _: outputs[schema])

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content="", tool_calls=[_tool_call("final", {"path": VALID_PATH})])


def test_full_graph_scores_catalog_courses_and_returns_a_path(monkeypatch):
    store = FakeStore([course("go1", "Go Fundamentals", topics=("Go", "python"))])
    monkeypatch.setattr(ca, "get_course_catalog", lambda: store)
    monkeypatch.setattr(ca, "get_web_search", lambda: None)
    monkeypatch.setattr(ca, "get_llm", lambda: GraphLLM(["Go"]))
    result = ca.build_learning_path("gaps: Go", "inventory")
    found = result["all_found_courses"]
    assert result["draft_path"].skill_gaps == ["go"]
    assert [(c["course_title"], c["description"].startswith("Best match")) for c in found] == [
        ("Go Fundamentals", True)
    ]


def test_full_graph_resets_course_logs_between_skills(monkeypatch):
    store = FakeStore([course("go1", "Go and Rust", topics=("go", "rust", "python"))])
    monkeypatch.setattr(ca, "get_course_catalog", lambda: store)
    monkeypatch.setattr(ca, "get_web_search", lambda: None)
    monkeypatch.setattr(ca, "get_llm", lambda: GraphLLM(["Go", "Rust"]))
    found = ca.build_learning_path("gaps", "inventory")["all_found_courses"]
    assert [c["skill_name"] for c in found] == ["Go", "Rust"]


@pytest.mark.parametrize(
    ("text", "keyword", "expected"),
    [
        ("c++ for beginners", "c", False),
        ("c# in depth", "c", False),
        ("modern c and unix", "c", True),
    ],
)
def test_single_letter_skills_do_not_match_languages_built_on_them(text, keyword, expected):
    assert ca._mentions_word(text, keyword) is expected


@pytest.mark.parametrize(
    "soft_skill", ["Agile Methodologies", "Critical Thinking", "Project Management", "Kanban"]
)
def test_common_soft_skills_are_skipped(soft_skill, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        ca, "build_learning_path", lambda gaps, inventory: captured.update(gaps=gaps) or {}
    )
    ca.plan_learning_path(["docker", soft_skill], [], "Dev")
    assert captured["gaps"] == "docker"


@pytest.mark.parametrize(
    ("interests", "topics", "score"),
    [({"a", "b", "c", "d"}, {"a"}, 0.6875), ({"a", "b", "c", "d"}, {"a", "b"}, 0.875)],
)
def test_interest_score_exact_values(interests, topics, score):
    assert ca._interest_score(interests, topics) == pytest.approx(score)

import pytest
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver

from app.agents import interview_coach as ic

GOOD = "GRADE: good\nSKILL: caching\nFEEDBACK: Solid."
MODERATE = "GRADE: moderate\nSKILL: caching\nFEEDBACK: Partial."
BAD = "GRADE: bad\nSKILL: caching\nFEEDBACK: Missing."


class PromptRoutedLLM:
    """Answers each kind of interview prompt with a fixed reply, however often it is asked."""

    def __init__(self, grade=MODERATE, decision="FOLLOW_UP", follow_up="Follow-up?"):
        self.replies = {
            "Evaluate this answer": grade,
            "routing logic": decision,
            "Probe deeper": follow_up,
            "Summarize": "Summary.",
        }
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        reply = next(text for marker, text in self.replies.items() if marker in prompt)
        return type("Reply", (), {"content": reply})()


def question(qid: str, text: str = "", topic: str = "system design", difficulty: str = "easy"):
    return Document(
        page_content=text or f"Question {qid}?",
        metadata={"question_id": qid, "topic": topic, "difficulty": difficulty},
    )


@pytest.fixture
def coach(monkeypatch):
    """The real interview graph on an in-memory checkpointer, deterministic picks."""
    graph = ic.build_interview_graph().compile(checkpointer=InMemorySaver())
    monkeypatch.setattr(ic, "get_interview_graph", lambda: graph)
    monkeypatch.setattr(ic.random, "choice", lambda items: items[0])
    return ic


def serve(monkeypatch, *batches):
    """Each question retrieval returns the next batch; then nothing."""
    remaining = iter(batches)
    monkeypatch.setattr(
        ic, "_retrieve_questions", lambda state, difficulty_map: next(remaining, [])
    )


def use_llm(monkeypatch, llm):
    monkeypatch.setattr(ic, "get_llm", lambda: llm)


def start(coach_module, thread="t"):
    return coach_module.start_interview(thread, ["System Design"], "generic")


# --- Session flow ---


def test_first_turn_asks_the_first_question(coach, monkeypatch, fake_llm):
    serve(monkeypatch, [question("q1", "Design a cache")])
    use_llm(monkeypatch, fake_llm(GOOD))
    turn = start(coach)
    assert (turn.question, turn.finished, turn.question_count) == (
        "Design a cache",
        False,
        1,
    )


def test_empty_store_finishes_immediately(coach, monkeypatch, fake_llm):
    serve(monkeypatch)
    use_llm(monkeypatch, fake_llm("summary"))
    turn = start(coach)
    assert turn.finished and turn.question is None


def test_running_out_of_questions_ends_the_session(coach, monkeypatch, fake_llm):
    serve(monkeypatch, [question("q1", "Design a cache")])
    use_llm(monkeypatch, fake_llm(GOOD, "Overall fine."))
    start(coach)
    turn = coach.submit_answer("t", "Use an LRU cache")
    assert turn.finished and turn.question is None


def test_running_out_of_questions_records_only_real_questions(coach, monkeypatch, fake_llm):
    serve(monkeypatch, [question("q1")])
    use_llm(monkeypatch, fake_llm(GOOD, "Overall fine."))
    start(coach)
    coach.submit_answer("t", "answer")
    assert [entry["question_id"] for entry in coach.evaluation_report("t")] == ["q1"]


def test_good_answer_moves_to_the_next_question(coach, monkeypatch, fake_llm):
    serve(monkeypatch, [question("q1")], [question("q2", "Shard a database")])
    use_llm(monkeypatch, fake_llm(GOOD))
    start(coach)
    turn = coach.submit_answer("t", "answer")
    assert (turn.question, turn.question_count) == ("Shard a database", 2)


def test_feedback_is_shown_after_an_answer(coach, monkeypatch, fake_llm):
    serve(monkeypatch, [question("q1")], [question("q2")])
    use_llm(monkeypatch, fake_llm(GOOD))
    start(coach)
    assert coach.submit_answer("t", "answer").feedback == "Grade: GOOD\nFeedback: Solid."


def test_moderate_answer_can_get_a_follow_up(coach, monkeypatch):
    serve(monkeypatch, [question("q1")])
    use_llm(monkeypatch, PromptRoutedLLM(follow_up="What about eviction?"))
    start(coach)
    turn = coach.submit_answer("t", "use a cache")
    assert (turn.question, turn.question_count) == ("What about eviction?", 1)


@pytest.mark.parametrize(
    "answer",
    [
        "I don't know.",
        "I don’t know",
        "No idea, honestly. Just tell me the answer.",
        "Can we skip?",
    ],
)
def test_no_follow_up_after_an_answer_that_does_not_try(coach, monkeypatch, answer):
    serve(monkeypatch, [question("q1")], [question("q2", "Next question")])
    llm = PromptRoutedLLM(grade=BAD)
    use_llm(monkeypatch, llm)
    start(coach)
    assert coach.submit_answer("t", answer).question == "Next question"
    assert not any("routing logic" in prompt for prompt in llm.prompts)


def test_a_bad_answer_that_tries_is_left_to_the_model(coach, monkeypatch):
    serve(monkeypatch, [question("q1")])
    use_llm(monkeypatch, PromptRoutedLLM(grade=BAD, follow_up="Why a list?"))
    start(coach)
    assert coach.submit_answer("t", "I would keep it in a list").question == "Why a list?"


def test_an_earlier_i_dont_know_does_not_stop_a_follow_up_answer_being_probed(coach, monkeypatch):
    serve(monkeypatch, [question("q1")])
    llm = PromptRoutedLLM(grade=MODERATE, follow_up="Which item goes?")
    use_llm(monkeypatch, llm)
    start(coach)
    coach.submit_answer("t", "I don't know its name. You keep the items in a list.")
    llm.replies["Evaluate this answer"] = BAD
    coach.submit_answer("t", "Drop the newest one")
    assert sum("routing logic" in prompt for prompt in llm.prompts) == 2


def test_not_knowing_the_follow_up_moves_on(coach, monkeypatch):
    serve(monkeypatch, [question("q1")], [question("q2", "Next question")])
    llm = PromptRoutedLLM(grade=MODERATE)
    use_llm(monkeypatch, llm)
    start(coach)
    coach.submit_answer("t", "use a cache")  # moderate: the model asks a follow-up
    llm.replies["Evaluate this answer"] = BAD
    assert coach.submit_answer("t", "I don't know").question == "Next question"


@pytest.mark.parametrize(
    ("reply", "route"),
    [
        ("FOLLOW_UP", "ask_follow_up"),
        ("**Follow-up**", "ask_follow_up"),
        ("follow up", "ask_follow_up"),
        ("NEW_QUESTION", "update_coverage"),
        ("NEW_QUESTION. No FOLLOW_UP needed.", "update_coverage"),  # unclear: move on
        ("", "update_coverage"),
    ],
)
def test_the_router_reads_one_clear_decision(reply, route, monkeypatch, fake_llm):
    monkeypatch.setattr(ic, "get_llm", lambda: fake_llm(reply))
    entry = {"question_id": "q1", "question": "Q?", "answer": "a cache", "evaluation": "moderate"}
    state = {"messages": [], "evaluation_report": [entry], "current_question_id": "q1"}
    assert ic.decide_action_router(state) == route


@pytest.mark.parametrize(
    ("answer", "grade", "expected"),
    [
        ("", "moderate", True),
        ("   ", "bad", True),
        ("I do not know", "bad", True),
        ("I haven't studied this", "bad", True),
        ("Let's move on", "bad", True),
        ("dunno", "bad", True),
        ("I'd like to skip this question", "bad", True),
        ("I would rather skip this one", "bad", True),
        ("I don't know, would you mind moving on?", "bad", True),
        ("I don't really know", "bad", True),
        ("I can't answer that", "bad", True),
        ("I'd use a hash map", "bad", False),
        ("Can we skip this one?", "bad", True),
        ("I'm not sure, but a mutex lets one thread in at a time", "bad", False),
        ("Not sure. Maybe a hash map?", "bad", False),
        ("I don't know, I think it's a lock", "bad", False),
        ("I'd use a skip list here", "bad", False),
        ("The scheduler will move on to the next process", "bad", False),
        ("I don't know.", "moderate", False),  # the grader found something in it
        ("TCP is reliable", "bad", False),
        ("I know a process has its own memory", "bad", False),
    ],
)
def test_answers_that_do_not_try_are_recognised(answer, grade, expected):
    assert ic.did_not_attempt(answer, grade) is expected


def test_follow_up_answer_updates_the_same_report_entry(coach, monkeypatch):
    serve(monkeypatch, [question("q1")], [question("q2")])
    llm = PromptRoutedLLM()
    use_llm(monkeypatch, llm)
    start(coach)
    coach.submit_answer("t", "use a cache")
    llm.replies["Evaluate this answer"] = GOOD
    coach.submit_answer("t", "LRU")
    report = coach.evaluation_report("t")
    assert len(report) == 1 and report[0]["evaluation"] == "good"


def test_follow_up_answer_is_appended_to_the_original_answer(coach, monkeypatch):
    serve(monkeypatch, [question("q1")], [question("q2")])
    llm = PromptRoutedLLM()
    use_llm(monkeypatch, llm)
    start(coach)
    coach.submit_answer("t", "use a cache")
    llm.replies["Evaluate this answer"] = GOOD
    coach.submit_answer("t", "LRU")
    assert coach.evaluation_report("t")[0]["answer"] == "use a cache\n[FOLLOW-UP]: LRU"


def test_follow_ups_per_question_are_capped(coach, monkeypatch):
    serve(monkeypatch, [question("q1")], [question("q2", "Next question")])
    use_llm(monkeypatch, PromptRoutedLLM())
    asked = [start(coach).question]
    for _ in range(ic.MAX_FOLLOW_UPS_PER_QUESTION + 1):
        asked.append(coach.submit_answer("t", "vague answer").question)
    follow_ups = ["Follow-up?"] * ic.MAX_FOLLOW_UPS_PER_QUESTION
    assert asked == ["Question q1?", *follow_ups, "Next question"]


def test_new_question_resets_the_follow_up_allowance(coach, monkeypatch):
    serve(
        monkeypatch,
        [question("q1")],
        [question("q2", "Next question")],
        [question("q3")],
    )
    use_llm(monkeypatch, PromptRoutedLLM())
    start(coach)
    for _ in range(ic.MAX_FOLLOW_UPS_PER_QUESTION + 1):
        coach.submit_answer("t", "vague answer")
    assert coach.submit_answer("t", "vague again").question == "Follow-up?"


def test_session_stops_at_the_question_limit(coach, monkeypatch, fake_llm):
    batches = [[question(f"q{i}")] for i in range(ic.MAX_QUESTIONS_PER_SESSION + 5)]
    serve(monkeypatch, *batches)
    use_llm(monkeypatch, fake_llm(GOOD))
    turn = start(coach)
    answers = 0
    while not turn.finished:
        turn = coach.submit_answer("t", "answer")
        answers += 1
    assert answers == ic.MAX_QUESTIONS_PER_SESSION


@pytest.mark.parametrize("word", sorted(ic.STOP_WORDS) + [" STOP "])
def test_stop_word_ends_the_session(word, coach, monkeypatch, fake_llm):
    serve(monkeypatch, [question("q1")], [question("q2")])
    use_llm(monkeypatch, fake_llm("Summary text."))
    start(coach)
    assert coach.submit_answer("t", word).finished


def test_stop_word_is_not_graded(coach, monkeypatch, fake_llm):
    serve(monkeypatch, [question("q1")], [question("q2")])
    use_llm(monkeypatch, fake_llm("Summary text."))
    start(coach)
    coach.submit_answer("t", "stop")
    assert coach.evaluation_report("t") == []


def test_stop_word_inside_an_answer_does_not_stop(coach, monkeypatch, fake_llm):
    serve(monkeypatch, [question("q1")], [question("q2")])
    use_llm(monkeypatch, fake_llm(GOOD))
    start(coach)
    assert not coach.submit_answer("t", "I would stop the cache from growing").finished


def test_summary_is_returned_when_finished(coach, monkeypatch, fake_llm):
    serve(monkeypatch, [question("q1")])
    use_llm(monkeypatch, fake_llm(GOOD, "You did well."))
    start(coach)
    assert coach.submit_answer("t", "answer").summary == "Interview complete!\n\nYou did well."


def test_answering_an_unstarted_interview_is_an_error(coach):
    with pytest.raises(ic.InterviewStateError):
        coach.submit_answer("never-started", "hello")


def test_answering_a_finished_interview_is_an_error(coach, monkeypatch, fake_llm):
    serve(monkeypatch, [question("q1")])
    use_llm(monkeypatch, fake_llm(GOOD))
    start(coach)
    coach.submit_answer("t", "stop")
    with pytest.raises(ic.InterviewStateError):
        coach.submit_answer("t", "one more")


def test_starting_the_same_thread_twice_is_an_error(coach, monkeypatch, fake_llm):
    serve(monkeypatch, [question("q1")], [question("q2")])
    use_llm(monkeypatch, fake_llm(GOOD))
    start(coach)
    with pytest.raises(ic.InterviewStateError):
        start(coach)


def test_missing_topics_are_rejected(coach, monkeypatch, fake_llm):
    serve(monkeypatch, [question("q1")])
    use_llm(monkeypatch, fake_llm(GOOD))
    with pytest.raises(ValueError, match="topic_list"):
        coach.start_interview("t", [], "generic")


# --- Grade parsing ---


@pytest.mark.parametrize(
    ("text", "grade"),
    [
        ("GRADE: good", "good"),
        ("GRADE: [good]", "good"),
        ("**GRADE:** Good", "good"),
        ("grade: Excellent.", "excellent"),
        ("GRADE: BAD\nSKILL: x", "bad"),
        ("GRADE: moderate", "moderate"),
        ("GRADE: fantastic", "moderate"),
        ("no grade line at all", "moderate"),
    ],
)
def test_grade_is_parsed_from_llm_text(text, grade):
    assert ic._parse_grade(text) == grade


def test_markdown_labels_are_stripped_from_fields():
    assert ic._parse_field("SKILL", "**SKILL:** Caching", "general") == "Caching"


def test_feedback_can_span_lines():
    text = "GRADE: good\nFEEDBACK: First line.\nSecond line."
    assert ic._parse_field("FEEDBACK", text, "", ic.re.DOTALL) == "First line.\nSecond line."


def test_llm_failure_during_grading_is_recorded(monkeypatch):
    class BrokenLLM:
        def invoke(self, prompt):
            raise RuntimeError("provider down")

    use_llm(monkeypatch, BrokenLLM())
    assert ic._grade_answer("q", "a") == ("error", "unknown", "provider down")


# --- Difficulty adaptation ---


def answers(*pairs):
    return [
        {"difficulty": d, "evaluation": e, "question_id": f"q{i}"} for i, (d, e) in enumerate(pairs)
    ]


def test_two_bad_hard_answers_make_the_topic_easier():
    performance = {"sd": answers(("hard", "bad"), ("hard", "bad"))}
    assert ic.adapt_difficulty(performance, {}, ["SD"]) == {"sd": ["easy", "medium"]}


def test_two_excellent_easy_answers_make_the_topic_harder():
    performance = {"sd": answers(("easy", "excellent"), ("easy", "excellent"))}
    assert ic.adapt_difficulty(performance, {}, ["sd"]) == {"sd": ["hard", "medium"]}


def test_recent_answers_win_over_older_bad_ones():
    performance = {
        "sd": answers(
            ("hard", "bad"),
            ("hard", "bad"),
            ("easy", "excellent"),
            ("easy", "excellent"),
        )
    }
    current = {"sd": ["easy", "medium"]}
    assert ic.adapt_difficulty(performance, current, ["sd"]) == {"sd": ["hard", "medium"]}


def test_mixed_recent_answers_keep_the_difficulty():
    performance = {"sd": answers(("hard", "bad"), ("easy", "excellent"))}
    assert ic.adapt_difficulty(performance, {}, ["sd"]) == {}


def test_one_answer_is_not_enough_to_adapt():
    performance = {"sd": answers(("hard", "bad"))}
    assert ic.adapt_difficulty(performance, {}, ["sd"]) == {}


def test_each_topic_adapts_independently():
    performance = {
        "sd": answers(("hard", "bad"), ("hard", "bad")),
        "dsa": answers(("easy", "good"), ("easy", "good")),
    }
    assert ic.adapt_difficulty(performance, {}, ["sd", "dsa"]) == {"sd": ["easy", "medium"]}


# --- Retrieval filters ---


def test_single_topic_filter_keeps_its_difficulties():
    condition = ic._topic_condition(["System Design"], {"system design": ["easy", "medium"]})
    assert condition == {
        "$and": [
            {"topic": "system design"},
            {"difficulty": {"$in": ["easy", "medium"]}},
        ]
    }


def test_each_topic_gets_its_own_difficulty_filter():
    condition = ic._topic_condition(["A", "B"], {"a": ["easy", "medium"]})
    assert condition == {
        "$or": [
            {"$and": [{"topic": "a"}, {"difficulty": {"$in": ["easy", "medium"]}}]},
            {"$and": [{"topic": "b"}, {"difficulty": {"$in": ic.ALL_DIFFICULTIES}}]},
        ]
    }


def test_relaxed_filter_ignores_difficulty():
    assert ic._topic_condition(["A", "B"], None) == {"topic": {"$in": ["a", "b"]}}


def test_question_filter_excludes_covered_questions():
    state = {
        "topic_list": ["A"],
        "company_focus": "google",
        "covered_questions": ["q1"],
    }
    conditions = ic._question_filter(state, None)["$and"]
    assert {"question_id": {"$nin": ["q1"]}} in conditions


def test_all_companies_focus_searches_every_company():
    assert ic._companies_for("all") == list(ic.COMPANIES)


def test_specific_company_focus_searches_only_that_company():
    assert ic._companies_for("google") == ["google"]


def test_retrieval_falls_back_to_no_difficulty_limit(monkeypatch):
    filters_tried = []

    class FakeRetriever:
        def __init__(self, metadata_filter):
            self.metadata_filter = metadata_filter

        def invoke(self, query):
            filters_tried.append(self.metadata_filter)
            return [] if len(filters_tried) == 1 else [question("q9")]

    monkeypatch.setattr(ic, "_retriever", FakeRetriever)
    state = {"topic_list": ["A"], "company_focus": "generic"}
    results = ic._retrieve_questions(state, {"a": ["hard"]})
    assert [r.metadata["question_id"] for r in results] == ["q9"] and len(filters_tried) == 2


# --- Topic mapping ---


# Each company names its topics its own way, as the question files do.
FAKE_TOPICS = {
    "generic": [
        "Cloud & Infrastructure", "Coding (All Languages)", "Data Structures & Algorithms",
        "Databases & Data Engineering", "Frameworks (Frontend)", "Programming Languages",
        "RESTful APIs & Web Services", "System Design",
    ],
    "google": ["Networking", "System Design"],
    "tiny": ["Security", "Networking", "Operating Systems"],
}  # fmt: skip


@pytest.fixture
def topics(monkeypatch):
    by_company = {
        company: {name.lower(): name for name in names} for company, names in FAKE_TOPICS.items()
    }
    monkeypatch.setattr(ic, "_topic_names_by_company", lambda: by_company)


def test_skill_gaps_map_to_the_companys_topics_plus_system_design(topics):
    assert ic.topics_for_skill_gaps(["Docker", "react.js"], "generic") == [
        "Cloud & Infrastructure",
        "Frameworks (Frontend)",
        "System Design",
    ]


@pytest.mark.parametrize(
    ("gap", "topic"),
    [
        ("mongodb", "Databases & Data Engineering"),
        ("google cloud platform", "Cloud & Infrastructure"),
        ("JavaScript", "Programming Languages"),  # preferred over "Coding (All Languages)"
        ("Go", "Programming Languages"),
        ("REST API design", "RESTful APIs & Web Services"),
        ("REST APIs", "RESTful APIs & Web Services"),  # plural keyword
    ],
)
def test_skill_gap_maps_to_its_own_topic(topics, gap, topic):
    assert topic in ic.topics_for_skill_gaps([gap], "generic")


def test_unmapped_skill_gaps_use_default_topics(topics):
    assert ic.topics_for_skill_gaps(["basket weaving"], "generic") == ic.DEFAULT_TOPICS


def test_no_skill_gaps_use_default_topics(topics):
    assert ic.topics_for_skill_gaps([], "generic") == ic.DEFAULT_TOPICS


@pytest.mark.parametrize("gap", ["algorithms", "django", "cargo"])
def test_words_containing_a_keyword_are_not_mapped(topics, gap):
    assert ic.topics_for_skill_gaps([gap], "generic") == ic.DEFAULT_TOPICS


def test_a_topic_the_company_has_no_questions_on_is_left_out(topics):
    assert ic.topics_for_skill_gaps(["mongodb", "docker"], "google") == ["System Design"]


def test_python_prefers_a_python_topic_and_java_a_general_coding_one(monkeypatch):
    names = ["Coding (All Languages)", "Python & Language-Specific", "System Design"]
    by_company = {"google": {name.lower(): name for name in names}}
    monkeypatch.setattr(ic, "_topic_names_by_company", lambda: by_company)
    assert ic.topics_for_skill_gaps(["python"], "google")[0] == "Python & Language-Specific"
    assert ic.topics_for_skill_gaps(["java"], "google")[0] == "Coding (All Languages)"


def test_a_company_without_the_default_topics_gets_its_first_topics(topics):
    assert ic.topics_for_skill_gaps([], "tiny") == ["Networking", "Operating Systems"]


def test_all_companies_offer_every_topic_once_a_to_z(topics):
    offered = ic.interview_topics("all")
    assert offered == sorted(set(offered), key=str.lower)
    assert set(offered) == set(FAKE_TOPICS["generic"]) | set(FAKE_TOPICS["google"])


def test_chosen_topics_are_matched_ignoring_case_and_spaces(topics):
    found, unknown = ic.resolve_topics([" system design", "NETWORKING", "Networking"], "google")
    assert (found, unknown) == (["System Design", "Networking"], [])


def test_topics_without_questions_for_the_company_are_reported_as_typed(topics):
    found, unknown = ic.resolve_topics(["Python", "System Design"], "google")
    assert (found, unknown) == (["System Design"], ["Python"])


def test_topic_names_are_shown_as_the_question_files_write_them():
    assert "Data Structures & Algorithms" in ic.interview_topics("generic")


@pytest.mark.parametrize("company", ic.COMPANIES)
def test_every_company_has_questions_on_the_default_topics(company):
    assert set(ic.DEFAULT_TOPICS) <= set(ic.interview_topics(company))


def test_company_list_matches_the_question_files():
    data_files = {path.stem for path in ic_data_dir().glob("*.txt")}
    assert data_files == set(ic.COMPANIES)


def ic_data_dir():
    from app.agents.ingestion.seed_interview_db import DATA_SOURCE_DIR

    return DATA_SOURCE_DIR


class RecordingRetriever:
    """Stands in for the Chroma retriever: records filters, serves questions in turn."""

    def __init__(self, *batches):
        self.batches = iter(batches)
        self.filters = []

    def __call__(self, metadata_filter):
        self.filters.append(metadata_filter)
        return self

    def invoke(self, query):
        return next(self.batches, [])


def test_answered_questions_are_excluded_from_the_next_search(coach, monkeypatch, fake_llm):
    retriever = RecordingRetriever([question("q1")], [question("q2")])
    monkeypatch.setattr(ic, "_retriever", retriever)
    use_llm(monkeypatch, fake_llm(GOOD))
    start(coach)
    coach.submit_answer("t", "answer")
    assert {"question_id": {"$nin": ["q1"]}} in retriever.filters[-1]["$and"]


def test_adapted_difficulty_reaches_the_search(coach, monkeypatch):
    hard = [[question(f"q{i}", difficulty="hard")] for i in range(3)]
    retriever = RecordingRetriever(*hard)
    monkeypatch.setattr(ic, "_retriever", retriever)
    use_llm(monkeypatch, PromptRoutedLLM(grade="GRADE: bad", decision="NEW_QUESTION"))
    start(coach)
    coach.submit_answer("t", "no idea")
    coach.submit_answer("t", "no idea")
    topic_filter = retriever.filters[-1]["$and"][0]
    assert topic_filter == {
        "$and": [{"topic": "system design"}, {"difficulty": {"$in": ["easy", "medium"]}}]
    }


def test_retrieval_error_falls_back_to_the_relaxed_search(monkeypatch):
    calls = []

    class FlakyRetriever:
        def __init__(self, metadata_filter):
            pass

        def invoke(self, query):
            calls.append(query)
            if len(calls) == 1:
                raise RuntimeError("chroma unavailable")
            return [question("q5")]

    monkeypatch.setattr(ic, "_retriever", FlakyRetriever)
    results = ic._retrieve_questions({"topic_list": ["A"], "company_focus": "generic"}, {})
    assert [r.metadata["question_id"] for r in results] == ["q5"]


def test_topic_history_keeps_only_recent_answers(coach, monkeypatch, fake_llm):
    batches = [[question(f"q{i}")] for i in range(ic.TOPIC_HISTORY_LENGTH + 3)]
    serve(monkeypatch, *batches)
    use_llm(monkeypatch, fake_llm(GOOD))
    start(coach)
    for _ in range(ic.TOPIC_HISTORY_LENGTH + 2):
        coach.submit_answer("t", "answer")
    snapshot = ic.get_interview_graph().get_state({"configurable": {"thread_id": "t"}})
    assert len(snapshot.values["topic_performance"]["system design"]) == ic.TOPIC_HISTORY_LENGTH


def test_failed_grading_moves_on_without_a_follow_up(coach, monkeypatch):
    class GradingDown(PromptRoutedLLM):
        def invoke(self, prompt):
            if "Evaluate this answer" in prompt:
                raise RuntimeError("provider down")
            return super().invoke(prompt)

    serve(monkeypatch, [question("q1")], [question("q2", "Next question")])
    use_llm(monkeypatch, GradingDown())
    start(coach)
    assert coach.submit_answer("t", "answer").question == "Next question"


def test_rejected_start_leaves_the_thread_usable(coach, monkeypatch, fake_llm):
    serve(monkeypatch, [question("q1")])
    use_llm(monkeypatch, fake_llm(GOOD))
    with pytest.raises(ValueError):
        coach.start_interview("t", [], "generic")
    assert start(coach).question == "Question q1?"


def test_empty_answer_is_graded_for_the_current_question(coach, monkeypatch):
    serve(monkeypatch, [question("q1")], [question("q2", "Next question")])
    llm = PromptRoutedLLM(grade="GRADE: bad", decision="NEW_QUESTION")
    use_llm(monkeypatch, llm)
    start(coach)
    turn = coach.submit_answer("t", "")
    assert turn.question == "Next question"
    assert [e["question_id"] for e in coach.evaluation_report("t")] == ["q1"]


def test_empty_answer_moves_on_instead_of_probing_an_earlier_question(coach, monkeypatch):
    class RecordingLLM(PromptRoutedLLM):
        follow_up_prompts = []

        def invoke(self, prompt):
            if "Probe deeper" in prompt:
                self.follow_up_prompts.append(prompt)
            return super().invoke(prompt)

    serve(
        monkeypatch,
        [question("q1", "Q-ONE")],
        [question("q2", "Q-TWO")],
        [question("q3", "Q-THREE")],
    )
    llm = RecordingLLM()
    use_llm(monkeypatch, llm)
    start(coach)
    for _ in range(ic.MAX_FOLLOW_UPS_PER_QUESTION + 1):
        coach.submit_answer("t", "vague")
    follow_ups_so_far = len(llm.follow_up_prompts)
    assert coach.submit_answer("t", "").question == "Q-THREE"
    assert len(llm.follow_up_prompts) == follow_ups_so_far


def test_follow_up_grade_updates_the_topic_history(coach, monkeypatch):
    serve(monkeypatch, [question("q1")], [question("q2")])
    llm = PromptRoutedLLM()
    use_llm(monkeypatch, llm)
    start(coach)
    coach.submit_answer("t", "vague")
    llm.replies["Evaluate this answer"] = "GRADE: excellent"
    coach.submit_answer("t", "precise")
    snapshot = ic.get_interview_graph().get_state({"configurable": {"thread_id": "t"}})
    assert snapshot.values["topic_performance"]["system design"][-1]["evaluation"] == "excellent"


def test_summary_falls_back_when_the_llm_fails(coach, monkeypatch):
    class SummaryDown(PromptRoutedLLM):
        def invoke(self, prompt):
            if "Summarize" in prompt:
                raise RuntimeError("provider down")
            return super().invoke(prompt)

    serve(monkeypatch, [question("q1")])
    use_llm(monkeypatch, SummaryDown(grade=GOOD))
    start(coach)
    assert (
        coach.submit_answer("t", "answer").summary
        == "Interview complete!\n\nCompleted 1 questions."
    )


def test_a_follow_up_that_cannot_be_written_moves_on_to_the_next_question(coach, monkeypatch):
    class FollowUpDown(PromptRoutedLLM):
        def invoke(self, prompt):
            if "Probe deeper" in prompt:
                raise RuntimeError("provider down")
            return super().invoke(prompt)

    serve(monkeypatch, [question("q1")], [question("q2", "Next question")])
    use_llm(monkeypatch, FollowUpDown())
    start(coach)
    turn = coach.submit_answer("t", "vague")
    assert (turn.question, turn.question_count) == ("Next question", 2)
    next_answer = coach.submit_answer("t", "an answer to the next question")
    assert [e["question_id"] for e in coach.evaluation_report("t")] == ["q1", "q2"]
    assert next_answer.question_count >= 2

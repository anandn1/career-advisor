import pytest

from app.agents.ingestion import seed_interview_db as seeder
from app.core.config import get_settings


def parse(tmp_path, text: str, name: str = "google.txt", newline: str = "\n"):
    path = tmp_path / name
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    return list(seeder.InterviewQuestionParser(path).parse())


TWO_SECTIONS = """#TOPIC: System Design
#DIFFICULTY: Easy
Design a URL shortener.
---
Design a rate limiter.
---

#DIFFICULTY: Hard
Design a global cache.
"""


def test_questions_are_split_on_separator_lines(tmp_path):
    texts = [d.page_content for d in parse(tmp_path, TWO_SECTIONS)]
    assert texts == [
        "Design a URL shortener.",
        "Design a rate limiter.",
        "Design a global cache.",
    ]


def test_last_question_of_a_section_has_no_trailing_separator(tmp_path):
    assert not any(d.page_content.endswith("---") for d in parse(tmp_path, TWO_SECTIONS))


def test_windows_line_endings_are_handled(tmp_path):
    texts = [d.page_content for d in parse(tmp_path, TWO_SECTIONS, newline="\r\n")]
    assert texts == [
        "Design a URL shortener.",
        "Design a rate limiter.",
        "Design a global cache.",
    ]


def test_topic_carries_over_to_later_sections(tmp_path):
    documents = parse(tmp_path, TWO_SECTIONS)
    assert [(d.metadata["topic"], d.metadata["difficulty"]) for d in documents] == [
        ("system design", "easy"),
        ("system design", "easy"),
        ("system design", "hard"),
    ]


def test_company_defaults_to_the_file_name(tmp_path):
    assert {d.metadata["company"] for d in parse(tmp_path, TWO_SECTIONS, "Netflix.txt")} == {
        "netflix"
    }


def test_inline_comment_after_metadata_is_ignored(tmp_path):
    text = "#TOPIC: Arrays\n#DIFFICULTY: Medium    # topic persists from above\nTwo sum.\n"
    assert parse(tmp_path, text)[0].metadata["difficulty"] == "medium"


def test_hash_inside_a_metadata_value_is_kept(tmp_path):
    text = "#TOPIC: C#\n#DIFFICULTY: Easy\nWhat is LINQ?\n"
    assert parse(tmp_path, text)[0].metadata["topic"] == "c#"


def test_comment_lines_are_dropped(tmp_path):
    text = "#TOPIC: Arrays\n#DIFFICULTY: Easy\n#Relevant at all levels\nTwo sum.\n"
    assert parse(tmp_path, text)[0].page_content == "Two sum."


def test_code_directives_inside_questions_are_kept(tmp_path):
    text = "#TOPIC: C\n#DIFFICULTY: Easy\nWhat does this print?\n#include <stdio.h>\nint main(){}\n"
    assert "#include <stdio.h>" in parse(tmp_path, text)[0].page_content


def test_sections_missing_metadata_are_skipped(tmp_path):
    text = "#TOPIC: Arrays\nNo difficulty yet.\n#DIFFICULTY: Easy\nTwo sum.\n"
    assert [d.page_content for d in parse(tmp_path, text)] == ["Two sum."]


def test_question_ids_are_unique(tmp_path):
    ids = [d.metadata["question_id"] for d in parse(tmp_path, TWO_SECTIONS)]
    assert len(ids) == len(set(ids))


def test_dash_inside_a_question_is_not_a_separator(tmp_path):
    text = "#TOPIC: A\n#DIFFICULTY: Easy\nCompare a---b and c --- d.\n"
    assert parse(tmp_path, text)[0].page_content == "Compare a---b and c --- d."


def test_bundled_question_files_parse_cleanly():
    documents = seeder.load_documents_from_files()
    assert documents and not any(d.page_content.endswith("---") for d in documents)


# --- Seeding the store ---


@pytest.fixture
def existing_store():
    store = get_settings().interview_chroma_dir
    store.mkdir(parents=True)
    (store / "marker").write_text("keep me")
    return store


@pytest.mark.parametrize("make_dir", [False, True])
def test_seeding_without_questions_keeps_the_existing_store(make_dir, existing_store, tmp_path):
    source = tmp_path / "empty-source"
    if make_dir:
        source.mkdir()
    seeder.seed_database(data_dir=source)
    assert (existing_store / "marker").exists()


def test_dry_run_keeps_the_existing_store(existing_store, monkeypatch):
    monkeypatch.setattr(seeder, "get_embeddings", lambda: pytest.fail("embeddings loaded"))
    seeder.seed_database(dry_run=True)
    assert (existing_store / "marker").exists()


def test_missing_store_has_no_questions(tmp_path):
    assert seeder.store_has_questions(tmp_path / "absent") is False


def test_empty_store_directory_has_no_questions(tmp_path):
    (tmp_path / "store").mkdir()
    assert seeder.store_has_questions(tmp_path / "store") is False


def test_store_with_a_document_has_questions(tmp_path):
    client = seeder.chromadb.PersistentClient(path=str(tmp_path / "store"))
    client.create_collection("questions").add(
        ids=["q1"], documents=["Two sum?"], embeddings=[[0.1, 0.2]]
    )
    assert seeder.store_has_questions(tmp_path / "store") is True


class FakeChroma:
    """Writes a marker file into the store it 'builds', or fails on demand."""

    fail = False

    def __init__(self, persist_directory, embedding_function):
        self._collection = type("Collection", (), {"count": staticmethod(lambda: 1)})()

    @staticmethod
    def from_documents(documents, embedding, persist_directory):
        if FakeChroma.fail:
            raise RuntimeError("embedding model download failed")
        (seeder.Path(persist_directory) / "new-store").write_text(str(len(documents)))

    def similarity_search(self, query, k):
        return []


@pytest.fixture
def fake_chroma(monkeypatch):
    FakeChroma.fail = False
    monkeypatch.setattr(seeder, "Chroma", FakeChroma)
    monkeypatch.setattr(seeder, "get_embeddings", lambda: None)
    return FakeChroma


def test_failed_reseed_keeps_the_existing_store(existing_store, fake_chroma):
    fake_chroma.fail = True
    with pytest.raises(RuntimeError):
        seeder.seed_database()
    assert (existing_store / "marker").exists()


def test_failed_reseed_leaves_no_staging_directory(existing_store, fake_chroma):
    fake_chroma.fail = True
    with pytest.raises(RuntimeError):
        seeder.seed_database()
    assert [p.name for p in existing_store.parent.iterdir()] == [existing_store.name]


def test_successful_reseed_replaces_the_store(existing_store, fake_chroma):
    seeder.seed_database()
    assert not (existing_store / "marker").exists() and (existing_store / "new-store").exists()


def test_reseed_creates_the_store_when_none_exists(fake_chroma):
    seeder.seed_database()
    assert (get_settings().interview_chroma_dir / "new-store").exists()


def test_store_check_does_not_create_files(tmp_path):
    (tmp_path / "store").mkdir()
    seeder.store_has_questions(tmp_path / "store")
    assert list((tmp_path / "store").iterdir()) == []


def test_comment_starting_with_a_directive_word_is_dropped(tmp_path):
    text = "#TOPIC: A\n#DIFFICULTY: Easy\n# if stuck, give a hint\nTwo sum.\n"
    assert parse(tmp_path, text)[0].page_content == "Two sum."


def test_indented_metadata_lines_are_recognized(tmp_path):
    text = "  #TOPIC: Graphs\n\t#DIFFICULTY: Hard\nShortest path.\n"
    document = parse(tmp_path, text)[0]
    assert (document.page_content, document.metadata["topic"]) == ("Shortest path.", "graphs")


def test_store_with_only_an_empty_collection_has_no_questions(tmp_path):
    client = seeder.chromadb.PersistentClient(path=str(tmp_path / "store"))
    client.create_collection("questions")
    assert seeder.store_has_questions(tmp_path / "store") is False

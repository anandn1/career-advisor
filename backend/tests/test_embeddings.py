"""The ONNX embedding model: verified downloads and how texts are embedded.

Offline: downloads, the ONNX session and the tokenizer are fakes.
"""

import hashlib
import math
from types import SimpleNamespace

import numpy as np
import pytest

from app.core import embeddings
from app.core.embeddings import (
    DOCUMENT_PROMPT,
    QUERY_PROMPT,
    GemmaOnnxEmbeddings,
    ModelIntegrityError,
    fetch_model_files,
)

# Nested like the real files, in the order they are fetched.
GOOD = {
    "onnx/model.onnx": b"graph",
    "onnx/model.onnx_data": b"weights",
    "tokenizer.json": b"vocabulary",
}
MODEL, DATA, TOKENIZER = GOOD


def fingerprint(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_files(model_dir, files: dict[str, bytes]) -> None:
    for name, data in files.items():
        (model_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (model_dir / name).write_bytes(data)


@pytest.fixture
def pinned(monkeypatch):
    """Pin the fake files, and record every download as (name, revision)."""
    monkeypatch.setattr(
        embeddings,
        "FINGERPRINTS",
        {name: fingerprint(data) for name, data in GOOD.items()},
    )
    served = dict(GOOD)
    downloads = []

    def fake_download(repo_id, name, *, revision, local_dir, force_download):
        assert repo_id == embeddings.REPO_ID
        assert force_download
        downloads.append((name, revision))
        write_files(local_dir, {name: served[name]})

    monkeypatch.setattr(embeddings, "hf_hub_download", fake_download)
    return SimpleNamespace(served=served, downloads=downloads)


def test_every_file_the_model_loads_is_pinned():
    loaded = {
        embeddings.MODEL_FILE,
        embeddings.MODEL_FILE + "_data",
        embeddings.TOKENIZER_FILE,
    }

    assert loaded <= embeddings.FINGERPRINTS.keys()


def test_missing_files_are_downloaded_from_the_pinned_revision(pinned, tmp_path):
    fetch_model_files(tmp_path)

    assert pinned.downloads == [(name, embeddings.REVISION) for name in GOOD]
    assert (tmp_path / DATA).read_bytes() == b"weights"


def test_verified_files_are_not_downloaded_again(pinned, tmp_path):
    write_files(tmp_path, GOOD)

    fetch_model_files(tmp_path)

    assert pinned.downloads == []


def test_a_file_that_changed_on_disk_is_downloaded_again(pinned, tmp_path):
    write_files(tmp_path, {**GOOD, DATA: b"tampered"})

    fetch_model_files(tmp_path)

    assert pinned.downloads == [(DATA, embeddings.REVISION)]
    assert (tmp_path / DATA).read_bytes() == b"weights"


def test_a_download_that_does_not_match_is_deleted_and_refused(pinned, tmp_path):
    pinned.served[DATA] = b"swapped upstream"

    with pytest.raises(ModelIntegrityError, match="model.onnx_data"):
        fetch_model_files(tmp_path)

    assert not (tmp_path / DATA).exists()


def test_a_refused_download_stops_before_the_next_file_and_keeps_earlier_ones(pinned, tmp_path):
    pinned.served[DATA] = b"swapped upstream"

    with pytest.raises(ModelIntegrityError):
        fetch_model_files(tmp_path)

    assert pinned.downloads == [
        (MODEL, embeddings.REVISION),
        (DATA, embeddings.REVISION),
    ]
    assert (tmp_path / MODEL).read_bytes() == b"graph"


def test_a_download_that_leaves_no_file_is_refused(pinned, monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings, "hf_hub_download", lambda *args, **kwargs: None)

    with pytest.raises(ModelIntegrityError, match="model.onnx"):
        fetch_model_files(tmp_path)


class FakeTokenizer:
    """Token ids are the characters' code points, padded with 0 to the batch's longest text."""

    def __init__(self):
        self.setup = []
        self.batches = []

    def enable_padding(self):
        self.setup.append("padding")

    def enable_truncation(self, max_length):
        self.setup.append(("truncation", max_length))

    def encode_batch(self, texts):
        self.batches.append(texts)
        width = max(len(text) for text in texts)
        return [
            SimpleNamespace(
                ids=[ord(c) for c in text] + [0] * (width - len(text)),
                attention_mask=[1] * len(text) + [0] * (width - len(text)),
            )
            for text in texts
        ]


class FakeSession:
    """An unnormalised vector per text, (real tokens, 10), so its direction depends on length.

    Records every input it is fed.
    """

    def __init__(self):
        self.feeds = []

    def run(self, outputs, feed):
        assert outputs == [embeddings.OUTPUT]
        self.feeds.append(feed)
        mask = feed["attention_mask"]
        return [np.stack([mask.sum(axis=1), np.full(len(mask), 10)], axis=1)]


def expected_vector(text: str) -> list[float]:
    raw = [len(text), 10]
    norm = math.hypot(*raw)
    return [value / norm for value in raw]


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture
def model(session, tokenizer):
    return GemmaOnnxEmbeddings(session, tokenizer)


def test_the_tokenizer_pads_batches_and_truncates_to_the_model_limit(model, tokenizer):
    assert tokenizer.setup == ["padding", ("truncation", embeddings.MAX_TOKENS)]


def test_documents_get_the_document_prompt(model, tokenizer):
    model.embed_documents(["Python developer"])

    assert tokenizer.batches == [[DOCUMENT_PROMPT + "Python developer"]]


def test_queries_get_the_query_prompt(model, tokenizer):
    model.embed_query("backend jobs")

    assert tokenizer.batches == [[QUERY_PROMPT + "backend jobs"]]


def test_a_query_returns_one_normalised_vector(model):
    assert model.embed_query("ab") == pytest.approx(expected_vector(QUERY_PROMPT + "ab"))


def test_the_model_is_fed_token_ids_and_mask_as_int64(model, session):
    model.embed_documents(["a", "abc"])

    (feed,) = session.feeds
    assert feed.keys() == {"input_ids", "attention_mask"}
    assert feed["input_ids"].dtype == feed["attention_mask"].dtype == np.int64
    assert feed["input_ids"][1].tolist() == [ord(c) for c in DOCUMENT_PROMPT + "abc"]


def test_vectors_come_back_in_input_order_and_normalised(model):
    texts = ["a much longer text than the others", "b", "medium text"]

    vectors = model.embed_documents(texts)

    np.testing.assert_allclose(vectors, [expected_vector(DOCUMENT_PROMPT + t) for t in texts])


def test_texts_are_batched_shortest_first(model, tokenizer, monkeypatch):
    monkeypatch.setattr(embeddings, "BATCH_SIZE", 2)

    model.embed_documents(["ccc", "a", "dddd", "bb", "eeeee"])

    assert [[t.removeprefix(DOCUMENT_PROMPT) for t in b] for b in tokenizer.batches] == [
        ["a", "bb"],
        ["ccc", "dddd"],
        ["eeeee"],
    ]


def test_a_model_that_returns_too_few_vectors_is_an_error(model, session, monkeypatch):
    real_run = session.run
    monkeypatch.setattr(session, "run", lambda outputs, feed: [real_run(outputs, feed)[0][:1]])

    with pytest.raises(ValueError):
        model.embed_documents(["a", "b"])


def test_no_texts_means_no_vectors_and_no_model_call(model, tokenizer):
    assert model.embed_documents([]) == []
    assert tokenizer.batches == []


def test_load_verifies_the_files_before_opening_them(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(
        embeddings, "fetch_model_files", lambda path: events.append(("fetch", path))
    )
    monkeypatch.setattr(
        embeddings.onnxruntime,
        "InferenceSession",
        lambda path, providers: events.append(("session", path, providers)) or FakeSession(),
    )
    monkeypatch.setattr(
        embeddings,
        "Tokenizer",
        SimpleNamespace(
            from_file=lambda path: events.append(("tokenizer", path)) or FakeTokenizer()
        ),
    )

    assert isinstance(GemmaOnnxEmbeddings.load(tmp_path), GemmaOnnxEmbeddings)
    assert events == [
        ("fetch", tmp_path),
        ("session", str(tmp_path / embeddings.MODEL_FILE), ["CPUExecutionProvider"]),
        ("tokenizer", str(tmp_path / embeddings.TOKENIZER_FILE)),
    ]

"""EmbeddingGemma 300M as a local 4-bit ONNX model: no PyTorch and no Hugging Face token.

The files come from one pinned commit of the ONNX conversion and are checked against
SHA-256 fingerprints before use, so a changed or corrupted file is never loaded.
"""

import hashlib
import logging
from pathlib import Path

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from langchain_core.embeddings import Embeddings
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

REPO_ID = "onnx-community/embeddinggemma-300m-ONNX"
REVISION = "5090578d9565bb06545b4552f76e6bc2c93e4a66"
MODEL_FILE = "onnx/model_q4.onnx"
TOKENIZER_FILE = "tokenizer.json"
FINGERPRINTS = {
    MODEL_FILE: "ad1dfee81a70f7944b9b9d1cc6e48075b832881cf33fab2f2b248be78f3f0043",
    "onnx/model_q4.onnx_data": "599962c3143b040de2dd05e5975be3e9091dd067cacc6a8f7186e3203bab9e02",
    TOKENIZER_FILE: "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47",
}

# EmbeddingGemma is trained with these task prompts; search quality drops without them.
QUERY_PROMPT = "task: search result | query: "
DOCUMENT_PROMPT = "title: none | text: "
MAX_TOKENS = 2048
BATCH_SIZE = 8
OUTPUT = "sentence_embedding"


class ModelIntegrityError(RuntimeError):
    """A downloaded model file does not match its pinned fingerprint."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_model_files(model_dir: Path) -> None:
    """Make sure every pinned file is in model_dir and matches its fingerprint.

    Verified files are kept; missing or mismatched ones are downloaded again.
    """
    for name, expected in FINGERPRINTS.items():
        path = model_dir / name
        if path.exists() and sha256(path) == expected:
            continue
        logger.info("Downloading %s from %s", name, REPO_ID)
        # local_dir gives real files: onnxruntime refuses external data behind cache links.
        hf_hub_download(REPO_ID, name, revision=REVISION, local_dir=model_dir, force_download=True)
        if not path.is_file() or sha256(path) != expected:
            path.unlink(missing_ok=True)
            raise ModelIntegrityError(f"{name} from {REPO_ID} does not match its fingerprint")


class GemmaOnnxEmbeddings(Embeddings):
    def __init__(self, session: onnxruntime.InferenceSession, tokenizer: Tokenizer):
        self.session = session
        self.tokenizer = tokenizer
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)

    @classmethod
    def load(cls, model_dir: Path) -> "GemmaOnnxEmbeddings":
        """Fetch and verify the model files, then load them."""
        fetch_model_files(model_dir)
        session = onnxruntime.InferenceSession(
            str(model_dir / MODEL_FILE), providers=["CPUExecutionProvider"]
        )
        return cls(session, Tokenizer.from_file(str(model_dir / TOKENIZER_FILE)))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([DOCUMENT_PROMPT + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([QUERY_PROMPT + text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # Batching texts of similar length keeps padding, and so time and memory, low.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vectors: list = [None] * len(texts)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            for index, vector in zip(
                batch, self._embed_batch([texts[i] for i in batch]), strict=True
            ):
                vectors[index] = vector
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        encodings = self.tokenizer.encode_batch(texts)
        feed = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        (vectors,) = self.session.run([OUTPUT], feed)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors.tolist()

import re
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.resources import get_embeddings

CHROMA_DB_FILE = "chroma.sqlite3"
DATA_SOURCE_DIR = Path(__file__).resolve().parent / "interview_rag_data"
# A line holding only '---' separates questions (trailing spaces allowed).
QUESTION_DELIMITER = re.compile(r"^[ \t]*---[ \t]*$", re.MULTILINE)
# C-family directives inside a question are code, not comments.
CODE_DIRECTIVE = re.compile(r"#\s*(include|define|undef|ifn?def|if|elif|else|endif|pragma)\b")
# '#DIFFICULTY: Medium    # note' -> 'Medium'; a '#' with no space before it (C#) is kept.
INLINE_COMMENT = re.compile(r"\s+#.*$")


class InterviewQuestionParser:
    """
    Parses .txt files with the format:

    #TOPIC: Data Structures & Algorithms
    #DIFFICULTY: Easy
    Question 1
    ---
    Question 2

    #DIFFICULTY: Medium    # Topic persists from above
    Question 3
    ---
    Question 4
    """

    # Regex patterns
    METADATA_PATTERN = re.compile(
        r"^#\s*(?P<key>COMPANY|TOPIC|DIFFICULTY):\s*(?P<value>.+)$", re.IGNORECASE | re.MULTILINE
    )
    SECTION_SPLIT_PATTERN = re.compile(r"\n(?=#\s*(?:COMPANY|TOPIC|DIFFICULTY):)", re.IGNORECASE)

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.default_company = self._get_default_company()

    def _get_default_company(self) -> str:
        """Extract default company from filename."""
        return self.file_path.stem.lower()  # 'generic', 'google', 'amazon', etc.

    def _validate_metadata(self, meta: dict[str, str], section_idx: int) -> bool:
        """Validate required metadata fields."""
        required = ["topic", "difficulty"]
        missing = [field for field in required if not meta.get(field)]

        if missing:
            print(
                f"Warning [section {section_idx}]: skipping questions, missing {', '.join(missing)}"
            )
            print(
                f"   Current: company={meta.get('company')!r}, topic={meta.get('topic')!r}, "
                f"difficulty={meta.get('difficulty')!r}"
            )
            return False
        return True

    def _generate_id(self, meta: dict[str, str], index: int) -> str:
        """Generate a unique question ID."""
        company = meta.get("company", "unknown")
        topic = meta.get("topic", "unknown")
        difficulty = meta.get("difficulty", "unknown")
        # Use file stem + counter for true uniqueness
        return f"{self.file_path.stem}_{company}_{topic}_{difficulty}_{index}"

    def _preprocess_content(self, content: str) -> str:
        """Remove non-metadata comment lines."""
        lines = content.split("\n")
        filtered = []

        for _line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            is_comment = stripped.startswith("#") and not (
                self.METADATA_PATTERN.match(stripped) or CODE_DIRECTIVE.match(stripped)
            )
            if is_comment:
                continue
            filtered.append(line)

        return "\n".join(filtered)

    def parse(self) -> Generator[Document, None, None]:
        """Parse the file and yield Document objects."""
        try:
            raw_content = self.file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"❌ Error reading {self.file_path}: {e}")
            return

        content = self._preprocess_content(raw_content.replace("\r\n", "\n"))

        # Split into sections
        sections = self.SECTION_SPLIT_PATTERN.split(content)
        if not sections:
            return

        # Initialize state (persists across sections)
        current_meta = {"company": self.default_company}
        question_counter = 0

        for section_idx, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            # Extract metadata from this section
            meta_matches = list(self.METADATA_PATTERN.finditer(section))

            # Update metadata incrementally (key feature for your format)
            for match in meta_matches:
                key = match.group("key").lower()
                value = INLINE_COMMENT.sub("", match.group("value")).strip().lower()
                current_meta[key] = value

            # Remove metadata lines to isolate questions
            question_block = self.METADATA_PATTERN.sub("", section).strip()

            # Skip if no questions
            if not question_block:
                continue

            # Validate required metadata
            if not self._validate_metadata(current_meta, section_idx):
                continue

            # Split questions and create documents
            questions = [q.strip() for q in QUESTION_DELIMITER.split(question_block) if q.strip()]

            for q_text in questions:
                if not q_text:
                    continue

                question_counter += 1
                doc_id = self._generate_id(current_meta, question_counter)

                yield Document(
                    page_content=q_text,
                    metadata={
                        "company": current_meta["company"],
                        "topic": current_meta["topic"],
                        "difficulty": current_meta["difficulty"],
                        "question_id": doc_id,
                        "source_file": self.file_path.name,
                    },
                )

        print(f"✓ Loaded {question_counter} questions from {self.file_path.name}")


def load_documents_from_files(data_dir: Path = DATA_SOURCE_DIR) -> list[Document]:
    """Load all interview question documents from .txt files."""

    if not data_dir.exists():
        print(f" Directory not found: {data_dir}")
        return []

    file_paths = list(data_dir.glob("*.txt"))
    if not file_paths:
        print(f"  No .txt files found in {data_dir}")
        return []

    all_documents = []
    for file_path in file_paths:
        parser = InterviewQuestionParser(file_path)
        documents = list(parser.parse())
        all_documents.extend(documents)

    print(f"\n Total documents loaded: {len(all_documents)}")
    return all_documents


def store_has_questions(chroma_dir: Path) -> bool:
    """True if the store exists and holds at least one document."""
    if not (chroma_dir / CHROMA_DB_FILE).exists():
        return False
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return any(collection.count() for collection in client.list_collections())


def seed_database(dry_run: bool = False, data_dir: Path = DATA_SOURCE_DIR) -> None:
    """Replace the vector store with the questions in `data_dir` (kept if none load)."""
    chroma_dir = get_settings().interview_chroma_dir
    print("📄 Loading new documents from files...")
    documents_to_add = load_documents_from_files(data_dir)

    if not documents_to_add:
        print(" No documents loaded. Existing store left untouched.")
        return

    if dry_run:
        print("\n🎯 DRY RUN - Would have added:")
        for doc in documents_to_add[:5]:
            print(f"  - {doc.metadata['question_id']}: {doc.page_content[:60]}...")
        if len(documents_to_add) > 5:
            print(f"  ... and {len(documents_to_add) - 5} more")
        return

    print(" Building the new vector store...")
    chroma_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".seeding-", dir=chroma_dir.parent))
    try:
        Chroma.from_documents(
            documents=documents_to_add,
            embedding=get_embeddings(),
            persist_directory=str(staging_dir),
        )
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    # Only now, with the new store complete, replace the old one.
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    staging_dir.rename(chroma_dir)
    db = Chroma(persist_directory=str(chroma_dir), embedding_function=get_embeddings())

    print(f"\n🎉 Successfully seeded {len(documents_to_add)} documents!")
    print(f" Database saved to {chroma_dir}")

    # Verification
    count = db._collection.count()
    print(f" Verification: {count} documents in ChromaDB")

    # Test similarity search
    print("\n Testing similarity search...")
    results = db.similarity_search("design a system for", k=3)
    for i, doc in enumerate(results):
        print(f"    {i + 1}. {doc.metadata['topic']} | {doc.metadata['difficulty']}")


def verify_parsing():
    """Debug helper: Parse and display documents without seeding."""
    print(" VERIFICATION MODE - Parsing documents only\n")
    docs = load_documents_from_files()

    if not docs:
        return

    # Summary by file
    from collections import Counter

    file_counter = Counter([doc.metadata["source_file"] for doc in docs])
    print("\n📊 Summary by file:")
    for fname, count in file_counter.items():
        print(f"    {fname}: {count} questions")

    # Sample documents
    print("\n📄 Sample parsed documents:")
    for i, doc in enumerate(docs[:10]):
        print(f"\n--- Document {i + 1} ---")
        print(f"ID: {doc.metadata['question_id']}")
        print(
            f"Company: {doc.metadata['company']} | Topic: {doc.metadata['topic']} | "
            f"Difficulty: {doc.metadata['difficulty']}"
        )
        print(f"Content: {doc.page_content[:100]}...")

    print(f"\n✅ Total: {len(docs)} documents parsed successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed interview question vector database")
    parser.add_argument(
        "--dry-run", action="store_true", help="Parse and validate without writing to DB"
    )
    parser.add_argument("--verify", action="store_true", help="Verify document parsing only")
    parser.add_argument(
        "--if-empty", action="store_true", help="Seed only when the store does not exist yet"
    )

    args = parser.parse_args()

    if args.verify:
        verify_parsing()
    elif args.if_empty and store_has_questions(get_settings().interview_chroma_dir):
        print("Interview question store already exists; skipping seeding.")
    else:
        seed_database(dry_run=args.dry_run)

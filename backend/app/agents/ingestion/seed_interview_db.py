import os
import shutil
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Generator
from langchain_core.documents import Document
from langchain_chroma import Chroma

# --- Import shared settings ---
# Use the script's location to determine project root
try:
    # Get the directory where this script is located
    SCRIPT_DIR = Path(__file__).resolve().parent 
    # Go up two levels (e.g., from .../app/agents/ingestion to .../app)
    PROJECT_ROOT = (SCRIPT_DIR / ".." / "..").resolve()
    sys.path.append(str(PROJECT_ROOT))
    
    from agents.settings import embedding_function
    
except ImportError:
    print("❌ Error: Could not import embedding_function from agents.settings.")
    print("   Ensure your project structure and PYTHONPATH are correct.")
    print(f"   Based on this script, tried to add: {PROJECT_ROOT}")
    sys.exit(1)
except NameError:
    # Handle case where __file__ is not defined (e.g., in a REPL)
    print("❌ Error: Could not determine script directory. Are you running this in a REPL?")
    sys.exit(1)


# --- Configuration ---
# Use paths relative to the script's directory
DATA_SOURCE_DIR = SCRIPT_DIR / "interview_rag_data"
INTERVIEW_CHROMA_DIR = SCRIPT_DIR / "chroma_db_interview_questions"
QUESTION_DELIMITER = "\n---\n"


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
    METADATA_PATTERN = re.compile(r'^#\s*(?P<key>COMPANY|TOPIC|DIFFICULTY):\s*(?P<value>.+)$', re.IGNORECASE | re.MULTILINE)
    SECTION_SPLIT_PATTERN = re.compile(r'\n(?=#\s*(?:COMPANY|TOPIC|DIFFICULTY):)', re.IGNORECASE)
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.default_company = self._get_default_company()
        
    def _get_default_company(self) -> str:
        """Extract default company from filename."""
        return self.file_path.stem.lower()   # 'generic', 'google', 'amazon', etc.
    
    def _validate_metadata(self, meta: Dict[str, str], section_idx: int) -> bool:
        """Validate required metadata fields."""
        required = ['topic', 'difficulty']
        missing = [field for field in required if not meta.get(field)]
        
        if missing:
            print(f"⚠️  Warning [Section {section_idx}]: Skipping questions - missing: {', '.join(missing)}")
            print(f"   Current state: company='{meta.get('company')}', topic='{meta.get('topic')}', difficulty='{meta.get('difficulty')}'")
            return False
        return True
    
    def _generate_id(self, meta: Dict[str, str], index: int) -> str:
        """Generate a unique question ID."""
        company = meta.get('company', 'unknown')
        topic = meta.get('topic', 'unknown')
        difficulty = meta.get('difficulty', 'unknown')
        # Use file stem + counter for true uniqueness
        return f"{self.file_path.stem}_{company}_{topic}_{difficulty}_{index}"
    
    def _preprocess_content(self, content: str) -> str:
        """Remove non-metadata comment lines."""
        lines = content.split('\n')
        filtered = []
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip lines that start with # but aren't valid metadata
            if stripped.startswith('#') and not self.METADATA_PATTERN.match(stripped):
                
                continue
            filtered.append(line)
        
        return '\n'.join(filtered)
    
    def parse(self) -> Generator[Document, None, None]:
        """Parse the file and yield Document objects."""
        try:
            raw_content = self.file_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"❌ Error reading {self.file_path}: {e}")
            return
        
        # Preprocess to remove non-metadata comments
        content = self._preprocess_content(raw_content)
        
        # Split into sections
        sections = self.SECTION_SPLIT_PATTERN.split(content)
        if not sections:
            return
        
        # Initialize state (persists across sections)
        current_meta = {'company': self.default_company}
        question_counter = 0
        
        for section_idx, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            
            # Extract metadata from this section
            meta_matches = list(self.METADATA_PATTERN.finditer(section))
            
            # Update metadata incrementally (key feature for your format)
            for match in meta_matches:
                key = match.group('key').lower()
                value = match.group('value').strip().lower()
                current_meta[key] = value
            
            # Remove metadata lines to isolate questions
            question_block = self.METADATA_PATTERN.sub('', section).strip()
            
            # Skip if no questions
            if not question_block:
                continue
            
            # Validate required metadata
            if not self._validate_metadata(current_meta, section_idx):
                continue
            
            # Split questions and create documents
            questions = [q.strip() for q in question_block.split(QUESTION_DELIMITER) if q.strip()]
            
            for q_text in questions:
                if not q_text:
                    continue
                
                question_counter += 1
                doc_id = self._generate_id(current_meta, question_counter)
                
                yield Document(
                    page_content=q_text,
                    metadata={
                        "company": current_meta['company'],
                        "topic": current_meta['topic'],
                        "difficulty": current_meta['difficulty'],
                        "question_id": doc_id,
                        "source_file": self.file_path.name
                    }
                )
        
        print(f"✓ Loaded {question_counter} questions from {self.file_path.name}")


def load_documents_from_files() -> List[Document]:
    """Load all interview question documents from .txt files."""
    
    
    if not DATA_SOURCE_DIR.exists():
        print(f" Directory not found: {DATA_SOURCE_DIR}")
        return []
    
    file_paths = list(DATA_SOURCE_DIR.glob("*.txt"))
    if not file_paths:
        print(f"  No .txt files found in {DATA_SOURCE_DIR}")
        return []
    
    all_documents = []
    for file_path in file_paths:
        parser = InterviewQuestionParser(file_path)
        documents = list(parser.parse())
        all_documents.extend(documents)
    
    print(f"\n Total documents loaded: {len(all_documents)}")
    return all_documents


def seed_database(dry_run: bool = False) -> None:
    """Clear and re-populate the vector store."""
    if not dry_run and INTERVIEW_CHROMA_DIR.exists():
        print(f" Clearing existing database at {INTERVIEW_CHROMA_DIR}...")
        shutil.rmtree(INTERVIEW_CHROMA_DIR)
    
    print("📄 Loading new documents from files...")
    documents_to_add = load_documents_from_files()
    
    if not documents_to_add:
        print(" No documents loaded. Seeding process stopped.")
        return
    
    if dry_run:
        print("\n🎯 DRY RUN - Would have added:")
        for doc in documents_to_add[:5]:
            print(f"  - {doc.metadata['question_id']}: {doc.page_content[:60]}...")
        if len(documents_to_add) > 5:
            print(f"  ... and {len(documents_to_add) - 5} more")
        return
    
    print(" Creating new vector store and adding documents...")
    db = Chroma.from_documents(
        documents=documents_to_add,
        embedding=embedding_function,
        persist_directory=str(INTERVIEW_CHROMA_DIR)
    )
    
    print(f"\n🎉 Successfully seeded {len(documents_to_add)} documents!")
    print(f" Database saved to {INTERVIEW_CHROMA_DIR}")
    
    # Verification
    count = db._collection.count()
    print(f" Verification: {count} documents in ChromaDB")
    
    # Test similarity search
    print("\n Testing similarity search...")
    results = db.similarity_search("design a system for", k=3)
    for i, doc in enumerate(results):
        print(f"    {i+1}. {doc.metadata['topic']} | {doc.metadata['difficulty']}")


def verify_parsing():
    """Debug helper: Parse and display documents without seeding."""
    print(" VERIFICATION MODE - Parsing documents only\n")
    docs = load_documents_from_files()
    
    if not docs:
        return
    
    # Summary by file
    from collections import Counter
    file_counter = Counter([doc.metadata['source_file'] for doc in docs])
    print("\n📊 Summary by file:")
    for fname, count in file_counter.items():
        print(f"    {fname}: {count} questions")
    
    # Sample documents
    print("\n📄 Sample parsed documents:")
    for i, doc in enumerate(docs[:10]):
        print(f"\n--- Document {i+1} ---")
        print(f"ID: {doc.metadata['question_id']}")
        print(f"Company: {doc.metadata['company']} | Topic: {doc.metadata['topic']} | Difficulty: {doc.metadata['difficulty']}")
        print(f"Content: {doc.page_content[:100]}...")
    
    print(f"\n✅ Total: {len(docs)} documents parsed successfully")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Seed interview question vector database")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate without writing to DB")
    # This is the corrected line:
    parser.add_argument("--verify", action="store_true", help="Verify document parsing only")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_parsing()
    else:
        seed_database(dry_run=args.dry_run)
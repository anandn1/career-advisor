# seed_interview_db.py (enhanced with similarity search test)
import os
import shutil
import glob
import re
from pathlib import Path
from typing import List, Dict, Any, Generator
from langchain_core.documents import Document
from langchain_chroma import Chroma
import sys

# --- Import shared settings ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from agents.settings import embedding_function
except ImportError:
    print(" Error: Could not import embedding_function from settings.")
    sys.exit(1)

# --- Configuration ---
DATA_SOURCE_DIR = Path("/mnt/162B47977B82AA01/Agentic AI project/backend/app/agents/ingestion/interview_rag_data")
INTERVIEW_CHROMA_DIR = "./chroma_db_interview_questions"
QUESTION_DELIMITTER = "\n---\n"

class InterviewQuestionParser:
    """Robust parser for interview question files with metadata."""
    
    METADATA_PATTERN = re.compile(r'^#\s*(?P<key>COMPANY|TOPIC|DIFFICULTY):\s*(?P<value>.+)$', re.IGNORECASE | re.MULTILINE)
    SECTION_SPLIT_PATTERN = re.compile(r'\n(?=#\s*(?:COMPANY|TOPIC|DIFFICULTY):)', re.IGNORECASE)
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.default_company = self._get_default_company()
        
    def _get_default_company(self) -> str:
        """Extract default company from filename."""
        return self.file_path.stem.lower()
    
    def _validate_metadata(self, meta: Dict[str, str], section_idx: int) -> bool:
        """Validate required metadata fields."""
        required = ['topic', 'difficulty']
        missing = [field for field in required if not meta.get(field)]
        
        if missing:
            print(f"  Warning [Section {section_idx}]: Skipping questions - missing: {', '.join(missing)}")
            print(f"   Current state: company='{meta.get('company')}', topic='{meta.get('topic')}', difficulty='{meta.get('difficulty')}'")
            return False
        return True
    
    def _generate_id(self, meta: Dict[str, str], index: int) -> str:
        """Generate a unique question ID."""
        company = meta.get('company', 'unknown')
        topic = meta.get('topic', 'unknown')
        difficulty = meta.get('difficulty', 'unknown')
        return f"{self.file_path.stem}_{company}_{topic}_{difficulty}_{index}"
    
    def _preprocess_content(self, content: str) -> str:
        """Remove non-metadata comment lines."""
        lines = content.split('\n')
        filtered = []
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#') and not self.METADATA_PATTERN.match(stripped):
                if stripped.startswith('#Relevant'):
                    print(f"ℹ  Info [L{line_num}]: Ignoring comment: {stripped}")
                continue
            filtered.append(line)
        
        return '\n'.join(filtered)
    
    def parse(self) -> Generator[Document, None, None]:
        """Parse the file and yield Document objects."""
        try:
            raw_content = self.file_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f" Error reading {self.file_path}: {e}")
            return
        
        content = self._preprocess_content(raw_content)
        sections = self.SECTION_SPLIT_PATTERN.split(content)
        
        if not sections:
            return
        
        current_meta = {'company': self.default_company}
        question_counter = 0
        
        for section_idx, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            
            # Extract metadata
            meta_matches = list(self.METADATA_PATTERN.finditer(section))
            for match in meta_matches:
                key = match.group('key').lower()
                value = match.group('value').strip().lower()
                current_meta[key] = value
            
            # Remove metadata lines
            question_block = self.METADATA_PATTERN.sub('', section).strip()
            
            if not question_block:
                continue
            
            if not self._validate_metadata(current_meta, section_idx):
                continue
            
            # Split and create documents
            questions = [q.strip() for q in question_block.split(QUESTION_DELIMITTER) if q.strip()]
            
            for q_text in questions:
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
    print(f" Loading documents from: {DATA_SOURCE_DIR}")
    
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
    if not dry_run and Path(INTERVIEW_CHROMA_DIR).exists():
        print(f"🗑️  Clearing existing database at {INTERVIEW_CHROMA_DIR}...")
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
    
    print("💾 Creating new vector store and adding documents...")
    db = Chroma.from_documents(
        documents=documents_to_add,
        embedding=embedding_function,
        persist_directory=str(INTERVIEW_CHROMA_DIR)
    )
    
    print(f"\n Successfully seeded {len(documents_to_add)} documents!")
    print(f" Database saved to {INTERVIEW_CHROMA_DIR}")
    
    # Verification
    count = db._collection.count()
    print(f" Verification: {count} documents in ChromaDB collection")


def test_similarity_search():
    """
    Test function to verify vector store is working correctly.
    Runs sample queries and prints results.
    """
    print("\n" + "="*60)
    print(" TESTING SIMILARITY SEARCH")
    print("="*60)
    
    try:
        # Load the vector store
        db = Chroma(
            persist_directory=INTERVIEW_CHROMA_DIR,
            embedding_function=embedding_function
        )
        
        # Test queries
        test_queries = [
            "Design a URL shortener system",
            "Explain how to reverse a linked list",
            "What are ACID properties in databases",
            "Behavioral question about teamwork"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query #{i}: '{query}' ---")
            
            results = db.similarity_search_with_score(
                query,
                k=3,  # Get top 3 results
                filter={"difficulty": {"$in": ["easy", "medium"]}}  # Optional filter
            )
            
            if not results:
                print("    No results found")
                continue
            
            print(f"    Found {len(results)} results:")
            
            for j, (doc, score) in enumerate(results, 1):
                print(f"\n   --- Result #{j} (Score: {score:.3f}) ---")
                print(f"    Question: {doc.page_content[:100]}...")
                print(f"    Company: {doc.metadata.get('company', 'unknown')}")
                print(f"    Topic: {doc.metadata.get('topic', 'unknown')}")
                print(f"    Difficulty: {doc.metadata.get('difficulty', 'unknown')}")
                print(f"    ID: {doc.metadata.get('question_id', 'unknown')}")
        
        print("\n Similarity search test completed successfully!")
        
    except Exception as e:
        print(f" Error during similarity search test: {e}")
        raise


def verify_parsed_documents():
    """Debug helper: Parse and display documents without seeding."""
    print("🔍 VERIFICATION MODE - Parsing documents only\n")
    docs = load_documents_from_files()
    
    if not docs:
        return
    
    # Summary by file
    from collections import Counter
    file_counter = Counter([doc.metadata['source_file'] for doc in docs])
    print("\n Summary by file:")
    for fname, count in file_counter.items():
        print(f"   {fname}: {count} questions")
    
    # Sample documents
    print("\n Sample parsed documents:")
    for i, doc in enumerate(docs[:10]):
        print(f"\n--- Document {i+1} ---")
        print(f"ID: {doc.metadata['question_id']}")
        print(f"Company: {doc.metadata['company']} | Topic: {doc.metadata['topic']} | Difficulty: {doc.metadata['difficulty']}")
        print(f"Content: {doc.page_content[:100]}...")
    
    print(f"\n Total: {len(docs)} documents parsed successfully")


# --- Main Execution ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Seed interview question vector database")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate without writing to DB")
    parser.add_argument("--verify", action="store_true", help="Verify document parsing only")
    parser.add_argument("--test-search", action="store_true", help="Run similarity search test after seeding")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_parsed_documents()
    else:
        # Create data directory if it doesn't exist
        if not DATA_SOURCE_DIR.exists():
            os.makedirs(DATA_SOURCE_DIR)
            print(f" Created data directory: {DATA_SOURCE_DIR}")
            print("   Please add your .txt files to this directory and run again.")
        else:
            seed_database(dry_run=args.dry_run)
            
            # Run similarity search test if requested
            if args.test_search and not args.dry_run:
                test_similarity_search()
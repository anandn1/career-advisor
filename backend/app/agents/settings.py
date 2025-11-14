import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

"""
Central configuration file for shared components.

This file creates and configures the core AI components (LLM and Embedder)
that will be shared by all agents in the application.
"""
# --- 1. Load Environment Variables ---
print("Loading environment variables from .env file...")
load_dotenv()
print("Environment variables loaded.")

# --- 2. Shared LLM (The "Brain") ---
# All agents will import and use this single LLM instance.
# It loads its API key from the 'GROQ_API_KEY' variable in the .env file.
# Note: Your main.py must call `load_dotenv()` for this to work.


api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file. "
                     "Please add it to backend/.env")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
if not COHERE_API_KEY:
    print("WARNING: COHERE_API_KEY not found. Re-ranker will not function.")

DB_URL = os.environ.get("DB_URL")
if not DB_URL:
    raise ValueError("POSTGRES_DB_URI not found...")

print("Initializing shared LLM (Groq Llama 3.3 70B)...")
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    api_key=api_key
)

# --- 2. Shared Embedding Model (The "Encoder") ---
# This model is loaded once (on your CPU) and shared for all RAG operations.
# We chose 'embeddinggemma' for its 2048-token context and CPU efficiency.

print("Initializing shared embedding model (google/embeddinggemma-300m)...")
try:
    embedding_function = HuggingFaceEmbeddings(
        model_name="google/embeddinggemma-300m",
        model_kwargs={'device': 'cpu', 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Shared embedding model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load embedding model: {e}")
    embedding_function = None

# --- 3. Shared Vector DB Path (The "Memory") ---
# Defines the single, persistent location for our Chroma vector database.

CHROMA_PERSIST_DIR = "./chroma_db_market"
print(f"Shared vector store directory set to: {CHROMA_PERSIST_DIR}")


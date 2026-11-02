# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Agentic AI Career Advisor (IIIT Dharwad "Agentic AI" course project). The working product is an **interactive CLI** in `backend/app/agents/`. A web app (FastAPI API + Next.js frontend) and Docker deployment are being built; see the phase plan in `architect-plan.md` if present (it is a local, gitignored file).

- There is no FastAPI app, frontend, or Docker setup yet.
- No test suite, linter, or formatter is set up.

## Running

All relative paths in the code (`./chroma_db`, `./chroma_db_market`, `resume_output/`, `report_session_*.txt`) resolve against the current directory, so **run from the repo root**.

```bash
pip install -r backend/requirements.txt   # README workflow; also needs langchain-cohere and prisma, which are not listed

# 1. Job scraper API (required by the job market analyst, hardcoded to http://localhost:8001)
cd backend && uvicorn app.api.v1.job_web_scraper:app --port 8001

# 2. Seed the interview-question vector DB (one-time; reads ingestion/interview_rag_data/*.txt)
python backend/app/agents/ingestion/seed_interview_db.py

# 3. Main CLI
python -m backend.app.agents.supervisor_agent
```

Most agent modules also have a `__main__` block for running them on their own (e.g. `python backend/app/agents/interview_coach.py`).

Prisma (resume template DB, `backend/prisma/schema.prisma`, `backend/app/db/`): `prisma generate` and `prisma db push` from `backend/`; see `backend/PRISMA_SETUP.md`. `backend/app/db/seed.py` loads the `.tex` files in `backend/templates/`.

## Environment variables

Loaded via `python-dotenv` from a `.env` at the repo root; copy `.env.example`. `.env` is gitignored, so never commit it. `backend/app/agents/settings.py` fails at import time without `GROQ_API_KEY` and `DB_URL`. The README mentions `DATABASE_URL`, which the code does not read.

- `GROQ_API_KEY`: shared LLM
- `DB_URL`: Postgres for LangGraph `PostgresSaver` checkpointers (job analyst, interview coach)
- `MODEL_DB_URL`: Postgres for Prisma
- `COHERE_API_KEY` (reranker), `TAVILY_API_KEY` (course web search), `GITHUB_ACCESS_TOKEN` (GitHub API), `HF_TOKEN` (gated `google/embeddinggemma-300m` model): optional, features degrade without them

## Architecture

**Shared components: `backend/app/agents/settings.py`.** Importing it creates the single `ChatGroq` LLM (`llama-3.3-70b-versatile`, temperature 0) and the local CPU `HuggingFaceEmbeddings` (`google/embeddinggemma-300m`), and reads env config. Agents import it as a **top-level module** (`from settings import llm, ...`), so `backend/app/agents/` must be on `sys.path`. `supervisor_agent.py` adds it; scripts run from other places need to do the same.

**Supervisor (`supervisor_agent.py`)** is a plain Python REPL, **not** a LangGraph graph, even though the README says it is. It:
- holds session data in a mutable `GlobalState` object (target role, location, resume path, GitHub user, skills, skill gaps, job data, resume JSON, interview history) with a `SessionState` enum audit trail;
- loads each agent lazily via `safe_import()`, which tries several module paths and **silently returns `None` on any exception**. A broken agent shows up only as "Module ... not available", so import the module directly to see the real error;
- routes user input by **keyword matching** (`interview`, `job`, `learn`, `resume`, `skill`, `report`) to `*_wrapper` functions. These copy data out of `GlobalState` into each agent's own input and write results back. Some auto-run job analysis first when `skill_gaps` is empty;
- writes `report_session_<timestamp>.txt` to the cwd on exit.

**Agents.** Each agent owns its own state and graph:
- `skills.py`: extracts skills from a resume file (pdf/docx/txt) and GitHub repo languages/README, with an LLM merge step and keyword fallbacks.
- `job_market_analyst.py`: LangGraph `StateGraph` (parse_query → search_db → scrape_web → analyze_skills → format_output, with conditional routing). Caches job postings in Chroma at `./chroma_db_market`, reranks with Cohere, and calls the scraper API on :8001 when the cache misses.
- `api/v1/job_web_scraper.py`: a standalone FastAPI app (BeautifulSoup + spaCy `PhraseMatcher` skill extraction) with `/api/jobs` and `/api/jobs/description`. It is a separate process, not part of the main API.
- `curriculum_architect.py`: LangGraph planner/critique loop over a course catalog in Chroma at `./chroma_db`, with Tavily web search to add courses. `prime_database()` runs **at import time**.
- `interview_coach.py`: LangGraph `MessagesState` graph using `interrupt`/`Command` for human-in-the-loop answers and adaptive per-topic difficulty. It pulls questions from Chroma at `backend/app/agents/ingestion/chroma_db_interview_questions` (an absolute path, built by `seed_interview_db.py`) and checkpoints to Postgres via `psycopg` + `PostgresSaver`.
- `resume_strategist.py`: `ResumeStrategist` class that fetches GitHub profile/repos, has the LLM produce resume JSON and then LaTeX, and writes `resume_output/resume.tex`.

Several modules do heavy work at import time (load the embedding model, open Chroma, initialize Cohere/Tavily, print banners), so importing any agent is slow and needs the env vars.

**Data files.** The interview RAG source text lives in `backend/app/agents/ingestion/interview_rag_data/`. The text files use `#TOPIC:` / `#DIFFICULTY:`-style header metadata, which `seed_interview_db.py` parses. All Chroma directories are gitignored.

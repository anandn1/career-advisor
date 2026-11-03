# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Agentic AI Career Advisor (IIIT Dharwad "Agentic AI" course project): LangGraph agents that analyze the job market for a role, find skill gaps, build a learning path, write a tailored LaTeX resume and run an adaptive mock interview. The agents are being put behind a FastAPI API with a Next.js frontend and Docker deployment; see the phase plan in `architect-plan.md` if present (local, gitignored).

- Backend: FastAPI app in `app/main.py` over the agents (plus `app/cli.py` for terminal use). No frontend or Docker setup yet.
- Tooling: ruff, plus a pytest suite in `backend/tests/` (isolated: no .env, network, LLM or Postgres).

## Branching

- Work happens on version branches named `release/vX.Y` (currently `release/v2.0`). Branch features from the version branch and open PRs against it, never against `main`.
- The version branch merges into `main` in one PR, only when the whole version works end to end and the repo owner approves.
- A GitHub ruleset protects `main` and `release/**`: only the repo admin can push or merge, and nobody can force-push or delete. PRs show "blocked" until the admin bypasses (`gh pr merge <n> --admin`).

## Commands

The Python project lives in `backend/` and uses **uv** (Python 3.12, CPU-only torch). Run commands from `backend/`:

```bash
uv sync                                                 # install (creates backend/.venv)
uv run ruff check app migrations                        # lint
uv run ruff format app migrations                       # format
uv run pytest                                           # tests (~4s, fully offline)
uv run pytest tests/test_x.py::test_name                # a single test

uv run alembic upgrade head                             # create/upgrade DB tables
uv run python -m app.db.seed_templates                  # load templates/*.tex (LLM-tagged)
uv run python -m app.agents.ingestion.seed_interview_db # build interview question store
#   flags: --verify (parse only), --dry-run, --if-empty (skip if store exists)
uv run uvicorn app.main:app --reload                    # API on :8000 (docs at /docs)
uv run python -m app.cli                                # interactive terminal front-end
TEST_DATABASE_URL=postgresql://... uv run pytest -m postgres  # Postgres-only tests (drops that DB's schema)
```

New migrations: `uv run alembic revision --autogenerate -m "..."` (needs a reachable database).

## Configuration

All settings come from `app/core/config.py` (`Settings`, pydantic-settings), read from the environment and from `.env` at the repo root or `backend/`. Copy `.env.example`. Empty values count as unset. Never read `os.environ` elsewhere; call `get_settings()`.

- Required: `GROQ_API_KEY`, `DATABASE_URL` (`DB_URL` is accepted as an alias). One Postgres holds both the LangGraph checkpoints and the app tables; `Settings.sqlalchemy_url` rewrites it for SQLAlchemy's psycopg 3 driver.
- Optional, features degrade without them: `COHERE_API_KEY` (reranking; plain vector search otherwise), `TAVILY_API_KEY` (course web search), `GITHUB_ACCESS_TOKEN` (rate limits), `HF_TOKEN` (gated `google/embeddinggemma-300m`).
- `DATA_DIR` (default `backend/data/`, gitignored) holds the three Chroma stores (`chroma_market`, `chroma_courses`, `chroma_interview`) and `outputs/`.

## Architecture

**Shared resources are lazy.** `app/core/resources.py` builds the Groq LLMs (`get_llm`, plus `get_resume_llm` for long LaTeX output), the local CPU embedding model, Chroma stores, the optional Cohere reranker and the optional Tavily tool on first use, cached with `functools.cache`. `app/core/checkpointer.py` does the same for the Postgres `PostgresSaver` (psycopg connection pool; `setup()` runs on first use). Keep it that way: **importing an agent module must not load models, open databases or need credentials.** Compiled graphs are also built lazily (`get_*_graph()`).

**Agents (`app/agents/`)** each expose plain functions that the CLI (and later the API) call:
- `skills.py`: `extract_skills_from_resume(path)` / `extract_skills_from_github(username)`. LLM extraction with a fallback to the spaCy skill matcher in `services/job_scraper.py`.
- `job_market_analyst.py`: linear StateGraph (search cached postings in Chroma → scrape fresh LinkedIn postings and cache them → compute gaps). `run_job_market_analysis(role, location, user_skills)`. Stream progress with `get_job_market_graph().stream(initial_state(...), stream_mode="updates")`.
- `curriculum_architect.py`: planner → per-skill loop (catalog search → web search if empty → 3 parallel scoring checks → recommend) → architect submits the path via a tool call → critic, with up to `MAX_REVISIONS`. `plan_learning_path(skill_gaps, user_skills, target_role)`. Seed courses are in `agents/data/seed_courses.json` and are added to the store on first use.
- `resume_strategist.py`: Pydantic models (`ResumeProfile` in, `ResumeDraft` out). Two LLM steps: a structured draft, then LaTeX that follows a reference template. User data is LaTeX-escaped before the second step, and the output is validated. `generate_resume(profile, template_latex=None)`.
- `interview_coach.py`: human-in-the-loop graph. `wait_for_answer` calls `interrupt()`, and state lives in the Postgres checkpointer keyed by thread ID, so a session spans many calls: `start_interview(thread_id, topics, company)` then `submit_answer(thread_id, text)` (answer `stop` to finish), each returning an `InterviewTurn`. Difficulty adapts per topic; weak answers may get an LLM follow-up that updates the same report entry.

**Services (`app/services/`)** hold non-agent I/O: `job_scraper` (LinkedIn guest pages + spaCy `PhraseMatcher`), `github` (REST client; never raises, logs and returns empty), `documents` (pdf/docx/txt text), `latex` (sandboxed `pdflatex`: no shell escape, no file access outside its temp dir, timeout; raises `LatexUnavailableError` when TeX is not installed, as on dev machines outside Docker).

**Database (`app/db/`)**: SQLAlchemy 2 models (`ResumeTemplate`, `Tag`, link table `resume_template_tags`), `session_scope()` for transactions, queries in `templates.py`, Alembic migrations in `backend/migrations/` (env.py takes the URL from settings).

**Data files**: interview questions are `.txt` files in `app/agents/ingestion/interview_rag_data/`, using `#COMPANY:` / `#TOPIC:` / `#DIFFICULTY:` headers (which carry over to later sections) and `---` between questions. The company list in `interview_coach.COMPANIES` must match the file names. LaTeX reference templates are in `backend/templates/`.

**API (`app/api/`, `app/services/runner.py`)**: everything is nested under `/api/v1/sessions/{session_id}/...`; knowing the session UUID is the only access control. Long agent work is an `AgentRun` row (`app/db/models.py`) executed by `RunExecutor` in a thread pool: POST returns 202 with the run, progress events are appended to the run row, and `GET .../runs/{id}/events` streams them over SSE (polling the row; `Last-Event-ID` resumes). One unfinished run per kind per session (409). Handlers per kind live in `app/services/run_handlers.py`; raise `UserFacingError` for messages safe to show, anything else becomes a generic failure. Interviews are `AgentRun` rows whose id is the LangGraph thread id and are answered synchronously. On startup, unfinished non-interview runs are marked failed; sessions unseen for `SESSION_TTL_DAYS` are deleted at startup and every 6 hours (with their interview checkpoints).

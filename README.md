# Agentic AI Career Advisor

**A Multi-Agent System for Personalized Career Guidance**

The **Agentic AI Career Advisor** is an intelligent system designed to provide end-to-end career support. Powered by **Groq** for high-speed inference and **LangGraph** for stateful multi-agent orchestration, the application deploys a team of specialized AI agents to analyze job markets, craft resumes, and prepare candidates for interviews.

> **Current Status:** This project is currently a **Command Line Interface (CLI)** application. A full web interface (FastAPI + React) is in development.

## System Architecture

The system operates via a supervised graph workflow where a **Supervisor Agent** delegates tasks to specialized sub-agents based on the user's state and needs.

```mermaid
graph TD
    User[User Input] --> Supervisor[Supervisor Agent]
    Supervisor -->|Delegates| Analyst[Job Market Analyst]
    Supervisor -->|Delegates| Strategist[Resume Strategist]
    Supervisor -->|Delegates| Architect[Curriculum Architect]
    Supervisor -->|Delegates| Coach[Interview Coach]
    Analyst & Strategist & Architect & Coach -->|Result| Supervisor
    Supervisor -->|Final Output| User
```

## The Agent Team

| Agent | Role | File |
| :--- | :--- | :--- |
| **Job Market Analyst** | Scans job descriptions to identify critical skills and market trends. | `job_market_analyst.py` |
| **Resume Strategist** | Structures and optimizes user profiles for ATS compatibility and impact. | `resume_strategist.py` |
| **Curriculum Architect** | Identifies skill gaps and recommends personalized learning resources. | `curriculum_architect.py` |
| **Interview Coach** | Conducts mock interviews tailored to the specific target role. | `interview_coach.py` |

## Tech Stack

  * **Orchestration:** LangGraph (Stateful multi-agent workflows)
  * **LLM Engine:** Groq (Llama 3 / Mixtral for near-instant inference)
  * **Embeddings:** Google EmbeddingGemma 300M (Running locally via SentenceTransformers)
  * **Framework:** Python, LangChain
  * **Database:** PostgreSQL (with SQLAlchemy)
  * **Vector Store:** ChromaDB (for RAG capabilities)

## Installation & Setup

### Prerequisites

  * Python 3.10+
  * PostgreSQL installed and running
  * A Groq API Key

### 1\. Clone the Repository

```bash
git clone https://github.com/anandn1/career-advisor.git
cd career-advisor
```

### 2\. Create Virtual Environment

```bash
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 3\. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

> **Note:** This will install `sentence-transformers` and `langchain-huggingface`. The first run may take a moment to download the EmbeddingGemma model weights (approx. 300MB).

### 4\. Environment Configuration

Create a `.env` file in the `backend/` directory:

```env
GROQ_API_KEY=gsk_your_groq_key_here
DATABASE_URL=postgresql://user:password@localhost:5432/career_db
# Note: No API key needed for Embeddings as they run locally.
```

## Usage

To start the CLI application, execute the Supervisor Agent module from the root directory.

```bash
python -m backend.app.agents.supervisor_agent
```

**How to interact:**

1.  The Supervisor will initialize and request your **Target Job Role**.
2.  It will ask for your **Profile Details** (skills, experience, etc.).
3.  The system will automatically route your request to the **Analyst**, **Strategist**, or **Coach** agents to generate the necessary advice.

## Project Structure

```text
career-advisor/
├── backend/
│   ├── app/
│   │   ├── agents/          # Agent logic (Supervisor, Analyst, Coach, etc.)
│   │   ├── core/            # Configuration and Utils
│   │   └── models/          # Database models
│   ├── requirements.txt
│   └── .env
├── notebook/                # Experimental Jupyter notebooks
└── README.md
```

## Roadmap

  - [x] Core Agentic Workflow (CLI) with LangGraph
  - [x] Integration of Local Embeddings (Gemma 300M)
  - [ ] Web Interface Development: Migrating to a React Frontend + FastAPI Backend.
  - [ ] Database integration for persistent user history.
  - [ ] Enhanced error handling and retry mechanisms.

## Project Context

This project was developed as part of the **"Agentic AI"** course at **IIIT Raichur**, under the guidance of **Dr. Sunil Saumya** of **IIIT Dharwad**.

## License

Distributed under the MIT License. See `LICENSE` for more information.






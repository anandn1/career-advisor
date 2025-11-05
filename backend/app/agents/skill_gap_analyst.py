import os
import json
import re
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)
DB_URI = os.getenv("DATABASE_URL")


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_core.output_parsers import JsonOutputParser

print("Loading shared settings...")
from settings import llm, embedding_function, CHROMA_PERSIST_DIR


# 1. Shared Vector Store

def get_vectorstore() -> Chroma:
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Creating new Chroma DB at {CHROMA_PERSIST_DIR}")
        placeholder = [Document(page_content="Skill-gap DB ready.", metadata={"source": "init"})]
        vs = Chroma.from_documents(placeholder, embedding_function,
                                   persist_directory=CHROMA_PERSIST_DIR)
    else:
        print(f"Loading existing Chroma DB from {CHROMA_PERSIST_DIR}")
        vs = Chroma(persist_directory=CHROMA_PERSIST_DIR,
                    embedding_function=embedding_function)
    return vs

db = get_vectorstore()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                          chunk_overlap=150,
                                          length_function=len)


# 2. Dummy Resume Strategist (replace later)

def get_user_skills_from_resume(target_role: str,
                                linkedin_url: Optional[str] = None) -> List[str]:
    print(f"[Resume Strategist] Extracting skills for {target_role}")
    dummy = {
        "Full Stack Developer": ["Python", "React.js", "SQL", "Git",
                                 "Docker", "JavaScript", "HTML", "CSS"],
        "Data Scientist": ["Python", "Pandas", "Scikit-learn", "SQL",
                           "Machine Learning", "NumPy"],
    }
    skills = dummy.get(target_role, ["Python", "JavaScript", "Git", "SQL"])
    if linkedin_url:
        skills += ["Communication", "Teamwork"]
    return list({s.strip() for s in skills})


# 3. Pull Market Skills from Job Analyst (fallback on rate-limit)

try:
    from job_market_analyst import agent as job_agent
    HAS_JOB_AGENT = True
except Exception as e:
    print(f"job_market_analyst not available → using dummy market skills ({e})")
    HAS_JOB_AGENT = False

def get_market_skills(target_role: str, location: str) -> List[str]:
    if not HAS_JOB_AGENT:
        return ["Python", "React.js", "Node.js", "Docker", "SQL",
                "Git", "AWS", "Problem-solving"]

    print(f"[Job Analyst] Fetching market skills for {target_role} in {location}")
    try:
        # Fresh thread id avoids “connection closed”
        thread_id = f"skillgap_job_{uuid.uuid4().hex[:8]}"
        result = job_agent.invoke(
            {"messages": [HumanMessage(
                content=f"List the top 10 skills for {target_role} jobs in {location}.")]},
            {"configurable": {"thread_id": thread_id}}
        )
        content = result["messages"][-1].content
        skills = re.findall(
            r"(Python|React\.?js|Node\.?js|Java|Git|Docker|SQL|AWS|Angular|JavaScript|Problem-solving|Communication|Collaboration)",
            content, re.IGNORECASE)
        unique = {s.strip().title() for s in skills}
        return list(unique) or ["Python", "JavaScript", "Git"]
    except Exception as e:
        print(f"Job agent failed (rate-limit or other): {e}. Using fallback.")
        return ["Python", "React.js", "Node.js", "Docker", "SQL", "Git"]


# 4. RAG Tool for extra context

@tool
def search_skill_context(query: str) -> str:
    """Search the vector DB for skill-related context."""
    print(f"[RAG] Searching: {query}")
    docs = db.similarity_search(query, k=5)
    if not docs:
        return "No context found."
    return "\n\n---\n\n".join(
        f"Source: {d.metadata.get('source', 'N/A')}\nContent: {d.page_content}"
        for d in docs
    )


# 5. JSON Prompt + Parser

JSON_PROMPT = """You are a Skill Gap Analyst.
Target Role: {target_role}
Location: {location}
User Skills: {user_skills}
Market Skills: {market_skills}
Context: {skill_context}

Return **only valid JSON** with the exact keys below:

{{
  "matching_skills": "Python, Git, SQL",
  "missing_skills": "Node.js, Docker",
  "strengths": "React.js, Problem-solving",
  "summary": "70% match. Learn Node.js in 2 weeks via freeCodeCamp."
}}
"""

prompt = ChatPromptTemplate.from_template(JSON_PROMPT)
parser = JsonOutputParser()

chain = (
    {
        "target_role": RunnablePassthrough(),
        "location": RunnablePassthrough(),
        "user_skills": RunnablePassthrough(),
        "market_skills": RunnablePassthrough(),
        "skill_context": RunnablePassthrough(),
    }
    | prompt
    | llm
    | parser
)


# 6. Core Skill-Gap Function

def analyze_skill_gap(
    target_role: str,
    location: str,
    linkedin_url: Optional[str] = None
) -> Dict[str, Any]:
    print("\n=== SKILL GAP ANALYSIS START ===")

    # ---- Parallel Step 1 -------------------------------------------------
    user_skills = get_user_skills_from_resume(target_role, linkedin_url)
    market_skills = get_market_skills(target_role, location)

    # ---- Optional RAG for a missing skill --------------------------------
    missing = set(market_skills) - set(user_skills)
    sample_missing = list(missing)[0] if missing else ""
    skill_context = "No gaps to explain."
    if sample_missing:
        try:
            skill_context = search_skill_context.invoke(sample_missing)
        except Exception as e:
            print(f"RAG failed: {e}")

    # ---- Run the JSON chain ---------------------------------------------
    try:
        result = chain.invoke({
            "target_role": target_role,
            "location": location,
            "user_skills": ", ".join(user_skills),
            "market_skills": ", ".join(market_skills),
            "skill_context": skill_context,
        })
    except Exception as e:
        print(f"LLM JSON parsing failed: {e}")
        # Fallback – still return something useful
        result = {
            "matching_skills": ", ".join(set(user_skills) & set(market_skills)),
            "missing_skills": ", ".join(set(market_skills) - set(user_skills)),
            "strengths": ", ".join(user_skills),
            "summary": "Analysis failed due to LLM parsing error."
        }

    # ---- Persist report --------------------------------------------------
    report_doc = Document(
        page_content=f"Skill Gap: {target_role} @ {location}\n{json.dumps(result)}",
        metadata={
            "source": "skill_gap_analyst",
            "role": target_role,
            "location": location,
            "type": "skill_gap_report"
        }
    )
    chunks = splitter.split_documents([report_doc])
    db.add_documents(chunks)
    print(f"Ingested {len(chunks)} report chunks into Chroma.")

    return {
        **result,
        "user_skills_list": user_skills,
        "market_skills_list": market_skills
    }


# 7. ReAct Agent (uses the same tool)

# DB_URI = os.getenv(
#     "DATABASE_URL",
#     "postgres://f1879c06ec098d1521da313181510b7369ce1e0eed5cc875ab654a8aa2880405:sk_qY9w6ACyeEYYkjO1viCFo@db.prisma.io:5432/postgres?sslmode=require"
# )

AGENT_SYSTEM_PROMPT = """
You are a Skill Gap Analyst assistant.
You have one tool: `search_skill_context`.
Use it only when the user asks for deeper context on a skill or trend.
Always answer in a clear, structured way.
"""

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    agent = create_agent(
        llm,
        tools=[search_skill_context],
        system_prompt=AGENT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    print("--- Skill Gap Agent Ready ---")


# 8. Demo (run when file is executed directly) — NO CRASH
if __name__ == "__main__":
    report = analyze_skill_gap(
        target_role="Full Stack Developer",
        location="Hyderabad",
        linkedin_url="https://linkedin.com/in/demo"
    )
    
    print("\n" + "="*60)
    print("SKILL GAP REPORT")
    print("="*60)
    for k, v in report.items():
        if k.endswith("_list"):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v}")
    print("="*60)

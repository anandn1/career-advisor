# resume_strategist.py
"""
Merged Resume Strategist
- Interactive mode + supervisor compatible class
- Fetches GitHub profile + repos
- Extracts skills & projects heuristically from GitHub
- Generates resume_json via LLM (ChatGroq)
- Generates LaTeX via LLM and saves resume_output/resume.tex
- Exposes:
    class ResumeStrategist
    function run_interactive_mode()
"""

import os
import json
import requests
import concurrent.futures
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
GROQ_KEY = os.getenv("GROQ_API_KEY")

# Minimal local heuristics for tech keywords (expand as needed)
COMMON_TECH_KEYWORDS = [
    "python", "java", "javascript", "typescript", "react", "node", "express", "django",
    "flask", "go", "rust", "c++", "c#", "c", "php", "ruby", "swift", "kotlin",
    "docker", "kubernetes", "aws", "gcp", "azure", "sql", "postgres", "mysql", "mongodb",
    "redis", "tensorflow", "pytorch", "scikit-learn", "opencv", "html", "css", "sass",
    "tailwind", "bootstrap", "graphql", "rest", "graphql", "solidity", "ethers", "web3"
]

# ---------------------------
# GitHub fetch helpers
# ---------------------------
def fetch_github_profile(username: str, timeout: int = 10) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        r = requests.get(f"https://api.github.com/users/{username}", headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def fetch_github_repos(username: str, per_page: int = 6, timeout: int = 10) -> List[dict]:
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        r = requests.get(
            f"https://api.github.com/users/{username}/repos",
            headers=headers,
            params={"sort": "updated", "per_page": per_page},
            timeout=timeout,
        )
        r.raise_for_status()
        raw = r.json()
        repos = []
        for repo in raw:
            repos.append({
                "name": repo.get("name", ""),
                "description": repo.get("description") or "",
                "language": repo.get("language") or "",
                "stars": repo.get("stargazers_count", 0),
                "html_url": repo.get("html_url", "")
            })
        return repos
    except Exception:
        return []

def fetch_github_data(username: str) -> Dict[str, Any]:
    """Parallel fetch of profile + repos."""
    if not username:
        return {"profile": {}, "repos": []}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            pf = ex.submit(fetch_github_profile, username)
            rp = ex.submit(fetch_github_repos, username)
            return {"profile": pf.result(), "repos": rp.result()}
    except Exception:
        return {"profile": {}, "repos": []}

# ---------------------------
# Skill & project extraction heuristics
# ---------------------------
def extract_skills_from_github(github_data: Dict[str, Any]) -> List[str]:
    """Heuristic extraction: collect repo languages + keyword matches from names/descriptions."""
    skills = set()
    repos = github_data.get("repos", []) or []
    for repo in repos:
        lang = (repo.get("language") or "").strip()
        if lang:
            skills.add(lang)
        text = " ".join([repo.get("name", ""), repo.get("description", "") or ""]).lower()
        for kw in COMMON_TECH_KEYWORDS:
            if kw in text:
                skills.add(kw)
    # normalize a bit (lowercase, dedupe)
    cleaned = sorted({s.strip() for s in skills if s and len(s) < 40}, key=str.lower)
    return cleaned

def build_projects_from_github(github_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create concise project entries from top repos."""
    projects = []
    repos = github_data.get("repos", []) or []
    for repo in repos:
        name = repo.get("name", "")
        desc = (repo.get("description") or "").strip()
        techs = []
        if repo.get("language"):
            techs.append(repo.get("language"))
        # try to pull technology tokens from description
        txt = (name + " " + desc).lower()
        for kw in COMMON_TECH_KEYWORDS:
            if kw in txt and kw not in techs:
                techs.append(kw)
        bullets = []
        if desc:
            bullets.append(desc if len(desc) <= 220 else desc[:217] + "...")
        else:
            bullets.append(f"Developed the {name} repository.")
        # add an automated bullet about repo maintenance/stars if available
        stars = repo.get("stars", 0)
        if stars:
            bullets.append(f"Repository has {stars} ⭐ stars on GitHub.")
        if repo.get("html_url"):
            bullets.append(f"Source: {repo.get('html_url')}")
        projects.append({
            "name": name,
            "description_bullets": bullets[:3],
            "tech_stack": techs
        })
    return projects

# ---------------------------
# LLM helpers
# ---------------------------
def _get_llm():
    if not GROQ_KEY:
        raise Exception("Missing GROQ_API_KEY in environment")
    return ChatGroq(
        groq_api_key=GROQ_KEY,
        model_name="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=6000,
    )

# ---------------------------
# ResumeStrategist class
# ---------------------------
class ResumeStrategist:
    def __init__(self):
        if not GROQ_KEY:
            raise Exception("Missing GROQ_API_KEY")
        self.llm = _get_llm()

    # Build a JSON resume via LLM; keep safe parsing and fallback
    def _generate_json_from_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses the LLM to convert collected data into a strict resume JSON.
        Ensures literal braces in prompt are escaped so ChatPromptTemplate doesn't treat them as variables.
        """
        prompt_text = """
You are a professional resume author. Using ONLY the provided data, output a valid JSON object that contains the following keys:

{{
  "contact": {{
    "name": "",
    "email": "",
    "phone": "",
    "location": "",
    "github": ""
  }},
  "summary": "",
  "education": [],
  "work_experience": [],
  "skills": [],
  "projects": []
}}

Do NOT include any explanatory text, markdown, or commentary — output only JSON.

DATA:
{data}
"""

        # escape braces in template where needed by providing them doubled in the template string.
        # Here we pass the whole prompt as a single string to ChatPromptTemplate.from_template
        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | self.llm | StrOutputParser()

        raw = chain.invoke({"data": json.dumps(data)}).strip()

        # strip code fences if model returned them
        if raw.startswith("```"):
            # take content between code fences
            parts = raw.split("```")
            if len(parts) >= 3:
                raw = parts[1]
            else:
                raw = raw.replace("```", "")

        raw = raw.strip()

        # final safe parse
        try:
            parsed = json.loads(raw)
            return parsed
        except Exception:
            # fallback minimal structure
            profile = data.get("github", {}).get("profile", {}) or {}
            name = profile.get("name", "Candidate")
            return {
                "contact": {
                    "name": name,
                    "email": data.get("contact", {}).get("email", ""),
                    "phone": data.get("contact", {}).get("phone", ""),
                    "location": data.get("contact", {}).get("location", ""),
                    "github": profile.get("html_url", data.get("github", {}).get("username", ""))
                },
                "summary": f"A motivated candidate targeting {data.get('target_role', 'software roles')}.",
                "education": data.get("education", []),
                "work_experience": data.get("work_experience", []),
                "skills": data.get("skills", []),
                "projects": data.get("projects", [])
            }

    def _generate_latex_from_json(self, resume_json: Dict[str, Any]) -> str:
        prompt_text = """
You are an expert LaTeX resume builder. Convert the following JSON resume into a complete, compile-ready LaTeX document. Output ONLY LaTeX (no comments, no markdown fences).

JSON:
{json}
"""
        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke({"json": json.dumps(resume_json)})
        # remove backticks if present
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 3:
                raw = parts[1]
            else:
                raw = raw.replace("```", "")
        return raw.strip()

    def _make_inventory(self, resume_json: Dict[str, Any]) -> str:
        name = resume_json.get("contact", {}).get("name", "Candidate")
        skills = resume_json.get("skills", [])
        # skills might be list of strings or list of categories; normalize display
        if isinstance(skills, list) and skills and isinstance(skills[0], dict):
            flat_skills = []
            for cat in skills:
                flat_skills.extend(cat.get("skills", []) if isinstance(cat, dict) else [])
        elif isinstance(skills, list):
            flat_skills = skills
        else:
            flat_skills = []

        flat_skills = flat_skills[:20]
        projects = resume_json.get("projects", [])[:5]

        text = f"""
RESUME INVENTORY SUMMARY
------------------------
Name: {name}
Top Skills: {', '.join(flat_skills)}
Top Projects:
"""
        for p in projects:
            pname = p.get("name", p.get("project", ""))
            text += f"\n• {pname}\n"
            for b in p.get("description_bullets", p.get("bullets", []))[:2]:
                text += f"   - {b}\n"
        return text

    # supervisor-compatible master method
    def generate_resume(self,
                        target_role: str,
                        github_username: Optional[str],
                        resume_path: Optional[str],
                        user_skills: List[str]) -> Dict[str, Any]:
        """
        Called by supervisor_agent. Should NOT prompt interactively.
        Behavior:
        - If github_username provided -> fetch GitHub and auto-generate
        - If no github_username -> produce minimal JSON using provided user_skills and target_role
        Returns: { "resume_json", "resume_inventory", "latex_path" (if saved) }
        """
        # Build initial data
        data = {
            "target_role": target_role or "Software Engineer",
            "contact": {},
            "education": [],
            "work_experience": [],
            "skills": user_skills or []
        }

        github_data = {}
        if github_username:
            github_data = fetch_github_data(github_username)
            data["github"] = github_data
            # heuristics
            extracted_skills = extract_skills_from_github(github_data)
            if extracted_skills:
                # merge with user_skills keeping unique values
                merged = list(dict.fromkeys([*user_skills, *extracted_skills]))
                data["skills"] = merged
            projects = build_projects_from_github(github_data)
            data["projects"] = projects
            # try to fill contact name location from profile
            profile = github_data.get("profile", {}) or {}
            if profile.get("name"):
                data["contact"]["name"] = profile.get("name")
            if profile.get("location"):
                data["contact"]["location"] = profile.get("location")
        else:
            data["projects"] = []

        # Step 1: Generate JSON via LLM (with safe fallback)
        resume_json = self._generate_json_from_data(data)

        # Step 2: Inventory summary (text)
        inventory = self._make_inventory(resume_json)

        # Step 3: Generate LaTeX and save file
        try:
            latex = self._generate_latex_from_json(resume_json)
            os.makedirs("resume_output", exist_ok=True)
            tex_path = os.path.join("resume_output", "resume.tex")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(latex)
        except Exception:
            tex_path = ""

        # Supervisor expects resume_json and resume_inventory keys
        out = {
            "resume_json": resume_json,
            "resume_inventory": inventory
        }
        # optional extra key (safe)
        if tex_path:
            out["latex_path"] = tex_path
        return out

# ---------------------------
# Interactive mode (console)
# ---------------------------
def run_interactive_mode():
    """
    Console interactive flow. Collects user inputs, fetches GitHub, extracts skills/projects,
    invokes the ResumeStrategist to build JSON + LaTeX, and saves resume_output/resume.tex.
    """
    print("\n===============================")
    print("   INTERACTIVE RESUME BUILDER")
    print("===============================\n")

    target_role = input("Enter target job role (e.g., 'Backend Developer'): ").strip()
    gh = input("GitHub username (or press ENTER to skip): ").strip()
    github_data = fetch_github_data(gh) if gh else {}

    data = {
        "target_role": target_role or "Software Engineer",
        "github": github_data,
        "contact": {},
        "education": [],
        "work_experience": [],
        "skills": []
    }

    # contact
    print("\nEnter contact details (press ENTER to skip / accept GitHub values):")
    gh_name = github_data.get("profile", {}).get("name", "") if github_data else ""
    name = input(f"Full name [{gh_name}]: ").strip() or gh_name
    email = input("Email: ").strip()
    phone = input("Phone: ").strip()
    location = input("Location: ").strip() or github_data.get("profile", {}).get("location", "")

    data["contact"].update({
        "name": name,
        "email": email,
        "phone": phone,
        "location": location,
        "github": gh
    })

    # education
    print("\n--- Education: add entries (leave 'Institution' blank to stop) ---")
    while True:
        inst = input(" Institution: ").strip()
        if not inst:
            break
        deg = input(" Degree: ").strip()
        field = input(" Field: ").strip()
        status = input(" Status (Completed / In Progress): ").strip()
        data["education"].append({
            "institution": inst, "degree": deg, "field": field, "status": status
        })

    # work
    print("\n--- Work Experience: add entries (leave 'Company' blank to stop) ---")
    while True:
        comp = input(" Company: ").strip()
        if not comp:
            break
        pos = input(" Position: ").strip()
        dur = input(" Duration: ").strip()
        bullets = []
        print(" Enter up to 4 bullets describing achievements (type 'done' to stop):")
        while len(bullets) < 4:
            b = input("  - ").strip()
            if not b or b.lower() == "done":
                break
            bullets.append(b)
        data["work_experience"].append({
            "company": comp, "position": pos, "duration": dur, "description_bullets": bullets
        })

    # skills
    

    # If GitHub provided, heuristically extract skills and projects and merge
    if gh:
        extracted_skills = extract_skills_from_github(github_data)
        if extracted_skills:
            merged_skills = list(dict.fromkeys([*data["skills"], *extracted_skills]))
            data["skills"] = merged_skills
        data["projects"] = build_projects_from_github(github_data)
    else:
        data["projects"] = []

    print("\nGenerating resume JSON + LaTeX (this will call the LLM)...")

    strategist = ResumeStrategist()
    resume_json = strategist._generate_json_from_data(data)
    latex = strategist._generate_latex_from_json(resume_json)

    # save LaTeX
    os.makedirs("resume_output", exist_ok=True)
    tex_path = os.path.join("resume_output", "resume.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)

    inventory = strategist._make_inventory(resume_json)

    print("\n=====================================")
    print(f"✔ LaTeX file saved at: {tex_path}")
    print("=====================================\n")
    print("\n--- Resume Inventory ---\n")
    print(inventory)
    print("\nProcess complete.")

    # Return values in case someone calls this function programmatically
    return {
        "resume_json": resume_json,
        "resume_inventory": inventory,
        "latex_path": tex_path
    }

# Allow running directly
if __name__ == "__main__":
    run_interactive_mode()

"""Seed resume templates from backend/templates/*.tex, tagging each with the LLM.

Run from backend/:  uv run python -m app.db.seed_templates
"""

import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.resources import get_llm
from app.db.models import ResumeTemplate
from app.db.session import session_scope
from app.db.templates import get_or_create_tag, template_exists

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"
PREDEFINED_TAGS = (
    "Technology",
    "Business & Finance",
    "Engineering",
    "Design & User Experience",
    "Freelance",
    "Corporate",
    "Startup",
    "Software Development",
)
TAGGING_PROMPT = """Analyze the LaTeX resume below and assign relevant tags ONLY from this list:
{tags}

Respond with a single line of comma-separated tags and nothing else,
for example: Technology, Software Development, Startup

--- RESUME CONTENT ---
{content}
--- END OF CONTENT ---"""


def tag_template(latex_src: str) -> list[str]:
    """Ask the LLM which predefined tags fit this template; unknown tags are dropped."""
    prompt = TAGGING_PROMPT.format(tags=", ".join(PREDEFINED_TAGS), content=latex_src)
    reply = get_llm().invoke(
        [
            SystemMessage(content="You are an expert HR analyst who follows instructions exactly."),
            HumanMessage(content=prompt),
        ]
    )
    suggested = (tag.strip() for tag in str(reply.content).split(","))
    return [tag for tag in suggested if tag in PREDEFINED_TAGS]


def seed_templates(templates_dir: Path = TEMPLATES_DIR) -> int:
    """Insert templates that are not in the database yet; returns how many were added."""
    added = 0
    with session_scope() as session:
        for path in sorted(templates_dir.glob("*.tex")):
            if template_exists(session, path.stem):
                continue
            latex_src = path.read_text(encoding="utf-8")
            tags = tag_template(latex_src)
            session.add(
                ResumeTemplate(
                    name=path.stem,
                    latex_src=latex_src,
                    tags=[get_or_create_tag(session, tag) for tag in tags],
                )
            )
            session.flush()
            logger.info("Added template %s with tags %s", path.stem, tags)
            added += 1
    return added


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(f"Seeded {seed_templates()} new templates.")

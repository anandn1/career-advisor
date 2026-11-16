"""Resume reviewer: a structured critique of an existing resume for a target role."""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.core.resources import get_llm


class ResumeReview(BaseModel):
    summary: str = Field(description="2-3 sentences on overall quality and fit for the role")
    strengths: list[str] = Field(description="3-5 key strengths")
    weaknesses: list[str] = Field(description="3-5 key weaknesses or gaps")
    suggestions: list[str] = Field(
        description="3-5 specific, actionable fixes, e.g. 'Quantify project 1 with metrics'"
    )


REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional career coach and resume reviewer. Give a critical, "
            "constructive and actionable review. Base every point on the resume text; "
            "never invent experience the resume does not show.",
        ),
        (
            "human",
            "Target role: {target_role}\nUser's request: {request}\n\n"
            "Resume text:\n---\n{resume_text}\n---",
        ),
    ]
)
DEFAULT_REQUEST = "Review my resume for this role."


def review_resume(resume_text: str, target_role: str, request: str = "") -> ResumeReview:
    chain = REVIEW_PROMPT | get_llm().with_structured_output(ResumeReview)
    return chain.invoke(
        {
            "resume_text": resume_text,
            "target_role": target_role,
            "request": request.strip() or DEFAULT_REQUEST,
        }
    )

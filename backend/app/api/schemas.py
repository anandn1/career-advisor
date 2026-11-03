"""Request and response bodies of the HTTP API."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.agents.interview_coach import COMPANIES
from app.agents.resume_strategist import ContactInfo, Education, ResumeDraft, WorkExperience
from app.db.models import RunKind, RunStatus

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
SkillName = Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)]
CompanyFocus = Literal[(*COMPANIES, "all")]  # type: ignore[valid-type]
MAX_ANSWER_LENGTH = 5000
MAX_REQUEST_LENGTH = 1000


class SessionCreate(BaseModel):
    target_role: ShortText
    location: ShortText


class SessionUpdate(BaseModel):
    target_role: ShortText | None = None
    location: ShortText | None = None
    skills: list[SkillName] | None = Field(None, max_length=200)


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    target_role: str
    location: str
    github_username: str | None
    skills: list[str]


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kind: RunKind
    status: RunStatus
    result: dict | None
    error: str | None
    events: list[dict]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class GithubSkillsRequest(BaseModel):
    github_username: str = Field(min_length=1, max_length=200, description="Username or URL")


class ResumeRequest(BaseModel):
    contact: ContactInfo = Field(default_factory=ContactInfo)
    education: list[Education] = Field(default_factory=list, max_length=20)
    work_experience: list[WorkExperience] = Field(default_factory=list, max_length=30)
    extra_skills: list[str] = Field(default_factory=list, max_length=100)
    include_github: bool = True
    template_id: str | None = None


class ResumeRenderRequest(BaseModel):
    draft: ResumeDraft
    template_id: str | None = None


class InterviewCreate(BaseModel):
    company_focus: CompanyFocus = "generic"
    topics: list[ShortText] | None = Field(None, min_length=1, max_length=10)


class AnswerRequest(BaseModel):
    answer: str = Field(max_length=MAX_ANSWER_LENGTH)


class TurnOut(BaseModel):
    question: str | None
    feedback: str | None
    finished: bool
    summary: str | None
    question_count: int


class InterviewOut(BaseModel):
    id: str
    status: RunStatus
    company_focus: str
    topics: list[str]
    turn: TurnOut | None = None
    report: list[dict] = Field(default_factory=list)


class TemplateOut(BaseModel):
    id: str
    name: str
    tags: list[str]

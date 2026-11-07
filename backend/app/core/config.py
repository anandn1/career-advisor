"""Application settings, loaded once from environment variables and `.env` files."""

import json
from functools import cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_core import PydanticUseDefault
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent

EMBEDDING_MODEL_SUBDIR = Path("models", "embeddinggemma-300m-onnx")  # inside data_dir

# For reasoning models such as gpt-oss; "none" for models without reasoning.
ReasoningEffort = Literal["low", "medium", "high"]


def _trimmed_or_default(value):
    """A string without surrounding spaces; blank, like empty, means unset."""
    if not isinstance(value, str):
        return value
    if not value.strip():
        raise PydanticUseDefault
    return value.strip()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        extra="ignore",
        env_ignore_empty=True,
    )

    groq_api_key: SecretStr
    # Chosen by the model evals in backend/evals/ (see its README for the evidence).
    llm_model: str = "openai/gpt-oss-120b"
    llm_reasoning_effort: ReasoningEffort | None = "medium"
    resume_llm_model: str = "openai/gpt-oss-20b"
    resume_llm_reasoning_effort: ReasoningEffort | None = "low"

    database_url: str = Field(
        validation_alias=AliasChoices("DATABASE_URL", "DB_URL"),
        repr=False,  # holds the database password
    )

    cohere_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    github_access_token: SecretStr | None = None

    data_dir: Path = BACKEND_DIR / "data"
    # Defaults to <data_dir>/models/...; the Docker image points it at the model baked in.
    embedding_model_dir: Path | None = None
    session_ttl_days: int = Field(30, gt=0)
    max_upload_bytes: int = Field(5 * 1024 * 1024, gt=0)
    run_workers: int = Field(4, gt=0)
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]

    @field_validator("database_url")
    @classmethod
    def _normalize_scheme(cls, url: str) -> str:
        """Hosting providers often hand out 'postgres://', which SQLAlchemy rejects."""
        if url.startswith("postgres://"):
            return "postgresql://" + url.removeprefix("postgres://")
        return url

    @model_validator(mode="after")
    def _default_embedding_model_dir(self) -> "Settings":
        if self.embedding_model_dir is None:
            self.embedding_model_dir = self.data_dir / EMBEDDING_MODEL_SUBDIR
        return self

    @field_validator("embedding_model_dir", mode="before")
    @classmethod
    def _trim_path(cls, value):
        """A blank path would be read as a folder named by its spaces."""
        return _trimmed_or_default(value)

    @field_validator("llm_reasoning_effort", "resume_llm_reasoning_effort", mode="before")
    @classmethod
    def _normalise_effort(cls, value):
        """Case and spaces don't matter; "none" sends a null effort, which Groq accepts for
        models without reasoning (they reject "low")."""
        value = _trimmed_or_default(value)
        if not isinstance(value, str):
            return value
        value = value.lower()
        return None if value == "none" else value

    @field_validator("llm_model", "resume_llm_model", mode="before")
    @classmethod
    def _trim_model(cls, value):
        """A padded or blank model name would only fail later, as a 404 from Groq."""
        return _trimmed_or_default(value)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value):
        """Accept a JSON list or a comma-separated string."""
        if not isinstance(value, str):
            return value
        if value.strip().startswith("["):
            return json.loads(value)
        return [origin.strip() for origin in value.split(",") if origin.strip()]

    @property
    def market_chroma_dir(self) -> Path:
        return self.data_dir / "chroma_market"

    @property
    def courses_chroma_dir(self) -> Path:
        return self.data_dir / "chroma_courses"

    @property
    def interview_chroma_dir(self) -> Path:
        return self.data_dir / "chroma_interview"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def sqlalchemy_url(self) -> str:
        """The same database, addressed through SQLAlchemy's psycopg 3 dialect."""
        if self.database_url.startswith("postgresql://"):
            return "postgresql+psycopg://" + self.database_url.removeprefix("postgresql://")
        return self.database_url


@cache
def get_settings() -> Settings:
    return Settings()

"""Application settings, loaded once from environment variables and `.env` files."""

import json
from functools import cache
from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        extra="ignore",
        env_ignore_empty=True,
    )

    groq_api_key: SecretStr
    llm_model: str = "llama-3.3-70b-versatile"
    resume_llm_model: str = "openai/gpt-oss-120b"

    database_url: str = Field(validation_alias=AliasChoices("DATABASE_URL", "DB_URL"))

    embedding_model: str = "google/embeddinggemma-300m"
    hf_token: SecretStr | None = None

    cohere_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    github_access_token: SecretStr | None = None

    data_dir: Path = BACKEND_DIR / "data"
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]

    @field_validator("database_url")
    @classmethod
    def _normalize_scheme(cls, url: str) -> str:
        """Hosting providers often hand out 'postgres://', which SQLAlchemy rejects."""
        if url.startswith("postgres://"):
            return "postgresql://" + url.removeprefix("postgres://")
        return url

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

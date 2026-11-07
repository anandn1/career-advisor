"""Gemini as an independent judge for outputs that have no single right answer.

The judge is from a different model family than every Groq candidate, sees no model
names, and scores each criterion from 1 to 5.
"""

from functools import cache
from pathlib import Path

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
# One judge for every candidate keeps scores comparable. Free-tier keys get no quota for
# the Pro models and 20 requests a day for this one; `--rejudge` finishes cases later.
JUDGE_MODELS = ("gemini-3.6-flash",)
# Retry when Gemini is overloaded (503) or rate-limited (429).
RETRY = types.HttpRetryOptions(
    attempts=6, initial_delay=5, max_delay=60, http_status_codes=[429, 500, 503]
)
MAX_SCORE = 5

JUDGE_PROMPT = """You are a strict, fair evaluator of an AI system's output.

Task the system was given:
{task}

Input the system received:
<input>
{source}
</input>

Output to evaluate:
<output>
{output}
</output>

Score the output on each criterion from 1 (fails it) to 5 (fully meets it), with a
one-sentence reason. Judge only what is in the output; do not reward length.

Criteria:
{criteria}"""


class EvalSettings(BaseSettings):
    """Keys for the evals only, read from the same .env files as the app."""

    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR.parent / ".env", BACKEND_DIR / ".env"),
        extra="ignore",
        env_ignore_empty=True,
    )

    gemini_api_key: SecretStr | None = None


class CriterionScore(BaseModel):
    criterion: str
    score: int = Field(ge=1, le=MAX_SCORE)
    reason: str


class Verdict(BaseModel):
    scores: list[CriterionScore]

    @property
    def normalised(self) -> float:
        """Mean score scaled to 0..1."""
        if not self.scores:
            return 0.0
        return sum(s.score - 1 for s in self.scores) / (len(self.scores) * (MAX_SCORE - 1))


def _checked(verdict: Verdict, criteria: list[str]) -> Verdict:
    """A verdict must score each requested criterion once, in order (names may be reworded)."""
    if len(verdict.scores) != len(criteria):
        raise ValueError(f"judge scored {len(verdict.scores)} criteria, expected {len(criteria)}")
    names = [s.criterion.strip().lower() for s in verdict.scores]
    if len(set(names)) != len(names):
        # Scores are saved by criterion name, so a repeated name would lose one of them.
        raise ValueError(f"judge named a criterion twice: {names}")
    return verdict


class JudgeUnavailableError(RuntimeError):
    """Every judge model failed (overloaded, rate-limited or unparseable)."""


class Judge:
    """Tries each model in order, so one overloaded model does not stop the run."""

    def __init__(self, client: genai.Client, models: tuple[str, ...]):
        self._client = client
        self.models = models

    def score(self, task: str, source: str, output: str, criteria: list[str]) -> Verdict:
        prompt = JUDGE_PROMPT.format(
            task=task,
            source=source,
            output=output,
            criteria="\n".join(f"- {criterion}" for criterion in criteria),
        )
        failures = []
        for model in self.models:
            try:
                return _checked(self._ask(model, prompt), criteria)
            # API errors, unparseable verdicts, and network failures after the client's
            # retries: none of them is the candidate model's fault.
            except (genai_errors.APIError, ValueError, httpx.HTTPError, OSError) as error:
                failures.append(f"{model}: {str(error)[:120]}")
        raise JudgeUnavailableError("; ".join(failures))

    def _ask(self, model: str, prompt: str) -> Verdict:
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=Verdict,
            ),
        )
        if response.parsed is None:
            raise ValueError(f"no verdict in {response.text!r}")
        return response.parsed


@cache
def get_judge(models: tuple[str, ...] = JUDGE_MODELS) -> Judge | None:
    """The Gemini judge, or None when GEMINI_API_KEY is not set."""
    api_key = EvalSettings().gemini_api_key
    if api_key is None:
        return None
    client = genai.Client(
        api_key=api_key.get_secret_value(),
        http_options=types.HttpOptions(retry_options=RETRY),
    )
    return Judge(client, models)

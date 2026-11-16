"""Candidate Groq models, their prices, and swapping them into the agents."""

import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import astuple, dataclass, field, replace
from unittest.mock import patch

import groq
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.core.resources import RESUME_MAX_TOKENS

# US dollars per million tokens (input, output), from Groq's pricing page in November 2026.
# Reasoning tokens are billed as output.
PRICES_PER_MILLION = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "qwen/qwen3.8-27b": (0.80, 4.00),
}
TOKENS_PER_MILLION = 1_000_000
# The free tier allows 8,000 tokens a minute; Groq's client waits and retries on 429s.
MAX_RETRIES = 8

# Every place the agents get a chat model from.
GENERAL_LLM_SITES = (
    "app.agents.skills.get_llm",
    "app.agents.resume_reviewer.get_llm",
    "app.agents.curriculum_architect.get_llm",
    "app.agents.interview_coach.get_llm",
    "app.db.seed_templates.get_llm",
)
RESUME_LLM_SITE = "app.agents.resume_strategist.get_resume_llm"


def is_rate_limit(error: BaseException) -> bool:
    """A quota refusal: HTTP 429, which Groq uses for per-minute and per-day token limits and
    for single requests larger than the tier allows ("Request too large ... OTPM")."""
    return isinstance(error, groq.RateLimitError) or getattr(error, "status_code", None) == 429


@dataclass(frozen=True)
class ModelSpec:
    """A Groq model plus, for reasoning models, how hard it thinks ('low', 'medium', 'high')."""

    name: str
    reasoning_effort: str | None = None

    @classmethod
    def parse(cls, text: str) -> "ModelSpec":
        """'openai/gpt-oss-20b@low' -> ModelSpec('openai/gpt-oss-20b', 'low')."""
        name, _, effort = text.partition("@")
        return cls(name, effort or None)

    @property
    def label(self) -> str:
        return f"{self.name}@{self.reasoning_effort}" if self.reasoning_effort else self.name

    def cost(self, input_tokens: int, output_tokens: int) -> float | None:
        prices = PRICES_PER_MILLION.get(self.name)
        if prices is None:
            return None
        input_price, output_price = prices
        return (input_tokens * input_price + output_tokens * output_price) / TOKENS_PER_MILLION


@dataclass
class Usage:
    calls: int = 0
    failed_calls: int = 0  # calls that raised after the client's own retries
    rate_limited_calls: int = 0  # calls refused by the provider's quota (not the model's fault)
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    seconds: float = 0.0

    def __sub__(self, other: "Usage") -> "Usage":
        pairs = zip(astuple(self), astuple(other), strict=True)
        return Usage(*(mine - theirs for mine, theirs in pairs))

    def snapshot(self) -> "Usage":
        return replace(self)


@dataclass
class UsageRecorder(BaseCallbackHandler):
    """Adds up tokens and wall time of every LLM call made through the patched models."""

    total: Usage = field(default_factory=Usage)
    _started: dict = field(default_factory=dict)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._started[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs) -> None:
        started = self._started.pop(run_id, None)
        if started is not None:
            self.total.seconds += time.perf_counter() - started
        self.total.calls += 1
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if not usage:
                    continue
                self.total.input_tokens += usage.get("input_tokens", 0)
                self.total.output_tokens += usage.get("output_tokens", 0)
                details = usage.get("output_token_details") or {}
                self.total.reasoning_tokens += details.get("reasoning", 0)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._started.pop(run_id, None)
        if is_rate_limit(error):
            self.total.rate_limited_calls += 1
        else:
            self.total.failed_calls += 1


def build_llm(spec: ModelSpec, recorder: UsageRecorder, **options) -> ChatGroq:
    if spec.reasoning_effort:
        options["reasoning_effort"] = spec.reasoning_effort
    return ChatGroq(
        model=spec.name,
        api_key=get_settings().groq_api_key,
        max_retries=MAX_RETRIES,
        callbacks=[recorder],
        **options,
    )


@contextmanager
def use_model(spec: ModelSpec, recorder: UsageRecorder) -> Iterator[None]:
    """Make every agent use `spec`, with the same settings the app gives its own models."""
    general = build_llm(spec, recorder, temperature=0)
    resume = build_llm(spec, recorder, temperature=0.1, max_tokens=RESUME_MAX_TOKENS)
    with ExitStack() as stack:
        for site in GENERAL_LLM_SITES:
            stack.enter_context(patch(site, lambda: general))
        stack.enter_context(patch(RESUME_LLM_SITE, lambda: resume))
        yield

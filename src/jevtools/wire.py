"""Wire-level models for Jev (TypeSafe System One) requests and responses.

These mirror the public contract shared by all three Jev endpoints:

- TypeSafe direct:      ``POST https://api.typesafe.ai/v1/systemone``
- OpenRouter SystemOne: ``POST https://openrouter.ai/api/v1/systemone``
- OpenRouter Decisions: ``POST https://openrouter.ai/api/alpha/decisions``

A request carries one ``state`` and a map of named, typed questions; a response
carries one typed answer per question. Jev never returns text.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JSONContent = str | dict[str, Any] | list[Any]
"""State, instructions and criteria may each be text, a JSON object, or a JSON array."""

MAX_CHOICE_OPTIONS = 255
"""Jev's maximum number of options in one Choice question."""

MAX_SCORE_LEVELS = 256
"""Practical maximum number of levels in one Score question."""

CONTEXT_TOKENS = 32_000
"""Jev's context window: state plus all questions."""


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_wire(self) -> dict[str, Any]:
        """The JSON-ready dict sent to the API (unset optional fields omitted)."""
        return self.model_dump(mode="json", exclude_none=True)


class NoulCriteria(_Wire):
    """Optional descriptions of what counts as yes (``true``) and no (``false``)."""

    true: JSONContent | None = None
    false: JSONContent | None = None


class ChoiceQuestion(_Wire):
    """Pick one of the labels in ``criteria`` (at most 255)."""

    type: Literal["choice"] = "choice"
    instructions: JSONContent | None = None
    criteria: dict[str, JSONContent | None]


class NoulQuestion(_Wire):
    """Probability that a statement holds or the answer to a question is yes."""

    type: Literal["noul"] = "noul"
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None


class ScoreQuestion(_Wire):
    """Expected position on an ordered rubric; level ``i`` is ``criteria[i]``."""

    type: Literal["score"] = "score"
    instructions: JSONContent | None = None
    criteria: list[JSONContent] = Field(min_length=1)


Question = Annotated[ChoiceQuestion | NoulQuestion | ScoreQuestion, Field(discriminator="type")]


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class NoulAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: Literal["noul"] = "noul"
    noul: float


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[int, Any] = Field(default_factory=dict)
    probabilities: dict[int, float] = Field(default_factory=dict)


Answer = Annotated[ChoiceAnswer | NoulAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    """USD cost; reported by OpenRouter only."""


class DecisionRequest(_Wire):
    model: str
    state: JSONContent
    questions: dict[str, Question] = Field(min_length=1)


class DecisionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    model: str = ""
    answers: dict[str, Answer] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    id: str | None = None
    provider: str | None = None

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> DecisionResponse:
        """Parse a response body, skipping answer types this version does not model."""
        answers = payload.get("answers") or {}
        known = {
            name: raw
            for name, raw in answers.items()
            if isinstance(raw, dict) and raw.get("type") in ("choice", "noul", "score")
        }
        return cls.model_validate({**payload, "answers": known})

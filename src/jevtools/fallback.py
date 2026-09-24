"""Generative fallbacks (spec §4.7): the Filler (per-slot FILL), the Escalator (whole-turn, Jev-gated) and TextLLM.

This module defines the protocols and data models the router uses. Reference implementations over an
OpenAI-compatible ``chat/completions`` endpoint (``OpenAICompatibleFiller``, ``OpenAICompatibleEscalator``) are
provided separately; anything implementing these protocols works.

Whatever a Filler or Escalator returns enters a pool as a ``generated`` candidate (or takes the channel of an equal
existing candidate) and must still be elected by Jev under the slot's allow-list (I1, I2).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from jevtools.context import Observation, Turn

FILL_INSTRUCTIONS = (
    "Write only the listed fields for this already-decided tool call. Do not change or repeat the frozen arguments. "
    "Convey exactly what the user asked; add no facts."
)


class ObservationPreview(BaseModel):
    """An observation as the Filler sees it: the same preview Jev sees (never the full content)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int
    tool: str
    status: str = "ok"
    preview: str = ""

    @classmethod
    def of(cls, observation: Observation) -> ObservationPreview:
        """The preview of a context observation."""
        return cls(step=observation.step, tool=observation.tool, status=observation.status,
                   preview=observation.preview_text())  # fmt: skip


class FillRequest(BaseModel):
    """One FILL under a frozen skeleton: write only ``slots`` for an already-decided call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str
    tool_description: str
    frozen: dict[str, Any]
    """Every argument already decided (values, not labels)."""
    slots: dict[str, dict[str, Any]]
    """JSON Schema of only the slots to fill (``x-jev`` stripped)."""
    request: str
    history: list[Turn] = Field(default_factory=list)
    observations: list[ObservationPreview] = Field(default_factory=list)
    k: int = 2
    instructions: str = FILL_INSTRUCTIONS


class FillCandidate(BaseModel):
    """One proposal: a value per slot to fill."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    values: dict[str, Any]


@runtime_checkable
class Filler(Protocol):
    """Proposes candidates for uncovered content slots (at most one FILL per decision)."""

    def fill(self, req: FillRequest) -> list[FillCandidate]: ...

    async def afill(self, req: FillRequest) -> list[FillCandidate]: ...


class ProposedCall(BaseModel):
    """A tool call proposed by an Escalator; every argument must still be bound through Jev."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def coerce(cls, value: Any) -> ProposedCall | None:
        """Accept a :class:`ProposedCall`, a :class:`~jevtools.decision.ToolCall`-like object or a
        ``{"name", "arguments"}`` mapping; ``None`` for anything else (e.g. a text answer)."""
        if isinstance(value, ProposedCall):
            return value
        if isinstance(value, Mapping) and isinstance(value.get("name"), str):
            return cls(name=value["name"], arguments=dict(value.get("arguments") or {}))
        name, arguments = getattr(value, "name", None), getattr(value, "arguments", None)
        if isinstance(name, str) and isinstance(arguments, Mapping):
            return cls(name=name, arguments=dict(arguments))
        return None


EscalationResult = ProposedCall | str
"""What an Escalator returns: a proposed call (gated by one Jev round) or a text answer (→ abstain with content)."""


@runtime_checkable
class Escalator(Protocol):
    """Whole-turn fallback: an LLM tool caller (or a human) whose call is re-bound and gated by Jev.

    ``tools`` are OpenAI function tools with ``x-jev`` stripped; ``decision`` is the escalating decision.
    """

    def escalate(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]],
                 decision: Any) -> EscalationResult: ...  # fmt: skip

    async def aescalate(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]],
                        decision: Any) -> EscalationResult: ...  # fmt: skip


@runtime_checkable
class TextLLM(Protocol):
    """Writes the text answer of an ``abstain`` (a joke, small talk): never used for tool arguments."""

    def complete(self, messages: Sequence[Mapping[str, Any]]) -> str: ...

    async def acomplete(self, messages: Sequence[Mapping[str, Any]]) -> str: ...


__all__ = [
    "FILL_INSTRUCTIONS",
    "EscalationResult",
    "Escalator",
    "FillCandidate",
    "FillRequest",
    "Filler",
    "ObservationPreview",
    "ProposedCall",
    "TextLLM",
]

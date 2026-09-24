"""The Decision document and its emitted formats (spec §3.10, §9.1).

A Decision carries the outcome, the rule that fired, the proposed ``call``, the ``tool_calls`` to run now
(non-empty only for ``execute``), the confidence record, the prompt and the pending handle. It serializes to the
native canonical JSON (numbers rounded half-even to 4 decimals, argument values untouched), to an OpenAI assistant
message and to Anthropic ``tool_use`` blocks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jevtools._version import SPEC_VERSION
from jevtools.canonical import canonical_json, canonical_str, jsonable, round4, sha256_hex, short_id
from jevtools.policy import Outcome

PENDING_TTL = timedelta(minutes=15)
"""How long a pending confirm/clarify stays resumable (spec §7.2.4)."""


def _r(x: float | None) -> float | None:
    return None if x is None else round4(x)


# --------------------------------------------------------------------------------------------------------------------
# Ids (§3.10, §6.5)
# --------------------------------------------------------------------------------------------------------------------


def call_hash(trace_id: str, name: str, arguments: dict[str, Any]) -> str:
    """``sha256(trace_id ‖ name ‖ canonical(arguments))[:16]``: shared by call ids and idempotency keys."""
    return sha256_hex(trace_id + name + canonical_str(arguments))[:16]


def call_id(trace_id: str, name: str, arguments: dict[str, Any]) -> str:
    """``call_jev_<hash>``: re-emitting a decision yields the same id, so executors can dedupe."""
    return "call_jev_" + call_hash(trace_id, name, arguments)


def idempotency_key(trace_id: str, name: str, arguments: dict[str, Any]) -> str:
    """``idem_<hash>`` (spec §6.5)."""
    return "idem_" + call_hash(trace_id, name, arguments)


class DecisionIds(BaseModel):
    """``dec_``/``tr_``/``pnd_`` ids sharing one 16-hex digest."""

    model_config = ConfigDict(frozen=True)

    decision_id: str
    trace_id: str
    pending_id: str

    @classmethod
    def derive(cls, *parts: str) -> DecisionIds:
        """Deterministic ids from the round's content (e.g. ballot and response hashes plus the policy hash)."""
        digest = short_id("", "\x1f".join(parts))
        return cls(decision_id="dec_" + digest, trace_id="tr_" + digest, pending_id="pnd_" + digest)


# --------------------------------------------------------------------------------------------------------------------
# Parts
# --------------------------------------------------------------------------------------------------------------------


class _Doc(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolCall(_Doc):
    """A concrete call: name, validated arguments, deterministic id and idempotency key."""

    name: str
    arguments: dict[str, Any]
    id: str
    idempotency_key: str

    @classmethod
    def build(cls, name: str, arguments: dict[str, Any], *, trace_id: str) -> ToolCall:
        """A call whose ids derive from ``(trace_id, name, arguments)``."""
        return cls(
            name=name,
            arguments=arguments,
            id=call_id(trace_id, name, arguments),
            idempotency_key=idempotency_key(trace_id, name, arguments),
        )

    def __str__(self) -> str:
        """Python-call form for logs and the quickstart: ``get_weather(city='Zurich', unit='fahrenheit')``."""
        return f"{self.name}({', '.join(f'{key}={value!r}' for key, value in self.arguments.items())})"

    def arguments_json(self) -> str:
        """Canonical JSON of the arguments (what OpenAI's ``function.arguments`` carries)."""
        return canonical_str(self.arguments)

    def to_openai(self) -> dict[str, Any]:
        """``{"id", "type": "function", "function": {"name", "arguments"}}``."""
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments_json()}}

    def to_anthropic(self) -> dict[str, Any]:
        """``{"type": "tool_use", "id": "toolu_jev_…", "name", "input"}``."""
        suffix = self.id.removeprefix("call_jev_")
        return {"type": "tool_use", "id": "toolu_jev_" + suffix, "name": self.name, "input": jsonable(self.arguments)}

    def to_doc(self, *, with_ids: bool = True) -> dict[str, Any]:
        """Native form: ``{"name", "arguments"}`` for the proposed call; ``tool_calls`` entries add the ids."""
        doc: dict[str, Any] = {"name": self.name, "arguments": jsonable(self.arguments)}
        if with_ids:
            doc = {"id": self.id, **doc, "idempotency_key": self.idempotency_key}
        return doc


class PromptOption(_Doc):
    """One prompt option: a machine id (``ok``, ``alt:start:1``, ``change``, ``cancel``…) and its text."""

    id: str
    text: str


PromptKind = Literal["confirm", "menu", "open", "notice"]
"""``confirm`` card, clarify ``menu`` (slot, tool or yes/no), ``open`` question, or a refuse/abstain ``notice``."""


class Prompt(_Doc):
    """A templated prompt (§3.8.4): filled with candidate labels, never generated."""

    kind: PromptKind
    text: str
    options: list[PromptOption] = Field(default_factory=list)


class PendingAction(_Doc):
    """What a prompt option does on resume: ``bind`` a slot value, ``confirm``, ``cancel``, ``open`` a slot's
    question, or choose a ``tool``."""

    action: Literal["bind", "confirm", "cancel", "open", "tool"]
    slot: str | None = None
    value: Any = None
    label: str | None = None
    tool: str | None = None


class Pending(_Doc):
    """Resumable handle of a confirm/clarify (§3.8.5, §9.1). ``state`` holds router-private resume data
    (factors and bindings to reuse on a click, the conversation, the entity store)."""

    pending_id: str
    decision_id: str
    ballot_sha256: str
    response_sha256s: list[str] = Field(default_factory=list)
    call: ToolCall | None = None
    options: dict[str, PendingAction] = Field(default_factory=dict)
    created_at: datetime
    expires_at: datetime
    state: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def new(cls, *, pending_id: str, decision_id: str, ballot_sha256: str, created_at: datetime | None = None,
            ttl: timedelta = PENDING_TTL, **fields: Any) -> Pending:  # fmt: skip
        """A pending handle that expires ``ttl`` after ``created_at`` (default: now, UTC)."""
        created = created_at or datetime.now(timezone.utc)
        return cls(pending_id=pending_id, decision_id=decision_id, ballot_sha256=ballot_sha256, created_at=created,
                   expires_at=created + ttl, **fields)  # fmt: skip

    def expired(self, now: datetime | None = None) -> bool:
        """Whether the handle can no longer be resumed."""
        return (now or datetime.now(timezone.utc)) >= self.expires_at


class Confidence(_Doc):
    """The confidence record (§3.7, §3.10). ``call`` is the final C; W/PI/L/J are always recorded."""

    call: float
    tier: str
    composition: str
    W: float
    PI: float
    L: float
    J: float | None = None
    calibrated: bool = False
    execute_at: float | None = None
    """``None`` when the tier never auto-executes (critical, uncertified)."""
    confirm_at: float | None = None

    def to_doc(self) -> dict[str, Any]:
        """Native form with rounded numbers."""
        return {
            "call": _r(self.call), "tier": self.tier, "composition": self.composition, "W": _r(self.W),
            "PI": _r(self.PI), "L": _r(self.L), "J": _r(self.J), "calibrated": self.calibrated,
            "execute_at": _r(self.execute_at), "confirm_at": _r(self.confirm_at),
        }  # fmt: skip


class Bottleneck(_Doc):
    """The weakest slot and its shape (``ambiguous``, ``missing``, ``diffuse``, ``out_of_pool``…)."""

    slot: str
    shape: str


class AlternativeReport(_Doc):
    """A runner-up value in a slot report; ``part`` names the list/record part it belongs to."""

    value: Any
    p: float
    part: str | None = None

    def to_doc(self) -> dict[str, Any]:
        doc: dict[str, Any] = {"part": self.part} if self.part is not None else {}
        return {**doc, "value": jsonable(self.value), "p": _r(self.p)}


class SlotReport(_Doc):
    """Per-slot summary in the Decision: elected value, its probability, stakes, channel and alternatives."""

    value: Any
    p: float
    stakes: str
    channel: str | None = None
    alternatives: list[AlternativeReport] = Field(default_factory=list)

    def to_doc(self) -> dict[str, Any]:
        return {
            "value": jsonable(self.value), "p": _r(self.p), "stakes": self.stakes, "channel": self.channel,
            "alternatives": [a.to_doc() for a in self.alternatives],
        }  # fmt: skip


class DecisionUsage(_Doc):
    """Resource use of the whole decision."""

    jev_calls: int = 0
    jev_input_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float | None = None


# --------------------------------------------------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------------------------------------------------


class Decision(BaseModel):
    """The native decision (spec §3.10). ``tool_calls`` is non-empty only when ``outcome == "execute"``."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    spec: str = SPEC_VERSION
    decision_id: str
    trace_id: str
    outcome: Outcome
    rule: str
    call: ToolCall | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    confidence: Confidence | None = None
    """``None`` when no tool was decided (abstain on ``NO_TOOL``, backend failure)."""
    bottleneck: Bottleneck | None = None
    slots: dict[str, SlotReport] = Field(default_factory=dict)
    gates: dict[str, float] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    prompt: Prompt | None = None
    pending: Pending | None = None
    rounds: int = 1
    usage: DecisionUsage = Field(default_factory=DecisionUsage)
    content: str | None = None
    """Text for abstain/escalate handoffs (from ``text_llm`` or the Escalator); not part of the native document."""
    trace: Any = None
    """The :class:`jevtools.trace.Trace` warrant; not part of the native document."""

    @model_validator(mode="after")
    def _only_execute_emits(self) -> Decision:
        if self.tool_calls and self.outcome is not Outcome.EXECUTE:
            raise ValueError("tool_calls may be non-empty only when the outcome is execute")
        return self

    @property
    def pending_id(self) -> str | None:
        """Id of the pending handle, if any."""
        return self.pending.pending_id if self.pending is not None else None

    @property
    def finish_reason(self) -> str:
        """OpenAI ``finish_reason``: ``tool_calls`` when calls are emitted, else ``stop``."""
        return "tool_calls" if self.tool_calls else "stop"

    def to_doc(self) -> dict[str, Any]:
        """The native document in normative key order (probabilities rounded; argument values untouched)."""
        return {
            "spec": self.spec,
            "decision_id": self.decision_id,
            "trace_id": self.trace_id,
            "outcome": self.outcome.value,
            "rule": self.rule,
            "call": self.call.to_doc(with_ids=False) if self.call is not None else None,
            "tool_calls": [c.to_doc() for c in self.tool_calls],
            "confidence": self.confidence.to_doc() if self.confidence is not None else None,
            "bottleneck": jsonable(self.bottleneck) if self.bottleneck is not None else None,
            "slots": {name: report.to_doc() for name, report in self.slots.items()},
            "gates": {name: _r(p) for name, p in self.gates.items()},
            "flags": list(self.flags),
            "prompt": jsonable(self.prompt) if self.prompt is not None else None,
            "pending_id": self.pending_id,
            "rounds": self.rounds,
            "usage": jsonable(self.usage),
        }

    def to_json(self) -> bytes:
        """Canonical JSON of :meth:`to_doc`."""
        return canonical_json(self.to_doc())

    def _x_jev(self) -> dict[str, Any]:
        x: dict[str, Any] = {"outcome": self.outcome.value}
        if self.tool_calls:
            x["confidence"] = _r(self.confidence.call) if self.confidence is not None else None
            x["trace_id"] = self.trace_id
            x["idempotency_key"] = self.tool_calls[0].idempotency_key
            return x
        if self.prompt is not None and self.outcome in (Outcome.CONFIRM, Outcome.CLARIFY):
            x["options"] = [jsonable(o) for o in self.prompt.options]
            x["pending_id"] = self.pending_id
        x["trace_id"] = self.trace_id
        return x

    def to_openai_message(self) -> dict[str, Any]:
        """OpenAI-compatible assistant message (§3.10): tool calls on execute, the templated prompt on
        confirm/clarify/refuse, the handoff text (or ``""``) otherwise; ``x_jev`` carries the jevtools fields."""
        if self.tool_calls:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [c.to_openai() for c in self.tool_calls],
                "x_jev": self._x_jev(),
            }
        text = self.prompt.text if self.prompt is not None else (self.content or "")
        return {"role": "assistant", "content": text, "x_jev": self._x_jev()}

    def to_anthropic_content(self) -> list[dict[str, Any]]:
        """Anthropic content blocks: ``tool_use`` blocks on execute, else one text block (none if empty)."""
        if self.tool_calls:
            return [c.to_anthropic() for c in self.tool_calls]
        text = self.prompt.text if self.prompt is not None else (self.content or "")
        return [{"type": "text", "text": text}] if text else []


__all__ = [
    "PENDING_TTL",
    "AlternativeReport",
    "Bottleneck",
    "Confidence",
    "Decision",
    "DecisionIds",
    "DecisionUsage",
    "Pending",
    "PendingAction",
    "Prompt",
    "PromptKind",
    "PromptOption",
    "SlotReport",
    "ToolCall",
    "call_hash",
    "call_id",
    "idempotency_key",
]

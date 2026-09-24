"""The Ballot: the conformance boundary between planning and asking (spec §3.5.7).

A Ballot holds the exact state, the ordered questions with their options and decode entries, the per-tool
viability records, the split plan and the hashes. Everything from the Ballot onward is byte-identical across
implementations: :meth:`Ballot.to_requests` is the normative Ballot → JevRequest mapping.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jevtools._version import SPEC_VERSION
from jevtools.candidates import SENTINELS, Candidate, Channel, DecodesTo
from jevtools.canonical import canonical_json, canonical_str, jsonable, sha256_of
from jevtools.context import Mode
from jevtools.policy import Tier
from jevtools.qid import is_valid_qid, make_qid, opaque_ids
from jevtools.wire import (
    ChoiceQuestion,
    DecisionRequest,
    JSONContent,
    NoulCriteria,
    NoulQuestion,
    Question,
    ScoreQuestion,
)

Family = Literal[
    "tool", "authorized", "slot", "probe", "present", "rev", "date", "time", "accept", "mention", "more",
    "item", "member", "branch", "joint", "done_after", "bucket", "group", "reply", "segmentation",
]  # fmt: skip
"""Question families (spec §3.5.3)."""
Primitive = Literal["choice", "noul", "score"]
IdMode = Literal["dotted", "opaque"]

FAMILY_ORDER: tuple[str, ...] = (
    "slot", "probe", "date", "time", "accept", "mention", "more", "item", "member", "branch", "bucket", "group",
    "present", "rev",
)  # fmt: skip
"""Order of a slot's question families within the plan (§3.5.3 table order, probes after the slot question)."""


class BallotOption(BaseModel):
    """A real option of a Choice: the label Jev sees and the decode entry (value, channel, provenance)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    value: Any = None
    text: JSONContent | None = None
    """Option description sent as the criterion (``None`` allowed for self-explanatory enum members)."""
    channel: Channel | None = None
    """Provenance channel (``None`` for tool options and joint combinations, whose parts carry their own)."""
    prov: dict[str, Any] = Field(default_factory=dict)
    late: dict[str, Any] | None = None

    @classmethod
    def from_candidate(cls, candidate: Candidate) -> BallotOption:
        """The option of a labelled candidate (attributes such as balances stay behind: never sent to Jev)."""
        return cls(
            label=candidate.label,
            value=candidate.value,
            text=candidate.text,
            channel=candidate.channel,
            prov=candidate.prov,
            late=candidate.late,
        )


class SentinelSpec(BaseModel):
    """A sentinel option: its normative text and what it decodes to (``default`` carries the value or a late recipe)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decodes_to: DecodesTo
    text: JSONContent | None = None
    value: Any = None
    """For ``decodes_to == "default"``: the default value (when known at compile time)."""
    late: dict[str, Any] | None = None
    """For a late-bound default: ``{"default_from": "from_account.currency"}``."""
    channel: Channel | None = None
    """Channel of the default value (``author`` for schema defaults, ``registry`` for context defaults)."""
    display: str | None = None
    """Display form of the default."""

    def to_doc(self) -> dict[str, Any]:
        """Ballot-document form: ``decodes_to`` and ``text`` always, the rest only when set."""
        doc: dict[str, Any] = {"decodes_to": self.decodes_to, "text": self.text}
        for key in ("value", "late", "channel", "display"):
            value = getattr(self, key)
            if value is not None:
                doc[key] = jsonable(value)
        return doc


class BallotQuestion(BaseModel):
    """One question with its decode map (spec §3.5.7).

    ``primitive`` is ``choice`` (``options`` + ``sentinels``), ``noul`` (optional ``criteria`` ``{"true", "false"}``)
    or ``score`` (``levels``). ``meta`` holds resolver-private decode information (anchor text, list part, reading
    names…); it is part of the Ballot, so decoding needs nothing else.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    qid: str
    family: Family
    tool: str | None = None
    """Tool name (``None`` for ``tool``/``reply``/``segmentation``)."""
    path: tuple[str, ...] = ()
    """Slot path (empty for tool-level questions)."""
    kind: str | None = None
    stakes: str | None = None
    primitive: Primitive
    instructions: JSONContent
    options: list[BallotOption] = Field(default_factory=list)
    sentinels: dict[str, SentinelSpec] = Field(default_factory=dict)
    criteria: dict[str, JSONContent] | None = None
    """Noul criteria ``{"true": …, "false": …}`` (``None`` → sent without criteria)."""
    levels: list[JSONContent] | None = None
    """Score levels, lowest first."""
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_shape(self) -> BallotQuestion:
        if self.primitive == "choice":
            if self.criteria is not None or self.levels is not None:
                raise ValueError(f"{self.qid}: a choice has options and sentinels, not criteria/levels")
            unknown = set(self.sentinels) - SENTINELS
            if unknown:
                raise ValueError(f"{self.qid}: unknown sentinel label(s) {sorted(unknown)}")
        elif self.options or self.sentinels:
            raise ValueError(f"{self.qid}: only choices have options and sentinels")
        if self.primitive == "score" and not self.levels:
            raise ValueError(f"{self.qid}: a score needs levels")
        if self.primitive != "score" and self.levels is not None:
            raise ValueError(f"{self.qid}: only scores have levels")
        if self.primitive != "noul" and self.criteria is not None:
            raise ValueError(f"{self.qid}: only nouls have criteria")
        return self

    @property
    def criteria_null(self) -> bool:
        """``True`` when the wire question carries no ``criteria`` (a Noul without criteria)."""
        return self.primitive == "noul" and self.criteria is None

    @property
    def labels(self) -> list[str]:
        """Choice labels in wire order: options, then sentinels."""
        return [o.label for o in self.options] + list(self.sentinels)

    def decode_label(self, label: str) -> BallotOption | SentinelSpec | None:
        """The decode entry of a label (``None`` for labels this question did not send)."""
        if label in self.sentinels:
            return self.sentinels[label]
        for option in self.options:
            if option.label == label:
                return option
        return None

    def to_wire(self, *, object_instructions: bool = True) -> Question:
        """The wire question (normative: option criteria first, then sentinel texts).

        ``object_instructions=False`` (a backend that rejects JSON-object instructions, §8.7) flattens
        ``{"question", <key>: <value>}`` to ``question + "\n<Key>: " + canonical JSON of value``.
        """
        instructions = self.instructions if object_instructions else flatten_instructions(self.instructions)
        if self.primitive == "choice":
            criteria: dict[str, JSONContent | None] = {o.label: o.text for o in self.options}
            criteria.update({label: s.text for label, s in self.sentinels.items()})
            return ChoiceQuestion(instructions=instructions, criteria=criteria)
        if self.primitive == "noul":
            noul = NoulCriteria(**self.criteria) if self.criteria is not None else None
            return NoulQuestion(instructions=instructions, criteria=noul)
        assert self.levels is not None
        return ScoreQuestion(instructions=instructions, criteria=list(self.levels))

    def to_doc(self) -> dict[str, Any]:
        """Ballot-document form in normative key order."""
        doc: dict[str, Any] = {
            "qid": self.qid,
            "family": self.family,
            "tool": self.tool,
            "path": list(self.path),
            "kind": self.kind,
            "stakes": self.stakes,
            "primitive": self.primitive,
            "instructions": self.instructions,
            "criteria_null": self.criteria_null,
            "options": [jsonable(o) for o in self.options],
            "sentinels": {label: s.to_doc() for label, s in self.sentinels.items()},
        }
        if self.criteria is not None:
            doc["criteria"] = self.criteria
        if self.levels is not None:
            doc["levels"] = self.levels
        doc["meta"] = jsonable(self.meta)
        return doc

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any]) -> BallotQuestion:
        """Inverse of :meth:`to_doc`."""
        return cls.model_validate({k: v for k, v in doc.items() if k != "criteria_null"})


class ToolViability(BaseModel):
    """Per-tool planning record (spec §3.5.7 ``tools``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    tier: Tier
    viable: str
    """``ok``, ``empty:<slot>``, ``channel_blocked:<slot>`` or ``budget``."""
    speculated: bool


class Ballot(BaseModel):
    """The conformance-boundary document of one round (spec §3.5.7)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec: str = SPEC_VERSION
    catalog_sha256: str
    context_sha256: str
    policy_version: str
    mode: Mode = "turn"
    state: JSONContent
    tools: list[ToolViability] = Field(default_factory=list)
    questions: list[BallotQuestion] = Field(default_factory=list)
    calls: list[list[str]] = Field(default_factory=list)
    """Split plan: qids per Jev call (all calls share the state). Left empty at construction → one call with every
    question."""

    @model_validator(mode="before")
    @classmethod
    def _default_plan(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and not data.get("calls"):
            questions = data.get("questions") or []
            qids = [q.qid if isinstance(q, BallotQuestion) else q["qid"] for q in questions]
            data = {**data, "calls": [qids] if qids else []}
        return data

    @model_validator(mode="after")
    def _check_ids(self) -> Ballot:
        qids = [q.qid for q in self.questions]
        if len(set(qids)) != len(qids):
            duplicates = sorted({q for q in qids if qids.count(q) > 1})
            raise ValueError(f"duplicate question ids: {duplicates}")
        if self.calls:
            planned = [qid for call in self.calls for qid in call]
            if sorted(planned) != sorted(qids):
                raise ValueError("calls must list every question exactly once")
        return self

    @property
    def by_qid(self) -> dict[str, BallotQuestion]:
        """Questions keyed by qid."""
        return {q.qid: q for q in self.questions}

    def question(self, qid: str) -> BallotQuestion:
        """The question ``qid`` (``KeyError`` if absent)."""
        for q in self.questions:
            if q.qid == qid:
                return q
        raise KeyError(qid)

    def questions_for(self, tool: str, path: Iterable[str] | None = None) -> list[BallotQuestion]:
        """Questions of one tool (optionally of one slot path), in ballot order."""
        wanted = tuple(path) if path is not None else None
        return [q for q in self.questions if q.tool == tool and (wanted is None or q.path == wanted)]

    def call_plan(self) -> list[list[str]]:
        """The split plan (a single call unless the planner split the questions)."""
        return [list(call) for call in self.calls]

    def wire_ids(self, id_mode: IdMode = "dotted") -> dict[str, str]:
        """``qid → id sent on the wire`` (identity in dotted mode, ``q0001…`` in opaque mode)."""
        qids = [q.qid for q in self.questions]
        return {qid: qid for qid in qids} if id_mode == "dotted" else opaque_ids(qids)

    def to_requests(
        self, model: str, *, id_mode: IdMode = "dotted", object_instructions: bool = True
    ) -> list[DecisionRequest]:
        """Normative Ballot → JevRequest (spec §3.5.7): one request per planned call, questions in call order.

        ``id_mode`` and ``object_instructions`` come from the probed :class:`~jevtools.validate.Limits`.
        """
        ids = self.wire_ids(id_mode)
        by_qid = self.by_qid
        return [
            DecisionRequest(
                model=model,
                state=self.state,
                questions={ids[qid]: by_qid[qid].to_wire(object_instructions=object_instructions) for qid in call},
            )
            for call in self.call_plan()
        ]

    def request_bytes(self, model: str, *, id_mode: IdMode = "dotted", object_instructions: bool = True) -> list[bytes]:
        """Canonical JSON of every request (what golden fixtures compare)."""
        requests = self.to_requests(model, id_mode=id_mode, object_instructions=object_instructions)
        return [canonical_json(r.to_wire()) for r in requests]

    def _body(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "catalog_sha256": self.catalog_sha256,
            "context_sha256": self.context_sha256,
            "policy_version": self.policy_version,
            "mode": self.mode,
            "state": jsonable(self.state),
            "tools": [jsonable(t) for t in self.tools],
            "questions": [q.to_doc() for q in self.questions],
            "calls": self.call_plan(),
        }

    @property
    def sha256(self) -> str:
        """``ballot_sha256``: digest of the document without its own hash field."""
        return sha256_of(self._body())

    def to_doc(self) -> dict[str, Any]:
        """The Ballot document in normative key order (``ballot_sha256`` second)."""
        body = self._body()
        return {"spec": body.pop("spec"), "ballot_sha256": self.sha256, **body}

    def to_json(self) -> bytes:
        """Canonical JSON of :meth:`to_doc`."""
        return canonical_json(self.to_doc())

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any]) -> Ballot:
        """Load a stored Ballot document; a ``ballot_sha256`` that does not match its content is rejected."""
        data = {k: v for k, v in doc.items() if k != "ballot_sha256"}
        data["questions"] = [BallotQuestion.from_doc(q) for q in data.get("questions", [])]
        ballot = cls.model_validate(data)
        stored = doc.get("ballot_sha256")
        if stored is not None and stored != ballot.sha256:
            raise ValueError(f"ballot_sha256 mismatch: stored {stored}, computed {ballot.sha256}")
        return ballot


def flatten_instructions(instructions: JSONContent) -> JSONContent:
    """Text form of object instructions: ``question`` then one ``<Key>: <canonical JSON>`` line per other key."""
    if not isinstance(instructions, dict) or not isinstance(instructions.get("question"), str):
        return instructions
    lines = [instructions["question"]]
    lines += [
        f"{key[:1].upper()}{key[1:]}: {canonical_str(value)}"
        for key, value in instructions.items()
        if key != "question"
    ]
    return "\n".join(lines)


def slot_qid(tool_id: str, qpath: str, *suffix: str | int, seg: int | None = None) -> str:
    """``[s<seg>.]<tool_id>.<qpath>[.<suffix>…]``: ids of slot-level families (``.present``, ``.accept.0``, ``.m1``)."""
    return make_qid(tool_id, qpath, *suffix, seg=seg)


def tool_qid(tool_id: str, family: str, *suffix: str | int, seg: int | None = None) -> str:
    """``<tool_id>.authorized``, ``<tool_id>.joint[.G]``, ``<tool_id>.done_after``."""
    return make_qid(tool_id, family, *suffix, seg=seg)


__all__ = [
    "FAMILY_ORDER",
    "Ballot",
    "BallotOption",
    "BallotQuestion",
    "Family",
    "IdMode",
    "Primitive",
    "SentinelSpec",
    "ToolViability",
    "flatten_instructions",
    "is_valid_qid",
    "slot_qid",
    "tool_qid",
]

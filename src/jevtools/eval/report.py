"""Records of an evaluation run (spec §11.2): one :class:`EvalRecord` per case and replay, collected in an
:class:`EvalReport`.

Every record carries what the metrics and the tuner need without re-running anything: the outcome and rule, the
proposed call and whether it matches gold, the four compositions (W, Π, L, J) and the final C, the **stage** a
failure is attributed to (extractor/source miss, model error, policy…), pool coverage per gold argument, and the
per-question calibration records read off the trace. Reports serialize to JSON so ``jevtools tune`` can run later.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jevtools.confidence import Composition, prior_of
from jevtools.policy import Composition as CompositionRule

Stage = Literal["ok", "backend", "plan", "extractor", "model", "policy", "error"]
"""Where a case failed (spec §11.2 "each failure is traced to its stage"):

- ``ok``: the decision is correct;
- ``backend``: Jev failed (P0, fail closed);
- ``plan``: the gold tool was never speculated (a speculation miss or a non-viable tool);
- ``extractor``: a gold value was not in any candidate pool (extractor, source or retrieval miss);
- ``model``: gold was offered but Jev's answer elected another tool or value;
- ``policy``: the proposed call is right but the outcome is not one of ``outcomes_ok``;
- ``error``: the harness caught an exception (a jevtools bug or a broken case).
"""


class QuestionRecord(BaseModel):
    """One calibration observation from a trace: the probability Jev gave and whether that event was true.

    For Choices ``p`` is the top option's mass (top-label calibration) unless ``sentinel`` names a sentinel whose
    own mass is recorded; for Nouls ``p`` is ``P(yes)``. ``family`` is the question family (``tool``, ``slot``,
    ``accept``…); ``sub`` refines it (``content``/``cosmetic`` stakes of accepts, the slot kind of slot questions).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    qid: str
    family: str
    sub: str | None = None
    tool: str | None = None
    primitive: str
    p: float
    correct: bool
    label: str | None = None
    sentinel: str | None = None
    round: int = 1
    on_gold_tool: bool = False
    """The question belongs to the gold tool (its premise is true)."""


class PoolHit(BaseModel):
    """Coverage of one gold argument in the pools sent to Jev."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    in_pool: bool
    rank: int | None = None
    """1-based rank of the gold value among the real candidates (``None`` when absent or reached by a default)."""
    size: int = 0
    via_default: bool = False
    """Gold is reachable only through the ``NOT_STATED`` default."""
    kind: str | None = None
    """The slot kind (``ref``, ``temporal``, ``text``…) for recall@K per kind."""


class EvalRecord(BaseModel):
    """The outcome of one case in one replay."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    replay: int = 0
    tags: list[str] = Field(default_factory=list)
    locale: str | None = None
    outcome: str
    rule: str
    outcomes_ok: list[str]
    gold_tool: str | None = None
    tool: str | None = None
    """Name of the proposed call (``None`` when no call was proposed)."""
    arguments: dict[str, Any] | None = None
    tier: str | None = None
    composition: str | None = None
    C: float | None = None
    W: float | None = None
    PI: float | None = None
    L: float | None = None
    J: float | None = None
    calibrated: bool = False
    outcome_ok: bool = False
    call_match: bool = False
    """The proposed call is the gold call (or neither exists)."""
    correct: bool = False
    """``outcome_ok`` and, when the outcome shows a call (execute/confirm), ``call_match``."""
    executed: bool = False
    wrong_if_executed: bool = True
    """Executing the proposed call would be wrong: not the gold call, or gold does not allow ``execute``."""
    stage: Stage = "ok"
    stage_detail: str | None = None
    tool_speculated: bool | None = None
    tool_top: str | None = None
    """The tool Choice's top label (``None`` when no tool question was asked)."""
    pool: dict[str, PoolHit] = Field(default_factory=dict)
    menu: list[dict[str, Any]] | None = None
    menu_has_gold: bool | None = None
    rounds: int = 0
    jev_calls: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    cost_usd: float | None = None
    flags: list[str] = Field(default_factory=list)
    planted_hit: bool | None = None
    """For injection cases (``meta.planted``): a planted value reached a call shown to the host."""
    questions: list[QuestionRecord] = Field(default_factory=list)
    decision: dict[str, Any] | None = None
    trace: dict[str, Any] | None = None
    error: str | None = None

    def composition_of(self) -> Composition | None:
        """W/Π/L/J as a :class:`~jevtools.confidence.Composition` (``None`` without a confidence record)."""
        if self.W is None or self.PI is None or self.L is None:
            return None
        return Composition(W=self.W, PI=self.PI, L=self.L, J=self.J)

    def score(self, rule: CompositionRule | Literal["C"] = "C") -> float | None:
        """The confidence under a composition rule (``W``, ``PI``, ``L``, ``J``, ``MIN_L_J``) or the final ``C``."""
        if rule == "C":
            return self.C
        comp = self.composition_of()
        return prior_of(comp, rule) if comp is not None else None


class EvalReport(BaseModel):
    """All records of a run plus its provenance (policy, backend, model, replays)."""

    model_config = ConfigDict(extra="forbid")

    meta: dict[str, Any] = Field(default_factory=dict)
    records: list[EvalRecord] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def replays(self) -> int:
        """Number of replays per case."""
        return 1 + max((r.replay for r in self.records), default=0)

    def first(self) -> list[EvalRecord]:
        """Records of replay 0 (one per case): what call-level metrics count."""
        return [r for r in self.records if r.replay == 0]

    def by_case(self) -> dict[str, list[EvalRecord]]:
        """Records grouped by case id (replays in order)."""
        out: dict[str, list[EvalRecord]] = {}
        for record in self.records:
            out.setdefault(record.case_id, []).append(record)
        return out

    def with_tags(self, *tags: str) -> list[EvalRecord]:
        """Replay-0 records carrying any of ``tags``."""
        wanted = set(tags)
        return [r for r in self.first() if wanted & set(r.tags)]

    def summary(self) -> dict[str, Any]:
        """The headline metrics (see :func:`jevtools.eval.metrics.summarize`)."""
        from jevtools.eval.metrics import summarize

        return summarize(self)

    def failures(self) -> list[EvalRecord]:
        """Replay-0 records that are not correct."""
        return [r for r in self.first() if not r.correct]

    def to_doc(self, *, traces: bool = True) -> dict[str, Any]:
        """JSON-ready document (``traces=False`` drops stored traces and decisions to keep files small)."""
        exclude = {"records": {"__all__": {"trace", "decision"}}} if not traces else None
        return self.model_dump(mode="json", exclude=exclude)

    def save(self, path: str | os.PathLike[str], *, traces: bool = True) -> None:
        """Write the report as JSON."""
        Path(path).write_text(json.dumps(self.to_doc(traces=traces), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> EvalReport:
        """Read a report written by :meth:`save`."""
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def merge(cls, reports: Iterable[EvalReport]) -> EvalReport:
        """Concatenate reports (their ``meta`` is kept under ``parts``)."""
        items = list(reports)
        return cls(meta={"parts": [r.meta for r in items]}, records=[rec for r in items for rec in r.records])


def new_meta(**fields: Any) -> dict[str, Any]:
    """Report metadata with a UTC ``created_at``."""
    return {"created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), **fields}


def records_of(data: EvalReport | Sequence[EvalRecord]) -> list[EvalRecord]:
    """Replay-0 records of a report, or the given records as they are."""
    return data.first() if isinstance(data, EvalReport) else list(data)


def tagged(records: Sequence[EvalRecord], tags: Mapping[str, Any] | Iterable[str]) -> list[EvalRecord]:
    """Records carrying any of ``tags``."""
    wanted = set(tags)
    return [r for r in records if wanted & set(r.tags)]


__all__ = [
    "EvalRecord",
    "EvalReport",
    "PoolHit",
    "QuestionRecord",
    "Stage",
    "new_meta",
    "records_of",
    "tagged",
]

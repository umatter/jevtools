"""Labels from confirm cards, for tuning a policy on an app's own traffic (spec §11.3).

Thresholds and calibration do not transfer between apps (BENCH "Does one calibration fit every bench?"), so they are
tuned on the app's own decisions. A confirm card labels its proposed call for free: the user **accepts** it (the call
was right), **edits** it or **cancels** it (executing it would have been wrong). In ``shadow`` mode every call the
policy would execute is shown as a card too, so the labels cover the execute band as well.

- :class:`FeedbackLog` appends one JSONL row per labelled card. A :class:`~jevtools.router.Router` given
  ``feedback=`` logs the outcome of each confirm card when it is resumed; an app can also log a label itself
  (``log.add(decision, "undone")`` when a user reverts an executed call).
- :meth:`FeedbackLog.report` turns the rows into an :class:`~jevtools.eval.EvalReport`, the input of
  :func:`jevtools.eval.tune`.
- :func:`check_policy` measures a policy on rows it was not tuned on (the newest ones): how many calls it would
  execute, how many of those were wrong, and whether the wrong-execution bound meets each tier's budget.
  ``jevtools tune --feedback LOG`` tunes on the older rows and checks on the newer ones before a policy is switched.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from jevtools.confidence import Composition, prior_of
from jevtools.policy import Policy, Tier
from jevtools.policy import _clears as clears

if TYPE_CHECKING:  # jevtools.eval imports the router, which imports this module: eval is imported lazily
    from jevtools.decision import Decision
    from jevtools.eval.report import EvalRecord, EvalReport

Label = Literal["accepted", "edited", "cancelled", "undone"]
"""``accepted``: the proposed call was right. ``edited`` / ``cancelled``: the user changed or refused it before it
ran. ``undone``: the user reverted a call that had run. Only ``accepted`` counts as right."""
LABELS: tuple[str, ...] = ("accepted", "edited", "cancelled", "undone")

__all__ = ["LABELS", "FeedbackLog", "FeedbackRecord", "Label", "TierCheck", "check_policy", "label_of", "split"]


class FeedbackRecord(BaseModel):
    """One labelled decision: its call, confidence parts and the user's verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: str
    at: datetime
    label: Label
    tool: str
    arguments: dict[str, Any]
    outcome: str
    rule: str
    tier: str
    composition: str
    C: float
    W: float
    PI: float
    L: float
    J: float | None = None
    shadow: bool = False
    """The card was shown only because the policy runs in shadow mode (the call would have executed)."""

    @property
    def right(self) -> bool:
        return self.label == "accepted"

    @classmethod
    def of(cls, decision: Decision, label: Label, *, at: datetime | None = None) -> FeedbackRecord | None:
        """The record of a decision with a call and a confidence record (``None`` otherwise: nothing to label)."""
        if label not in LABELS:
            raise ValueError(f"unknown feedback label {label!r}; expected one of {LABELS}")
        conf, call = decision.confidence, decision.call
        if conf is None or call is None:
            return None
        outcome = getattr(decision.trace, "outcome", None)
        caps = outcome.get("caps", []) if isinstance(outcome, Mapping) else []
        return cls(
            decision_id=decision.decision_id, at=at or datetime.now(timezone.utc), label=label, tool=call.name,
            arguments=dict(call.arguments), outcome=str(decision.outcome), rule=decision.rule, tier=conf.tier,
            composition=conf.composition, C=conf.call, W=conf.W, PI=conf.PI, L=conf.L, J=conf.J,
            shadow="shadow" in caps,
        )  # fmt: skip

    def to_eval_record(self) -> EvalRecord:
        """The row as :func:`jevtools.eval.tune` reads it: right = ``call_match``, wrong if executed otherwise."""
        from jevtools.eval.report import EvalRecord

        return EvalRecord(
            case_id=self.decision_id, outcome=self.outcome, rule=self.rule,
            outcomes_ok=["execute", "confirm"] if self.right else ["clarify"], gold_tool=self.tool if self.right
            else None, tool=self.tool, arguments=self.arguments, tier=self.tier, composition=self.composition,
            C=self.C, W=self.W, PI=self.PI, L=self.L, J=self.J, call_match=self.right, correct=self.right,
            wrong_if_executed=not self.right,
        )  # fmt: skip


def label_of(action: str | None) -> Label:
    """The label of a confirm card's resume: ``confirm`` → accepted, ``cancel`` → cancelled, anything else (a
    changed value, another tool, a free-text reply) → edited."""
    if action == "confirm":
        return "accepted"
    if action == "cancel":
        return "cancelled"
    return "edited"


class FeedbackLog:
    """An append-only JSONL file of :class:`FeedbackRecord` rows (thread-safe within a process)."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def add(self, decision: Decision, label: Label, *, at: datetime | None = None) -> FeedbackRecord | None:
        """Append the label of ``decision`` (ignored, returning ``None``, when it has no call to label)."""
        record = FeedbackRecord.of(decision, label, at=at)
        if record is None:
            return None
        line = record.model_dump_json() + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        return record

    def records(self) -> list[FeedbackRecord]:
        """Every row, oldest first."""
        if not self.path.exists():
            return []
        rows = [FeedbackRecord.model_validate(json.loads(line))
                for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]  # fmt: skip
        return sorted(rows, key=lambda r: r.at)

    def report(self, records: Iterable[FeedbackRecord] | None = None) -> EvalReport:
        """The rows (default: all) as an :class:`EvalReport` for :func:`jevtools.eval.tune`."""
        from jevtools.eval.report import EvalReport

        rows = list(self.records() if records is None else records)
        return EvalReport(meta={"source": "feedback", "log": str(self.path), "rows": len(rows)},
                          records=[r.to_eval_record() for r in rows])  # fmt: skip


def split(records: Sequence[FeedbackRecord], check: float = 0.2) -> tuple[list[FeedbackRecord], list[FeedbackRecord]]:
    """``(fit, check)``: the oldest rows to tune on and the newest ``check`` share to test the tuned policy on."""
    rows = sorted(records, key=lambda r: r.at)
    cut = len(rows) - max(1, round(len(rows) * check)) if rows else 0
    return rows[:cut], rows[cut:]


@dataclass(frozen=True)
class TierCheck:
    """A policy measured on labelled rows of one tier."""

    tier: str
    n: int
    executed: int
    wrong: int
    upper: float | None
    """One-sided 95% Clopper–Pearson upper bound of the wrong-execution rate among the executed rows."""
    alpha: float
    threshold: float | None

    @property
    def automation(self) -> float:
        return self.executed / self.n if self.n else 0.0

    @property
    def ok(self) -> bool:
        """Whether the wrong-execution bound meets the tier's budget (true when nothing would execute)."""
        return self.upper is None or self.upper <= self.alpha

    @property
    def status(self) -> Literal["ok", "unproven", "over"]:
        """``ok``: the bound meets the budget; ``unproven``: the observed rate does, but too few rows to show it;
        ``over``: the observed rate exceeds the budget."""
        if self.ok:
            return "ok"
        return "unproven" if self.wrong / self.executed <= self.alpha else "over"


def check_policy(records: Sequence[FeedbackRecord], policy: Policy,
                 alphas: Mapping[str, float] | None = None) -> dict[str, TierCheck]:  # fmt: skip
    """Per tier: which rows ``policy`` would execute (its composition, execute threshold and hysteresis, re-applied to
    each row's W/Π/L/J, as the runtime policy tests them) and how many of those the user did not accept."""
    from jevtools.eval.stats import clopper_pearson_upper
    from jevtools.eval.tuning import DEFAULT_ALPHAS

    budget = {**DEFAULT_ALPHAS, **dict(alphas or {})}
    out: dict[str, TierCheck] = {}
    for tier in Tier:
        rows = [r for r in records if r.tier == tier.value]
        if not rows:
            continue
        settings = getattr(policy.tiers, tier.value)
        threshold = settings.execute if isinstance(settings.execute, (int, float)) else None
        scores = [prior_of(Composition(W=r.W, PI=r.PI, L=r.L, J=r.J), settings.composition) for r in rows]
        executed = [r for r, score in zip(rows, scores, strict=True)
                    if threshold is not None and clears(score, threshold, policy.hysteresis)]  # fmt: skip
        wrong = sum(not r.right for r in executed)
        upper = clopper_pearson_upper(wrong, len(executed), 0.95) if executed else None
        out[tier.value] = TierCheck(tier=tier.value, n=len(rows), executed=len(executed), wrong=wrong, upper=upper,
                                    alpha=budget[tier.value], threshold=threshold)  # fmt: skip
    return out

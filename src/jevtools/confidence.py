"""Confidence composition (spec §3.7): factors → W, Π, L, J → tier prior → calibrator → coherence cap.

- **W** = ``min F`` (Fréchet upper bound, the cookbook's weakest-judgment rule).
- **Π** = ``∏ F`` (exact under independence; conservative under positive dependence).
- **L** = ``max(0, 1 − Σ(1 − f))`` (Fréchet lower bound, valid under any dependence).
- **J** = the joint Choice mass on the factorized MAP (only when a joint question was asked).

``L ≤ Π ≤ W`` always. The tier picks the prior (read W; write/external Π; critical ``min(L, J)``), an optional
isotonic calibrator maps it to C, and the coherence cap ``C ≤ W`` keeps the call no surer than its weakest part.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from jevtools.policy import Composition as CompositionRule
from jevtools.policy import Policy, Tier

# --------------------------------------------------------------------------------------------------------------------
# Factors and compositions
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Factors:
    """The factor set F of the elected tool (§3.7.1), in order: ``tool``, ``authorized``, then slots.

    Cosmetic slots, ``present``, ``done_after`` and ``joint`` are never factors.
    """

    items: dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(
        cls, *, tool: float, authorized: float | None = None, slots: Mapping[str, float | None] | None = None
    ) -> Factors:
        """Assemble factors; ``None`` slot factors (cosmetic, unasked defaults) are skipped."""
        items: dict[str, float] = {"tool": tool}
        if authorized is not None:
            items["authorized"] = authorized
        for name, f in (slots or {}).items():
            if f is not None:
                items[name] = f
        return cls(items)

    def values(self) -> list[float]:
        """Factor values in order."""
        return list(self.items.values())

    def slot_items(self) -> dict[str, float]:
        """Slot factors only (not ``tool``/``authorized``): the inputs of ``Q_t`` (§3.7.4)."""
        return {k: v for k, v in self.items.items() if k not in ("tool", "authorized")}

    def weakest(self) -> str | None:
        """Name of the smallest factor (first one on ties)."""
        return min(self.items, key=lambda k: self.items[k]) if self.items else None


@dataclass(frozen=True)
class Composition:
    """All four compositions plus the tier prior, calibration and final C (§3.7.2, §3.7.3)."""

    W: float
    PI: float
    L: float
    J: float | None = None
    tier: str | None = None
    rule: str | None = None
    """How the tier composed its prior (``W``, ``PI``, ``MIN_L_J``…)."""
    prior: float | None = None
    C: float | None = None
    calibrator: str | None = None
    """Name of the calibrator applied (``None`` when uncalibrated)."""

    @property
    def calibrated(self) -> bool:
        """Whether a fitted calibrator produced C."""
        return self.calibrator is not None

    def to_doc(self) -> dict[str, Any]:
        """Trace form: ``{W, PI, L, J, tier, rule, calibrator, C}``."""
        return {"W": self.W, "PI": self.PI, "L": self.L, "J": self.J, "tier": self.tier, "rule": self.rule,
                "calibrator": self.calibrator, "C": self.C}  # fmt: skip


def compose(factors: Factors | Mapping[str, float] | Iterable[float], J: float | None = None) -> Composition:
    """W, Π and L of a factor set (an empty set composes to 1.0: nothing can be wrong), and J as given."""
    values = _values(factors)
    for f in values:
        if not 0.0 <= f <= 1.0 or math.isnan(f):
            raise ValueError(f"factor {f!r} is not a probability")
    W = min(values, default=1.0)
    PI = math.prod(values)
    L = max(0.0, 1.0 - sum(1.0 - f for f in values))
    return Composition(W=W, PI=PI, L=L, J=J)


def _values(factors: Factors | Mapping[str, float] | Iterable[float]) -> list[float]:
    if isinstance(factors, Factors):
        return factors.values()
    if isinstance(factors, Mapping):
        return [float(v) for v in factors.values()]
    return [float(v) for v in factors]


def prior_of(comp: Composition, rule: CompositionRule) -> float:
    """The prior a composition rule selects (``MIN_L_J`` falls back to L when J was not asked)."""
    if rule == "W":
        return comp.W
    if rule == "PI":
        return comp.PI
    if rule == "L":
        return comp.L
    if rule == "J":
        return comp.J if comp.J is not None else comp.L
    return min(comp.L, comp.J) if comp.J is not None else comp.L


def tier_prior(comp: Composition, tier: Tier | str, policy: Policy | None = None) -> float:
    """``C_prior`` of a tier (§3.7.3): read W, write Π, external Π, critical ``min(L, J)``."""
    policy = policy or Policy()
    return prior_of(comp, policy.tier(tier).composition)


def finalize(
    comp: Composition,
    tier: Tier | str,
    policy: Policy | None = None,
    calibrators: Mapping[str, IsotonicCalibrator] | None = None,
) -> Composition:
    """Tier prior → calibrator (if one is fitted for the tier) → coherence cap ``C ≤ W``."""
    policy = policy or Policy()
    tier = Tier(tier)
    rule = policy.tier(tier).composition
    prior = prior_of(comp, rule)
    calibrator = (calibrators or {}).get(tier.value)
    c = calibrator.predict(prior) if calibrator is not None and calibrator.fitted else prior
    name = calibrator.name if calibrator is not None and calibrator.fitted else None
    return Composition(W=comp.W, PI=comp.PI, L=comp.L, J=comp.J, tier=tier.value, rule=rule, prior=prior,
                       C=min(c, comp.W), calibrator=name)  # fmt: skip


def confidence(
    factors: Factors,
    tier: Tier | str,
    *,
    J: float | None = None,
    policy: Policy | None = None,
    calibrators: Mapping[str, IsotonicCalibrator] | None = None,
) -> Composition:
    """:func:`compose` then :func:`finalize`."""
    return finalize(compose(factors, J), tier, policy, calibrators)


# --------------------------------------------------------------------------------------------------------------------
# Call MAP (§3.7.4)
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CallMap:
    """Tool selection vs call MAP: ``t* = argmax P(t)``, ``t_S = argmax P(t)·Q_t`` over speculated tools."""

    chosen: str | None
    call_map: str | None
    scores: dict[str, float]

    @property
    def disagrees(self) -> bool:
        """``call_map_disagrees``: both are defined and differ."""
        return self.chosen is not None and self.call_map is not None and self.chosen != self.call_map


CALL_MAP_MIN_P = 0.10
"""A tool competes for the call MAP only from this tool probability: an infeasible call of the tool the user asked
for is a consistency problem of that call (``infeasible``, P8), not a reason to offer an unrelated tool."""


def call_map(tool_p: Mapping[str, float], q: Mapping[str, float]) -> CallMap:
    """``S(t) = P(t)·Q_t`` for every tool with a ``Q_t``; ``t*`` is the argmax of ``tool_p`` (first on ties).

    ``tool_p`` may contain sentinel labels (``NO_TOOL``…); only tools present in ``q`` get a score. ``t_S`` is the
    argmax of ``S`` over ``t*`` and the tools with ``P(t) ≥ CALL_MAP_MIN_P``.
    """
    chosen = max(tool_p, key=lambda t: tool_p[t]) if tool_p else None
    scores = {t: tool_p.get(t, 0.0) * qt for t, qt in q.items()}
    rivals = {t: s for t, s in scores.items() if t == chosen or tool_p.get(t, 0.0) >= CALL_MAP_MIN_P - 1e-9}
    best = max(rivals, key=lambda t: rivals[t]) if rivals else None
    return CallMap(chosen=chosen, call_map=best, scores=scores)


# --------------------------------------------------------------------------------------------------------------------
# Isotonic calibration (pure-Python PAV)
# --------------------------------------------------------------------------------------------------------------------


class IsotonicCalibrator:
    """Monotone (non-decreasing) calibration map fitted by pool-adjacent-violators on ``(C_prior, correct)`` pairs.

    ``predict`` interpolates linearly between block centres and is constant outside the fitted range.
    """

    def __init__(self, name: str = "isotonic") -> None:
        self.name = name
        self.xs: list[float] = []
        self.ys: list[float] = []

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has run on at least one pair."""
        return bool(self.xs)

    def fit(
        self, xs: Sequence[float], ys: Sequence[float | bool], weights: Sequence[float] | None = None
    ) -> IsotonicCalibrator:
        """Fit on predictions ``xs`` and outcomes ``ys`` (booleans or rates in [0, 1]).

        Tied predictions are pooled first (one point per distinct ``x``, the secondary tie approach): a calibration
        map is a function of ``x``, so every observation at one ``x`` must end in the same block."""
        if len(xs) != len(ys):
            raise ValueError("xs and ys must have the same length")
        w = list(weights) if weights is not None else [1.0] * len(xs)
        pooled: dict[float, list[float]] = {}  # x → [sum_wy, sum_w]
        for x, y, weight in zip((float(x) for x in xs), (float(y) for y in ys), w, strict=True):
            point = pooled.setdefault(x, [0.0, 0.0])
            point[0] += weight * y
            point[1] += weight
        blocks: list[list[float]] = []  # [sum_wx, sum_wy, sum_w]
        for x in sorted(pooled):
            sum_wy, sum_w = pooled[x]
            if sum_w <= 0:
                continue
            blocks.append([sum_w * x, sum_wy, sum_w])
            while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] >= blocks[-1][1] / blocks[-1][2]:
                last = blocks.pop()
                for i in range(3):
                    blocks[-1][i] += last[i]
        self.xs = [b[0] / b[2] for b in blocks]
        self.ys = [b[1] / b[2] for b in blocks]
        return self

    def predict(self, x: float) -> float:
        """Calibrated value of ``x`` (identity before fitting)."""
        if not self.xs:
            return float(x)
        if x <= self.xs[0]:
            return self.ys[0]
        if x >= self.xs[-1]:
            return self.ys[-1]
        i = bisect.bisect_right(self.xs, x)
        x0, x1, y0, y1 = self.xs[i - 1], self.xs[i], self.ys[i - 1], self.ys[i]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def to_dict(self) -> dict[str, Any]:
        """Serializable form (``{"name", "xs", "ys"}``)."""
        return {"name": self.name, "xs": list(self.xs), "ys": list(self.ys)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IsotonicCalibrator:
        """Inverse of :meth:`to_dict`."""
        cal = cls(str(data.get("name", "isotonic")))
        cal.xs = [float(x) for x in data.get("xs", [])]
        cal.ys = [float(y) for y in data.get("ys", [])]
        return cal


__all__ = [
    "CALL_MAP_MIN_P",
    "CallMap",
    "Composition",
    "Factors",
    "IsotonicCalibrator",
    "call_map",
    "compose",
    "confidence",
    "finalize",
    "prior_of",
    "tier_prior",
]

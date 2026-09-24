"""Threshold tuning, calibration and certification (spec §11.3, §11.4).

For each risk tier, :func:`tune` chooses the composition (W, Π, L, J, ``MIN_L_J``), ``τ_execute`` and ``τ_confirm``
that **maximize automation** subject to the one-sided 95% Clopper–Pearson upper bound on the wrong-execution rate
among the cases the policy would execute being ≤ ``α_tier`` (defaults: read 5%, write 2%, external 1%,
critical 0.1%). ``method="crc"`` uses split conformal risk control instead. Optionally an isotonic calibrator
(:class:`~jevtools.confidence.IsotonicCalibrator`) is fitted per tier on ``(C_prior, call correct)`` pairs.

The output is a **versioned** :class:`~jevtools.policy.Policy` (TOML via :meth:`TuningResult.to_toml`) plus the
calibrators document; the policy's ``notes.tuning`` records how it was tuned and cites the calibrators' hash, and
every trace cites the policy hash.

Rules the tuner enforces:

- A tuned threshold reproduces the policy's hysteresis test (``C − τ ≥ h`` clears): the bound is computed on
  exactly the cases the policy would let through, with thresholds rounded **up** to 4 decimals.
- A tier without a threshold meeting its bound gets ``execute = "never"`` (read/write/external); a tier without
  data keeps its prior thresholds.
- The critical tier stays **confirm-only** unless at least 3,000 labelled critical-tier cases back the threshold
  (§11.4, the rule of three); only then is ``tiers.critical.auto_execute`` set, with
  ``certified.critical_cases``.

Thresholds do not transfer across datasets: tune on the adopter's own labelled traffic (collected in shadow mode).
Offline scripted or simulated runs exercise this code; they certify nothing about Jev.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from jevtools.canonical import sha256_of
from jevtools.confidence import IsotonicCalibrator, prior_of
from jevtools.eval.metrics import hysteresis_width
from jevtools.eval.report import EvalRecord, EvalReport, records_of
from jevtools.eval.stats import clopper_pearson_upper
from jevtools.policy import CRITICAL_CERTIFICATION_CASES, Policy, Tier
from jevtools.policy import Composition as CompositionRule

Method = Literal["cp", "crc"]
TierStatus = Literal["tuned", "never", "confirm_only", "no_data"]
DEFAULT_ALPHAS: dict[str, float] = {"read": 0.05, "write": 0.02, "external": 0.01, "critical": 0.001}
"""Default wrong-execution budgets per tier (spec §11.3)."""
DEFAULT_CONFIRM_ALPHA = 0.5
"""A confirm card is worth showing when its proposed call is right more often than not (95% bound)."""
COMPOSITIONS: tuple[CompositionRule, ...] = ("W", "PI", "L", "J", "MIN_L_J")
_EPS = 1e-9

__all__ = [
    "COMPOSITIONS",
    "DEFAULT_ALPHAS",
    "DEFAULT_CONFIRM_ALPHA",
    "Certification",
    "Method",
    "ThresholdChoice",
    "TierTuning",
    "TuningResult",
    "calibrators_doc",
    "certify_critical",
    "clopper_pearson_upper",
    "confirm_threshold",
    "execute_threshold",
    "fit_calibrators",
    "load_calibrators",
    "policy_to_toml",
    "save_calibrators",
    "tune",
    "tune_thresholds",
]


# --------------------------------------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------------------------------------


def _ceil4(x: float) -> float:
    """Round up to 4 decimals (a threshold never admits more than the evaluated set)."""
    return math.ceil(x * 10_000 - 1e-7) / 10_000


def _clears(score: float, tau: float, h: float) -> bool:
    """The policy's hysteresis test (``jevtools.policy``): ``C − τ ≥ h`` clears."""
    return score - tau >= h - _EPS


def _tier_records(records: Sequence[EvalRecord], tier: str) -> list[EvalRecord]:
    return [r for r in records if r.tier == tier and r.W is not None and r.PI is not None and r.L is not None]


def _score(record: EvalRecord, rule: CompositionRule, calibrator: IsotonicCalibrator | None) -> float:
    comp = record.composition_of()
    assert comp is not None
    prior = prior_of(comp, rule)
    if calibrator is not None and calibrator.fitted:
        return min(calibrator.predict(prior), comp.W)
    return prior


def fit_calibrators(
    data: EvalReport | Sequence[EvalRecord],
    policy: Policy | None = None,
    *,
    rules: Mapping[str, CompositionRule] | None = None,
    min_cases: int = 1,
) -> dict[str, IsotonicCalibrator]:
    """Isotonic calibrators per tier, fitted on ``(C_prior, proposed call is the gold call)`` over the tier's
    decisions; ``C_prior`` uses the tier's composition from ``rules`` (default: the policy's). Tiers with fewer than
    ``min_cases`` decisions get none."""
    policy = policy or Policy()
    records = records_of(data)
    out: dict[str, IsotonicCalibrator] = {}
    for tier in Tier:
        rule = (rules or {}).get(tier.value) or policy.tier(tier).composition
        items = _tier_records(records, tier.value)
        if len(items) < max(1, min_cases):
            continue
        xs = [_score(r, rule, None) for r in items]
        out[tier.value] = IsotonicCalibrator(f"isotonic.{tier.value}.{rule}").fit(xs, [r.call_match for r in items])
    return out


def calibrators_doc(calibrators: Mapping[str, IsotonicCalibrator]) -> dict[str, Any]:
    """The calibrators document: ``{"tiers": {tier: {name, xs, ys}}, "sha256": …}``."""
    tiers = {tier: cal.to_dict() for tier, cal in sorted(calibrators.items())}
    return {"tiers": tiers, "sha256": sha256_of(tiers)}


def save_calibrators(calibrators: Mapping[str, IsotonicCalibrator], path: str | os.PathLike[str]) -> None:
    """Write :func:`calibrators_doc` as JSON."""
    Path(path).write_text(json.dumps(calibrators_doc(calibrators), ensure_ascii=False, indent=1), encoding="utf-8")


def load_calibrators(source: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, IsotonicCalibrator]:
    """Calibrators from a document or file written by :func:`save_calibrators` (``Router(calibrators=…)``)."""
    doc = source if isinstance(source, Mapping) else json.loads(Path(source).read_text(encoding="utf-8"))
    tiers = doc.get("tiers", doc)
    return {str(tier): IsotonicCalibrator.from_dict(data) for tier, data in tiers.items()}


# --------------------------------------------------------------------------------------------------------------------
# Threshold search
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdChoice:
    """A threshold and the evidence behind it: the ``kept`` cases it lets through, the ``errors`` among them, the
    bound and the automation share (``kept / n``)."""

    composition: str
    threshold: float
    kept: int
    errors: int
    n: int
    upper: float

    @property
    def automation(self) -> float:
        """Share of the tier's decisions the threshold lets through."""
        return self.kept / self.n if self.n else 0.0


def _candidates(scores: Sequence[float], h: float) -> list[float]:
    """Thresholds worth testing: each distinct score minus ``h`` (rounded up), highest first."""
    taus = {_ceil4(max(0.0, s - h)) for s in scores if s >= h - _EPS}
    return sorted(taus, reverse=True)


class _Kept:
    """The cases a threshold lets through, in O(log n) per threshold: scores sorted with prefix counts of flagged
    cases, reproducing the policy's hysteresis test ``C − τ ≥ h``."""

    def __init__(self, scores: Sequence[float], flags: Sequence[bool], h: float) -> None:
        if len(scores) != len(flags):
            raise ValueError("scores and flags must have the same length")
        pairs = sorted(zip(scores, flags, strict=True), key=lambda sf: -sf[0])
        self.h = h
        self.scores = [s for s, _ in pairs]
        self.prefix = [0]
        for _, flag in pairs:
            self.prefix.append(self.prefix[-1] + int(flag))

    def __call__(self, tau: float) -> tuple[int, int]:
        """``(flagged, kept)`` among the cases with ``score − τ ≥ h``."""
        lo, hi = 0, len(self.scores)
        while lo < hi:  # first index whose score does not clear
            mid = (lo + hi) // 2
            if _clears(self.scores[mid], tau, self.h):
                lo = mid + 1
            else:
                hi = mid
        return self.prefix[lo], lo


def execute_threshold(
    scores: Sequence[float],
    wrong: Sequence[bool],
    alpha: float,
    *,
    conf: float = 0.95,
    hysteresis: float = 0.0,
    method: Method = "cp",
    composition: str = "C",
) -> ThresholdChoice | None:
    """The threshold letting through the most cases whose wrong-execution risk meets ``alpha``.

    ``cp``: the one-sided Clopper–Pearson upper bound (level ``conf``) of the wrong rate among the kept cases is
    ≤ ``alpha``. ``crc``: split conformal risk control on the loss ``wrong ∧ kept``:
    ``(Σ loss + 1) / (n + 1) ≤ alpha``. ``None`` when no threshold qualifies.
    """
    count = _Kept(scores, wrong, hysteresis)
    n = len(scores)
    best: ThresholdChoice | None = None
    for tau in _candidates(scores, hysteresis):
        k, m = count(tau)
        if not m or (best is not None and m <= best.kept):
            continue
        if method == "cp":
            if k / m > alpha:
                continue
            bound = clopper_pearson_upper(k, m, conf)
        else:
            bound = (k + 1) / (n + 1)
        if bound <= alpha:
            best = ThresholdChoice(composition=composition, threshold=tau, kept=m, errors=k, n=n, upper=bound)
    return best


def confirm_threshold(
    scores: Sequence[float],
    call_wrong: Sequence[bool],
    alpha: float,
    *,
    conf: float = 0.95,
    hysteresis: float = 0.0,
    ceiling: float | None = None,
    composition: str = "C",
) -> ThresholdChoice | None:
    """The lowest threshold (≤ ``ceiling``) whose kept cases have a proposed call that is wrong at most ``alpha``
    of the time (Clopper–Pearson bound): below it a confirm card is not worth showing."""
    count = _Kept(scores, call_wrong, hysteresis)
    n = len(scores)
    best: ThresholdChoice | None = None
    for tau in _candidates(scores, hysteresis):
        if ceiling is not None and tau > ceiling + _EPS:
            continue
        k, m = count(tau)
        if not m or k / m > alpha or (best is not None and m <= best.kept):
            continue
        bound = clopper_pearson_upper(k, m, conf)
        if bound <= alpha:
            best = ThresholdChoice(composition=composition, threshold=tau, kept=m, errors=k, n=n, upper=bound)
    return best


# --------------------------------------------------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Certification:
    """The §11.4 certification of critical auto-execution: labelled critical cases, the evidence at the chosen
    threshold, and whether auto-execution is unlocked (≥ 3,000 cases and the bound ≤ α)."""

    cases: int
    alpha: float
    certified: bool
    threshold: float | None = None
    kept: int = 0
    errors: int = 0
    upper: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class TierTuning:
    """The tuning outcome of one tier."""

    tier: str
    n: int
    alpha: float
    status: TierStatus
    composition: str
    execute: float | Literal["never"] | None
    confirm: float | None
    execute_choice: ThresholdChoice | None = None
    confirm_choice: ThresholdChoice | None = None
    calibrator: str | None = None

    def to_note(self) -> dict[str, Any]:
        """TOML-safe summary for ``policy.notes.tuning.tiers``."""
        doc: dict[str, Any] = {"n": self.n, "alpha": self.alpha, "status": self.status,
                               "composition": self.composition}  # fmt: skip
        if self.execute_choice is not None:
            c = self.execute_choice
            doc.update(kept=c.kept, errors=c.errors, upper=round(c.upper, 6), automation=round(c.automation, 4))
        if self.confirm_choice is not None:
            doc["confirm_kept"] = self.confirm_choice.kept
        if self.calibrator:
            doc["calibrator"] = self.calibrator
        return doc


@dataclass
class TuningResult:
    """A tuned policy with its per-tier evidence, calibrators and the critical certification."""

    policy: Policy
    tiers: dict[str, TierTuning]
    certification: Certification
    calibrators: dict[str, IsotonicCalibrator] = field(default_factory=dict)
    hysteresis: float = 0.03
    method: Method = "cp"

    def to_toml(self) -> str:
        """The versioned ``policy.toml``."""
        return policy_to_toml(self.policy)

    def calibrators_doc(self) -> dict[str, Any]:
        """The calibrators document (cited by ``policy.notes.tuning.calibrators_sha256``)."""
        return calibrators_doc(self.calibrators)

    def save(self, directory: str | os.PathLike[str], *, name: str = "policy") -> tuple[Path, Path | None]:
        """Write ``<name>.toml`` and, when calibrators were fitted, ``<name>.calibrators.json``."""
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        toml_path = folder / f"{name}.toml"
        toml_path.write_text(self.to_toml(), encoding="utf-8")
        cal_path = None
        if self.calibrators:
            cal_path = folder / f"{name}.calibrators.json"
            save_calibrators(self.calibrators, cal_path)
        return toml_path, cal_path


def tune(
    data: EvalReport | Sequence[EvalRecord],
    alphas: Mapping[str, float] | None = None,
    *,
    base_policy: Policy | None = None,
    compositions: Sequence[CompositionRule] | None = None,
    method: Method = "cp",
    conf: float = 0.95,
    confirm_alpha: float | None = DEFAULT_CONFIRM_ALPHA,
    calibrate: bool | EvalReport | Sequence[EvalRecord] = False,
    hysteresis: float | None = None,
    version: str | None = None,
) -> TuningResult:
    """Tune every tier's composition and thresholds on labelled decisions (spec §11.3).

    - ``alphas``: wrong-execution budgets per tier (merged over :data:`DEFAULT_ALPHAS`).
    - ``compositions``: the compositions to try (default: all five; the tier's current one wins ties).
    - ``calibrate``: fit isotonic calibrators per tier and composition — on a held-out report when one is given,
      else in-sample (``True``; optimistic, recorded in the notes).
    - ``hysteresis``: the width ``h`` (default: ``max(0.03, q95(|ΔC|))`` from the report's replays, else the base
      policy's).
    - ``confirm_alpha``: the confirm band starts where the proposed call is wrong at most this often (95% bound);
      ``None`` keeps the base confirm thresholds.
    """
    base = base_policy or Policy()
    budgets = {**DEFAULT_ALPHAS, **dict(alphas or {})}
    records = records_of(data)
    if hysteresis is not None:
        h = hysteresis
    elif isinstance(data, EvalReport) and data.replays > 1:
        h = hysteresis_width(data)
    else:
        h = base.hysteresis
    held_out = records_of(calibrate) if isinstance(calibrate, (EvalReport, list, tuple)) else None
    fit_in_sample = calibrate is True
    tiers: dict[str, TierTuning] = {}
    calibrators: dict[str, IsotonicCalibrator] = {}
    certification = Certification(cases=0, alpha=budgets["critical"], certified=False, reason="no critical data")
    for tier in Tier:
        items = _tier_records(records, tier.value)
        prior_rule = base.tier(tier).composition
        options = list(dict.fromkeys([prior_rule, *(compositions or COMPOSITIONS)]))
        tuning, cal, cert = _tune_tier(
            tier, items, base, budgets[tier.value], options, method=method, conf=conf, h=h,
            confirm_alpha=confirm_alpha, calibration=held_out if held_out is not None else items if fit_in_sample
            else None,
        )  # fmt: skip
        tiers[tier.value] = tuning
        if cal is not None:
            calibrators[tier.value] = cal
        if cert is not None:
            certification = cert
    policy = _tuned_policy(base, tiers, certification, calibrators, h=h, method=method, conf=conf,
                           budgets=budgets, confirm_alpha=confirm_alpha, data=data, version=version,
                           in_sample=fit_in_sample)  # fmt: skip
    return TuningResult(policy=policy, tiers=tiers, certification=certification, calibrators=calibrators,
                        hysteresis=h, method=method)  # fmt: skip


def tune_thresholds(
    data: EvalReport | Sequence[EvalRecord], alphas: Mapping[str, float] | None = None, **kw: Any
) -> Policy:
    """:func:`tune` returning only the tuned :class:`~jevtools.policy.Policy` (spec §9 signature)."""
    return tune(data, alphas, **kw).policy


def _tune_tier(
    tier: Tier,
    items: list[EvalRecord],
    base: Policy,
    alpha: float,
    options: Sequence[CompositionRule],
    *,
    method: Method,
    conf: float,
    h: float,
    confirm_alpha: float | None,
    calibration: Sequence[EvalRecord] | None,
) -> tuple[TierTuning, IsotonicCalibrator | None, Certification | None]:
    tp = base.tier(tier)
    critical = tier is Tier.CRITICAL
    if not items:
        empty = Certification(cases=0, alpha=alpha, certified=False, reason="no critical data") if critical else None
        return (TierTuning(tier=tier.value, n=0, alpha=alpha, status="no_data", composition=tp.composition,
                           execute=tp.execute, confirm=tp.confirm), None, empty)  # fmt: skip
    best: tuple[ThresholdChoice | None, CompositionRule, IsotonicCalibrator | None] | None = None
    for rule in options:
        cal = _fit(tier, rule, calibration)
        scores = [_score(r, rule, cal) for r in items]
        choice = execute_threshold(scores, [r.wrong_if_executed for r in items], alpha, conf=conf, hysteresis=h,
                                   method=method, composition=rule)  # fmt: skip
        if best is None or (choice is not None and (best[0] is None or choice.kept > best[0].kept)):
            best = (choice, rule, cal)
    assert best is not None
    choice, rule, cal = best
    scores = [_score(r, rule, cal) for r in items]
    call_wrong = [not r.call_match for r in items]
    cert: Certification | None = None
    if critical:
        cases = len({r.case_id for r in items})
        cert = _certify(cases, alpha, choice)
        execute: float | Literal["never"] | None = tp.execute
        status: TierStatus = "tuned" if cert.certified else "confirm_only"
    elif choice is None:
        execute, status = "never", "never"
    else:
        execute, status = choice.threshold, "tuned"
    ceiling = choice.threshold if choice is not None and (not critical or cert is not None and cert.certified) \
        else None  # fmt: skip
    confirm = tp.confirm
    confirm_choice = None
    if confirm_alpha is not None and tp.confirm is not None:
        confirm_choice = confirm_threshold(scores, call_wrong, confirm_alpha, conf=conf, hysteresis=h,
                                           ceiling=ceiling, composition=rule)  # fmt: skip
        if confirm_choice is not None:
            confirm = confirm_choice.threshold
        elif ceiling is not None:
            confirm = ceiling
    tuning = TierTuning(tier=tier.value, n=len(items), alpha=alpha, status=status, composition=rule, execute=execute,
                        confirm=confirm, execute_choice=choice, confirm_choice=confirm_choice,
                        calibrator=cal.name if cal is not None else None)  # fmt: skip
    return tuning, cal, cert


def _fit(tier: Tier, rule: CompositionRule, records: Sequence[EvalRecord] | None) -> IsotonicCalibrator | None:
    if records is None:
        return None
    items = _tier_records(list(records), tier.value)
    if not items:
        return None
    xs = [_score(r, rule, None) for r in items]
    return IsotonicCalibrator(f"isotonic.{tier.value}.{rule}").fit(xs, [r.call_match for r in items])


def _certify(cases: int, alpha: float, choice: ThresholdChoice | None) -> Certification:
    """§11.4: auto-execution of the critical tier needs ≥ 3,000 labelled critical cases and the bound met."""
    common: dict[str, Any] = {}
    if choice is not None:
        common = {"threshold": choice.threshold, "kept": choice.kept, "errors": choice.errors, "upper": choice.upper}
    if cases < CRITICAL_CERTIFICATION_CASES:
        return Certification(cases=cases, alpha=alpha, certified=False, **common,
                             reason=f"{cases} labelled critical cases < {CRITICAL_CERTIFICATION_CASES}")  # fmt: skip
    if choice is None:
        return Certification(cases=cases, alpha=alpha, certified=False,
                             reason="no threshold meets the critical wrong-execution bound")  # fmt: skip
    return Certification(cases=cases, alpha=alpha, certified=True, reason="certified", **common)


def certify_critical(
    data: EvalReport | Sequence[EvalRecord],
    *,
    alpha: float = DEFAULT_ALPHAS["critical"],
    composition: CompositionRule = "MIN_L_J",
    conf: float = 0.95,
    hysteresis: float = 0.03,
    calibrator: IsotonicCalibrator | None = None,
) -> Certification:
    """Check the §11.4 rule on labelled critical-tier decisions: the best Clopper–Pearson threshold for ``alpha``
    and whether at least 3,000 cases stand behind it."""
    items = _tier_records(records_of(data), Tier.CRITICAL.value)
    scores = [_score(r, composition, calibrator) for r in items]
    choice = execute_threshold(scores, [r.wrong_if_executed for r in items], alpha, conf=conf,
                               hysteresis=hysteresis, composition=composition)  # fmt: skip
    return _certify(len({r.case_id for r in items}), alpha, choice)


def _tuned_policy(
    base: Policy,
    tiers: Mapping[str, TierTuning],
    certification: Certification,
    calibrators: Mapping[str, IsotonicCalibrator],
    *,
    h: float,
    method: Method,
    conf: float,
    budgets: Mapping[str, float],
    confirm_alpha: float | None,
    data: EvalReport | Sequence[EvalRecord],
    version: str | None,
    in_sample: bool,
) -> Policy:
    tier_updates: dict[str, dict[str, Any]] = {}
    for name, t in tiers.items():
        if t.status == "no_data":
            continue
        update: dict[str, Any] = {"composition": t.composition, "confirm": t.confirm}
        if name == Tier.CRITICAL.value:
            update["auto_execute"] = certification.threshold if certification.certified else None
        else:
            update["execute"] = t.execute
        tier_updates[name] = update
    meta = data.meta if isinstance(data, EvalReport) else {}
    tuning: dict[str, Any] = {
        "method": "clopper_pearson" if method == "cp" else "crc", "conf": conf, "hysteresis": h,
        "alphas": dict(budgets), "tiers": {name: t.to_note() for name, t in tiers.items()},
        "certification": certification.to_dict(),
        "dataset": {k: meta[k] for k in ("dataset", "created_at", "backend", "model", "policy_version") if k in meta},
        "cases": len({r.case_id for r in records_of(data)}),
    }  # fmt: skip
    if confirm_alpha is not None:
        tuning["confirm_alpha"] = confirm_alpha
    if calibrators:
        tuning["calibrators_sha256"] = calibrators_doc(calibrators)["sha256"]
        tuning["calibration"] = "in_sample" if in_sample else "held_out"
    tuning = _drop_none(tuning)
    digest = sha256_of({"base": base.sha256, "tuning": tuning}).removeprefix("sha256:")
    tuned_version = version or f"{_base_version(base.version)}+tuned.{digest[:12]}"
    notes = {**base.notes, "tuning": tuning}
    return Policy.from_dict({**base.to_dict(), "version": tuned_version, "hysteresis": h,
                             "tiers": _merge(base.to_dict()["tiers"], tier_updates),
                             "certified": {"critical_cases": certification.cases}, "notes": notes})  # fmt: skip


def _base_version(version: str) -> str:
    return version.split("+tuned.", 1)[0]


def _merge(tiers: Mapping[str, Any], updates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {name: {**dict(values), **dict(updates.get(name, {}))} for name, values in tiers.items()}


def _drop_none(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value if v is not None]
    return value


# --------------------------------------------------------------------------------------------------------------------
# TOML emission (no third-party writer)
# --------------------------------------------------------------------------------------------------------------------

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(key: str) -> str:
    return key if _BARE_KEY.match(key) else json.dumps(key, ensure_ascii=False)


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        text = repr(value)
        return text if any(c in text for c in ".en") else text + ".0"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_scalar(v) for v in value if v is not None) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{_key(str(k))} = {_scalar(v)}" for k, v in value.items() if v is not None) + "}"
    raise TypeError(f"cannot write {type(value).__name__} to TOML")


def _emit(table: Mapping[str, Any], path: list[str], lines: list[str]) -> None:
    scalars = [(k, v) for k, v in table.items() if v is not None and not (isinstance(v, Mapping) and v)]
    tables = [(k, v) for k, v in table.items() if isinstance(v, Mapping) and v]
    if path and (scalars or not tables):
        if lines:
            lines.append("")
        lines.append("[" + ".".join(_key(p) for p in path) + "]")
    for key, value in scalars:
        lines.append(f"{_key(str(key))} = {_scalar(value)}")
    for key, value in tables:
        _emit(value, [*path, str(key)], lines)


def policy_to_toml(policy: Policy) -> str:
    """The policy as a ``policy.toml`` document (``None`` fields are left out, so they read back as their
    defaults). Raises ``ValueError`` if the text would not read back as the same policy."""
    lines: list[str] = []
    _emit(policy.to_dict(), [], lines)
    text = "\n".join(lines) + "\n"
    try:
        again = Policy.from_toml_text(text)
    except ImportError:  # pragma: no cover - Python 3.10 without tomli: the check needs a TOML reader
        return text
    if again != policy:
        raise ValueError("the policy does not round-trip through TOML (a None overrides a non-None default)")
    return text

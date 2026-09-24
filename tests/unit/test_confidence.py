"""Confidence composition (spec §3.7) against the hand-computed numbers of §13."""

from __future__ import annotations

import random

import pytest

from jevtools.confidence import (
    Composition,
    Factors,
    IsotonicCalibrator,
    call_map,
    compose,
    confidence,
    finalize,
    tier_prior,
)
from jevtools.policy import Policy, Tier


def test_r2_external_composition() -> None:
    factors = Factors.build(tool=0.96, authorized=0.95, slots={"to": 0.86, "subject": None, "body": 0.91})
    assert list(factors.items) == ["tool", "authorized", "to", "body"]  # cosmetic subject is not a factor
    comp = confidence(factors, Tier.EXTERNAL)
    assert comp.PI == pytest.approx(0.714, abs=5e-4)
    assert comp.W == pytest.approx(0.86)
    assert comp.L == pytest.approx(0.68)
    assert comp.C == pytest.approx(comp.PI) and comp.rule == "PI" and not comp.calibrated


def test_r3_critical_uses_min_l_j() -> None:
    factors = Factors.build(
        tool=0.98, authorized=0.98, slots={"from_account": 0.95, "to_account": 0.97, "amount": 0.99, "currency": 0.97}
    )
    comp = confidence(factors, Tier.CRITICAL, J=0.92)
    assert comp.L == pytest.approx(0.84)
    assert comp.C == pytest.approx(0.84) and comp.rule == "MIN_L_J"
    assert confidence(factors, Tier.CRITICAL, J=0.80).C == pytest.approx(0.80)
    assert confidence(factors, Tier.CRITICAL).C == pytest.approx(0.84)  # J not asked → L


def test_r5_all_compositions() -> None:
    factors = Factors.build(
        tool=0.95, authorized=0.96, slots={"title": None, "start": 0.80, "duration_minutes": 0.96, "attendees": 0.7857}
    )
    comp = confidence(factors, "external")
    assert (round(comp.PI, 3), round(comp.W, 3), round(comp.L, 3)) == (0.550, 0.786, 0.456)
    assert comp.PI == pytest.approx(0.5503, abs=5e-5) and comp.L == pytest.approx(0.4557, abs=5e-5)


def test_r1_and_r6_read_tier_uses_w() -> None:
    assert confidence(Factors.build(tool=0.98, slots={"city": 0.97, "unit": 0.97}), "read").C == pytest.approx(0.97)
    assert confidence(Factors.build(tool=0.84, slots={"path": 0.83}), "read").C == pytest.approx(0.83)


def test_empty_factor_set_and_bad_factors() -> None:
    assert compose([]) == Composition(W=1.0, PI=1.0, L=1.0)
    with pytest.raises(ValueError, match="not a probability"):
        compose([1.2])
    assert compose({"a": 0.5, "b": 0.4}).L == pytest.approx(0.0)  # clamped at 0


def test_l_le_pi_le_w_on_random_factor_sets() -> None:
    rng = random.Random(7)
    for _ in range(2000):
        values = [rng.random() for _ in range(rng.randint(1, 8))]
        comp = compose(values)
        assert comp.L <= comp.PI + 1e-12 <= comp.W + 2e-12


def test_tier_prior_follows_policy() -> None:
    comp = compose([0.9, 0.8], J=0.95)
    assert tier_prior(comp, "read") == pytest.approx(0.8)
    assert tier_prior(comp, "write") == pytest.approx(0.72)
    assert tier_prior(comp, "critical") == pytest.approx(0.7)
    custom = Policy.from_dict({"tiers": {"read": {"composition": "L"}}})
    assert tier_prior(comp, "read", custom) == pytest.approx(0.7)


def test_coherence_cap_and_calibration() -> None:
    comp = compose([0.9, 0.6])
    optimistic = IsotonicCalibrator().fit([0.1, 0.5, 0.9], [1, 1, 1])
    out = finalize(comp, "write", calibrators={"write": optimistic})
    assert out.calibrated and out.calibrator == "isotonic"
    assert out.C == pytest.approx(0.6)  # calibrator says 1.0; capped at W
    unfitted = finalize(comp, "write", calibrators={"write": IsotonicCalibrator()})
    assert not unfitted.calibrated and unfitted.C == pytest.approx(0.54)


def test_isotonic_calibrator_is_monotone_and_pools_violators() -> None:
    cal = IsotonicCalibrator().fit([0.1, 0.2, 0.3, 0.4, 0.5], [0, 1, 0, 1, 1])
    assert cal.ys == sorted(cal.ys)
    assert cal.predict(0.0) == pytest.approx(0.0) and cal.predict(1.0) == pytest.approx(1.0)
    rng = random.Random(3)
    xs = [rng.random() for _ in range(300)]
    ys = [1 if rng.random() < x else 0 for x in xs]
    cal = IsotonicCalibrator().fit(xs, ys)
    grid = [cal.predict(i / 100) for i in range(101)]
    assert all(a <= b + 1e-12 for a, b in zip(grid, grid[1:], strict=False))
    assert IsotonicCalibrator.from_dict(cal.to_dict()).predict(0.42) == pytest.approx(cal.predict(0.42))
    assert IsotonicCalibrator().predict(0.3) == 0.3
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit([0.1], [1, 0])


def test_isotonic_duplicate_x_values() -> None:
    cal = IsotonicCalibrator().fit([0.5, 0.5, 0.5, 0.9], [0, 1, 1, 1])
    assert cal.predict(0.5) == pytest.approx(2 / 3)
    assert cal.predict(0.7) == pytest.approx(2 / 3 + (1 - 2 / 3) * 0.5)


def test_call_map_disagreement() -> None:
    tool_p = {"read_file": 0.55, "search_web": 0.40, "NO_TOOL": 0.05}
    agree = call_map(tool_p, {"read_file": 0.9, "search_web": 0.95})
    assert agree.chosen == agree.call_map == "read_file" and not agree.disagrees
    infeasible = call_map(tool_p, {"read_file": 0.0, "search_web": 0.95})
    assert infeasible.call_map == "search_web" and infeasible.disagrees
    assert infeasible.scores["search_web"] == pytest.approx(0.38)
    assert call_map({"NO_TOOL": 0.9}, {}).call_map is None


def test_isotonic_pools_ties_after_a_violator_merge() -> None:
    """Review #11: tied x values split across blocks once a violator pool moved the centroid (upward bias)."""
    cal = IsotonicCalibrator().fit([0.8] * 10 + [0.9] * 10, [1] * 10 + [0] * 5 + [1] * 5)
    assert cal.predict(0.9) == pytest.approx(0.75)
    assert len(cal.xs) == 1
    cal2 = IsotonicCalibrator("isotonic.write.PI").fit([0.7] * 5 + [1.0] * 10, [1] * 5 + [0] * 4 + [1] * 6)
    assert cal2.predict(1.0) == pytest.approx(11 / 15)
    out = finalize(compose({"a": 1.0, "b": 1.0}), Tier.WRITE, None, {"write": cal2})
    assert out.C == pytest.approx(11 / 15)
    weighted = IsotonicCalibrator().fit([0.5, 0.5, 0.6], [1, 0, 1], weights=[0.0, 2.0, 1.0])
    assert weighted.predict(0.5) == pytest.approx(0.0) and weighted.predict(0.6) == pytest.approx(1.0)

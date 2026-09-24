"""Token estimation with the running correction factor (spec §5.5)."""

from __future__ import annotations

import pytest

from jevtools.budget import TokenEstimator
from jevtools.canonical import canonical_str
from jevtools.wire import DecisionRequest, NoulQuestion, Usage

REQUEST = DecisionRequest(model="m", state="x" * 700, questions={"a.b": NoulQuestion(instructions="Is it?")})


def test_estimate_is_chars_over_3_5() -> None:
    est = TokenEstimator()
    chars = len(canonical_str(REQUEST.to_wire()))
    assert est.raw(REQUEST) == pytest.approx(chars / 3.5)
    assert est.est(REQUEST) == -(-chars * 2 // 7)  # ceil(chars / 3.5)


def test_ema_update_clamp_and_keys() -> None:
    est = TokenEstimator()
    raw = est.raw(REQUEST)
    r1 = est.observe(REQUEST, Usage(input_tokens=round(raw * 1.5)), backend="typesafe", model="jev-latest")
    assert r1 == pytest.approx(0.7 + 0.3 * round(raw * 1.5) / raw)
    assert est.ratio("typesafe", "jev-latest") == r1 and est.ratio("openrouter_decisions", "x") == 1.0
    for _ in range(50):
        est.observe(REQUEST, Usage(input_tokens=round(raw * 5)), backend="typesafe", model="jev-latest")
    assert est.ratio("typesafe", "jev-latest") == 1.6
    for _ in range(50):
        est.observe(REQUEST, Usage(input_tokens=1), backend="typesafe", model="jev-latest")
    assert est.ratio("typesafe", "jev-latest") == 0.7
    assert est.est(REQUEST, backend="typesafe", model="jev-latest") == -(-est.raw(REQUEST) * 0.7 // 1)
    assert est.observe(REQUEST, Usage(), backend="b") == 1.0  # no input_tokens: unchanged
    assert set(est.ratios) == {("typesafe", "jev-latest")}


def test_invalid_chars_per_token() -> None:
    with pytest.raises(ValueError):
        TokenEstimator(0)

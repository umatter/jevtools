"""§11.2 metrics on synthetic records: calibration, call-level rates, risk–coverage, flips, exact statistics."""

from __future__ import annotations

import math

import pytest

from jevtools.eval import metrics as m
from jevtools.eval.report import EvalReport, PoolHit, QuestionRecord
from jevtools.eval.stats import binom_cdf, clopper_pearson_upper, quantile, rule_of_three_cases
from tests.eval.support import record


def q(family: str, p: float, correct: bool, **kw: object) -> QuestionRecord:
    return QuestionRecord.model_validate({"qid": f"t.{family}", "family": family, "primitive": "choice", "p": p,
                                          "correct": correct, **kw})  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# Exact statistics
# --------------------------------------------------------------------------------------------------------------------


def test_clopper_pearson_upper_bound() -> None:
    assert clopper_pearson_upper(0, 0) == 1.0 and clopper_pearson_upper(5, 5) == 1.0
    assert clopper_pearson_upper(0, 3000) == pytest.approx(1 - 0.05 ** (1 / 3000))
    assert clopper_pearson_upper(0, 2995) <= 0.001 < clopper_pearson_upper(0, 2994)
    assert rule_of_three_cases(0.001) == 2995 and rule_of_three_cases(0.05) == 59
    upper = clopper_pearson_upper(1, 10)
    assert upper == pytest.approx(0.3942, abs=1e-4)  # Beta(2, 9) 95th percentile
    assert binom_cdf(1, 10, upper) == pytest.approx(0.05, abs=1e-9)
    assert clopper_pearson_upper(3, 100, 0.99) > clopper_pearson_upper(3, 100, 0.95) > 0.03
    with pytest.raises(ValueError):
        clopper_pearson_upper(3, 2)
    with pytest.raises(ValueError):
        clopper_pearson_upper(1, 2, conf=1.0)


def test_binom_cdf_and_quantile() -> None:
    assert binom_cdf(-1, 5, 0.5) == 0.0 and binom_cdf(5, 5, 0.5) == 1.0
    assert binom_cdf(2, 4, 0.5) == pytest.approx(11 / 16)
    assert quantile([3, 1, 2], 0.5) == 2 and quantile([0, 10], 0.95) == pytest.approx(9.5)
    with pytest.raises(ValueError):
        quantile([], 0.5)


# --------------------------------------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------------------------------------


def test_ece_brier_and_reliability() -> None:
    probs, labels = [0.9, 0.9, 0.1, 0.1], [True, False, False, False]
    assert m.brier(probs, labels) == pytest.approx((0.01 + 0.81 + 0.01 + 0.01) / 4)
    assert m.ece(probs, labels) == pytest.approx(0.5 * abs(0.5 - 0.9) + 0.5 * abs(0.0 - 0.1))
    bins = m.reliability([1.0, 0.0], [1, 0], bins=10)
    assert [(b.lo, b.n) for b in bins] == [(0.0, 1), (0.9, 1)]  # p = 1 falls in the last bin
    assert m.ece([], []) is None and m.brier([], []) is None
    perfect = m.calibration([1.0, 0.0], [True, False])
    assert (perfect.n, perfect.ece, perfect.brier, perfect.accuracy) == (2, 0.0, 0.0, 0.5)
    with pytest.raises(ValueError):
        m.ece([1.5], [1])
    with pytest.raises(ValueError):
        m.brier([0.5], [])


def test_family_calibration_groups() -> None:
    items = [
        q("tool", 0.9, True), q("slot", 0.8, True, sub="ref"), q("slot", 0.6, False, sub="enum"),
        q("accept", 0.7, True, sub="content"), q("slot", 0.05, False, sentinel="NONE_OF_THESE"),
    ]  # fmt: skip
    stats = m.family_calibration(items)
    assert set(stats) == {"tool", "slot", "slot.ref", "slot.enum", "accept", "accept.content",
                          "sentinel.NONE_OF_THESE"}  # fmt: skip
    assert stats["slot"].n == 2 and stats["sentinel.NONE_OF_THESE"].n == 1
    assert stats["slot.enum"].accuracy == 0.0 and stats["tool"].ece == pytest.approx(0.1)
    assert set(m.family_calibration(items, families=False)) == {"tool", "slot.ref", "slot.enum", "accept.content",
                                                                "sentinel.NONE_OF_THESE"}  # fmt: skip
    report = EvalReport(records=[record(questions=items[:2]), record(case_id="d", questions=items[2:])])
    assert m.family_calibration(report)["slot"].n == 2 and len(m.question_records(report)) == 5


# --------------------------------------------------------------------------------------------------------------------
# Call level
# --------------------------------------------------------------------------------------------------------------------


def call_level_records() -> list:
    return [
        record(case_id="a"),
        record(case_id="b", tier="external", wrong_if_executed=True, call_match=False, correct=False,
               outcome_ok=True),
        record(case_id="c", outcome="confirm", executed=False, tier="external"),
        record(case_id="d", outcome="clarify", executed=False, correct=False, outcome_ok=False, menu_has_gold=True,
               menu=[{"id": "pick:to:0"}]),
        record(case_id="e", outcome="clarify", executed=False, correct=False, outcome_ok=False,
               menu_has_gold=False, menu=[{"id": "pick:to:0"}]),
        record(case_id="f", outcome="clarify", executed=False, correct=False, outcome_ok=False),
        record(case_id="g", outcome="abstain", executed=False, gold_tool=None, tool=None, outcomes_ok=["abstain"]),
        record(case_id="h", outcome="abstain", executed=False, outcome_ok=False, correct=False, call_match=False),
        record(case_id="i", outcome="refuse", executed=False, outcome_ok=True),
    ]  # fmt: skip


def test_call_level_metrics() -> None:
    records = call_level_records()
    rates = m.wrong_execution_rate(records)
    assert (rates["all"].k, rates["all"].n) == (1, 2)
    assert (rates["read"].k, rates["read"].n, rates["external"].rate) == (0, 1, 1.0)
    assert m.exact_match(records) == pytest.approx(7 / 9) and m.accuracy(records) == pytest.approx(4 / 9)
    assert m.clarify_rate(records) == pytest.approx(3 / 9)
    usefulness = m.clarify_usefulness(records)
    assert (usefulness.k, usefulness.n) == (1, 2)  # the open question (no menu) is not counted
    precision = m.abstention_precision(records)
    assert (precision.k, precision.n) == (2, 3)
    assert m.outcome_counts(records) == {"abstain": 2, "clarify": 3, "confirm": 1, "execute": 2, "refuse": 1}
    assert m.exact_match([]) is None and m.clarify_usefulness([]).rate is None
    assert m.Rate(0, 0).upper95 == 1.0 and m.Rate(1, 4).to_dict()["rate"] == 0.25


def test_usage_stage_counts_and_by_tag() -> None:
    records = [
        record(case_id="a", rounds=1, jev_calls=1, cost_usd=0.0001, tags=["x"]),
        record(case_id="b", rounds=3, jev_calls=2, stage="extractor", correct=False, tags=["x", "y"]),
    ]
    use = m.usage(records)
    assert use["rounds_per_decision"] == 2.0 and use["jev_calls"] == 3 and use["cost_usd"] == pytest.approx(0.0001)
    assert m.usage([record()])["cost_usd"] is None  # never estimated
    assert m.stage_counts(records) == {"extractor": 1, "ok": 1}
    assert m.by_tag(EvalReport(records=records)) == {"x": 0.5, "y": 0.0}


def test_recall_at_k_and_pool_recall() -> None:
    assert m.recall_at_k([1, 3, None, 40], 3) == 0.5 and m.recall_at_k([], 5) is None
    with pytest.raises(ValueError):
        m.recall_at_k([1], 0)
    records = [
        record(case_id="a", locale="en", pool={"to": PoolHit(in_pool=True, rank=2, size=3, kind="ref")}),
        record(case_id="b", locale="de", pool={"to": PoolHit(in_pool=False, size=3, kind="ref"),
                                               "unit": PoolHit(in_pool=True, via_default=True, kind="enum")}),
    ]  # fmt: skip
    assert {k: (r.k, r.n) for k, r in m.pool_recall(records).items()} == {"enum": (1, 1), "ref": (1, 2)}
    assert {k: (r.k, r.n) for k, r in m.pool_recall(records, k=1).items()} == {"enum": (0, 1), "ref": (0, 2)}
    assert set(m.pool_recall(records, by="locale")) == {"de", "en"}


def test_risk_coverage() -> None:
    points = m.risk_coverage([0.9, 0.8, 0.8, 0.3], [True, True, False, False])
    assert [(p.threshold, p.kept, p.errors) for p in points] == [(0.9, 1, 0), (0.8, 3, 1), (0.3, 4, 2)]
    assert points[1].coverage == 0.75 and points[1].risk == pytest.approx(1 / 3)
    assert m.aurc(points) == pytest.approx(0.25 * 0 + 0.5 * (1 / 3) + 0.25 * 0.5)
    assert m.aurc([]) is None
    records = [record(case_id="a", W=0.9, PI=0.8, L=0.7, J=None, C=0.8),
               record(case_id="b", W=0.6, PI=0.5, L=0.4, J=0.3, C=0.5, call_match=False)]  # fmt: skip
    curves = m.risk_coverage_curves(records)
    assert set(curves) == {"W", "PI", "L", "J", "C"}
    assert [p.threshold for p in curves["J"]] == [0.7, 0.3]  # J falls back to L when no joint was asked
    assert curves["C"][-1].risk == 0.5


def test_injection_flips_and_hysteresis() -> None:
    assert (m.injection_success_rate([record(planted_hit=False), record(planted_hit=True), record()]).k,
            m.injection_success_rate([record(planted_hit=False)]).n) == (1, 1)  # fmt: skip
    stable = EvalReport(records=[record(case_id="a", replay=i, C=0.8) for i in range(3)])
    assert m.flip_rate(stable) == 0.0 and m.hysteresis_width(stable) == 0.03
    shaky = EvalReport(records=[
        record(case_id="a", replay=0, C=0.80), record(case_id="a", replay=1, C=0.70, outcome="confirm"),
        record(case_id="b", replay=0, C=0.50), record(case_id="b", replay=1, C=0.52),
    ])  # fmt: skip
    assert m.flip_rate(shaky) == 0.5
    assert sorted(m.confidence_deltas(shaky)) == pytest.approx([0.02, 0.10])
    assert m.hysteresis_width(shaky) == pytest.approx(0.02 + 0.95 * 0.08)
    assert m.flip_rate(EvalReport(records=[record()])) is None


def test_summary_document_is_json_ready() -> None:
    report = EvalReport(records=call_level_records())
    summary = m.summarize(report)
    assert summary["cases"] == 9 and summary["replays"] == 1 and math.isclose(summary["accuracy"], 4 / 9)
    assert summary["wrong_execution_rate"]["all"] == {
        "k": 1,
        "n": 2,
        "rate": 0.5,
        "upper95": pytest.approx(clopper_pearson_upper(1, 2)),
    }
    assert set(summary) >= {"calibration", "risk_coverage", "pool_recall", "flip_rate", "hysteresis", "usage"}
    assert report.summary() == summary

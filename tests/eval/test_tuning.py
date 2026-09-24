"""§11.3/§11.4 tuning: Clopper–Pearson thresholds per tier, CRC, calibrators, certification and the versioned
policy.toml. Synthetic records stand in for labelled traffic (they say nothing about Jev)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jevtools.eval import tuning
from jevtools.eval.harness import run
from jevtools.eval.report import EvalRecord, EvalReport
from jevtools.eval.stats import clopper_pearson_upper
from jevtools.policy import Outcome, Policy, PolicyInput, SlotState, Tier, evaluate
from jevtools.router import Router
from tests.eval.support import factory, record, scenario_cases
from tests.scenario import scripts
from tests.scenario.fixtures import scenario_messages


def tier_records(tier: str, n: int, *, errors_below: float, start: int = 0, prefix: str = "c") -> list[EvalRecord]:
    """``n`` decisions with scores spread over (0, 1]; the proposed call is wrong exactly when the score is below
    ``errors_below``."""
    out = []
    for i in range(n):
        score = round((i + 1) / n, 6)
        wrong = score < errors_below
        out.append(record(case_id=f"{prefix}{start + i}", tier=tier, W=min(1.0, score + 0.05), PI=score,
                          L=max(0.0, score - 0.1), J=None, C=score, call_match=not wrong,
                          wrong_if_executed=wrong, correct=not wrong))  # fmt: skip
    return out


def kept(records: list[EvalRecord], rule: str, tau: float, h: float) -> list[EvalRecord]:
    return [r for r in records if (r.score(rule) or 0.0) - tau >= h - 1e-9]  # type: ignore[arg-type]


def test_execute_threshold_meets_the_bound() -> None:
    records = tier_records("write", 400, errors_below=0.3)
    result = tuning.tune(EvalReport(records=records))
    write = result.tiers["write"]
    assert write.status == "tuned" and isinstance(write.execute, float) and write.execute_choice is not None
    through = kept(records, write.composition, write.execute, result.hysteresis)
    errors = sum(r.wrong_if_executed for r in through)
    assert len(through) == write.execute_choice.kept and errors == write.execute_choice.errors
    assert clopper_pearson_upper(errors, len(through)) <= 0.02
    # the cut sits at the error boundary: a 2% budget over ~280 cases lets at most one error through
    assert errors <= 1 and 0.29 <= write.execute + result.hysteresis <= 0.32
    assert write.execute == round(write.execute, 4)  # thresholds are rounded up to 4 decimals
    policy = result.policy
    assert policy.tier("write").execute == write.execute and policy.execute_at("write") == write.execute
    assert policy.version.startswith("jevtools-default-0.1+tuned.") and policy.notes["tuning"]["method"] == \
        "clopper_pearson"  # fmt: skip


def test_the_tuned_policy_reproduces_the_evaluated_set() -> None:
    records = tier_records("external", 300, errors_below=0.2)
    result = tuning.tune(records, {"external": 0.05})
    ext = result.tiers["external"]
    assert isinstance(ext.execute, float)
    for r in records:
        inp = PolicyInput(tools={"t": 1.0}, chosen="t", tier=Tier.EXTERNAL, C=r.score(ext.composition),  # type: ignore[arg-type]
                          authorized=1.0, slots=[SlotState(name="x", factor=0.99)])  # fmt: skip
        executes = evaluate(inp, result.policy).outcome is Outcome.EXECUTE
        assert executes == (r in kept(records, ext.composition, ext.execute, result.hysteresis))


def test_tiers_without_data_or_without_a_threshold() -> None:
    records = tier_records("read", 50, errors_below=0.99)  # nearly everything is wrong
    result = tuning.tune(records)
    assert result.tiers["read"].status == "never" and result.policy.tier("read").execute == "never"
    assert result.policy.execute_at("read") is None
    base = Policy()
    for name in ("write", "external"):
        assert result.tiers[name].status == "no_data"
        assert result.policy.tier(name) == base.tier(name)
    assert result.certification.certified is False and result.certification.reason == "no critical data"


def test_confirm_threshold_and_composition_choice() -> None:
    records = tier_records("write", 400, errors_below=0.3)
    result = tuning.tune(records, compositions=["W", "PI", "L"])
    write = result.tiers["write"]
    assert write.confirm is not None and isinstance(write.execute, float) and write.confirm <= write.execute
    assert write.composition in ("W", "PI", "L")
    unchanged = tuning.tune(records, confirm_alpha=None)
    assert unchanged.tiers["write"].confirm == Policy().tier("write").confirm


def test_crc_method() -> None:
    records = tier_records("write", 400, errors_below=0.3)
    cp = tuning.tune(records)
    crc = tuning.tune(records, method="crc")
    assert crc.policy.notes["tuning"]["method"] == "crc" and crc.tiers["write"].status == "tuned"
    assert crc.tiers["write"].execute_choice is not None and cp.tiers["write"].execute_choice is not None
    # CRC bounds the expected loss over all cases, so it may admit a few errors that CP (conditional) does not
    assert crc.tiers["write"].execute_choice.kept >= cp.tiers["write"].execute_choice.kept


def test_critical_tier_needs_3000_cases() -> None:
    few = tier_records("critical", 500, errors_below=0.0)
    result = tuning.tune(few)
    crit = result.tiers["critical"]
    assert crit.status == "confirm_only" and result.policy.tiers.critical.auto_execute is None
    assert result.policy.certified.critical_cases == 500 and result.policy.critical_auto_execute() is None
    assert not result.certification.certified and "500 labelled critical cases" in result.certification.reason
    assert result.policy.tier("critical").execute == "never"

    many = tier_records("critical", 3100, errors_below=0.0)
    certified = tuning.tune(many)
    cert = certified.certification
    assert cert.certified and cert.cases == 3100 and cert.upper is not None and cert.upper <= 0.001
    assert certified.policy.critical_auto_execute() == cert.threshold
    assert certified.policy.execute_at("critical") == cert.threshold
    check = tuning.certify_critical(many, composition=certified.tiers["critical"].composition)  # type: ignore[arg-type]
    assert check.certified and check.cases == 3100


def test_policy_toml_round_trip(tmp_path: Path) -> None:
    result = tuning.tune(tier_records("write", 200, errors_below=0.2), calibrate=True)
    text = result.to_toml()
    assert Policy.from_toml_text(text) == result.policy
    assert 'version = "jevtools-default-0.1+tuned.' in text and "[tiers.write]" in text
    toml_path, cal_path = result.save(tmp_path)
    assert Policy.from_toml(toml_path) == result.policy and cal_path is not None
    loaded = tuning.load_calibrators(cal_path)
    assert set(loaded) == {"write"} and loaded["write"].xs == result.calibrators["write"].xs
    assert result.policy.notes["tuning"]["calibrators_sha256"] == result.calibrators_doc()["sha256"]
    assert result.policy.notes["tuning"]["calibration"] == "in_sample"
    assert tuning.policy_to_toml(Policy()).startswith('version = "jevtools-default-0.1"')
    retuned = tuning.tune(tier_records("write", 200, errors_below=0.2), base_policy=result.policy)
    assert retuned.policy.version.count("+tuned.") == 1


def test_fit_calibrators_is_monotone_and_held_out() -> None:
    records = tier_records("external", 200, errors_below=0.5)
    cals = tuning.fit_calibrators(records)
    cal = cals["external"]
    assert cal.fitted and cal.name == "isotonic.external.PI"
    assert all(a <= b for a, b in zip(cal.ys, cal.ys[1:], strict=False))
    assert cal.predict(0.1) == 0.0 and cal.predict(0.9) == 1.0
    held = tuning.tune(records, calibrate=EvalReport(records=records))
    assert held.policy.notes["tuning"]["calibration"] == "held_out" and "external" in held.calibrators
    assert tuning.fit_calibrators(records, min_cases=500) == {}


def test_hysteresis_comes_from_replays() -> None:
    base = tier_records("write", 200, errors_below=0.2)
    replayed = [r.model_copy(update={"replay": 1, "C": (r.C or 0) + 0.06}) for r in base]
    result = tuning.tune(EvalReport(records=base + replayed))
    assert result.hysteresis == pytest.approx(0.06) and result.policy.hysteresis == pytest.approx(0.06)
    assert tuning.tune(base, hysteresis=0.05).policy.hysteresis == 0.05


def test_threshold_helpers_edge_cases() -> None:
    assert tuning.execute_threshold([], [], 0.05) is None
    assert tuning.execute_threshold([0.9, 0.8], [True, True], 0.05) is None
    with pytest.raises(ValueError):
        tuning.execute_threshold([0.9], [], 0.05)
    choice = tuning.execute_threshold([0.9] * 100, [False] * 100, 0.05, hysteresis=0.0)
    assert choice is not None and choice.kept == 100 and choice.automation == 1.0
    confirm = tuning.confirm_threshold([0.9] * 30 + [0.3] * 30, [False] * 55 + [True] * 5, 0.5, ceiling=0.5)
    assert confirm is not None and confirm.threshold == 0.3 and (confirm.kept, confirm.errors) == (60, 5)
    assert tuning.confirm_threshold([0.9, 0.2], [False, True], 0.5) is None  # one right call: its bound is 0.95


def test_tuning_a_scenario_report_yields_a_usable_policy() -> None:
    report = run(scenario_cases(), factory)
    policy = tuning.tune_thresholds(report)
    assert policy.notes["tuning"]["cases"] == 7 and policy.tier("critical").execute == "never"
    router = factory(scenario_cases()[0])
    tuned = Router(router.catalog, backend=router.backend, policy=policy, context=router.context)
    decision = tuned.decide(scenario_messages(scripts.R1_REQUEST))
    assert decision.trace.policy["version"] == policy.version


def test_tuning_uses_the_policys_own_hysteresis_test() -> None:
    """Tuned thresholds are valid only under the runtime rule: tuning calls it instead of a copy (#18)."""
    from jevtools import policy
    from jevtools.eval import tuning

    assert tuning.clears is policy._clears and tuning._EPS == policy._EPS
    assert not hasattr(tuning, "_clears")

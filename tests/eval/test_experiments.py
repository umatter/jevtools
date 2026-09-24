"""E1–E10 definitions: live experiments are skipped offline; with ``allow_offline`` their mechanics run on scripted
routers (``evidence=False``: a script is never evidence about Jev); E8/E9/E10 measure code and run offline."""

from __future__ import annotations

import pytest

from jevtools.backends.scripted import ScriptedBackend
from jevtools.eval import experiments as ex
from jevtools.eval.harness import run
from jevtools.sources import Registry
from jevtools.wire import ChoiceQuestion, DecisionRequest
from tests.eval.support import case, factory, scenario_cases
from tests.scenario import scripts
from tests.scenario.fixtures import SCENARIO_CONTACTS


def by_id(cid: str) -> list:
    return [c for c in scenario_cases() if c.id == cid]


def test_every_experiment_is_defined() -> None:
    assert list(ex.EXPERIMENTS) == [f"E{i}" for i in range(1, 11)]
    assert [e.id for e in ex.EXPERIMENTS.values() if not e.live] == ["E8", "E9", "E10"]
    assert all(e.question and e.protocol and e.drives for e in ex.EXPERIMENTS.values())
    with pytest.raises(KeyError, match="unknown experiment"):
        ex.run_experiment("E11", scenario_cases(), factory)


@pytest.mark.parametrize("eid", ["E1", "E2", "E3", "E4", "E5", "E6", "E7"])
def test_live_experiments_are_skipped_offline(eid: str) -> None:
    result = ex.run_experiment(eid, scenario_cases()[:1], factory)
    assert result.status == "skipped" and result.evidence is False and "live Jev backend" in result.notes[0]


def test_offline_detection() -> None:
    assert ex.is_offline(ScriptedBackend()) and ex.is_offline(ex.ReversedOptions(ScriptedBackend()))

    class Live:
        model, name = "jev-latest", "typesafe"

    assert not ex.is_offline(Live())
    assert ex.run_experiment("E2", [], factory).status == "skipped"


def test_e2_gold_removed_mechanics() -> None:
    cases = by_id("r2")
    [variant] = ex.gold_removed(cases)
    assert variant.meta["remove"] == ["anna.keller@acme.com"] and variant.id == "r2~gold_removed"
    router = ex.gold_removed_factory(factory)(variant)
    assert router.context.sources["contacts"].lookup("anna.keller@acme.com") is None
    result = ex.run_experiment("E2", cases, factory, allow_offline=True)
    assert result.status in ("passed", "failed") and result.evidence is False
    assert result.metrics["slots_present"] == 1 and result.metrics["slots_removed"] == 1
    assert result.metrics["present_none_lt_0.10"] == 1.0  # the script puts 0.03 on NONE_OF_THESE
    assert "offline backend" in result.notes[-1]


def test_without_rows_keeps_the_registry_settings() -> None:
    reg = Registry("contacts", SCENARIO_CONTACTS, key="email", label="{name} <{email}>", match=["name"],
                   provides=["email"], recency="last")  # fmt: skip
    smaller = ex.without_rows(reg, ["anna.keller@acme.com"])
    assert len(smaller) == len(reg) - 1 and smaller.label == reg.label and smaller.recency == "last"
    assert smaller.provides == reg.provides and ex.without_rows("not a registry", ["x"]) == "not a registry"


def test_e3_masks_the_gold_mention() -> None:
    report = run(by_id("r2"), factory)
    mentions = ex.gold_mentions(by_id("r2")[0], report.first()[0])
    assert mentions == {"to": "Anna"}
    masked = ex.masked(by_id("r2")[0], "Anna")
    assert masked.request == "Email [someone] that I'll be 10 minutes late" and masked.meta["masked"] == "Anna"
    result = ex.run_experiment("E3", by_id("r2"), factory, allow_offline=True)
    assert result.metrics["masked_cases"] == 1 and result.evidence is False


def test_backend_wrappers_preserve_answers() -> None:
    backend = ScriptedBackend({"q": {"a": 0.7, "b": 0.3}})
    request = DecisionRequest(
        model="m",
        state="s",
        questions={
            "q": ChoiceQuestion(criteria={"a": None, "b": None}),
            "r": ChoiceQuestion(criteria={"x": None, "y": None}),
        },
    )
    reversed_ = ex.ReversedOptions(backend).decide(request)
    sent = backend.requests[-1].questions["q"]
    assert isinstance(sent, ChoiceQuestion) and list(sent.criteria) == ["b", "a"]
    assert reversed_.answers["q"].probabilities == pytest.approx({"a": 0.7, "b": 0.3})  # type: ignore[union-attr]
    split = ex.SplitQuestions(backend, parts=2).decide(request)
    assert set(split.answers) == {"q", "r"} and len(backend.requests) == 3
    ex.ReorderedQuestions(backend).decide(request)
    assert list(backend.requests[-1].questions) == ["r", "q"]


def test_e4_e5_e6_e7_run_offline_when_allowed() -> None:
    cases = by_id("r2") + by_id("r4")
    e4 = ex.run_experiment("E4", cases, factory, allow_offline=True)
    assert e4.metrics["choices"] > 0 and e4.metrics["top_flip_rate"] == 0.0  # the script answers by label
    e5 = ex.run_experiment("E5", cases, factory, allow_offline=True)
    assert e5.metrics["premise_agreed"]["n"] > 0 and e5.metrics["premise_counterfactual"] is None
    e6 = ex.run_experiment("E6", by_id("r4"), factory, allow_offline=True, ks=(5, 40))
    assert set(e6.metrics) == {"5", "40"}
    e7 = ex.run_experiment("E7", cases, factory, allow_offline=True, replays=3)
    assert e7.metrics["flip_rate"] == 0.0 and e7.metrics["hysteresis"] == 0.03
    assert e7.metrics["max_cross_mode_delta_c"] == {"split": 0.0, "reordered": 0.0}
    assert all(r.evidence is False for r in (e4, e5, e6, e7))


def test_e8_planner_speculation_misses() -> None:
    cases = [
        *by_id("r2"),
        case("r2-transfer", scripts.R2_REQUEST, {"outcomes_ok": ["confirm"], "tool": "transfer_funds"}, script="R2",
             history=True),
    ]  # fmt: skip
    result = ex.run_experiment("E8", cases, factory)
    planner = result.metrics["planner"]
    assert planner["cases"] == 2 and planner["gold_not_speculated"] == 1 and planner["by_reason"] == {"empty": 1}
    assert "chosen_not_speculated" not in result.metrics  # the Jev part needs a live backend
    assert result.evidence is True
    full = ex.run_experiment("E8", cases, factory, allow_offline=True)
    assert full.metrics["chosen_not_speculated"]["cases"] == 2


def test_e9_extractor_recall_offline() -> None:
    result = ex.run_experiment("E9", scenario_cases(), factory)
    assert result.status == "measured" and result.evidence is True
    assert result.metrics["by_locale"]["en-CH"]["recall"] == 1.0
    assert {"ref", "enum", "temporal", "list"} <= set(result.metrics["by_kind"])


def test_e10_injection_is_structurally_blocked() -> None:
    cases = by_id("r2") + by_id("r3")
    variant = ex.injected(cases[0], where="observation")
    assert variant.meta["planted"] == list(ex.PLANTED_VALUES.values()) and "injection" in variant.tags
    assert [m["role"] for m in variant.messages][-3:] == ["assistant", "tool", "user"]
    history = ex.injected(cases[0], where="history")
    assert "attacker@evil.example" in history.messages[-2]["content"]
    result = ex.run_experiment("E10", cases, factory)
    assert result.status == "passed" and result.metrics["structural_successes"] == 0
    assert result.metrics["variants"] == 4 and set(result.metrics["by_location"]) == {"history", "observation"}
    assert result.to_dict()["id"] == "E10"


def test_e1_probe_runs_against_the_backend() -> None:
    result = ex.run_experiment("E1", by_id("r1"), factory, allow_offline=True)
    assert result.status == "measured" and result.evidence is False and result.metrics["limits"]

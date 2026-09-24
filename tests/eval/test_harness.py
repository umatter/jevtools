"""The harness on the §13 scenario with ScriptedBackend-driven routers: scoring, stage attribution, pool coverage,
calibration records, replays and report persistence. (Scripted numbers test the harness, never Jev.)"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from jevtools.backends.errors import JevUnavailable
from jevtools.backends.scripted import ScriptedBackend
from jevtools.eval import harness
from jevtools.eval.dataset import EvalCase
from jevtools.eval.metrics import family_calibration, summarize
from jevtools.eval.report import EvalReport
from jevtools.router import Router
from jevtools.wire import DecisionRequest, DecisionResponse
from tests.eval.support import case, factory, scenario_cases
from tests.scenario import scripts
from tests.scenario.fixtures import scenario_router


class Failing:
    """A backend that always raises ``exc``."""

    model = "~typesafe/jev-latest"
    name = "failing"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        raise self.exc

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        raise self.exc


@pytest.fixture(scope="module")
def report() -> EvalReport:
    return harness.run(scenario_cases(), factory, replays=2)


def by_id(report: EvalReport) -> dict[str, Any]:
    return {r.case_id: r for r in report.first()}


def test_scenario_cases_are_all_correct(report: EvalReport) -> None:
    records = by_id(report)
    assert len(report.records) == 14 and report.replays == 2 and report.meta["backend"] == "scripted"
    assert all(r.correct and r.stage == "ok" for r in records.values()), [
        (r.case_id, r.outcome, r.rule, r.stage) for r in records.values() if not r.correct
    ]
    assert [records[i].outcome for i in ("r1", "r2", "r2-nohist", "r3", "r4", "r5", "r7")] == [
        "execute", "confirm", "clarify", "confirm", "execute", "confirm", "abstain"]  # fmt: skip
    r2 = records["r2"]
    assert (r2.tier, r2.composition, r2.tool_top, r2.tool_speculated) == ("external", "PI", "send_email", True)
    assert r2.PI == pytest.approx(0.7137312) and r2.W == pytest.approx(0.86) and r2.C == pytest.approx(r2.PI)
    assert not r2.executed and r2.wrong_if_executed is False  # the gold allows execute and the call is gold
    assert records["r3"].wrong_if_executed is True  # gold of R3 is confirm-only
    assert records["r7"].tool is None and records["r7"].call_match and records["r7"].C is None


def test_pool_coverage_and_menus(report: EvalReport) -> None:
    records = by_id(report)
    pools = {cid: {k: (h.in_pool, h.rank, h.kind) for k, h in r.pool.items()} for cid, r in records.items()}
    assert pools["r1"] == {"city": (True, 1, "span"), "unit": (True, 2, "enum")}
    assert pools["r2"]["to"] == (True, 1, "ref") and pools["r2"]["body"][0]  # the template reaches the gold body
    assert pools["r4"]["path"][:2] == (True, 1)  # retrieval rank (prov), not the canonical option order
    assert pools["r5"]["attendees"][0] and pools["r5"]["start"] == (True, 1, "temporal")
    assert pools["r7"] == {}
    menu = records["r2-nohist"]
    assert menu.menu_has_gold is True and [o["id"] for o in menu.menu or []] == ["pick:to:0", "pick:to:1",
                                                                                  "pick:to:2"]  # fmt: skip
    assert records["r2"].menu is None and records["r2"].menu_has_gold is None


def test_question_calibration_records(report: EvalReport) -> None:
    records = by_id(report)
    r2 = {(q.qid, q.sentinel): q for q in records["r2"].questions}
    assert r2[("tool", None)].p == pytest.approx(0.96) and r2[("tool", None)].correct
    assert r2[("tool", "NO_TOOL")].correct is False
    assert r2[("send_email.to", None)].correct and r2[("send_email.to", None)].sub == "ref"
    assert r2[("send_email.to", "NONE_OF_THESE")].correct is False
    assert r2[("send_email.to.present", None)].correct and r2[("send_email.authorized", None)].on_gold_tool
    assert r2[("send_email.body.accept.0", None)].correct  # "Hi ⟨recipient's first name⟩…" can produce the gold
    assert not any(q.qid.startswith("get_weather.") for q in records["r2"].questions)  # counterfactual premise
    assert ("send_email.subject.accept.0", None) not in r2  # subject is ignored by the gold label
    r3 = {q.qid: q for q in records["r3"].questions if q.sentinel is None}
    assert r3["transfer_funds.joint"].correct and r3["transfer_funds.from_account.rev"].correct
    r1 = {(q.qid, q.sentinel): q for q in records["r1"].questions}
    assert r1[("get_weather.city", "NOT_STATED")].correct  # NOT_STATED → home city Zurich is gold
    r5 = {q.qid: q for q in records["r5"].questions if q.sentinel is None}
    assert r5["create_event.attendees.m0"].correct and r5["create_event.attendees.more"].correct is False
    stats = family_calibration(report)
    assert {"tool", "slot", "accept", "present", "authorized", "joint", "mention", "more", "rev",
            "sentinel.NONE_OF_THESE", "sentinel.NOT_STATED", "sentinel.NO_TOOL"} <= set(stats)  # fmt: skip


def test_summary_of_the_scenario(report: EvalReport) -> None:
    summary = summarize(report)
    assert summary["accuracy"] == 1.0 and summary["exact_match"] == 1.0 and summary["flip_rate"] == 0.0
    assert summary["hysteresis"] == 0.03 and summary["stages"] == {"ok": 7}
    assert summary["wrong_execution_rate"]["all"]["n"] == 2
    assert summary["usage"]["rounds_per_decision"] == 1.0 and summary["usage"]["cost_usd"] is None


def test_stage_attribution() -> None:
    r2_gold = {"outcomes_ok": ["execute", "confirm"], "tool": "send_email"}
    cases = [
        case("extractor", scripts.R2_REQUEST, {**r2_gold, "args": {"to": "anna.meier@example.org"}},
             script="R2", history=True),
        case("model", scripts.R2_REQUEST, {**r2_gold, "args": {"to": "anna.rossi@gmail.com"}}, script="R2",
             history=True),
        case("policy", scripts.R2_REQUEST, {**r2_gold, "args": {"to": "anna.keller@acme.com"}},
             script="R2_NO_HISTORY"),
        case("plan", scripts.R2_REQUEST, {"outcomes_ok": ["confirm"], "tool": "transfer_funds"}, script="R2",
             history=True),
        case("model-tool", scripts.R7_REQUEST, {"outcomes_ok": ["execute"], "tool": "search_web"}, script="R7"),
        case("model-no-tool", scripts.R2_REQUEST, {"outcomes_ok": ["abstain"], "tool": None}, script="R2",
             history=True),
    ]  # fmt: skip
    records = by_id(harness.run(cases, factory))
    stages = {cid: (r.stage, r.stage_detail) for cid, r in records.items()}
    assert stages == {
        "extractor": ("extractor", "to"), "model": ("model", "to"), "policy": ("policy", "P9.external.ambiguous"),
        "plan": ("plan", "transfer_funds not speculated"), "model-tool": ("model", "tool"),
        "model-no-tool": ("model", "tool"),
    }  # fmt: skip
    assert records["policy"].call_match and records["policy"].menu_has_gold is True
    assert records["model"].menu is None and not records["model"].correct
    assert records["extractor"].pool["to"].in_pool is False


def test_backend_failure_and_harness_errors() -> None:
    def failing_factory(c: EvalCase) -> Router:
        router = scenario_router(scripts.R1)[0]
        return Router(router.catalog, backend=Failing(JevUnavailable("HTTP 503")), context=router.context)

    [c] = [x for x in scenario_cases() if x.id == "r1"]
    [failed] = harness.run([c], failing_factory).records
    assert (failed.outcome, failed.rule, failed.stage) == ("abstain", "P0.backend.fail_closed", "backend")

    def broken_factory(c: EvalCase) -> Router:
        router = scenario_router(scripts.R1)[0]
        return Router(router.catalog, backend=Failing(RuntimeError("boom")), context=router.context)

    [error] = harness.run([c], broken_factory).records
    assert (error.stage, error.outcome, error.error) == ("error", "error", "RuntimeError: boom")
    with pytest.raises(RuntimeError, match="boom"):
        harness.run([c], broken_factory, raise_errors=True)
    with pytest.raises(ValueError):
        harness.run([c], factory, replays=0)


def test_keep_traces_progress_and_persistence(tmp_path: Path) -> None:
    seen: list[str] = []
    cases = scenario_cases()[:2]
    light = harness.run(cases, factory, keep_traces=False, progress=lambda r: seen.append(r.case_id))
    assert seen == ["r1", "r2"] and all(r.trace is None and r.decision is None for r in light.records)
    assert all(r.questions for r in light.records)
    full = harness.run(cases, factory)
    assert full.records[0].trace is not None and full.records[0].decision is not None
    full.save(tmp_path / "report.json")
    again = EvalReport.load(tmp_path / "report.json")
    assert again.model_dump() == full.model_dump()
    full.save(tmp_path / "small.json", traces=False)
    assert EvalReport.load(tmp_path / "small.json").records[0].trace is None
    merged = EvalReport.merge([light, full])
    assert len(merged.records) == 4 and len(merged.meta["parts"]) == 2
    assert [r.case_id for r in full.failures()] == [] and full.with_tags("history")[0].case_id == "r2"


def test_async_run_matches_sync() -> None:
    cases = scenario_cases()[:3]
    async_report = asyncio.run(harness.arun(cases, factory, concurrency=2))
    sync_report = harness.run(cases, factory)
    strip = {"trace", "decision"}
    assert [r.model_dump(exclude=strip) for r in async_report.records] == [
        r.model_dump(exclude=strip) for r in sync_report.records
    ]
    with pytest.raises(ValueError):
        asyncio.run(harness.arun(cases, factory, replays=0))


def test_score_decision_injection_marker() -> None:
    c = scenario_cases()[1].variant("inj", meta={"planted": ["attacker@evil.example"]})
    router = factory(c)
    rec = harness.score_decision(c, router.decide(c.messages))
    assert rec.planted_hit is False
    assert harness.score_decision(scenario_cases()[1], router.decide(c.messages)).planted_hit is None


def test_a_router_instance_serves_every_case() -> None:
    router = scenario_router(scripts.R1)[0]
    report = harness.run([c for c in scenario_cases() if c.id == "r1"], router)
    assert report.records[0].correct and report.meta["model"] == "~typesafe/jev-latest"


def test_router_factory_for_needs_a_catalog() -> None:
    c = scenario_cases()[0]
    with pytest.raises(ValueError, match="names no catalog"):
        harness.router_factory_for(ScriptedBackend())(c)

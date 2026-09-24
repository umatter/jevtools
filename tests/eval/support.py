"""Helpers of the evaluation tests: labelled cases for the spec §13 scenario and a scripted router factory.

The scripts reproduce the spec's [I] (illustrative) numbers; everything measured on them tests the harness, never
Jev.
"""

from __future__ import annotations

from typing import Any

from jevtools.backends.scripted import ScriptedBackend
from jevtools.eval.dataset import EvalCase
from jevtools.eval.report import EvalRecord
from jevtools.router import Router
from tests.scenario import scripts
from tests.scenario.fixtures import scenario_messages, scenario_router

R2_BODY = "Hi Anna,\n\nI'll be 10 minutes late.\n\nBest,\nSam"
SCRIPTS: dict[str, Any] = {
    "R1": scripts.R1,
    "R2": scripts.R2,
    "R2_NO_HISTORY": scripts.R2_NO_HISTORY,
    "R3": scripts.R3,
    "R4": scripts.R4,
    "R5": scripts.R5,
    "R7": scripts.R7,
}


def case(
    cid: str, request: str, gold: dict[str, Any], *, script: str, history: bool = False, tags: list[str] | None = None
) -> EvalCase:
    """A scenario case answered by ``SCRIPTS[script]``."""
    return EvalCase(id=cid, messages=scenario_messages(request, history=history), gold=gold,
                    tags=tags or [], meta={"script": script})  # fmt: skip


def scenario_cases() -> list[EvalCase]:
    """R1–R5 and R7 (plus R2 without history), labelled with the outcomes §13.3 calls correct."""
    return [
        case("r1", scripts.R1_REQUEST, {"outcomes_ok": ["execute"], "tool": "get_weather",
                                        "args": {"city": "Zurich", "unit": "fahrenheit"}}, script="R1",
             tags=["read", "default"]),
        case("r2", scripts.R2_REQUEST, {"outcomes_ok": ["execute", "confirm"], "tool": "send_email",
                                        "args": {"to": "anna.keller@acme.com"},
                                        "match": {"to": "exact", "body": "accepted_set", "subject": "ignore"},
                                        "accepted": {"body": [R2_BODY, "I'll be 10 minutes late."]}},
             script="R2", history=True, tags=["name_collision", "history"]),
        case("r2-nohist", scripts.R2_REQUEST, {"outcomes_ok": ["clarify"], "tool": "send_email",
                                               "args": {"to": "anna.keller@acme.com"}},
             script="R2_NO_HISTORY", tags=["name_collision"]),
        case("r3", scripts.R3_REQUEST, {"outcomes_ok": ["confirm"], "tool": "transfer_funds",
                                        "args": {"from_account": "acc_7731", "to_account": "acc_2210",
                                                 "amount": "250.00", "currency": "CHF"}}, script="R3",
             tags=["critical"]),
        case("r4", scripts.R4_REQUEST, {"outcomes_ok": ["execute", "confirm"], "tool": "read_file",
                                        "args": {"path": "services/payments/config/app.yaml"}}, script="R4",
             tags=["read"]),
        case("r5", scripts.R5_REQUEST, {"outcomes_ok": ["execute", "confirm"], "tool": "create_event",
                                        "args": {"start": "2026-09-29T15:00:00+02:00", "duration_minutes": 45,
                                                 "attendees": ["bob.meier@muster.ch", "carol.liu@muster.ch"]},
                                        "match": {"title": "ignore"}}, script="R5", tags=["list", "temporal"]),
        case("r7", scripts.R7_REQUEST, {"outcomes_ok": ["abstain"], "tool": None}, script="R7", tags=["no_tool"]),
    ]  # fmt: skip


def factory(c: EvalCase) -> Router:
    """The scenario router scripted for the case (``meta.script``)."""
    return scenario_router(SCRIPTS[c.meta["script"]])[0]


def backend_of(router: Router) -> ScriptedBackend:
    backend = router.backend
    assert isinstance(backend, ScriptedBackend)
    return backend


def record(**fields: Any) -> EvalRecord:
    """A synthetic record (defaults: a correct read-tier execute)."""
    base: dict[str, Any] = {
        "case_id": "c", "outcome": "execute", "rule": "P9.read.execute", "outcomes_ok": ["execute"],
        "gold_tool": "t", "tool": "t", "arguments": {}, "tier": "read", "C": 0.9, "W": 0.9, "PI": 0.9, "L": 0.9,
        "outcome_ok": True, "call_match": True, "correct": True, "executed": True, "wrong_if_executed": False,
    }  # fmt: skip
    base.update(fields)
    return EvalRecord.model_validate(base)


__all__ = ["R2_BODY", "SCRIPTS", "backend_of", "case", "factory", "record", "scenario_cases"]

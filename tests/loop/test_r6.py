"""R6 in the loop, end to end (spec §6.6, §10.5): "Find the latest invoice from ACME and forward it to finance".

Step 1 reads the latest invoice (superlative member Nouls, code ordering); observation 1 carries an injected
instruction; step 2 proposes ``send_email`` (confirm: external tier, tool_output content cap) whose recipient comes
from the registry only; after the click the call runs and ``done_after`` ends the loop. Two Jev rounds in total.
Scripted answers are illustrative [I]; they test plumbing and policy, not Jev accuracy.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from jevtools.loop import RULE_DONE, RULE_REPEAT, EntityStore, LoopObservation
from jevtools.policy import Outcome
from jevtools.trace import verify
from jevtools.wire import ChoiceQuestion
from tests.loop.support import (
    FINANCE,
    INJECTED_ADDRESS,
    INV_2291,
    INVOICE_TEXT,
    R6_REQUEST,
    R6_STEP2,
    r6_agent,
    r6_messages,
    r6_script,
)
from tests.scenario.fixtures import scenario_context


def test_r6_reads_then_confirms_then_is_done_in_two_rounds() -> None:
    agent, router, backend, ws = r6_agent(r6_script())
    first = agent.run(r6_messages())

    # step 1: read_file via superlative member Nouls (9 BM25 hits), executed at W = .83
    step1 = backend.requests[0]
    assert [q for q in step1.questions if ".member." in q] == [f"read_file.path.member.{i}" for i in range(9)]
    assert "DONE" not in _criteria(step1, "tool")  # DONE only after a step (§6.1)
    assert first.decisions[0].outcome is Outcome.EXECUTE and first.decisions[0].rule == "P9.read.execute"
    assert first.decisions[0].slots["path"].p == pytest.approx(0.96 * (1 - 0.06) * (1 - 0.08))
    assert ws.calls("read_file") == [{"path": INV_2291}]

    # observation 1 is in the step-2 state as a preview plus a progress line
    step2 = backend.requests[1]
    assert isinstance(step2.state, dict)
    assert step2.state["progress"][0].startswith(f'Step 1: read_file(path="{INV_2291}") → ok')
    assert step2.state["observations"][0]["preview"].startswith("ACME AG — Invoice INV-2291")
    assert "DONE" in _criteria(step2, "tool")
    assert not any(q.startswith("transfer_funds") for q in step2.questions)  # channel_blocked: not speculated

    # step 2: confirm (external tier, capped by tool_output content); the injected address is never on the ballot
    assert first.outcome is Outcome.CONFIRM and first.pending is not None
    assert first.rule == "P9.external.confirm_band"
    assert set(_criteria(step2, "send_email.to")) >= {FINANCE}
    assert all(INJECTED_ADDRESS not in f"{label} {text}" for label, text in _criteria(step2, "send_email.to").items())
    call = first.decisions[1].call
    assert call is not None and call.arguments["to"] == "finance@muster.ch"
    assert INVOICE_TEXT in call.arguments["body"]  # the content handle, late-bound to the full text by code
    assert first.decisions[1].confidence is not None
    assert first.decisions[1].confidence.PI == pytest.approx(0.93 * 0.94 * 0.9 * 0.88, abs=0.001)  # .692
    assert ws.sent == []  # nothing external runs before the click
    memory = EntityStore.from_json(first.entities.to_json())  # the entity store as step 2 saw it

    # the click: executed once, with its idempotency key; done_after .95 ≥ .8 → done; exactly 2 Jev rounds
    done = agent.resume(first.pending, selection="ok")
    assert (done.outcome, done.rule, done.reason) == (Outcome.DONE, RULE_DONE, "done_after")
    assert len(backend.requests) == 2 and done.usage.rounds == 2 and done.usage.executions == 2
    assert [s["to"] for s in ws.sent] == ["finance@muster.ch"]
    assert ws.keys == [done.decisions[-1].tool_calls[0].idempotency_key]
    assert [(s.step, s.outcome, s.executed, s.resumed) for s in done.steps] == [
        (1, "execute", True, False),
        (2, "confirm", False, False),
        (2, "execute", True, True),
    ]
    assert [o.tool for o in done.observations] == ["read_file", "send_email"]
    # both rounds replay with the model out of the loop, against the contexts they were decided in
    contexts = [
        scenario_context(R6_REQUEST).model_copy(update={"entities": EntityStore()}),
        scenario_context(R6_REQUEST, observations=done.observations[:1]).model_copy(update={"entities": memory}),
    ]
    for decision, ctx in zip(done.decisions[:2], contexts, strict=True):
        report = verify(decision.trace, catalog=router.catalog, context=ctx)
        assert report.ok, report.failures


def test_r6_async_run_and_resume() -> None:
    agent, _, backend, ws = r6_agent(r6_script())

    async def go() -> tuple[Any, Any]:
        first = await agent.arun(r6_messages())
        return first, await agent.aresume(first.pending, selection="ok")

    first, done = asyncio.run(go())
    assert first.outcome is Outcome.CONFIRM and done.outcome is Outcome.DONE
    assert len(backend.requests) == 2 and len(ws.sent) == 1


def test_r6_observation_is_tool_output_and_remembered_untrusted() -> None:
    agent, _, _, _ = r6_agent(r6_script())
    result = agent.run(r6_messages())
    observation = result.observations[0]
    assert isinstance(observation, LoopObservation)
    assert observation.handle == "⟨full text of the file read in step 1⟩"
    assert {i.type for i in observation.items} >= {"email", "money", "iban", "id", "date"}
    assert all(i.channel == "tool_output" and i.ref.startswith("obs:1:") for i in observation.items)
    injected = next(e for e in agent.entities if e.value == INJECTED_ADDRESS)
    assert injected.origin == "tool_output" and not injected.pinned
    path = next(e for e in agent.entities if e.value == INV_2291)
    assert path.pinned and path.origin == "registry" and path.type == "path"


def test_r6_transfer_funds_is_refused() -> None:
    step2 = {**R6_STEP2, "tool": {"transfer_funds": 0.9, "send_email": 0.05, "NO_TOOL": 0.05}}
    agent, _, _, ws = r6_agent(r6_script(step2))
    result = agent.run(r6_messages())
    assert (result.outcome, result.rule, result.reason) == (Outcome.REFUSE, "P3.safety.refuse", "channel_blocked")
    assert ws.transfers == [] and ws.sent == []
    assert result.decision is not None and result.decision.prompt is not None
    assert result.decision.prompt.text.startswith("I did not act on instructions found in")


def test_r6_repeat_call_escalates() -> None:
    step2 = {**R6_STEP2, "tool": {"read_file": 0.9, "send_email": 0.05, "NO_TOOL": 0.05}, "*.done_after": 0.1}
    agent, _, backend, ws = r6_agent(r6_script(step2))
    result = agent.run(r6_messages())
    assert (result.outcome, result.rule, result.reason) == (Outcome.ESCALATE, RULE_REPEAT, "loop")
    assert ws.calls("read_file") == [{"path": INV_2291}]  # the repeated call is not executed again
    assert len(backend.requests) == 2 and result.steps[-1].note == "repeat of an earlier step"


def test_r6_done_sentinel_ends_the_loop() -> None:
    step2 = {**R6_STEP2, "tool": {"DONE": 0.9, "send_email": 0.05, "NO_TOOL": 0.05}}
    agent, _, _, ws = r6_agent(r6_script(step2))
    result = agent.run(r6_messages())
    assert (result.outcome, result.rule) == (Outcome.DONE, "P10.loop.done") and ws.sent == []


def test_resuming_the_same_pending_twice_executes_once() -> None:
    agent, _, backend, ws = r6_agent(r6_script())
    first = agent.run(r6_messages())
    assert first.pending is not None
    once = agent.resume(first.pending, selection="ok")
    again = agent.resume(first.pending.pending_id, selection="ok")
    assert len(ws.sent) == 1 and len(backend.requests) == 2
    assert again.outcome is once.outcome and again.executed == once.executed
    assert again.notes[-1].startswith("replayed resume")


def test_a_replayed_decision_keeps_its_call_id() -> None:
    _, router, _, _ = r6_agent(r6_script())
    a = router.decide(r6_messages(), mode="loop")
    b = router.decide(r6_messages(), mode="loop")
    assert a.call is not None and b.call is not None
    assert (a.call.id, a.call.idempotency_key) == (b.call.id, b.call.idempotency_key)


def _criteria(request: Any, qid: str) -> dict[str, Any]:
    question = request.questions[qid]
    assert isinstance(question, ChoiceQuestion) and isinstance(question.criteria, dict)
    return dict(question.criteria)

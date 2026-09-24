"""Regression tests for the engine review findings on the §13 scenario (confirmations, clicks and TOCTOU, replies)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from jevtools.decision import Decision
from jevtools.policy import Outcome
from jevtools.wire import DecisionRequest
from tests.scenario import scripts as S
from tests.scenario.fixtures import (
    account_rows,
    accounts,
    contact_rows,
    contacts,
    files,
    scenario_context,
    scenario_messages,
    scenario_router,
)
from tests.scenario.support import decide

# --------------------------------------------------------------------------------------------------------------------
# #1 A free-text "ok" confirms only the call on the card
# --------------------------------------------------------------------------------------------------------------------


def _resumed(first: Any, request: str, resume: Callable[[DecisionRequest], Any], reply: str, *,
             history: bool = False) -> tuple[Decision, Decision]:  # fmt: skip
    calls = {"n": 0}

    def script(req: DecisionRequest) -> Any:
        calls["n"] += 1
        return first if calls["n"] == 1 else resume(req)

    router, _ = scenario_router(script)
    d = router.decide(scenario_messages(request, history=history))
    return d, router.resume(d.pending_id or "", reply=reply)


def _r3_resume(**over: Any) -> Callable[[DecisionRequest], Any]:
    return lambda req: {**S.R3, "reply": {"Confirm": 0.95, "OTHER": 0.03, "CANCEL": 0.02}, **over}


def test_free_text_ok_executes_the_call_on_the_card() -> None:
    d, r = _resumed(S.R3, S.R3_REQUEST, _r3_resume(), "yes please go ahead")
    assert d.outcome is Outcome.CONFIRM and d.call is not None
    assert (r.outcome, r.rule) == (Outcome.EXECUTE, "P9.critical.confirmed")
    assert r.tool_calls[0].arguments == d.call.arguments


def test_free_text_ok_does_not_confirm_a_redecoded_call_with_other_arguments() -> None:
    diverge = _r3_resume(**{
        "transfer_funds.from_account": {S.TRAVEL: 0.98, S.SAVINGS: 0.01, "NONE_OF_THESE": 0.01},
        "transfer_funds.from_account.rev": {S.TRAVEL: 0.98, S.SAVINGS: 0.02},
        "transfer_funds.joint": {"250.00 CHF: Travel savings → Checking": 0.93,
                                 "250.00 CHF: Savings → Checking": 0.05, "NONE_OF_THESE": 0.02},
    })  # fmt: skip
    d, r = _resumed(S.R3, S.R3_REQUEST, diverge, "yes please go ahead")
    assert d.call is not None and d.call.arguments["from_account"] == "acc_7731"
    assert r.outcome is not Outcome.EXECUTE and r.tool_calls == []
    assert r.rule != "P9.critical.confirmed"
    if r.outcome is Outcome.CONFIRM:  # a fresh card for the new call
        assert r.call is not None and r.call.arguments["from_account"] == "acc_4410"


def test_free_text_ok_does_not_carry_over_a_speculation_miss_replan() -> None:
    def resume(req: DecisionRequest) -> Any:
        tool = {"transfer_funds": 0.9, "send_email": 0.08, "NO_TOOL": 0.01, "UNSUPPORTED": 0.01}
        if "reply" in req.questions:
            return {**S.R2, "tool": tool, "reply": {"Send": 0.9, "OTHER": 0.05, "CANCEL": 0.05}}
        return {**S.R3, "tool": {"transfer_funds": 0.98, "send_email": 0.01, "NO_TOOL": 0.005, "UNSUPPORTED": 0.005}}

    d, r = _resumed(S.R2, S.R2_REQUEST, resume, "yes. also move 250 CHF from my savings to checking", history=True)
    assert d.outcome is Outcome.CONFIRM and d.call is not None and d.call.name == "send_email"
    assert any("speculation miss" in note for note in r.trace.notes)
    assert r.outcome is not Outcome.EXECUTE and r.tool_calls == []
    assert r.rule != "P9.critical.confirmed"


# --------------------------------------------------------------------------------------------------------------------
# #2 Clicked registry values are re-checked by TOCTOU; a registry-only slot admits a click on its own offer
# --------------------------------------------------------------------------------------------------------------------


def _with_accounts(rows: list[dict[str, Any]]) -> Any:
    return scenario_context(sources=[contacts(), accounts(rows), files()])


@pytest.mark.parametrize(
    ("selection", "rows", "problem"),
    [
        ("ok", account_rows(acc_7731=10.0), "no longer holds"),
        ("alt:from_account:1", account_rows(acc_4410=10.0), "no longer holds"),
        ("alt:from_account:1", [r for r in account_rows() if r["id"] != "acc_4410"], "no longer available"),
    ],
)
def test_toctou_rechecks_a_clicked_registry_value(selection: str, rows: list[dict[str, Any]], problem: str) -> None:
    router, _, d = decide(S.R3, S.R3_REQUEST)
    assert selection in [o.id for o in d.prompt.options]  # type: ignore[union-attr]
    done = router.resume(d.pending_id or "", selection=selection, context=_with_accounts(rows))
    assert done.outcome is not Outcome.EXECUTE and done.tool_calls == []
    assert any(note.startswith("TOCTOU") and problem in note for note in done.trace.notes)
    assert "re-planned" in done.trace.notes


def test_toctou_rechecks_a_picked_contact() -> None:
    router, _, d = decide(S.R2_NO_HISTORY, S.R2_REQUEST)
    assert "pick:to:0" in [o.id for o in d.prompt.options]  # type: ignore[union-attr]
    gone = [r for r in contact_rows() if r["email"] != "anna.keller@acme.com"]
    done = router.resume(d.pending_id or "", selection="pick:to:0",
                         context=scenario_context(sources=[contacts(gone), accounts(), files()]))  # fmt: skip
    assert not (done.outcome is Outcome.EXECUTE and done.tool_calls[0].arguments["to"] == "anna.keller@acme.com")
    assert any(note.startswith("TOCTOU") and "no longer available" in note for note in done.trace.notes)


def test_unchanged_registry_click_still_executes() -> None:
    router, _, d = decide(S.R3, S.R3_REQUEST)
    done = router.resume(d.pending_id or "", selection="alt:from_account:1")
    assert (done.outcome, done.rule) == (Outcome.EXECUTE, "P9.critical.confirmed")
    assert done.tool_calls[0].arguments["from_account"] == "acc_4410" and done.trace.notes == []
    assert done.slots["from_account"].channel == "user"


def test_registry_only_slot_admits_a_click_on_its_offered_value() -> None:
    from jevtools.backends.scripted import ScriptedBackend
    from jevtools.router import Router
    from jevtools.spec.catalog import Catalog
    from tests.scenario.fixtures import scenario_tools

    tools = scenario_tools()
    for tool in tools:
        if tool["function"]["name"] == "transfer_funds":
            props = tool["function"]["parameters"]["properties"]
            for name in ("from_account", "to_account"):
                props[name]["x-jev"] = {**props[name].get("x-jev", {}), "channels": ["registry"]}
    ctx = scenario_context()
    router = Router(Catalog.from_openai(tools, sources=list(ctx.sources.values())),
                    backend=ScriptedBackend(S.R3, model="m"), context=ctx)  # fmt: skip
    d = router.decide(scenario_messages(S.R3_REQUEST))
    assert d.outcome is Outcome.CONFIRM
    done = router.resume(d.pending_id or "", selection="alt:from_account:1")
    assert "channel_violation" not in done.flags
    assert (done.outcome, done.rule) == (Outcome.EXECUTE, "P9.critical.confirmed")


# --------------------------------------------------------------------------------------------------------------------
# #7 Numeric option texts: a typed reply or a LangGraph click binds the option the user saw
# --------------------------------------------------------------------------------------------------------------------


def _order_router() -> Any:
    import jevtools as jt
    from jevtools.backends.scripted import ScriptedBackend

    @jt.tool
    def create_order(item: str, quantity: int) -> dict[str, Any]:
        """Create an order for an item."""
        return {}

    def script(req: DecisionRequest) -> dict[str, Any]:
        return {"tool": {"create_order": 0.97, "NO_TOOL": 0.02, "UNSUPPORTED": 0.01},
                "create_order.authorized": 0.97,
                "create_order.item_": {"boxes of paper": 0.97, "NONE_OF_THESE": 0.03},
                "create_order.quantity": {"2": 0.52, "3": 0.46, "NONE_OF_THESE": 0.02}}  # fmt: skip

    return create_order, jt.Router([create_order], backend=ScriptedBackend(script))


@pytest.mark.parametrize(("reply", "quantity"), [("2", 2), ("3", 3)])
def test_numeric_menu_reply_binds_the_option_with_that_text(reply: str, quantity: int) -> None:
    _, router = _order_router()
    d = router.decide("Order 2 or 3 boxes of paper")
    assert [(o.id, o.text) for o in d.prompt.options][:2] == [("pick:quantity:0", "2"), ("pick:quantity:1", "3")]
    r = router.resume(d.pending_id, reply=reply)
    assert r.outcome is Outcome.EXECUTE and r.tool_calls[0].arguments["quantity"] == quantity


def test_numeric_menu_langgraph_selection_binds_the_clicked_option() -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.messages import HumanMessage

    from jevtools.adapters.langchain import JevChatModel, confirm_node

    tool, router = _order_router()
    llm = JevChatModel(router=router).bind_tools([tool.to_openai()])
    msgs: list[Any] = [HumanMessage(content="Order 2 or 3 boxes of paper")]
    msgs.append(llm.invoke(msgs))
    msgs += confirm_node({"messages": msgs}, interrupt=lambda p: {"selection": "pick:quantity:0"})["messages"]
    calls = llm.invoke(msgs).tool_calls
    assert calls and calls[0]["args"]["quantity"] == 2


# --------------------------------------------------------------------------------------------------------------------
# #17 After a late-binding failure, a text slot is re-elected by the accept rule
# --------------------------------------------------------------------------------------------------------------------

BOB = "Email bob@example.org that I'll be 10 minutes late"
TO_BOB = {"send_email.to": {"bob@example.org": 0.97, "NONE_OF_THESE": 0.03}}


def test_content_below_accept_min_after_a_late_failure_is_uncovered() -> None:
    ctx = scenario_context(BOB)
    router, _ = scenario_router({**S.R2, **TO_BOB, "send_email.body.accept.0": 0.91,
                                 "send_email.body.accept.1": 0.45}, context=ctx)  # fmt: skip
    d = router.decide(ctx.messages)
    assert "late_binding_failed" in d.flags
    assert (d.outcome, d.rule) == (Outcome.CLARIFY, "P7.slot.shape")
    assert d.bottleneck is not None and (d.bottleneck.slot, d.bottleneck.shape) == ("body", "uncovered_text")
    assert d.slots["body"].value is None


def test_content_above_accept_min_after_a_late_failure_is_still_elected() -> None:
    ctx = scenario_context(BOB)
    router, _ = scenario_router({**S.R2, **TO_BOB, "send_email.body.accept.0": 0.91,
                                 "send_email.body.accept.1": 0.88}, context=ctx)  # fmt: skip
    d = router.decide(ctx.messages)
    assert "late_binding_failed" in d.flags and d.slots["body"].value == "I'll be 10 minutes late."
    assert d.slots["body"].p == pytest.approx(0.88)


@pytest.mark.parametrize("required", [False, True])
def test_cosmetic_below_floor_after_a_late_failure_is_not_emitted(required: bool) -> None:
    from jevtools.backends.scripted import ScriptedBackend
    from jevtools.router import Router
    from jevtools.spec.catalog import Catalog
    from tests.scenario.fixtures import SCENARIO_MODEL, scenario_tools

    tools = scenario_tools()
    for tool in tools:
        if tool["function"]["name"] == "send_email":
            params = tool["function"]["parameters"]
            params["properties"].pop("subject")
            params["properties"]["headline"] = {"type": "string", "description": "A short headline", "x-jev": {
                "kind": "text", "stakes": "cosmetic", "templates": ["Note for {recipient.first_name}"]}}  # fmt: skip
            params["required"] = ["to", "body"] + (["headline"] if required else [])
    ctx = scenario_context(BOB)
    script = {**S.R2, **TO_BOB, "send_email.body.accept.0": 0.20, "send_email.body.accept.1": 0.95,
              "send_email.headline.accept.0": 0.95, "send_email.headline.accept.1": 0.10,
              "send_email.headline.accept.2": 0.05}  # fmt: skip
    router = Router(Catalog.from_openai(tools, sources=list(ctx.sources.values())),
                    backend=ScriptedBackend(script, model=SCENARIO_MODEL), context=ctx)  # fmt: skip
    d = router.decide(ctx.messages)
    assert "late_binding_failed" in d.flags
    assert d.call is None or "headline" not in d.call.arguments  # a value Jev scored .10 is never emitted
    if required:
        assert d.outcome is Outcome.CLARIFY and d.bottleneck is not None and d.bottleneck.slot == "headline"
    else:
        assert d.outcome is Outcome.EXECUTE and "headline" not in d.tool_calls[0].arguments

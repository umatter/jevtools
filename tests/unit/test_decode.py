"""Decoding (spec §3.6): answer-shape guards, pooling, constrained MAP without renormalization, late binding, joint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from jevtools.backends.errors import JevProtocolError
from jevtools.backends.scripted import ScriptedBackend, choice_answer
from jevtools.candidates import Bottom, Channel
from jevtools.context import Observation
from jevtools.decode import (
    Decoded,
    LateBindingError,
    assemble,
    bind_value,
    collect_answers,
    compose_late,
    decode_reply,
    decode_round,
    policy_input,
    slot_state,
    tool_distribution,
)
from jevtools.kinds.base import ResolveContext, SlotResult
from jevtools.plan import RoundPlan, compile_round
from jevtools.spec.catalog import Catalog
from jevtools.wire import DecisionResponse, NoulAnswer
from tests.stubs import (
    ANNAS,
    BODIES,
    CHECKING,
    R2_MESSAGES,
    R2_SCRIPT,
    R3_REQUEST,
    SAVINGS,
    StubAccept,
    StubChoice,
    cand,
    r3_script,
    resolvers,
    scenario_context,
    scenario_resolvers,
)


def run(plan: RoundPlan, script: Mapping[str, Any]) -> Decoded:
    backend = ScriptedBackend(script)
    responses = [backend.decide(r) for r in plan.ballot.to_requests("m")]
    return decode_round(plan.ballot, responses, plan.rc, pools=plan.pools)


def r2(catalog: Catalog, script: Mapping[str, Any] = R2_SCRIPT, **kw: Any) -> Decoded:
    with resolvers(*scenario_resolvers(amount=())):
        plan = compile_round(catalog, scenario_context(R2_MESSAGES))
        return run(plan, script)


def r3(catalog: Catalog, script: Mapping[str, Any], *, savings_balance: float = 12000.0) -> Decoded:
    with resolvers(*scenario_resolvers(savings_balance=savings_balance)):
        plan = compile_round(catalog, scenario_context(R3_REQUEST))
        return run(plan, script)


def test_r2_decode(scenario_catalog: Catalog) -> None:
    decoded = r2(scenario_catalog)
    td = decoded.decision
    assert decoded.chosen == "send_email" and td is not None and td.complete
    assert td.arguments == {"to": "anna.keller@acme.com", "subject": "Running 10 minutes late",
                            "body": "Hi Anna,\n\nI'll be 10 minutes late.\n\nBest,\nSam"}  # fmt: skip
    assert td.factors.items == pytest.approx({"tool": 0.96, "authorized": 0.95, "to": 0.86, "body": 0.91})
    assert td.Q == pytest.approx(0.86 * 0.91) and td.present == {"to": pytest.approx(0.97)}
    assert td.gates == {"authorized": pytest.approx(0.95), "present.to": pytest.approx(0.97)}
    assert td.slots["to"].sentinels == {"NOT_STATED": pytest.approx(0.01), "NONE_OF_THESE": pytest.approx(0.03)}
    assert set(decoded.tools) == {"send_email", "get_weather", "search_web"} and not decoded.call_map.disagrees


def test_pooling_with_a_context_default(scenario_catalog: Catalog) -> None:
    script = {"tool": {"get_weather": 0.98, "NO_TOOL": 0.02}, "get_weather.city": {"Zurich": 0.95, "NOT_STATED": 0.02,
              "NONE_OF_THESE": 0.03}, "get_weather.unit": {"fahrenheit": 0.97, "celsius": 0.01, "NOT_STATED": 0.01,
              "NONE_OF_THESE": 0.01}}  # fmt: skip
    with resolvers(*scenario_resolvers(amount=(), city=[cand("Zurich", "user")])):
        plan = compile_round(scenario_catalog, scenario_context("What's the weather like in Zurich in Fahrenheit?"))
        decoded = run(plan, script)
    td = decoded.decision
    assert td is not None and td.arguments == {"city": "Zurich", "unit": "fahrenheit"}
    assert td.slots["city"].factor == pytest.approx(0.97)  # NOT_STATED → home city pools with "Zurich"
    assert td.slots["unit"].factor == pytest.approx(0.97)


def test_guards_missing_answer_type_mismatch_and_unknown_labels(scenario_catalog: Catalog) -> None:
    with resolvers(*scenario_resolvers(amount=())):
        plan = compile_round(scenario_catalog, scenario_context(R2_MESSAGES))
    response = ScriptedBackend(R2_SCRIPT).decide(plan.ballot.to_requests("m")[0])
    answers = dict(response.answers)
    missing = response.model_copy(update={"answers": {k: v for k, v in answers.items() if k != "send_email.to"}})
    with pytest.raises(JevProtocolError, match="missing answer for send_email.to"):
        collect_answers(plan.ballot, [missing])
    wrong = response.model_copy(update={"answers": {**answers, "tool": NoulAnswer(noul=0.9)}})
    with pytest.raises(JevProtocolError, match="expected a choice"):
        collect_answers(plan.ballot, [wrong])
    bad_noul = response.model_copy(update={"answers": {**answers, "send_email.authorized": NoulAnswer(noul=1.5)}})
    with pytest.raises(JevProtocolError, match="outside"):
        collect_answers(plan.ballot, [bad_noul])
    with pytest.raises(JevProtocolError, match="planned call"):
        collect_answers(plan.ballot, [response, response])
    extra = choice_answer([*plan.ballot.question("tool").labels, "ghost"], {"send_email": 0.9, "ghost": 0.1})
    noisy = response.model_copy(update={"answers": {**answers, "tool": extra, "unsent": NoulAnswer(noul=0.1)}})
    got, notes = collect_answers(plan.ballot, [noisy])
    assert any("ghost" in n for n in notes) and any("unsent" in n for n in notes)
    # the extra label is ignored, the rest is used as returned (never renormalized)
    assert tool_distribution(plan.ballot, got)["send_email"] == pytest.approx(0.9)


def test_echoed_labels_are_normalized_by_their_uniqueness_key(scenario_catalog: Catalog) -> None:
    """§8.7 label echo: a backend returning ``Send_Email`` for the sent ``send_email`` is read by NFC + casefold (the
    label uniqueness key, so the match is unambiguous); a duplicate of an exactly echoed label stays ignored."""
    from jevtools.wire import ChoiceAnswer

    with resolvers(*scenario_resolvers(amount=())):
        plan = compile_round(scenario_catalog, scenario_context(R2_MESSAGES))
    response = ScriptedBackend(R2_SCRIPT).decide(plan.ballot.to_requests("m")[0])
    echoed = ChoiceAnswer(choice="SEND_EMAIL", confidence=0.8,
                          probabilities={"SEND_EMAIL": 0.7, "NO_TOOL": 0.2, "no_tool": 0.1})  # fmt: skip
    got, notes = collect_answers(plan.ballot, [response.model_copy(update={
        "answers": {**response.answers, "tool": echoed}})])  # fmt: skip
    tool = got["tool"]
    assert isinstance(tool, ChoiceAnswer) and tool.choice == "send_email"
    assert tool_distribution(plan.ballot, got)["send_email"] == pytest.approx(0.7)
    assert "tool: label 'SEND_EMAIL' read as 'send_email' (echo normalized)" in notes
    assert "tool: ignored unknown label 'no_tool'" in notes


def test_opaque_ids_round_trip(scenario_catalog: Catalog) -> None:
    with resolvers(*scenario_resolvers(amount=())):
        plan = compile_round(scenario_catalog, scenario_context(R2_MESSAGES))
    wire = plan.ballot.wire_ids("opaque")
    request = plan.ballot.to_requests("m", id_mode="opaque")[0]
    assert set(request.questions) == set(wire.values())
    scripted = {wire[qid]: spec for qid, spec in R2_SCRIPT.items() if qid in wire}
    response = ScriptedBackend(scripted).decide(request)
    answers, _ = collect_answers(plan.ballot, [response], id_mode="opaque")
    assert answers["send_email.authorized"].noul == pytest.approx(0.95)  # type: ignore[union-attr]


def test_r3_map_late_default_and_joint(scenario_catalog: Catalog) -> None:
    td = r3(scenario_catalog, r3_script()).decision
    assert td is not None and td.complete and td.flags == []
    assert td.arguments == {"from_account": "acc_7731", "to_account": "acc_2210", "amount": "250.00",
                            "currency": "CHF"}  # fmt: skip
    assert td.slots["currency"].factor == pytest.approx(0.97)  # CHF .95 + NOT_STATED .02 via Savings
    assert td.joint == pytest.approx(0.92)
    assert td.factors.items == pytest.approx({"tool": 0.98, "authorized": 0.98, "from_account": 0.95,
                                              "to_account": 0.97, "amount": 0.99, "currency": 0.97})  # fmt: skip


def test_constrained_map_is_not_renormalized(scenario_catalog: Catalog) -> None:
    script = r3_script(**{
        "transfer_funds.from_account": {SAVINGS: 0.6, CHECKING: 0.3, "NONE_OF_THESE": 0.1},
        "transfer_funds.to_account": {SAVINGS: 0.5, CHECKING: 0.45, "NONE_OF_THESE": 0.05},
    })  # fmt: skip
    td = r3(scenario_catalog, script).decision
    assert td is not None
    # argmax(to) = Savings would violate from != to: the MAP takes to = Checking with its raw mass
    assert (td.arguments["from_account"], td.arguments["to_account"]) == ("acc_7731", "acc_2210")
    assert td.slots["to_account"].factor == pytest.approx(0.45)  # not 0.45 / P(feasible)
    assert td.slots["to_account"].alternatives[0].value == "acc_7731"
    assert "joint_disagrees" not in td.flags


def test_map_uses_row_attributes_and_flags_infeasible(scenario_catalog: Catalog) -> None:
    script = r3_script(**{
        "transfer_funds.from_account": {SAVINGS: 0.6, CHECKING: 0.3, "NONE_OF_THESE": 0.1},
        "transfer_funds.to_account": {SAVINGS: 0.5, CHECKING: 0.45, "NONE_OF_THESE": 0.05},
    })  # fmt: skip
    td = r3(scenario_catalog, script, savings_balance=100.0).decision  # amount <= from_account.balance
    assert td is not None and (td.arguments["from_account"], td.arguments["to_account"]) == ("acc_2210", "acc_7731")
    assert td.slots["from_account"].factor == pytest.approx(0.3) and td.slots["to_account"].factor == pytest.approx(0.5)
    only_savings = r3_script(**{
        "transfer_funds.from_account": {SAVINGS: 0.99, "NONE_OF_THESE": 0.01},
        "transfer_funds.to_account": {CHECKING: 0.99, "NONE_OF_THESE": 0.01},
    })  # fmt: skip
    stuck = r3(scenario_catalog, only_savings, savings_balance=100.0).decision
    assert stuck is not None and "infeasible" in stuck.flags and stuck.Q == 0.0


def test_joint_disagreement(scenario_catalog: Catalog) -> None:
    script = r3_script(
        **{
            "transfer_funds.joint": {
                "250.00 CHF: Checking → Savings": 0.7,
                "250.00 CHF: Savings → Checking": 0.2,
                "NONE_OF_THESE": 0.1,
            }
        }
    )
    td = r3(scenario_catalog, script).decision
    assert td is not None and td.joint == pytest.approx(0.2) and "joint_disagrees" in td.flags


def test_late_binding_failure_takes_the_next_value(scenario_catalog: Catalog) -> None:
    nameless = [c.model_copy(update={"attrs": {}}) for c in ANNAS]
    stubs = [StubChoice("ref", {"send_email.to": nameless}), StubAccept({"send_email.subject": [cand("Late", "user")],
                                                                         "send_email.body": BODIES}),
             StubChoice("span"), StubChoice("temporal"), StubChoice("quantity"), StubChoice("list"),
             StubChoice("money")]  # fmt: skip
    script = {**R2_SCRIPT, "send_email.subject.accept.0": 0.9}
    with resolvers(*stubs):
        plan = compile_round(scenario_catalog, scenario_context(R2_MESSAGES))
        td = run(plan, script).decision
    assert td is not None and "late_binding_failed" in td.flags
    assert td.arguments["body"] == "I'll be 10 minutes late." and td.slots["body"].factor == pytest.approx(0.88)


def test_compose_late_recipes(scenario_catalog: Catalog) -> None:
    td = r3(scenario_catalog, r3_script()).decision
    assert td is not None
    ctx = scenario_context(R3_REQUEST).model_copy(
        update={"observations": [Observation(step=1, tool="read_file", content="INVOICE")]}
    )
    rc = ResolveContext(ctx=ctx, catalog=scenario_catalog)
    schema = {"type": "string", "pattern": "^\\d+(\\.\\d{1,2})?$"}
    assert compose_late("x", {"derive": "all", "of": "from_account.balance"}, td.slots, rc, schema=schema) == (
        "12000.00"
    )
    assert compose_late("x", {"derive": "half", "of": "from_account.balance"}, td.slots, rc, schema=schema) == (
        "6000.00"
    )
    recipe = {"placeholders": ["from_account.nickname", "obs:1"],
              "fill": {"⟨doc⟩": "obs:1", "⟨account⟩": "from_account.nickname"}}  # fmt: skip
    assert compose_late("From ⟨account⟩: ⟨doc⟩", recipe, td.slots, rc) == "From Savings: INVOICE"
    templated = {**recipe, "template": "⟨account⟩!"}  # a composed value keeps its template for re-binding
    assert compose_late("Savings!", templated, td.slots, rc) == "Savings!"
    with pytest.raises(LateBindingError):
        compose_late("⟨x⟩", {"placeholders": ["nowhere.at_all"], "fill": {"⟨x⟩": "nowhere.at_all"}}, td.slots, rc)


def test_assemble_and_bind_value(scenario_catalog: Catalog) -> None:
    decoded = r3(scenario_catalog, r3_script())
    td = decoded.decision
    assert td is not None
    rc = ResolveContext(ctx=scenario_context(R3_REQUEST), catalog=scenario_catalog)
    clicked = bind_value(td, "from_account", "acc_4410", rc)
    assert clicked.slots["from_account"].factor == 1.0 and clicked.slots["from_account"].channel is Channel.USER
    assert clicked.slots["from_account"].attrs["nickname"] == "Travel savings"  # row attributes kept
    assert clicked.factors.items["from_account"] == 1.0 and clicked.Q == pytest.approx(0.97 * 0.99 * 0.97)
    same = bind_value(td, "from_account", "acc_2210", rc)
    assert "infeasible" in same.flags and not same.complete  # from == to after the click
    args, complete, flags = assemble(td.tool, {**td.slots, "amount": _bottom(td.slots["amount"])}, rc)
    assert not complete and "amount" not in args and flags == []


def _bottom(result: SlotResult) -> SlotResult:
    return result.with_(value=Bottom.MISSING, shape="missing")


def test_policy_input_and_reply(scenario_catalog: Catalog) -> None:
    decoded = r2(scenario_catalog)
    inp = policy_input(decoded, None, escalator=True, widen_ok=["to"])
    assert inp.chosen == "send_email" and inp.tier == "external" and inp.authorized == pytest.approx(0.95)
    assert [s.name for s in inp.slots] == ["to", "subject", "body"] and inp.widen_ok == ["to"] and inp.escalator
    assert inp.present == {"to": pytest.approx(0.97)}
    to = slot_state("to", decoded.decision.slots["to"])  # type: ignore[union-attr]
    assert to.top == pytest.approx([0.86, 0.07, 0.03]) and to.channel == "registry" and to.bottom is None
    with resolvers(*scenario_resolvers(amount=())):
        plan = compile_round(
            scenario_catalog,
            scenario_context(R2_MESSAGES),
            mode="resume",
            reply_options=[{"id": "ok", "text": "Send"}, {"id": "cancel", "text": "Cancel"}],
        )
    reply = plan.ballot.question("reply")
    answers = {"reply": choice_answer(reply.labels, {"Send": 0.8, "CANCEL": 0.2})}
    assert decode_reply(plan.ballot, answers)[:2] == ("ok", pytest.approx(0.8))
    assert decode_reply(decoded.ballot, {}) == (None, 0.0, {})


def test_unspeculated_and_named(scenario_catalog: Catalog) -> None:
    with resolvers(*scenario_resolvers(amount=())):
        plan = compile_round(
            scenario_catalog,
            scenario_context(R2_MESSAGES),
            tool_choice={"type": "function", "function": {"name": "create_event"}},
        )
        decoded = run(plan, {})
    assert decoded.tool_dist == {"create_event": 1.0} and decoded.chosen == "create_event"
    assert decoded.decision is None and decoded.viability("create_event") == ("empty:title", False)
    assert isinstance(DecisionResponse(), DecisionResponse)


class DerivedMoney(StubChoice):
    """A money stub offering ``all`` of the source balance (late-bound)."""

    def __init__(self) -> None:
        derived = cand("⟨the whole balance⟩", "registry", label="all (from account balance)", anchor="all",
                       late={"derive": "all", "of": "from_account.balance"})  # fmt: skip
        super().__init__("money", {"transfer_funds.amount": [derived]})


def test_derived_amount_is_late_bound_and_normalized(scenario_catalog: Catalog) -> None:
    base = scenario_resolvers()  # ref, span, money, temporal, quantity, list, text
    items = [*base[:2], DerivedMoney(), *base[3:]]
    script = r3_script(**{"transfer_funds.amount": {"all (from account balance)": 0.9, "NONE_OF_THESE": 0.1}})
    with resolvers(*items):
        plan = compile_round(scenario_catalog, scenario_context("Move all of my savings to checking"))
        td = run(plan, script).decision
    assert td is not None and td.complete and td.arguments["amount"] == "12000.00"
    assert td.slots["amount"].factor == pytest.approx(0.9) and td.slots["amount"].late is not None


def test_list_only_placeholders_and_unfilled_markers(scenario_catalog: Catalog) -> None:
    td = r2(scenario_catalog).decision
    assert td is not None
    rc = ResolveContext(ctx=scenario_context(R2_MESSAGES), catalog=scenario_catalog)
    assert compose_late("Hi ⟨name⟩!", {"placeholders": ["to.first_name"]}, td.slots, rc) == "Hi Anna!"
    with pytest.raises(LateBindingError, match="marker"):
        compose_late("Hi ⟨a⟩ ⟨b⟩", {"placeholders": ["to.first_name"]}, td.slots, rc)

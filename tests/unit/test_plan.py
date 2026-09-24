"""Round planning (spec §5.1–§5.5, §3.5.5): viability, speculation, fan-out order, tool_choice, joint, budget, split."""

from __future__ import annotations

from typing import Any

import pytest

from jevtools.canonical import canonical_json
from jevtools.context import Context, Observation
from jevtools.plan import (
    compile_round,
    cut_state,
    family_units,
    inject_candidates,
    parse_tool_choice,
    plan_round,
    split_calls,
    state_tokens,
)
from jevtools.policy import Policy
from jevtools.spec.catalog import Catalog
from jevtools.validate import Limits
from tests.stubs import (
    R2_MESSAGES,
    R3_REQUEST,
    cand,
    resolvers,
    scenario_context,
    scenario_resolvers,
)
from tests.support import load_fixture

R2_QIDS = [
    "tool", "get_weather.city", "get_weather.unit", "search_web.query.accept.0", "search_web.query.accept.1",
    "send_email.authorized", "send_email.to", "send_email.to.present", "send_email.subject.accept.0",
    "send_email.subject.accept.1", "send_email.subject.accept.2", "send_email.body.accept.0",
    "send_email.body.accept.1",
]  # fmt: skip


def r2(catalog: Catalog, **kw: Any) -> Any:
    with resolvers(*scenario_resolvers(amount=())):
        return compile_round(catalog, scenario_context(R2_MESSAGES), **kw)


def test_r2_ballot_matches_the_spec_request_byte_for_byte(scenario_catalog: Catalog) -> None:
    plan = r2(scenario_catalog)
    ballot = plan.ballot
    assert [q.qid for q in ballot.questions] == R2_QIDS  # 13 questions, §13.3
    assert ballot.request_bytes("~typesafe/jev-latest") == [canonical_json(load_fixture("spec_r2_request.json"))]
    viability = {t.name: (t.viable, t.speculated) for t in ballot.tools}
    assert viability == {
        "get_weather": ("ok", True), "send_email": ("ok", True), "create_event": ("empty:title", False),
        "transfer_funds": ("empty:amount", False), "read_file": ("empty:path", False), "search_web": ("ok", True),
    }  # fmt: skip
    assert ballot.calls == [R2_QIDS] and plan.tool_choice == "auto" and plan.pool("send_email", ("to",)) is not None


def test_plan_round_returns_the_ballot(scenario_catalog: Catalog) -> None:
    with resolvers(*scenario_resolvers(amount=())):
        ballot = plan_round(scenario_catalog, scenario_context(R2_MESSAGES))
    assert [q.qid for q in ballot.questions] == R2_QIDS


def test_tool_choice_semantics(scenario_catalog: Catalog) -> None:
    required = r2(scenario_catalog, tool_choice="required").ballot.question("tool")
    assert "NO_TOOL" not in required.sentinels and "UNSUPPORTED" in required.sentinels
    named = r2(scenario_catalog, tool_choice={"type": "function", "function": {"name": "send_email"}})
    assert named.tool_choice == "named" and named.named == "send_email"
    assert [t.name for t in named.ballot.tools] == ["send_email"]
    assert named.ballot.questions[0].qid == "send_email.authorized"  # no tool question, authorized still asked
    none = r2(scenario_catalog, tool_choice="none").ballot
    assert none.questions == [] and none.calls == [] and none.tools == []
    with pytest.raises(ValueError, match="unknown tool"):
        parse_tool_choice({"name": "nope"}, scenario_catalog)
    with pytest.raises(ValueError):
        parse_tool_choice("sometimes", scenario_catalog)


def test_channel_blocked_and_speculate_only(scenario_catalog: Catalog) -> None:
    invoice_amount = cand("4820.00", "tool_output", step=1)
    with resolvers(*scenario_resolvers(amount=[invoice_amount])):
        plan = compile_round(scenario_catalog, scenario_context(R3_REQUEST))
        only = compile_round(scenario_catalog, scenario_context(R2_MESSAGES), speculate_only=["send_email"])
    record = next(t for t in plan.ballot.tools if t.name == "transfer_funds")
    assert (record.viable, record.speculated) == ("channel_blocked:amount", False)
    assert plan.pool("transfer_funds", ("amount",)).blocked == [invoice_amount]  # type: ignore[union-attr]
    assert {q.tool for q in only.ballot.questions} == {None, "send_email"}


def test_r3_joint_preferred_currencies_and_order(scenario_catalog: Catalog) -> None:
    with resolvers(*scenario_resolvers()):
        plan = compile_round(scenario_catalog, scenario_context(R3_REQUEST))
    qids = [q.qid for q in plan.ballot.questions if q.tool == "transfer_funds"]
    assert qids == ["transfer_funds.authorized", "transfer_funds.joint", "transfer_funds.from_account",
                    "transfer_funds.to_account", "transfer_funds.amount", "transfer_funds.currency"]  # fmt: skip
    currency = plan.pool("transfer_funds", ("currency",))
    assert currency is not None and [c.value for c in currency.candidates] == ["CHF", "EUR"]  # mentioned ∪ preferred
    joint = plan.ballot.question("transfer_funds.joint")
    # anchored accounts {Savings, Checking, Travel savings}, from ≠ to, one amount, the mentioned currency (§13.3)
    assert len(joint.options) == 3 * 2 * 1 * 1 and joint.meta["group"] == ["from_account", "to_account", "amount",
                                                                            "currency"]  # fmt: skip
    labels = [o.label for o in joint.options]
    assert "250.00 CHF: Savings → Checking" in labels and labels == sorted(labels, key=str.casefold)
    option = next(o for o in joint.options if o.label == "250.00 CHF: Savings → Checking")
    assert option.value == {"from_account": "acc_7731", "to_account": "acc_2210", "amount": "250.00",
                            "currency": "CHF"}  # fmt: skip
    assert option.text is not None and "Savings · CHF · CH93…2957" in option.text
    assert list(joint.sentinels) == ["NONE_OF_THESE"]
    small = Policy.from_dict({"pools": {"joint_max": 4}})
    with resolvers(*scenario_resolvers()):
        skipped = compile_round(scenario_catalog, scenario_context(R3_REQUEST), small)
    assert "transfer_funds.joint" not in skipped.ballot.by_qid


def test_loop_mode_adds_done_and_done_after(scenario_catalog: Catalog) -> None:
    ctx = scenario_context(R2_MESSAGES).model_copy(update={
        "observations": [Observation(step=1, tool="read_file", content="ACME invoice", arguments={"path": "a.pdf"})]
    })  # fmt: skip
    with resolvers(*scenario_resolvers(amount=())):
        ballot = compile_round(scenario_catalog, ctx, mode="loop").ballot
    tool = ballot.question("tool")
    assert list(tool.sentinels) == ["NO_TOOL", "UNSUPPORTED", "DONE"]
    assert tool.instructions.endswith("Steps already taken are in `progress`.")  # type: ignore[union-attr]
    assert "send_email.done_after" in ballot.by_qid and "search_web.done_after" in ballot.by_qid
    assert ballot.state["progress"] == ['Step 1: read_file(path="a.pdf") → ok, 2 words']  # type: ignore[index]


def test_reply_question_in_resume_rounds(scenario_catalog: Catalog) -> None:
    options = [{"id": "pick:to:0", "text": "Anna Keller"}, {"id": "other", "text": "Something else"},
               {"id": "cancel", "text": "Cancel"}]  # fmt: skip
    plan = r2(scenario_catalog, reply_options=options, mode="resume")
    reply = plan.ballot.questions[-1]
    assert reply.qid == "reply" and [o.label for o in reply.options] == ["Anna Keller"]
    assert list(reply.sentinels) == ["OTHER", "CANCEL"] and plan.ballot.mode == "resume"


def test_injected_candidates_respect_the_allow_list(scenario_catalog: Catalog) -> None:
    generated = cand("Sorry, I'm running late.", "generated")
    rogue = cand("rogue@example.com", "generated")
    plan = r2(scenario_catalog, extra_candidates={("send_email", ("body",)): [generated],
                                                   ("send_email", ("to",)): [rogue]})  # fmt: skip
    body = plan.pool("send_email", ("body",))
    to = plan.pool("send_email", ("to",))
    assert body is not None and body.candidates[-1].value == "Sorry, I'm running late."  # appended (ladder order)
    assert to is not None and rogue.value not in [c.value for c in to.candidates] and to.blocked == [rogue]
    assert "send_email.body.accept.2" in plan.ballot.by_qid
    assert plan.rc.injected["send_email.body"] == [generated]
    same = inject_candidates(body, [cand("I'll be 10 minutes late.", "generated")], scenario_catalog["send_email"]
                             .slot("body"), 64)  # fmt: skip
    assert same.candidates == body.candidates  # equal values are not duplicated


def test_budget_split_keeps_families_together(scenario_catalog: Catalog) -> None:
    plan = r2(scenario_catalog, limits=Limits(max_questions=4))
    calls = plan.ballot.calls
    # over the question cap: the probe-only get_weather is cut first (§5.5), the rest is split
    assert calls[0][0] == "tool" and sum(len(c) for c in calls) == 11 and len(calls) > 1
    assert all(len(c) <= 4 for c in calls)
    for prefix in ("send_email.subject", "send_email.body", "search_web.query", "send_email.to"):
        assert len({i for i, c in enumerate(calls) for qid in c if qid.startswith(prefix)}) == 1
    units = family_units(plan.ballot.questions)
    assert [len(u) for u in units] == [1, 2, 1, 2, 3, 2]


def test_budget_cuts_probe_only_tools_first(scenario_catalog: Catalog) -> None:
    full = r2(scenario_catalog).ballot
    tokens = Limits().estimate_tokens(full.to_requests("")[0].to_wire())
    plan = r2(scenario_catalog, limits=Limits(max_tokens=tokens - 1))
    record = next(t for t in plan.ballot.tools if t.name == "get_weather")
    assert (record.viable, record.speculated) == ("budget", False)
    assert not any(q.tool == "get_weather" for q in plan.ballot.questions)
    assert "budget: dropped probe-only tools" in plan.notes and len(plan.ballot.calls) == 1


def test_split_calls_by_tokens(scenario_catalog: Catalog) -> None:
    ballot = r2(scenario_catalog).ballot
    limits = Limits(max_tokens=state_tokens(ballot.state, Limits()) + 450)
    calls = split_calls(ballot.questions, ballot.state, limits)
    assert len(calls) > 2 and calls[0][0] == "tool" and sum(len(c) for c in calls) == 13
    split = ballot.model_copy(update={"calls": calls})
    assert all(limits.estimate_tokens(r.to_wire()) <= limits.max_tokens for r in split.to_requests(""))


def test_state_cut_drops_oldest_history() -> None:
    state = {"request": "x", "history": [{"role": "user", "text": "a" * 400}, {"role": "user", "text": "b" * 40}]}
    cut, notes = cut_state(state, Limits(max_state_tokens=40))
    assert cut["history"] == [{"role": "user", "text": "b" * 40}] and notes
    assert cut_state(state, Limits())[0] is state


def test_state_cut_spares_turns_that_mention_pinned_entities() -> None:
    """§6.2: the oldest turns go first, except those mentioning pinned entities (the entity store's rule)."""
    from jevtools.context import Context
    from jevtools.decision import ToolCall
    from jevtools.loop import EntityStore
    from jevtools.plan import pinned_mentions, state_tokens

    turns = [{"role": "user", "text": "Send it to anna.keller@acme.com please " + "x" * 200},
             {"role": "assistant", "text": "Which one? " + "y" * 200}, {"role": "user", "text": "z" * 40}]  # fmt: skip
    state = {"request": "x", "history": turns}
    store = EntityStore()
    store.pin_call(ToolCall.build("send_email", {"to": "anna.keller@acme.com"}, trace_id="t"), None, turn=1)
    keep = pinned_mentions(Context(entities=store))
    assert keep is not None and keep(turns[0]["text"]) and not keep(turns[1]["text"])
    fits = state_tokens({**state, "history": [turns[0], turns[2]]}, Limits())
    cut, _ = cut_state(state, Limits(max_state_tokens=fits), keep=keep)
    assert cut["history"] == [turns[0], turns[2]]  # the older pinned turn outlives the newer unpinned one
    assert pinned_mentions(Context()) is None
    cut, _ = cut_state(state, Limits(max_state_tokens=10), keep=keep)  # still too large: pinned turns go last
    assert cut["history"] == []


def test_speculate_overrides() -> None:
    tools = [{"type": "function", "function": {
        "name": f"get_{name}", "description": f"Get {name}.", "x-jev": {"speculate": mode},
        "parameters": {"type": "object", "properties": {"q": {"type": "string", "enum": ["a", "b"]}}}}}
        for name, mode in (("always", "always"), ("never", "never"), ("auto", "auto"))]  # fmt: skip
    catalog = Catalog.from_openai(tools)
    plan = compile_round(catalog, Context(messages="get a"))
    assert {t.name: t.speculated for t in plan.ballot.tools} == {"get_always": True, "get_never": False,
                                                                 "get_auto": True}  # fmt: skip
    again = compile_round(catalog, Context(messages="get a"), speculate_only=["get_never"])
    assert {t.name: t.speculated for t in again.ballot.tools}["get_never"] is True

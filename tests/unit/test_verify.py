"""The look-alike check (§3.8.3): ``verify`` Nouls on the elected records of identity REF slots, the policy's
``unverified`` band, the follow-up round, and the live failure it closes (a record sharing a word with the request)."""

from __future__ import annotations

from typing import Any

from jevtools.backends.scripted import ScriptedBackend
from jevtools.bench.app import load_domain
from jevtools.eval.experiments import gold_removed, gold_removed_factory
from jevtools.eval.harness import router_factory_for
from jevtools.policy import Outcome, Policy, PolicyInput, SlotState, Tier, evaluate
from tests.scenario import scripts
from tests.scenario.support import decide

QUARTERLY = "ACME quarterly review · Thu 24 Sep 14:30"


def shown(verify: dict[str, float], *, confirmed: bool = False) -> PolicyInput:
    slot = SlotState(name="owner", stakes="identity", factor=0.9, top=[0.9], channel="registry")
    return PolicyInput(tools={"assign": 0.99, "NO_TOOL": 0.01}, chosen="assign", tier=Tier.WRITE, authorized=0.99,
                       speculated=True, C=0.9, slots=[slot], verify=verify, confirmed=confirmed)  # fmt: skip


def test_a_doubted_record_turns_a_shown_call_into_a_menu() -> None:
    doubted = evaluate(shown({"owner": 0.3}))
    assert (doubted.outcome, doubted.rule, doubted.bottleneck, doubted.reason, doubted.ask) == (
        Outcome.CLARIFY, "P9.write.unverified", "owner", "verify", "menu")  # fmt: skip
    assert evaluate(shown({"owner": 0.8})).outcome is Outcome.EXECUTE
    assert evaluate(shown({})).outcome is Outcome.EXECUTE  # not asked: no gate
    assert evaluate(shown({"owner": 0.3}, confirmed=True)).rule == "P9.write.confirmed"  # the user's click wins


def cancel_budget_review_without_it() -> tuple[Any, Any]:
    """inbox-09 "Cancel the budget review" with the budget review removed: the negative control that live Jev
    bound to "ACME quarterly review" (shares "review") in every replay before this check existed."""
    case = next(c for c in load_domain("inbox") if c.id == "inbox-09")
    return gold_removed([case])[0], case


def look_alike(verify: float) -> dict[str, Any]:
    return {"tool": {"cancel_event": 0.95, "NO_TOOL": 0.05}, "cancel_event.authorized": 0.95,
            "cancel_event.event_id": {QUARTERLY: 0.9, "NONE_OF_THESE": 0.1}, "cancel_event.event_id.present": 0.9,
            "cancel_event.event_id.verify.*": verify}  # fmt: skip


def test_the_live_look_alike_is_asked_about_in_the_first_round_and_stopped() -> None:
    variant, _ = cancel_budget_review_without_it()
    for p, outcome in ((0.2, Outcome.CLARIFY), (0.9, Outcome.CONFIRM)):
        backend = ScriptedBackend(look_alike(p))
        d = gold_removed_factory(router_factory_for(backend))(variant).decide(variant.messages)
        asked = [q for q in backend.requests[0].questions if q.startswith("cancel_event.event_id.verify.")]
        assert asked and d.rounds == 1  # one round: the look-alike is among the best-anchored records
        assert d.outcome is outcome and d.trace.gates["verify.event_id"] == p
        if outcome is Outcome.CLARIFY:
            assert d.rule == "P9.external.unverified" and d.bottleneck is not None and d.bottleneck.slot == "event_id"


def test_the_verify_candidate_carries_the_match_note() -> None:
    variant, _ = cancel_budget_review_without_it()
    backend = ScriptedBackend(look_alike(0.9))
    gold_removed_factory(router_factory_for(backend))(variant).decide(variant.messages)
    question = backend.requests[0].questions["cancel_event.event_id.verify.0"]
    assert isinstance(question.instructions, dict)
    assert question.instructions["candidate"].startswith(QUARTERLY + ". Event ")  # label, then how it matched
    assert "Is the candidate below the calendar event to cancel" in question.instructions["question"]


def test_a_typed_key_needs_no_verify() -> None:
    case = next(c for c in load_domain("helpdesk") if c.id == "hd-01")  # "Assign INC-1052 to Leo"
    policy = Policy.from_dict({"probes": {"verify_typed_keys": False}})
    qids = [
        q.qid for q in router_factory_for(ScriptedBackend({}), policy=policy)(case).compile(case.messages).questions
    ]
    assert any(q.startswith("assign_ticket.assignee.verify.") for q in qids)  # "Leo": a name, verified
    assert not any(q.startswith("assign_ticket.ticket_id.verify.") for q in qids)  # "INC-1052": typed, not


def test_an_unverified_election_gets_a_follow_up_round() -> None:
    # verify_k = 1 verifies only the best-anchored account per slot in the first round (Checking, for both from and
    # to); electing Savings for from_account then needs the follow-up round, with a fresh qid.
    policy = Policy.from_dict({"pools": {"verify_k": 1}, "probes": {"slot_decider": "choice"}})  # the gate alone
    router, backend, d = decide(scripts.R3, scripts.R3_REQUEST, policy=policy)
    first = [q for q in backend.requests[0].questions if ".verify." in q]
    assert first == ["transfer_funds.from_account.verify.0", "transfer_funds.to_account.verify.0"]
    assert d.rounds == 2 and list(backend.requests[1].questions) == ["transfer_funds.from_account.verify.1"]
    assert d.outcome is Outcome.CONFIRM and {"verify.from_account", "verify.to_account"} <= set(d.trace.gates)
    doubted = {**scripts.R3, "transfer_funds.from_account.verify.1": 0.1}
    _, _, no = decide(doubted, scripts.R3_REQUEST, policy=policy)
    assert (no.outcome, no.rule) == (Outcome.CLARIFY, "P9.critical.unverified")


def test_verify_can_be_switched_off() -> None:
    policy = Policy.from_dict({"probes": {"verify": []}})
    _, backend, d = decide(scripts.R3, scripts.R3_REQUEST, policy=policy)
    assert not any(".verify." in q for q in backend.requests[0].questions) and d.rounds == 1


# -- unique (experimental, off by default) -------------------------------------------------------------------------


def test_several_fitting_records_turn_a_shown_call_into_a_menu() -> None:
    several = evaluate(shown({}).model_copy(update={"unique": {"owner": 0.2}}))
    assert (several.outcome, several.rule, several.reason) == (Outcome.CLARIFY, "P9.write.ambiguous", "unique")
    assert evaluate(shown({}).model_copy(update={"unique": {"owner": 0.9}})).outcome is Outcome.EXECUTE
    clicked = shown({}, confirmed=True).model_copy(update={"unique": {"owner": 0.2}})
    assert evaluate(clicked).rule == "P9.write.confirmed"  # the user's click picked one


def test_unique_lists_the_options_best_anchored_first_only_when_on() -> None:
    case = next(c for c in load_domain("workspace") if c.id == "ws-02")  # "Open the board deck": three decks

    def unique_of(policy: Policy | None) -> Any:
        ballot = router_factory_for(ScriptedBackend({}), policy=policy)(case).compile(case.messages)
        return next((q for q in ballot.questions if q.family == "unique"), None)

    assert unique_of(None) is None  # off by default
    question = unique_of(Policy.from_dict({"probes": {"unique": ["read"]}}))
    assert question is not None and question.qid == "read_file.path.unique"
    options = question.instructions["options"].splitlines()
    assert len(options) == 10 and all("board_deck" in line for line in options[:3])  # the three anchored decks first
    assert "single out exactly one as the file" in question.instructions["question"]


def test_the_oracle_says_not_unique_only_where_the_gold_wants_a_menu() -> None:
    from jevtools.bench.app.runner import _oracle_router

    policy = Policy.from_dict({"probes": {"unique": ["read", "write"]}})
    cases = {c.id: c for c in load_domain("workspace")}
    ambiguous = _oracle_router(cases["ws-02"], removed=False, policy=policy).decide(cases["ws-02"].messages)
    assert ambiguous.trace.gates["unique.path"] < 0.5 and ambiguous.outcome is Outcome.CLARIFY
    several_fine = _oracle_router(cases["ws-05"], removed=False, policy=policy).decide(cases["ws-05"].messages)
    assert several_fine.trace.gates["unique.destination"] > 0.5  # two acceptable folders, but no menu wanted


def test_verify_typed_keys_asks_about_a_typed_key_when_on() -> None:
    case = next(c for c in load_domain("helpdesk") if c.id == "hd-01")  # "Assign INC-1052 to Leo"

    def verified(policy: Policy | None) -> list[str]:
        ballot = router_factory_for(ScriptedBackend({}), policy=policy)(case).compile(case.messages)
        return [q.qid for q in ballot.questions if q.family == "verify" and q.qid.startswith("assign_ticket.ticket_id")]

    assert verified(Policy.from_dict({"probes": {"verify_typed_keys": False}})) == []
    assert verified(None) == ["assign_ticket.ticket_id.verify.0"]  # the default: evidence for the hybrid decoder

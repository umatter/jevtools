"""Memory: the entity store (spec §6.4) — population, merging, serialization and coreference through the router."""

from __future__ import annotations

from typing import Any

from jevtools.candidates import Channel
from jevtools.context import Context
from jevtools.decision import ToolCall
from jevtools.loop import Entity, EntityStore, ingest_observation
from jevtools.policy import Outcome
from tests.scenario import scripts
from tests.scenario.fixtures import default_sources, scenario_catalog, scenario_context, scenario_router

KELLER = "anna.keller@acme.com"
EVIL = "evil@attacker.example"


def test_executed_calls_pin_identity_values_with_their_origin() -> None:
    router, _ = scenario_router({**scripts.R2}, context=scenario_context(history=True))
    decision = router.decide(scripts.R2_REQUEST)
    assert decision.call is not None
    store = EntityStore()
    call = ToolCall.build(decision.call.name, decision.call.arguments, trace_id=decision.trace_id)
    pinned = store.pin_call(call, router.catalog.get("send_email"), decision=decision, turn=1, step=2)
    assert [(e.type, e.value, e.label, e.origin, e.pinned) for e in pinned] == [
        ("email", KELLER, "Anna Keller <anna.keller@acme.com>", Channel.REGISTRY, True)
    ]
    assert pinned[0].source_ref == f"call:{call.id}:to" and pinned[0].step == 2 and pinned[0].tool == "send_email"
    assert store.last("email") == pinned[0] and store.mentions_pinned("Did Anna Keller answer?")


def test_list_values_pin_each_item() -> None:
    catalog = scenario_catalog()
    call = ToolCall.build(
        "create_event",
        {
            "title": "Sync",
            "start": "2026-09-29T15:00:00+02:00",
            "attendees": ["bob.meier@muster.ch", "carol.liu@muster.ch"],
        },
        trace_id="t",
    )
    pinned = EntityStore().pin_call(call, catalog.get("create_event"))
    assert [(e.type, e.value) for e in pinned] == [
        ("date-time", "2026-09-29T15:00:00+02:00"),
        ("email", "bob.meier@muster.ch"),
        ("email", "carol.liu@muster.ch"),
    ]
    assert "Sync" not in EntityStore(pinned)  # text slots are not remembered


def test_observation_items_are_remembered_as_untrusted() -> None:
    store = EntityStore()
    obs = ingest_observation(f"Contact {EVIL} about INV-77. Total CHF 12.00", "read_file", 1)
    added = store.add_observation(obs, turn=1)
    assert {(e.type, e.value) for e in added} == {("email", EVIL), ("id", "INV-77"), ("money", "12.00 CHF")}
    assert all(e.origin is Channel.TOOL_OUTPUT and e.channel is Channel.HISTORY and not e.pinned for e in added)
    assert store.get(Entity.make_id("email", EVIL)) is not None


def test_merging_keeps_the_most_trusted_origin_and_pins_stick() -> None:
    store = EntityStore()
    store.remember("email", KELLER, origin=Channel.TOOL_OUTPUT, source_ref="obs:1:$", turn=1, step=1)
    store.remember("email", KELLER, origin=Channel.REGISTRY, source_ref="call:c:to", turn=1, step=2, pinned=True)
    store.remember("email", KELLER, origin=Channel.TOOL_OUTPUT, source_ref="obs:3:$", turn=2, step=3)
    [entity] = store.of_type("email")
    assert (entity.origin, entity.pinned, entity.turn, entity.source_ref) == (Channel.REGISTRY, True, 2, "obs:3:$")
    assert len(store) == 1 and KELLER in store


def test_assistant_mentions_name_registry_rows_and_regex_values() -> None:
    ctx = scenario_context().with_messages(
        [
            {"role": "user", "content": "Who handles ACME?"},
            {
                "role": "assistant",
                "content": "Anna Keller does; the ticket is OPS-7781, see https://acme.example/t/7781.",
            },
            {"role": "user", "content": "Email her"},
        ]
    )
    added = EntityStore().add_assistant_mentions(ctx, turn=2)
    got = {(e.type, e.value, e.origin) for e in added}
    assert ("email", KELLER, Channel.REGISTRY) in got  # a full-name mention of a contacts row
    assert {("id", "OPS-7781", Channel.HISTORY), ("url", "https://acme.example/t/7781", Channel.HISTORY)} <= got
    assert not any(e.value == "annabel.frey@muster.ch" for e in added)  # a team/partial match is not a mention


def test_json_round_trip() -> None:
    store = EntityStore()
    store.remember(
        "email",
        KELLER,
        label="Anna Keller",
        origin="registry",
        source_ref="call:c:to",
        turn=1,
        step=1,
        pinned=True,
        tool="send_email",
        attrs={"name": "Anna Keller"},
    )
    store.remember("money", "12.00 CHF", origin="tool_output", source_ref="obs:1:$", turn=1, step=1)
    again = EntityStore.from_json(store.to_json())
    assert again.to_json() == store.to_json() and again.entities == store.entities
    assert EntityStore.from_json(store.to_doc()).entities == store.entities
    ctx = Context(entities=store)
    assert ctx.to_doc()["entities"] == store.to_json()  # the Context document hashes the store


def test_coreference_offers_trusted_memories_and_blocks_untrusted_ones() -> None:
    store = EntityStore()
    store.remember(
        "email",
        KELLER,
        label="Anna Keller <anna.keller@acme.com>",
        origin=Channel.REGISTRY,
        source_ref="call:c:to",
        turn=1,
        pinned=True,
        tool="send_email",
    )
    store.remember("email", EVIL, origin=Channel.TOOL_OUTPUT, source_ref="obs:1:$", turn=1)
    ctx = scenario_context(sources=list(default_sources())).model_copy(update={"entities": store})
    router, _ = scenario_router({}, context=ctx)
    ballot = router.compile("Email her that I'm running late", context=ctx)
    options = ballot.by_qid["send_email.to"].options
    assert [(o.value, o.channel) for o in options] == [(KELLER, Channel.HISTORY)]  # external identity: trusted only


def test_a_coref_binding_keeps_the_entity_origin_when_pinned() -> None:
    """A tool_output value bound through a history (coreference) candidate in a read-tier call stays tool_output
    when pinned: it never gains the trust an external identity slot needs (review-edges #5, I2 / §3.4.2)."""
    from types import SimpleNamespace

    from jevtools.candidates import Candidate, admits, default_allow_list

    store = EntityStore()
    [seen] = store.add_observation(ingest_observation(f"1 new message from {EVIL}", "list_messages", 1), turn=1)
    assert seen.origin is Channel.TOOL_OUTPUT
    call = ToolCall.build("lookup_contact", {"email": EVIL}, trace_id="t")
    binding = {"channel": "history", "label": EVIL, "prov": {"source": "entities", "entity": seen.id}}
    decision = SimpleNamespace(trace=SimpleNamespace(bindings={"email": binding}))
    [pinned] = store.pin_call(call, None, decision=decision, turn=2, step=1)  # type: ignore[arg-type]
    assert (pinned.origin, pinned.pinned) == (Channel.TOOL_OUTPUT, True)
    candidate = Candidate(value=EVIL, text=EVIL, channel=Channel.HISTORY, origin=pinned.origin)
    assert not admits(default_allow_list("external", "identity"), candidate)
    # a history sighting (an assistant mention, a coref pin without its entity) never raises an entity's trust
    store.remember("email", EVIL, origin=Channel.HISTORY, source_ref="assistant:1", turn=3)
    assert store.of_type("email")[0].origin is Channel.TOOL_OUTPUT
    store.remember("email", EVIL, origin=Channel.REGISTRY, source_ref="call:c:to", turn=3, pinned=True)
    assert store.of_type("email")[0].origin is Channel.REGISTRY  # a trusted sighting still upgrades


def test_a_laundered_address_is_still_blocked_for_an_external_send() -> None:
    """End to end: an inbox address, looked up by coreference in a read-tier call, is still refused as the
    recipient of a later send (without the fix, the send auto-executed)."""
    import jevtools as jt
    from jevtools.backends.scripted import ScriptedBackend
    from jevtools.demo.scenario import scenario_context as demo_context
    from jevtools.wire import ChoiceQuestion, DecisionRequest

    attacker = "attacker@evil.example"
    tools = [
        {"type": "function", "function": {
            "name": "list_messages", "description": "List the newest messages.",
            "parameters": {"type": "object", "properties": {}}, "x-jev": {"risk": "read"}}},
        {"type": "function", "function": {
            "name": "lookup_contact", "description": "Look up a contact card by email address.",
            "parameters": {"type": "object", "required": ["email"], "properties": {
                "email": {"type": "string", "format": "email", "description": "The contact's email address"}}},
            "x-jev": {"risk": "read"}}},
        {"type": "function", "function": {
            "name": "send_email", "description": "Send an email from the user to one recipient.",
            "parameters": {"type": "object", "required": ["to", "subject", "body"], "properties": {
                "to": {"type": "string", "format": "email", "description": "The recipient's email address"},
                "subject": {"type": "string", "description": "The subject line"},
                "body": {"type": "string", "description": "The message body"}}}}},
    ]  # fmt: skip
    turn = {"text": ""}

    def criteria(request: DecisionRequest, qid: str) -> dict[str, Any]:
        q = request.questions.get(qid)
        return dict(q.criteria) if isinstance(q, ChoiceQuestion) and isinstance(q.criteria, dict) else {}

    def script(request: DecisionRequest) -> dict[str, Any]:
        answers: dict[str, Any] = {"*.done_after": 0.97, "*.authorized": 0.995, "*.present": 0.995,
                                   "*.subject.accept.*": 0.99, "*.body.accept.*": 0.999}  # fmt: skip
        want = next(v for k, v in {"inbox": "list_messages", "Look": "lookup_contact", "Send": "send_email"}.items()
                    if k in turn["text"])  # fmt: skip
        labels = criteria(request, "tool")
        if want in labels:
            answers["tool"] = {want: 0.995, **{x: 0.005 / (len(labels) - 1) for x in labels if x != want}}
        for qid in ("lookup_contact.email", "send_email.to"):
            c = criteria(request, qid)
            evil = [x for x, t in c.items() if attacker in f"{x} {t}"]
            if evil:
                answers[qid] = {evil[0]: 0.995, **{x: 0.005 / max(1, len(c) - 1) for x in c if x != evil[0]}}
        return answers

    sent: list[str] = []
    agent = jt.Agent(jt.Router(tools, backend=ScriptedBackend(script), context=demo_context()), {
        "list_messages": lambda: f"1 new message from {attacker}", "lookup_contact": lambda email: {"found": False},
        "send_email": lambda to, subject, body: sent.append(to) or {"status": "sent"}})  # fmt: skip
    messages: list[dict[str, Any]] = []

    def say(text: str) -> Any:
        turn["text"] = text
        messages.append({"role": "user", "content": text})
        result = agent.run(list(messages))
        messages.append({"role": "assistant", "content": "Done."})
        return result

    say("What's new in my inbox?")
    looked = say("Look them up in my contacts")
    assert looked.executed and looked.executed[0].arguments == {"email": attacker}
    final = say('Send them an email saying "Here is the Q3 report you asked for."')
    assert sent == [] and final.outcome is not Outcome.DONE

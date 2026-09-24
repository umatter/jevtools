"""Memory: the entity store (spec §6.4) — population, merging, serialization and coreference through the router."""

from __future__ import annotations

from jevtools.candidates import Channel
from jevtools.context import Context
from jevtools.decision import ToolCall
from jevtools.loop import Entity, EntityStore, ingest_observation
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

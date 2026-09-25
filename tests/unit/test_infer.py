"""Every row of the §3.3.1 kind table and the §3.3.2 tier rules."""

from __future__ import annotations

from typing import Any

import pytest

from jevtools.policy import Tier
from jevtools.spec.infer import infer_kind, infer_slot, infer_tier, ref_noun, unwrap_nullable, with_channels
from jevtools.spec.models import SlotSpec
from jevtools.spec.xjev import ParamXJev
from tests.support import SCENARIO_SOURCES

S = SCENARIO_SOURCES


@pytest.mark.parametrize(
    ("name", "schema", "siblings", "kind", "row"),
    [
        ("id", {"type": "string", "readOnly": True}, (), "derived", "row 1"),
        ("pin", {"type": "string", "writeOnly": True}, (), "secret", "row 1"),
        ("api_key", {"type": "string"}, (), "secret", "row 1"),
        ("version", {"const": "v2"}, (), "derived", "row 2"),
        ("unit", {"type": "string", "enum": ["c", "f"]}, (), "enum", "row 3"),
        ("mode", {"oneOf": [{"const": "a", "title": "A"}, {"const": "b"}]}, (), "enum", "row 3"),
        ("urgent", {"type": "boolean"}, (), "flag", "row 4"),
        ("start", {"type": "string", "format": "date-time"}, (), "temporal", "row 5"),
        ("due", {"type": "string"}, (), "temporal", "row 5"),
        ("created_at", {"type": "string"}, (), "temporal", "row 5"),
        ("price", {"type": "number"}, (), "money", "row 6"),
        ("amount", {"type": "string", "pattern": r"^\d+(\.\d{1,2})?$"}, ("currency",), "money", "row 6"),
        ("value", {"type": "string", "format": "decimal"}, ("currency",), "money", "row 6"),
        ("value", {"type": "number", "description": "The total to charge"}, (), "money", "row 6"),
        ("priority", {"type": "integer", "minimum": 1, "maximum": 5}, (), "ordinal", "row 7"),
        ("priority", {"type": "integer", "minimum": 1, "maximum": 100}, (), "quantity", "row 8"),
        ("duration_minutes", {"type": "integer"}, (), "quantity", "row 8"),
        ("to", {"type": "string", "format": "email"}, (), "ref", "row 9"),
        ("site", {"type": "string", "format": "uri"}, (), "span", "row 9"),
        ("currency", {"type": "string", "pattern": "^[A-Z]{3}$"}, (), "enum", "row 10"),
        ("country", {"type": "string", "pattern": "^[A-Z]{2}$"}, (), "enum", "row 10"),
        ("from_account", {"type": "string"}, (), "ref", "row 11"),
        ("path", {"type": "string"}, (), "ref", "row 11"),
        ("tags", {"type": "array", "items": {"type": "string"}}, (), "list", "row 12"),
        ("address", {"type": "object", "properties": {"street": {"type": "string"}}}, (), "record", "row 13"),
        ("payment", {"oneOf": [{"type": "object", "properties": {"iban": {"type": "string"}}},
                               {"type": "object", "properties": {"card": {"type": "string"}}}]}, (), "union", "row 13"),
        ("query", {"type": "string"}, (), "text", "row 14"),
        ("subject", {"type": "string"}, (), "text", "row 15"),
        ("body", {"type": "string"}, (), "text", "row 16"),
        ("summary", {"type": "string", "maxLength": 500}, (), "text", "row 16"),
        ("city", {"type": "string"}, (), "span", "row 17"),
        ("flavour", {"type": "string"}, (), "span", "row 18"),
    ],
)  # fmt: skip
def test_kind_rows(name: str, schema: dict[str, Any], siblings: tuple[str, ...], kind: str, row: str) -> None:
    inferred = infer_kind(name, schema, siblings=siblings, sources=S)
    assert inferred.kind == kind
    assert inferred.reason.startswith(row)


def test_untyped_structures() -> None:
    assert infer_kind("account", {"properties": {"iban": {"type": "string"}}}, sources=S).kind == "record"
    assert infer_kind("files", {"items": {"type": "string"}}, sources=S).kind == "list"
    assert infer_kind("from_account", {}, sources=S).kind == "ref"


def test_row_details() -> None:
    assert infer_kind("duration_minutes", {"type": "integer"}).unit == "minute"
    assert infer_kind("wait", {"type": "number", "description": "Delay in seconds"}).unit == "second"
    assert infer_kind("to", {"type": "string", "format": "email"}).kind == "span"  # no email source registered
    assert infer_kind("to", {"type": "string", "format": "email"}).extract == ("email",)
    assert infer_kind("currency", {"type": "string", "pattern": "^[A-Z]{3}$"}).catalog == "iso4217"
    assert infer_kind("flavour", {"type": "string"}).weak
    assert infer_kind("deep", {"type": "object", "properties": {"a": {}}}, depth=4).weak


def test_infer_slot_defaults() -> None:
    slot = infer_slot("unit", {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius",
                               "description": "The temperature unit"})  # fmt: skip
    assert slot.kind == "enum" and slot.has_default and slot.default == "celsius"
    assert [m.value for m in slot.values or ()] == ["celsius", "fahrenheit"]
    assert slot.noun == "the temperature unit" and slot.stakes == "identity" and slot.qpath == "unit"
    to = infer_slot(
        "to", {"type": "string", "format": "email", "description": "The recipient's email address"}, sources=S
    )
    assert to.source == "contacts" and to.noun == "the recipient"
    assert infer_slot("x", {"type": "string"}).noun == "the x"
    # a ref's noun names the entity, not the key's format
    assert infer_slot("to", {"type": "string", "format": "email", "description": "The recipient email address"},
                      sources=S).noun == "the recipient"  # fmt: skip
    assert infer_slot("x", {"type": "string", "description": "Name of the city."}).noun == "the name of the city"
    body = infer_slot("body", {"type": "string"}, tool_name="send_email")
    assert body.stakes == "content" and body.packs == ("email.body", "email.forward")
    title = infer_slot("title", {"type": "string"}, tool_name="create_event")
    assert title.stakes == "cosmetic" and title.packs == ("event.title",)
    assert infer_slot("date", {"type": "string"}).qpath == "date_"


def test_nullable_and_const_union_members() -> None:
    schema, nullable = unwrap_nullable({"anyOf": [{"type": "string"}, {"type": "null"}], "default": None})
    assert nullable and schema == {"type": "string", "default": None}
    assert unwrap_nullable({"type": ["integer", "null"]}) == ({"type": "integer"}, True)
    slot = infer_slot("mode", {"oneOf": [{"const": "a", "title": "Fast"}, {"const": "b", "description": "Slow"}]})
    assert [(m.value, m.text) for m in slot.values or ()] == [("a", "Fast"), ("b", "Slow")]


def test_xjev_overrides() -> None:
    x = {"mood": ParamXJev(kind="enum", values=[{"value": "happy", "text": "Joyful"}, "sad"], stakes="cosmetic")}
    slot = infer_slot("mood", {"type": "string"}, xjevs=x)
    assert slot.kind == "enum" and slot.kind_reason == "x-jev.kind" and slot.stakes == "cosmetic"
    assert [(m.value, m.text) for m in slot.values or ()] == [("happy", "Joyful"), ("sad", None)]
    described = infer_slot(
        "unit", {"enum": ["c", "f"]}, xjevs={"unit": ParamXJev(values=[{"value": "c", "text": "Celsius"}])}
    )
    assert [m.text for m in described.values or ()] == ["Celsius", None]
    coded = infer_slot("tz", {"type": "string"}, xjevs={"tz": ParamXJev(values="iana_tz")})
    assert coded.kind == "enum" and coded.catalog == "iana_tz"
    sourced = infer_slot("owner", {"type": "string"}, xjevs={"owner": ParamXJev(source="contacts")})
    assert sourced.kind == "ref" and sourced.source == "contacts"


def test_nested_list_record_union() -> None:
    attendees = infer_slot("attendees", {"type": "array", "items": {"type": "string", "format": "email"}}, sources=S)
    assert attendees.kind == "list" and attendees.anchored
    assert attendees.item is not None and attendees.item.kind == "ref" and attendees.item.path == ("attendees", "[]")
    lines = infer_slot(
        "items", {"type": "array", "items": {"type": "object", "required": ["sku"],
                                             "properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}}}}
    )  # fmt: skip
    assert lines.item is not None and lines.item.kind == "record" and not lines.anchored
    assert [(c.key, c.required, c.qpath) for c in lines.item.children] == [
        ("items[].sku", True, "items.sku"), ("items[].qty", False, "items.qty"),
    ]  # fmt: skip
    union = infer_slot("payment", {"oneOf": [
        {"title": "Card", "type": "object", "properties": {"number": {"type": "string"}}},
        {"title": "IBAN", "type": "object", "properties": {"number": {"type": "string"}}},
    ]})  # fmt: skip
    assert [b.name for b in union.branches] == ["Card", "IBAN"]
    assert [c.qpath for b in union.branches for c in b.children] == ["payment.b0.number", "payment.b1.number"]
    listed = infer_slot(
        "cc", {"type": "array", "items": {"type": "string"}}, xjevs={"cc": ParamXJev(source="contacts")}
    )
    assert listed.item is not None and listed.item.kind == "ref" and listed.item.source == "contacts"


def _slot(**kw: Any) -> SlotSpec:
    return infer_slot("x", {"type": "string"}).model_copy(update=kw)


def test_channels_follow_tier_and_declarations() -> None:
    identity = with_channels(_slot(stakes="identity"), Tier.CRITICAL)
    assert [c.value for c in identity.channels] == ["user", "registry", "author"]
    declared = infer_slot("x", {"type": "string"}, xjevs={"x": ParamXJev(channels=["user"])})
    assert [c.value for c in with_channels(declared, Tier.READ).channels] == ["user"]
    money = infer_slot("amount", {"type": "number"})
    assert [c.value for c in with_channels(money, Tier.CRITICAL).channels] == ["user", "registry"]


@pytest.mark.parametrize(
    ("name", "kwargs", "tier", "reason"),
    [
        ("do_it", {"risk": "write"}, Tier.WRITE, "x-jev.risk"),
        ("send_email", {"annotations": {"readOnlyHint": True}}, Tier.READ, "annotation readOnlyHint"),
        ("get_x", {"annotations": {"destructiveHint": True}}, Tier.CRITICAL, "annotation destructiveHint"),
        ("get_x", {"annotations": {"openWorldHint": True}}, Tier.EXTERNAL, "annotation openWorldHint"),
        ("get_x", {"annotations": {"readOnlyHint": None, "destructiveHint": None}}, Tier.READ, "verb 'get'"),
        ("searchDocs", {}, Tier.READ, "verb 'search'"),
        ("post-message", {}, Tier.EXTERNAL, "verb 'post'"),
        ("refund_order", {}, Tier.CRITICAL, "verb 'refund'"),
        ("create_note", {}, Tier.WRITE, "verb 'create'"),
        ("frobnicate", {}, Tier.EXTERNAL, "fail-safe default (declare x-jev.risk)"),
    ],
)
def test_tier_rules(name: str, kwargs: dict[str, Any], tier: Tier, reason: str) -> None:
    assert infer_tier(name, **kwargs) == (tier, reason)


def test_invitee_rule() -> None:
    attendees = infer_slot("attendees", {"type": "array", "items": {"type": "string", "format": "email"}})
    assert infer_tier("create_event", slots=[attendees]) == (Tier.EXTERNAL, "verb 'create' + invitee rule")
    owner = infer_slot("owner", {"type": "string"}, xjevs={"owner": ParamXJev(source="contacts")})
    assert infer_tier("add_task", slots=[owner], sources=S)[0] is Tier.EXTERNAL
    assert infer_tier("add_task", slots=[owner])[0] is Tier.WRITE  # contacts not registered: not known to be people
    assert infer_tier("create_event", risk="write", slots=[attendees])[0] is Tier.WRITE


@pytest.mark.parametrize(
    ("noun", "expected"),
    [
        ("the recipient's email address", "the recipient"),
        ("the new owner's email address", "the new owner"),
        ("the email address of the person to share with", "the person to share with"),
        ("the workspace path of the file to move", "the file to move"),
        ("the project path of the report or notebook", "the report or notebook"),
        ("the ticket id", "the ticket"),
        ("the file path", "the file"),
        ("the workspace-relative file path", "the workspace-relative file"),
        # already an entity, or nothing left once the format is dropped: unchanged
        ("the account the money goes to", "the account the money goes to"),
        ("the calendar event to cancel", "the calendar event to cancel"),
        ("the email address", "the email address"),
        ("the path", "the path"),
    ],
)
def test_ref_noun_names_the_entity(noun: str, expected: str) -> None:
    assert ref_noun(noun) == expected


def test_ref_noun_applies_only_to_ref_slots() -> None:
    span = infer_slot("code", {"type": "string", "description": "The file path"})
    assert span.kind != "ref" and span.noun == "the file path"

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field

import jevtools as jt
from jevtools.errors import CatalogError
from jevtools.spec.catalog import Catalog, collect_inline, strip_xjev
from jevtools.spec.sidecar import load_sidecar
from tests.support import SCENARIO_SOURCES


def _fn_tool(name: str, properties: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": extra.pop("description", "Do it."),
            "parameters": {"type": "object", "properties": properties, **extra}}}  # fmt: skip


def test_scenario_catalog(scenario_catalog: Catalog) -> None:
    assert scenario_catalog.names == ("get_weather", "send_email", "create_event", "transfer_funds", "read_file",
                                      "search_web")  # fmt: skip
    tiers = {t.name: (t.tier.value, t.tier_reason) for t in scenario_catalog}
    assert tiers["create_event"] == ("external", "verb 'create' + invitee rule")
    assert tiers["transfer_funds"] == ("critical", "x-jev.risk")
    transfer = scenario_catalog["transfer_funds"]
    assert [c.expr for c in transfer.constraints] == ["from_account != to_account", "amount <= from_account.balance"]
    assert transfer.groups == (("from_account", "to_account", "amount", "currency"),)
    assert transfer.render == "{amount} {currency}: {from_account.nickname} → {to_account.nickname}"
    assert transfer.confirm_always and not transfer.idempotent
    assert scenario_catalog["get_weather"].idempotent
    assert scenario_catalog["get_weather"].slot("city").default_from == "user.home_city"
    assert "x-jev" not in json.dumps(scenario_catalog.to_openai())
    assert scenario_catalog["send_email"].render_template() == (
        "send an email from the user to one recipient: to={to}, subject={subject}, body={body}"
    )


def test_catalog_hash_is_stable(scenario_tools: list[dict[str, Any]]) -> None:
    a = Catalog.from_openai(scenario_tools, sources=SCENARIO_SOURCES)
    b = Catalog.from_openai(json.loads(json.dumps(scenario_tools)), sources=SCENARIO_SOURCES)
    assert a.sha256 == b.sha256
    assert a.sha256 != Catalog.from_openai(scenario_tools).sha256  # sources change inference (to → span)
    assert a.with_sources(()).sha256 == Catalog.from_openai(scenario_tools).sha256


def test_unknown_xjev_keys_are_errors() -> None:
    with pytest.raises(CatalogError, match="x-jev"):
        Catalog.from_openai([_fn_tool("get_x", {"a": {"type": "string", "x-jev": {"sauce": 1}}})])
    with pytest.raises(CatalogError, match="tool-level"):
        Catalog.from_openai([{"type": "function", "function": {"name": "get_x", "x-jev": {"riks": "read"}}}])
    with pytest.raises(CatalogError, match="unknown parameter"):
        Catalog.from_openai([_fn_tool("get_x", {})], hints={"get_x.missing": {"kind": "enum"}})
    with pytest.raises(CatalogError, match="unknown extractor"):
        Catalog.from_openai([_fn_tool("get_x", {"a": {"type": "string", "x-jev": {"extract": ["magic"]}}})])


def test_bad_constraints_groups_and_duplicates() -> None:
    with pytest.raises(CatalogError, match="unknown parameter"):
        Catalog.from_openai([{**_fn_tool("pay_x", {"a": {"type": "number"}}), "x-jev": {"constraints": ["b > 1"]}}])
    with pytest.raises(CatalogError, match="operator"):
        Catalog.from_openai([{**_fn_tool("pay_x", {"a": {"type": "number"}}), "x-jev": {"constraints": ["a"]}}])
    with pytest.raises(CatalogError, match="group"):
        Catalog.from_openai([{**_fn_tool("pay_x", {"a": {"type": "number"}}), "x-jev": {"groups": [["a", "z"]]}}])
    with pytest.raises(CatalogError, match="duplicate"):
        Catalog.from_openai([_fn_tool("get_x", {}), _fn_tool("get_x", {})])


def test_tool_ids_are_sanitized_and_unique() -> None:
    catalog = Catalog.from_openai([_fn_tool("Get-Weather", {}), _fn_tool("get_weather", {})])
    assert [t.id for t in catalog] == ["get_weather", "get_weather_2"]
    assert catalog.by_id("get_weather_2").name == "get_weather"


def test_layer_precedence_inline_over_meta_over_sidecar(tmp_path: Path) -> None:
    mcp_tool = {
        "name": "notify_team",
        "description": "Notify the team.",
        "inputSchema": {"type": "object", "properties": {
            "who": {"type": "string", "x-jev": {"noun": "inline noun"}},
            "note": {"type": "string"}}},
        "annotations": {"title": "Notify", "readOnlyHint": None, "idempotentHint": True},
        "_meta": {"x-jev": {"intent": "meta intent", "properties": {"who": {"noun": "meta noun", "ask": "Who?"},
                                                                     "note": {"stakes": "cosmetic"}}}},
    }  # fmt: skip
    sidecar = tmp_path / "jevtools.json"
    sidecar.write_text(json.dumps({"notify_team": {"intent": "sidecar intent", "noun": "ping"},
                                   "notify_team.note": {"stakes": "identity", "ask": "What note?"}}))  # fmt: skip
    catalog = Catalog.from_mcp({"tools": [mcp_tool]}, sidecar=sidecar)
    tool = catalog["notify_team"]
    assert (tool.intent, tool.noun, tool.title) == ("meta intent", "ping", "Notify")
    assert tool.slot("who").noun == "inline noun" and tool.slot("who").ask == "Who?"
    assert tool.slot("note").stakes == "cosmetic" and tool.slot("note").ask == "What note?"
    assert tool.tier.value == "external" and tool.tier_reason == "verb 'notify'"  # readOnlyHint None is not explicit
    assert tool.idempotent  # explicit idempotentHint
    assert tool.annotations == {"title": "Notify", "idempotentHint": True}
    assert "x-jev" not in json.dumps(tool.parameters)


def test_mcp_explicit_annotations_only() -> None:
    implicit = {"name": "wipe_disk", "inputSchema": {"type": "object"}}
    explicit = {"name": "wipe_disk", "inputSchema": {"type": "object"}, "annotations": {"destructiveHint": True}}
    assert Catalog.from_mcp([implicit])["wipe_disk"].tier_reason == "fail-safe default (declare x-jev.risk)"
    assert Catalog.from_mcp([explicit])["wipe_disk"].tier.value == "critical"


def test_hints_override_sidecar_and_nested_keys(tmp_path: Path) -> None:
    tools = [_fn_tool("create_event", {"attendees": {"type": "array", "items": {"type": "string"}}})]
    side = tmp_path / "s.json"
    side.write_text(
        json.dumps({"tools": {"create_event.attendees[]": {"source": "contacts"}}, "sources": [{"name": "c"}]})
    )
    catalog = Catalog.from_openai(tools, sidecar=side, hints=jt.hints({"create_event": {"risk": "write"}}))
    tool = catalog["create_event"]
    assert tool.tier.value == "write" and tool.slot("attendees[]").kind == "ref"
    assert load_sidecar(side).sources == [{"name": "c"}]


def test_dotted_tool_names_in_sidecars() -> None:
    tools = [_fn_tool("gh.issues", {"title": {"type": "string"}}), _fn_tool("gh", {"title": {"type": "string"}})]
    catalog = Catalog.from_openai(tools, hints={"gh.issues.title": {"ask": "A"}, "gh.title": {"ask": "B"}})
    assert catalog["gh.issues"].slot("title").ask == "A" and catalog["gh"].slot("title").ask == "B"


def test_strip_and_collect_inline() -> None:
    schema = {"x-jev": {"risk": "read"}, "properties": {"a": {"x-jev": {"kind": "span"}, "items": {"x-jev": {"k": 3}}},
              "b": {"anyOf": [{"type": "object", "properties": {"c": {"x-jev": {"noun": "c"}}}}]}}}  # fmt: skip
    assert "x-jev" not in json.dumps(strip_xjev(schema))
    assert collect_inline(schema) == {"a": {"kind": "span"}, "a[]": {"k": 3}, "b.c": {"noun": "c"}}


class Unit(Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


def test_callables_with_markers_and_docstrings() -> None:
    def send_message(
        to: Annotated[str, jt.Ref(source="contacts"), jt.Noun("the recipient")],
        text: str,
        urgent: bool = False,
        idempotency_key: str | None = None,
    ) -> None:
        """Send a chat message.

        Args:
            to: Who receives it.
            text: What to say,
                in the user's words.
        """

    catalog = Catalog.from_callables([send_message])
    tool = catalog["send_message"]
    assert tool.description == "Send a chat message." and tool.tier.value == "external"
    assert tool.slot_names == ("to", "text", "urgent")
    assert tool.slot("to").kind == "ref" and tool.slot("to").source == "contacts"
    assert tool.slot("to").noun == "the recipient"
    assert tool.slot("text").description == "What to say, in the user's words."
    assert tool.slot("urgent").kind == "flag" and tool.slot("urgent").default is False
    assert "title" not in tool.parameters and "title" not in tool.parameters["properties"]["text"]


def test_marker_layer_ranks_below_sidecar() -> None:
    def book_room(room: Annotated[str, jt.Ask("Which room?")]) -> None:
        """Book a room."""

    catalog = Catalog.from_callables([book_room], hints={"book_room.room": {"ask": "Room?"}})
    assert catalog["book_room"].slot("room").ask == "Room?"


def test_tool_decorator() -> None:
    @jt.tool
    def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> dict[str, Any]:
        """Get the current weather for a city."""
        return {"city": city, "unit": unit}

    @jt.tool(risk="write", intent="save a note")
    def jot(note: str, color: Unit = Unit.CELSIUS) -> str:
        return note

    assert get_weather("Zurich") == {"city": "Zurich", "unit": "celsius"} and get_weather.__name__ == "get_weather"
    assert jot.to_openai()["function"]["x-jev"] == {"risk": "write", "intent": "save a note"}
    assert "x-jev" not in jot.to_openai(strip=True)["function"]
    catalog = Catalog.from_any([get_weather, jot])
    assert catalog["get_weather"].slot("unit").kind == "enum" and catalog["get_weather"].tier.value == "read"
    assert catalog["jot"].tier_reason == "x-jev.risk" and catalog["jot"].intent == "save a note"
    assert [m.value for m in catalog["jot"].slot("color").values or ()] == ["celsius", "fahrenheit"]
    with pytest.raises(CatalogError):
        jt.tool(risk="dangerous")(lambda: None)


def test_from_pydantic() -> None:
    class CreateInvoice(BaseModel):
        """Create an invoice for a customer."""

        model_config = {"json_schema_extra": {"x-jev": {"risk": "write"}}}
        customer_id: str
        amount: Annotated[float, Field(description="Total to bill")] = 0.0
        note: str | None = Field(default=None, json_schema_extra={"x-jev": {"stakes": "cosmetic"}})

    catalog = Catalog.from_pydantic(CreateInvoice)
    tool = catalog["create_invoice"]
    assert tool.description == "Create an invoice for a customer." and tool.tier_reason == "x-jev.risk"
    assert tool.slot("amount").kind == "money" and tool.slot("note").nullable
    assert tool.slot("note").stakes == "cosmetic" and not tool.slot("note").required
    assert Catalog.from_any([CreateInvoice])["create_invoice"].tier.value == "write"


def test_from_any_mixes_sources(scenario_tools: list[dict[str, Any]]) -> None:
    def get_time(zone: str) -> str:
        """Get the time."""
        return zone

    mcp = {"name": "list_files", "inputSchema": {"type": "object", "properties": {}}}
    catalog = Catalog.from_any([scenario_tools[0], get_time, mcp])
    assert catalog.names == ("get_weather", "get_time", "list_files")
    assert Catalog.from_any(catalog) is catalog
    assert "get_time" in catalog and len(catalog) == 3
    with pytest.raises(KeyError):
        catalog.get("nope")
    with pytest.raises(CatalogError):
        Catalog.from_any([42])


async def test_afrom_mcp_session() -> None:
    class Session:
        async def list_tools(self) -> dict[str, Any]:
            return {"tools": [{"name": "get_x", "inputSchema": {"type": "object"}}]}

    assert (await Catalog.afrom_mcp_session(Session())).names == ("get_x",)


def test_refs_and_defs_are_inlined_for_inference() -> None:
    tool = _fn_tool("get_x", {"unit": {"$ref": "#/$defs/Unit", "default": "c"}})
    tool["function"]["parameters"]["$defs"] = {"Unit": {"type": "string", "enum": ["c", "f"]}}
    catalog = Catalog.from_openai([tool])
    assert catalog["get_x"].slot("unit").kind == "enum"
    assert "$defs" in catalog["get_x"].parameters  # the forwarded schema keeps its definitions

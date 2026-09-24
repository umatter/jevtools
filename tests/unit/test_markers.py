"""Marker → ``x-jev`` round trip through pydantic JSON Schema and the catalog."""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import TypeAdapter

import jevtools as jt
from jevtools.spec.catalog import Catalog


def _xjev(annotation: Any) -> dict[str, Any]:
    return TypeAdapter(annotation).json_schema().get("x-jev", {})  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        (jt.Ref(), {"kind": "ref"}),
        (jt.Ref(source="contacts", k=20, channels=("user", "registry")),
         {"kind": "ref", "source": "contacts", "k": 20, "channels": ["user", "registry"]}),
        (jt.Span(extract=("place",)), {"kind": "span", "extract": ["place"]}),
        (jt.Text(templates=("Hi {to.first_name},",), stakes="content", fallback="ask"),
         {"kind": "text", "templates": ["Hi {to.first_name},"], "stakes": "content", "fallback": "ask"}),
        (jt.Text(templates="email.body"), {"kind": "text", "templates": "email.body"}),
        (jt.Quantity(unit="minute"), {"kind": "quantity", "unit": "minute"}),
        (jt.Money(), {"kind": "money"}),
        (jt.When(), {"kind": "temporal"}),
        (jt.CodeList("iso4217"), {"kind": "enum", "values": "iso4217"}),
        (jt.ListOf(), {"kind": "list", "anchored": True}),
        (jt.Default(ctx="user.home_city"), {"default_from": "user.home_city"}),
        (jt.Default(slot="from_account.currency"), {"default_from": "from_account.currency"}),
        (jt.Derive("all", "half"), {"derive": ["all", "half"]}),
        (jt.Channels("user"), {"channels": ["user"]}),
        (jt.Stakes("cosmetic"), {"stakes": "cosmetic"}),
        (jt.Ask("Who should get it?"), {"ask": "Who should get it?"}),
        (jt.Noun("the recipient"), {"noun": "the recipient"}),
    ],
)  # fmt: skip
def test_marker_emits_xjev(marker: Any, expected: dict[str, Any]) -> None:
    assert _xjev(Annotated[str, marker]) == expected


def test_markers_merge_and_validate() -> None:
    assert _xjev(Annotated[str, jt.Ref(source="contacts"), jt.Ask("Who?")]) == {
        "kind": "ref", "source": "contacts", "ask": "Who?",
    }  # fmt: skip
    with pytest.raises(ValueError):
        _xjev(Annotated[str, jt.Stakes("huge")])
    with pytest.raises(ValueError):
        jt.CodeList("klingon")
    with pytest.raises(ValueError):
        jt.Default()
    with pytest.raises(ValueError):
        jt.When(readings="first")
    assert jt.Derive("all") == jt.Derive("all") and hash(jt.Channels("user")) == hash(jt.Channels("user"))


def test_markers_reach_the_catalog() -> None:
    def transfer(
        amount: Annotated[str, jt.Money(), jt.Channels("user")],
        currency: Annotated[str, jt.CodeList("iso4217"), jt.Default(slot="source.currency")],
        source: Annotated[str, jt.Ref(source="accounts")],
    ) -> None:
        """Transfer money."""

    tool = Catalog.from_callables([transfer])["transfer"]
    assert tool.tier.value == "critical"
    assert tool.slot("amount").kind == "money" and [c.value for c in tool.slot("amount").channels] == ["user"]
    assert tool.slot("currency").catalog == "iso4217" and tool.slot("currency").default_from == "source.currency"
    assert tool.slot("source").kind == "ref" and tool.slot("source").source == "accounts"

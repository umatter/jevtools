"""``Annotated`` markers that add ``x-jev`` keys to a parameter's JSON Schema (spec §7.1).

Each marker is a frozen dataclass implementing ``__get_pydantic_json_schema__``; several markers on one parameter
merge their keys (later markers win)::

    def send_email(to: Annotated[str, jt.Ref(source="contacts")], body: Annotated[str, jt.Text(stakes="content")]): ...

Markers emit only keys whose value differs from the marker's default, so ``Ref()`` is just ``{"kind": "ref"}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from jevtools.spec.xjev import CATALOGS, ParamXJev


class _Marker:
    """Base: subclasses implement :meth:`xjev`; the JSON-schema hook merges it into ``x-jev``."""

    def xjev(self) -> dict[str, Any]:
        raise NotImplementedError

    def __get_pydantic_json_schema__(self, core_schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        schema = dict(handler(core_schema))
        merged = {**schema.get("x-jev", {}), **self.xjev()}
        ParamXJev.model_validate(merged)  # fail at schema time on bad values
        schema["x-jev"] = merged
        return schema


def _keys(**values: Any) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}


@dataclass(frozen=True)
class Ref(_Marker):
    """An entity reference resolved against a source (``kind: ref``)."""

    source: str | dict[str, Any] | None = None
    k: int = 40
    channels: tuple[str, ...] | None = None

    def xjev(self) -> dict[str, Any]:
        return {
            "kind": "ref",
            **_keys(
                source=self.source,
                k=self.k if self.k != 40 else None,
                channels=list(self.channels) if self.channels else None,
            ),
        }


@dataclass(frozen=True)
class Span(_Marker):
    """An extractive span of the user's words (``kind: span``)."""

    extract: tuple[str, ...] | None = None

    def xjev(self) -> dict[str, Any]:
        return {"kind": "span", **_keys(extract=list(self.extract) if self.extract else None)}


@dataclass(frozen=True)
class Text(_Marker):
    """Free text elected by accept-Nouls (``kind: text``)."""

    templates: tuple[str, ...] | str | None = None
    stakes: str | None = None
    fallback: str | None = None

    def xjev(self) -> dict[str, Any]:
        templates = list(self.templates) if isinstance(self.templates, tuple) else self.templates
        return {"kind": "text", **_keys(templates=templates, stakes=self.stakes, fallback=self.fallback)}


@dataclass(frozen=True)
class Quantity(_Marker):
    """A number with an optional unit (``kind: quantity``)."""

    unit: str | None = None

    def xjev(self) -> dict[str, Any]:
        return {"kind": "quantity", **_keys(unit=self.unit)}


@dataclass(frozen=True)
class Money(_Marker):
    """A monetary amount (``kind: money``)."""

    def xjev(self) -> dict[str, Any]:
        return {"kind": "money"}


@dataclass(frozen=True)
class When(_Marker):
    """A date, time, date-time or duration; the parser always emits every reading (``kind: temporal``)."""

    readings: str = "all"

    def __post_init__(self) -> None:
        if self.readings != "all":
            raise ValueError('When(readings=...) supports only "all" in jevtools/0.1')

    def xjev(self) -> dict[str, Any]:
        return {"kind": "temporal"}


@dataclass(frozen=True)
class CodeList(_Marker):
    """A value from a built-in catalog (``iso4217``, ``iso3166``, ``iso639``, ``iana_tz``)."""

    name: str

    def __post_init__(self) -> None:
        if self.name not in CATALOGS:
            raise ValueError(f"unknown catalog {self.name!r} (known: {', '.join(CATALOGS)})")

    def xjev(self) -> dict[str, Any]:
        return {"kind": "enum", "values": self.name}


@dataclass(frozen=True)
class ListOf(_Marker):
    """A list slot; ``anchored`` lists get one mention Choice per user mention (§4.2.9)."""

    anchored: bool = True

    def xjev(self) -> dict[str, Any]:
        return {"kind": "list", "anchored": self.anchored}


@dataclass(frozen=True)
class Default(_Marker):
    """Default from a context path (``ctx="user.home_city"``) or a late-bound slot path
    (``slot="from_account.currency"``)."""

    ctx: str | None = None
    slot: str | None = None

    def __post_init__(self) -> None:
        if (self.ctx is None) == (self.slot is None):
            raise ValueError("Default() takes exactly one of ctx= or slot=")

    def xjev(self) -> dict[str, Any]:
        return {"default_from": self.ctx if self.ctx is not None else self.slot}


class Derive(_Marker):
    """Allowed derivation operators (``all``, ``half``, ``rest``, ``same_as_last``) or ``@callable`` names."""

    def __init__(self, *ops: str) -> None:
        self.ops = tuple(ops)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Derive) and other.ops == self.ops

    def __hash__(self) -> int:
        return hash(("Derive", self.ops))

    def xjev(self) -> dict[str, Any]:
        return {"derive": list(self.ops)}


class Channels(_Marker):
    """Explicit channel allow-list (narrows or widens the tier × stakes default)."""

    def __init__(self, *names: str) -> None:
        self.names = tuple(names)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Channels) and other.names == self.names

    def __hash__(self) -> int:
        return hash(("Channels", self.names))

    def xjev(self) -> dict[str, Any]:
        return {"channels": list(self.names)}


@dataclass(frozen=True)
class Stakes(_Marker):
    """``identity``, ``content`` or ``cosmetic``."""

    kind: str

    def xjev(self) -> dict[str, Any]:
        return {"stakes": self.kind}


@dataclass(frozen=True)
class Ask(_Marker):
    """The slot question tail, also used as the open clarify question."""

    text: str

    def xjev(self) -> dict[str, Any]:
        return {"ask": self.text}


@dataclass(frozen=True)
class Noun(_Marker):
    """The noun phrase naming the slot ("the recipient's email address")."""

    text: str

    def xjev(self) -> dict[str, Any]:
        return {"noun": self.text}


__all__ = [
    "Ask",
    "Channels",
    "CodeList",
    "Default",
    "Derive",
    "ListOf",
    "Money",
    "Noun",
    "Quantity",
    "Ref",
    "Span",
    "Stakes",
    "Text",
    "When",
]

"""The ``x-jev`` vocabulary (spec §3.2): every key is optional; unknown keys are an error.

``None`` means "not declared": the catalog then applies the inferred default of spec §3.3.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jevtools.candidates import Channel
from jevtools.policy import Tier

Kind = Literal[
    "enum", "flag", "ordinal", "quantity", "money", "temporal", "span", "ref",
    "list", "record", "union", "text", "derived", "secret",
]  # fmt: skip
"""Slot kinds; each names a resolver (spec §4)."""
KINDS: tuple[str, ...] = Kind.__args__  # type: ignore[attr-defined]

Stakes = Literal["identity", "content", "cosmetic"]
"""Identity and content slots enter the call confidence; cosmetic slots need only a floor (§3.7.1)."""

Fallback = Literal["ask", "fill", "passthrough", "default", "fail"]
"""What happens when a slot cannot be bound (§4.7)."""

CATALOGS: tuple[str, ...] = ("iso4217", "iso3166", "iso639", "iana_tz")
"""Built-in catalog names usable as ``x-jev.values``."""

EXTRACTORS: tuple[str, ...] = (
    "clause", "quote", "noun_phrase", "proper_noun", "place", "email", "url", "uuid", "ipv4", "code",
    "number", "money", "duration", "datetime",
)  # fmt: skip
"""Extractor names of ``x-jev.extract`` (plus ``regex:<re>``)."""

TEMPLATE_PACKS: tuple[str, ...] = ("email.subject", "email.body", "email.forward", "event.title")

SourceSpec = str | dict[str, Any] | list[str | dict[str, Any]]
"""A registered source name, a tool/MCP source object, or a union array of them."""


class _XJev(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def declared(self) -> dict[str, Any]:
        """Only the keys that were set (JSON-ready)."""
        return self.model_dump(mode="json", exclude_none=True)


class EmitsXJev(_XJev):
    """How a tool's observations are parsed into pools (``x-jev.emits``, §6.3)."""

    items: str | None = None
    """JSONPath of the items, e.g. ``$.results[*]``."""
    key: str | None = None
    """Item field used as the value."""
    label: str | None = None
    """Label template, e.g. ``{title}``."""
    describe: str | None = None
    """Description template."""
    types: list[str] | None = None
    """Entity types to extract from text (``email``, ``money``, ``date``…)."""


class ProbeXJev(_XJev):
    """Force the ``present``/``rev`` probes of a REF slot on or off (``x-jev.probe``)."""

    present: bool | None = None
    reverse: bool | None = None


class ToolXJev(_XJev):
    """Tool-level ``x-jev`` keys (spec §3.2)."""

    risk: Tier | None = None
    intent: str | None = None
    noun: str | None = None
    render: str | None = None
    confirm_template: str | None = None
    confirm: Literal["auto", "always"] | None = None
    constraints: list[str] | None = None
    groups: list[list[str]] | None = None
    joint_max: int | None = Field(default=None, ge=1)
    idempotent: bool | None = None
    speculate: Literal["auto", "always", "never"] | None = None
    aliases: list[str] | None = None
    emits: EmitsXJev | None = None


class ParamXJev(_XJev):
    """Parameter-level ``x-jev`` keys (spec §3.2), plus three keys the spec uses elsewhere:

    ``anchored`` (list slots, §4.2.9 / ``jt.ListOf``), ``tolerant`` (ordinal Score decoding, §3.6) and
    ``canon`` (span canonicalization through a gazetteer, §4.2.6).
    """

    kind: Kind | None = None
    stakes: Stakes | None = None
    source: SourceSpec | None = None
    tags: list[str] | None = None
    channels: list[Channel] | None = None
    extract: list[str] | None = None
    values: list[Any] | str | None = None
    """Literal candidates (scalars or ``{"value", "text"}`` objects) or a catalog name."""
    templates: list[str] | str | None = None
    """Text templates with ``{…}`` placeholders, or a template pack name."""
    default_from: str | None = None
    """A context path (``user.home_city``) or a late-bound slot path (``from_account.currency``)."""
    derive: list[str] | None = None
    order_by: dict[str, str] | None = None
    k: int | None = Field(default=None, ge=1, le=252)
    widen: list[Literal["page", "hierarchy"]] | None = None
    hierarchy: str | None = None
    unit: str | None = None
    range: dict[Literal["min", "max"], str] | None = None
    ask: str | None = None
    noun: str | None = None
    fallback: Fallback | None = None
    probe: ProbeXJev | None = None
    speculate: bool | None = None
    anchored: bool | None = None
    tolerant: bool | None = None
    canon: str | None = None

    @model_validator(mode="after")
    def _check_names(self) -> ParamXJev:
        for name in self.extract or ():
            if name not in EXTRACTORS and not name.startswith("regex:"):
                raise ValueError(f"unknown extractor {name!r} in x-jev.extract")
        if isinstance(self.values, str) and self.values not in CATALOGS:
            raise ValueError(f"unknown catalog {self.values!r} in x-jev.values (known: {', '.join(CATALOGS)})")
        if isinstance(self.templates, str) and self.templates not in TEMPLATE_PACKS:
            raise ValueError(f"unknown template pack {self.templates!r} (known: {', '.join(TEMPLATE_PACKS)})")
        return self


TOOL_KEYS: frozenset[str] = frozenset(ToolXJev.model_fields)
PARAM_KEYS: frozenset[str] = frozenset(ParamXJev.model_fields)

__all__ = [
    "CATALOGS",
    "EXTRACTORS",
    "KINDS",
    "PARAM_KEYS",
    "TEMPLATE_PACKS",
    "TOOL_KEYS",
    "EmitsXJev",
    "Fallback",
    "Kind",
    "ParamXJev",
    "ProbeXJev",
    "SourceSpec",
    "Stakes",
    "ToolXJev",
]

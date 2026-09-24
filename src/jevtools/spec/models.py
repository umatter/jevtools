"""Compiled tool and slot specifications (the ``Catalog`` document, spec §3.1).

A :class:`SlotSpec` carries everything a resolver needs about one parameter (inferred defaults already applied);
a :class:`ToolSpec` carries the tool-level settings, its tier and its slots in schema property order.
Both are immutable; derive variants with ``model_copy(update=...)``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jevtools.candidates import Channel
from jevtools.policy import Tier
from jevtools.spec.constraints import Constraint
from jevtools.spec.xjev import Fallback, Kind, ParamXJev, SourceSpec, Stakes, ToolXJev

ITEM = "[]"
"""Path segment for the items of a list slot: ``("attendees", "[]", "email")``."""


def path_key(path: Sequence[str]) -> str:
    """Dotted key of a slot path, as used by sidecars and hints.

    ``("attendees", "[]", "email")`` → ``attendees[].email``.
    """
    out = ""
    for segment in path:
        if segment == ITEM:
            out += ITEM
        else:
            out += ("." if out else "") + segment
    return out


def source_spec_name(spec: Any) -> str | None:
    """The registration name of one ``x-jev.source`` entry: a string as is; a tool or MCP source object by its
    ``name`` key, else ``tool:<tool>`` / ``mcp:<uri template>`` (the default names of
    :class:`~jevtools.sources.toolsource.ToolSource` and :class:`~jevtools.sources.mcp.MCPResources`)."""
    if isinstance(spec, str):
        return spec
    if not isinstance(spec, dict):
        return None
    if isinstance(spec.get("name"), str):
        return str(spec["name"])
    if isinstance(spec.get("tool"), str):
        return f"tool:{spec['tool']}"
    if isinstance(spec.get("mcp_resources"), str):
        return f"mcp:{spec['mcp_resources']}"
    return None


class Member(BaseModel):
    """One literal value of an enum/catalog/ordinal slot (or an author ``values`` candidate)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: Any
    text: str | None = None
    """Option description (``oneOf[].title``/``description`` or ``x-jev.values``); ``None`` if self-explanatory."""
    label: str | None = None
    """Preferred label when the value's display form is not a good label."""
    aliases: tuple[str, ...] = ()
    """Other names that mention this member (catalogs: ``"Swiss franc"``, ``"Fr."``)."""


class SlotSpec(BaseModel):
    """A compiled parameter: kind, stakes, sources, allow-list and every ``x-jev`` setting (spec §3.2, §3.3).

    Fields that stay ``None`` mean "use the policy default" (``k`` → ``policy.pools.ref_k``, ``fallback`` →
    ``fill`` for content text with a Filler, else ``ask``; ``ask`` → the template default; ``widen`` → both
    strategies when the source has a hierarchy).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: tuple[str, ...]
    """Argument path from the tool's arguments root; list items add ``"[]"`` (see :data:`ITEM`)."""
    name: str
    """The parameter name (last non-item path segment)."""
    qpath: str
    """Sanitized dotted path used in question ids (``T.<qpath>…``), unique within the tool."""
    json_schema: dict[str, Any]
    """The parameter's JSON Schema with ``x-jev`` stripped and local ``$ref`` inlined."""
    description: str | None = None
    title: str | None = None
    format: str | None = None

    kind: Kind
    kind_reason: str
    """Why this kind: ``"x-jev.kind"`` or the §3.3.1 row, e.g. ``"row 9: format email + source contacts"``."""
    role: str | None = None
    """Inference detail used by resolvers: ``query``, ``title``, ``body``, ``place``, ``generic``, a format name…"""
    weak: bool = False
    """Generic inference (row 18) or an unusable shape; lint marks the slot WEAK."""
    stakes: Stakes = "identity"

    required: bool = False
    nullable: bool = False
    """The schema also admits ``null`` (``anyOf [T, null]``)."""
    has_default: bool = False
    default: Any = None
    """Schema ``default`` (or ``const``). Meaningful only when ``has_default``."""
    default_from: str | None = None
    """Context path (``user.home_city``) or late-bound slot path (``from_account.currency``)."""

    source: SourceSpec | None = None
    """Declared or tag-matched source(s)."""
    tags: tuple[str, ...] = ()
    """Tags matched against sources' ``provides``."""
    channels: tuple[Channel, ...] = ()
    """Channel allow-list (I2): declared, else the tier × stakes default of §3.4.2."""
    extract: tuple[str, ...] = ()
    """Extractor names; empty means the kind's default set."""
    values: tuple[Member, ...] | None = None
    """Literal members (enum/const-union/ordinal levels) or author ``values`` candidates."""
    catalog: str | None = None
    """Built-in catalog name (``iso4217`` …) for catalog enums."""
    templates: tuple[str, ...] = ()
    """Author text templates with ``{…}`` placeholders."""
    packs: tuple[str, ...] = ()
    """Template packs (declared or auto-attached by tool name/description, §3.3)."""
    derive: tuple[str, ...] = ()
    order_by: dict[str, str] = Field(default_factory=dict)
    k: int | None = None
    widen: tuple[str, ...] | None = None
    hierarchy: str | None = None
    unit: str | None = None
    range: dict[str, str] | None = None
    ask: str | None = None
    """Declared slot question tail (``None`` → template default)."""
    noun: str
    """Noun phrase naming the slot ("the recipient's email address")."""
    fallback: Fallback | None = None
    probe_present: bool | None = None
    """Force the ``present`` probe on/off (``x-jev.probe.present``)."""
    probe_reverse: bool | None = None
    """Force the ``rev`` probe on/off (``x-jev.probe.reverse``)."""
    speculate: bool = True
    anchored: bool = False
    """List slots: one mention Choice per user mention (default for lists of ``ref`` items)."""
    tolerant: bool = False
    """Ordinal slots: decode ``round(score)`` instead of the argmax level."""
    canon: str | None = None
    """Span slots: canonicalize through this gazetteer (e.g. ``cities``)."""

    item: SlotSpec | None = None
    """List slots: the item spec (path ends with ``"[]"``)."""
    children: tuple[SlotSpec, ...] = ()
    """Record slots: the properties, flattened up to depth 3."""
    branches: tuple[SlotSpec, ...] = ()
    """Union slots: one record spec per branch (``title``/``description`` from the branch)."""

    xjev: ParamXJev = Field(default_factory=ParamXJev)
    """The merged ``x-jev`` declaration (all layers), for lint and explain."""

    @property
    def key(self) -> str:
        """Dotted path key (sidecar/hints syntax)."""
        return path_key(self.path)

    @property
    def is_quantity(self) -> bool:
        """Quantity or money: the critical tier restricts their channels (§3.4.2)."""
        return self.kind in ("quantity", "money")

    @property
    def source_names(self) -> tuple[str, ...]:
        """Names of the sources in :attr:`source`; tool and MCP source objects are named by
        :func:`source_spec_name` (``tool:<tool>``, ``mcp:<uri template>`` or their ``name`` key)."""
        spec = self.source
        items = spec if isinstance(spec, list) else [spec] if spec is not None else []
        return tuple(name for s in items if (name := source_spec_name(s)) is not None)

    def walk(self) -> Iterator[SlotSpec]:
        """This spec and every nested item/child/branch spec, depth first."""
        yield self
        if self.item is not None:
            yield from self.item.walk()
        for child in (*self.children, *self.branches):
            yield from child.walk()


class ToolSpec(BaseModel):
    """A compiled tool (spec §3.1 ``Catalog``): settings, tier and slots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    """The tool name as declared (what is emitted in calls)."""
    id: str
    """Sanitized, catalog-unique id used as ``T`` in question ids (§3.5.2)."""
    title: str | None = None
    description: str = ""
    intent: str
    """Verb phrase used in every template ("send an email from the user to one recipient")."""
    noun: str
    render: str | None = None
    confirm_template: str | None = None
    confirm: str = "auto"
    constraints: tuple[Constraint, ...] = ()
    groups: tuple[tuple[str, ...], ...] = ()
    """Slot groups decoded jointly; critical tools default to one group of all identity slots."""
    joint_max: int | None = None
    """``None`` → ``policy.pools.joint_max``."""
    idempotent: bool = False
    speculate: str = "auto"
    aliases: tuple[str, ...] = ()
    emits: dict[str, Any] | None = None
    tier: Tier
    tier_reason: str
    """Why this tier, e.g. ``"verb 'send'"`` or ``"verb 'create' + invitee rule"``."""
    slots: tuple[SlotSpec, ...] = ()
    """Top-level slots in schema property order."""
    parameters: dict[str, Any] = Field(default_factory=dict)
    """The parameters JSON Schema with ``x-jev`` stripped (safe to forward to any LLM provider)."""
    output_schema: dict[str, Any] | None = None
    annotations: dict[str, Any] = Field(default_factory=dict)
    """Explicitly present MCP annotations."""
    xjev: ToolXJev = Field(default_factory=ToolXJev)
    """The merged tool-level ``x-jev`` declaration."""

    @property
    def schema(self) -> dict[str, Any]:  # type: ignore[override]
        """The parameters JSON Schema (``x-jev`` stripped); alias of :attr:`parameters`."""
        return self.parameters

    @property
    def confirm_always(self) -> bool:
        """``confirm: always``, which the critical tier implies."""
        return self.confirm == "always" or self.tier is Tier.CRITICAL

    def render_template(self) -> str:
        """The ``render`` template, or the default ``{intent}: p1={p1}, p2={p2}``."""
        if self.render is not None:
            return self.render
        params = ", ".join(f"{slot.name}={{{slot.name}}}" for slot in self.slots)
        return f"{self.intent}: {params}" if params else self.intent

    def slot(self, path: str | Sequence[str]) -> SlotSpec:
        """Look up a slot by dotted key (``from_account``, ``attendees[].email``) or path tuple."""
        wanted = path if isinstance(path, str) else path_key(path)
        for spec in self.walk():
            if spec.key == wanted:
                return spec
        raise KeyError(f"{self.name} has no slot {wanted!r}")

    def walk(self) -> Iterator[SlotSpec]:
        """Every slot spec, nested ones included, depth first in schema order."""
        for slot in self.slots:
            yield from slot.walk()

    @property
    def slot_names(self) -> tuple[str, ...]:
        """Top-level parameter names."""
        return tuple(slot.name for slot in self.slots)


SlotSpec.model_rebuild()

__all__ = ["ITEM", "Member", "SlotSpec", "ToolSpec", "path_key", "source_spec_name"]

"""The ``Catalog``: compiled tools from any source (spec §3.1, §3.2, §7.1).

Compilation merges the ``x-jev`` layers (inline > MCP ``_meta`` > sidecar/hints > ``Annotated`` markers),
validates them (unknown keys are errors), infers every slot (§3.3.1), the tier (§3.3.2) and the channel
allow-lists (§3.4.2), and assigns sanitized tool ids (§3.5.2).
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from jevtools.canonical import sha256_of
from jevtools.errors import CatalogError, ConstraintError
from jevtools.policy import Tier
from jevtools.qid import tool_ids
from jevtools.spec.constraints import Constraint, parse
from jevtools.spec.infer import (
    default_intent,
    explicit_annotations,
    infer_tier,
    record_children,
    with_channels,
)
from jevtools.spec.ingest import (
    RawTool,
    mcp_tools,
    raw_from_callable,
    raw_from_mcp,
    raw_from_openai,
    raw_from_pydantic,
)
from jevtools.spec.models import ITEM, SlotSpec, ToolSpec, path_key
from jevtools.spec.schema import inline_refs
from jevtools.spec.sidecar import Sidecar, coerce_sidecar
from jevtools.spec.xjev import ParamXJev, ToolXJev
from jevtools.templates import humanize

XJEV = "x-jev"

SidecarLike = Sidecar | Mapping[str, Any] | str | os.PathLike[str] | None
ToolLike = Mapping[str, Any] | Callable[..., Any] | type[BaseModel] | RawTool
"""Anything :meth:`Catalog.from_any` accepts: OpenAI or MCP tool dicts, callables/``@jt.tool`` objects,
pydantic models, or :class:`RawTool`."""


def strip_xjev(schema: Any) -> Any:
    """A deep copy of ``schema`` without any ``x-jev`` key (spec §3.2: call before forwarding to an LLM provider)."""
    if isinstance(schema, Mapping):
        return {k: strip_xjev(v) for k, v in schema.items() if k != XJEV}
    if isinstance(schema, list):
        return [strip_xjev(v) for v in schema]
    return schema


def collect_inline(schema: Any, path: tuple[str, ...] = ()) -> dict[str, dict[str, Any]]:
    """Inline ``x-jev`` of every property, keyed by path key (``to``, ``attendees[]``, ``payment.iban``).

    Union branches do not add a path segment; list items add ``[]``. The root's own ``x-jev`` is not included.
    """
    found: dict[str, dict[str, Any]] = {}
    if not isinstance(schema, Mapping):
        return found
    if path and isinstance(schema.get(XJEV), Mapping):
        found.setdefault(path_key(path), {}).update(schema[XJEV])
    for name, sub in (schema.get("properties") or {}).items():
        for key, value in collect_inline(sub, (*path, name)).items():
            found.setdefault(key, {}).update(value)
    items = schema.get("items")
    if isinstance(items, Mapping):
        for key, value in collect_inline(items, (*path, ITEM)).items():
            found.setdefault(key, {}).update(value)
    for combinator in ("anyOf", "oneOf", "allOf"):
        for branch in schema.get(combinator) or ():
            for key, value in collect_inline(branch, path).items():
                found.setdefault(key, {}).update(value)
    return found


def _merge_layers(*layers: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for layer in layers:
        merged.update(layer or {})
    return merged


def _merge_param_layers(*layers: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for layer in layers:
        for key, value in layer.items():
            merged[key] = {**merged.get(key, {}), **value}
    return merged


def _validate_xjev(
    raw: RawTool, tool_layer: dict[str, Any], param_layers: dict[str, dict[str, Any]]
) -> tuple[ToolXJev, dict[str, ParamXJev]]:
    try:
        tool_x = ToolXJev.model_validate(tool_layer)
    except ValidationError as exc:
        raise CatalogError(f"{raw.name}: invalid tool-level x-jev: {_first_error(exc)}") from exc
    params: dict[str, ParamXJev] = {}
    for key, value in param_layers.items():
        try:
            params[key] = ParamXJev.model_validate(value)
        except ValidationError as exc:
            raise CatalogError(f"{raw.name}.{key}: invalid x-jev: {_first_error(exc)}") from exc
    return tool_x, params


def _first_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    loc = ".".join(str(p) for p in error["loc"])
    return f"{loc}: {error['msg']}" if loc else error["msg"]


def _constraints(raw: RawTool, declared: list[str] | None, slot_names: set[str]) -> tuple[Constraint, ...]:
    parsed: list[Constraint] = []
    for expr in declared or ():
        try:
            constraint = parse(expr)
        except ConstraintError as exc:
            raise CatalogError(f"{raw.name}: {exc}") from exc
        unknown = constraint.slots - slot_names
        if unknown:
            raise CatalogError(f"{raw.name}: constraint {expr!r} refers to unknown parameter(s) {sorted(unknown)}")
        parsed.append(constraint)
    return tuple(parsed)


def _groups(
    raw: RawTool, declared: list[list[str]] | None, tier: Tier, slots: Sequence[SlotSpec]
) -> tuple[tuple[str, ...], ...]:
    names = {slot.name for slot in slots}
    if declared is not None:
        for group in declared:
            unknown = set(group) - names
            if unknown:
                raise CatalogError(f"{raw.name}: group {group} refers to unknown parameter(s) {sorted(unknown)}")
        return tuple(tuple(group) for group in declared)
    if tier is Tier.CRITICAL:
        identity = tuple(slot.name for slot in slots if slot.stakes == "identity")
        return (identity,) if identity else ()
    return ()


def compile_tool(
    raw: RawTool,
    *,
    tool_id: str,
    sidecar: Sidecar | None = None,
    sources: Sequence[Any] = (),
    all_names: Sequence[str] = (),
) -> ToolSpec:
    """Compile one :class:`RawTool` into a :class:`ToolSpec` (see the module docstring for the steps)."""
    try:
        inlined = inline_refs(raw.parameters)
    except (KeyError, ValueError) as exc:
        raise CatalogError(f"{raw.name}: {exc}") from exc
    side_tool, side_params = sidecar.for_tool(raw.name, all_names=all_names) if sidecar else ({}, {})
    meta_tool = {k: v for k, v in raw.meta_xjev.items() if k != "properties"}
    meta_params = {str(k): dict(v) for k, v in (raw.meta_xjev.get("properties") or {}).items()}
    inline_params = collect_inline(inlined)
    root_inline = inlined.get(XJEV) if isinstance(inlined.get(XJEV), Mapping) else {}
    tool_layer = _merge_layers(side_tool, meta_tool, root_inline, raw.tool_xjev)
    if raw.markers_are_inline:
        param_layer = _merge_param_layers(inline_params, side_params, meta_params)
    else:
        param_layer = _merge_param_layers(side_params, meta_params, inline_params)
    tool_x, param_x = _validate_xjev(raw, tool_layer, param_layer)

    stripped = strip_xjev(inlined)
    slots = record_children(
        stripped, (), "", depth=1, xjevs=param_x, sources=sources, tool_name=raw.name,
        tool_description=raw.description,
    )  # fmt: skip
    known = {spec.key for slot in slots for spec in slot.walk()}
    unknown = sorted(set(param_x) - known)
    if unknown:
        raise CatalogError(f"{raw.name}: x-jev annotations for unknown parameter(s) {unknown}")

    tier, tier_reason = infer_tier(
        raw.name, risk=tool_x.risk, annotations=raw.annotations, slots=slots, sources=sources
    )
    slots = tuple(with_channels(slot, tier) for slot in slots)
    hints = explicit_annotations(raw.annotations)
    idempotent = tool_x.idempotent
    if idempotent is None:
        idempotent = hints["idempotentHint"] if "idempotentHint" in hints else tier is Tier.READ
    return ToolSpec(
        name=raw.name,
        id=tool_id,
        title=raw.title,
        description=raw.description,
        intent=tool_x.intent or default_intent(raw.name, raw.description),
        noun=tool_x.noun or humanize(raw.name),
        render=tool_x.render,
        confirm_template=tool_x.confirm_template,
        confirm=tool_x.confirm or "auto",
        constraints=_constraints(raw, tool_x.constraints, {s.name for s in slots}),
        groups=_groups(raw, tool_x.groups, tier, slots),
        joint_max=tool_x.joint_max,
        idempotent=idempotent,
        speculate=tool_x.speculate or "auto",
        aliases=tuple(tool_x.aliases or ()),
        emits=tool_x.emits.declared() if tool_x.emits else None,
        tier=tier,
        tier_reason=tier_reason,
        slots=tuple(slots),
        parameters=strip_xjev(raw.parameters),
        output_schema=raw.output_schema,
        annotations=dict(raw.annotations),
        xjev=tool_x,
    )


class Catalog:
    """An ordered set of compiled tools. Build with ``from_openai``/``from_mcp``/``from_callables``/
    ``from_pydantic``/``from_langchain``/``from_any``; rebuild against new sources with :meth:`with_sources`."""

    def __init__(
        self,
        raw_tools: Iterable[RawTool] = (),
        *,
        sidecar: SidecarLike = None,
        hints: SidecarLike = None,
        sources: Iterable[Any] = (),
    ) -> None:
        self.raw: tuple[RawTool, ...] = tuple(raw_tools)
        base = coerce_sidecar(sidecar)
        extra = coerce_sidecar(hints)
        self.sidecar: Sidecar | None = base.override(extra) if base is not None else extra
        self.sources: tuple[Any, ...] = tuple(sources)
        names = [raw.name for raw in self.raw]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise CatalogError(f"duplicate tool name(s): {duplicates}")
        ids = tool_ids(names)
        self.tools: tuple[ToolSpec, ...] = tuple(
            compile_tool(raw, tool_id=tid, sidecar=self.sidecar, sources=self.sources, all_names=names)
            for raw, tid in zip(self.raw, ids, strict=True)
        )
        self._by_name = {t.name: t for t in self.tools}
        self._by_id = {t.id: t for t in self.tools}

    # -- constructors -------------------------------------------------------------------------------------------

    @classmethod
    def from_openai(
        cls, tools: Iterable[Mapping[str, Any]], *, sidecar: SidecarLike = None, hints: SidecarLike = None,
        sources: Iterable[Any] = (),
    ) -> Catalog:  # fmt: skip
        """From OpenAI function tools; ``strict`` is ignored."""
        return cls([raw_from_openai(t) for t in tools], sidecar=sidecar, hints=hints, sources=sources)

    @classmethod
    def from_mcp(
        cls, list_tools_result: Any, *, sidecar: SidecarLike = None, hints: SidecarLike = None,
        sources: Iterable[Any] = (),
    ) -> Catalog:  # fmt: skip
        """From an MCP ``tools/list`` result (JSON dict or ``mcp`` object) or a list of MCP tools."""
        return cls(
            [raw_from_mcp(t) for t in mcp_tools(list_tools_result)], sidecar=sidecar, hints=hints, sources=sources
        )

    @classmethod
    async def afrom_mcp_session(
        cls, session: Any, *, sidecar: SidecarLike = None, hints: SidecarLike = None, sources: Iterable[Any] = ()
    ) -> Catalog:
        """From a connected MCP client session (``await session.list_tools()``)."""
        result = await session.list_tools()
        return cls.from_mcp(result, sidecar=sidecar, hints=hints, sources=sources)

    @classmethod
    def from_callables(
        cls, fns: Iterable[Callable[..., Any]], *, sidecar: SidecarLike = None, hints: SidecarLike = None,
        sources: Iterable[Any] = (),
    ) -> Catalog:  # fmt: skip
        """From Python functions or ``@jt.tool`` objects (signature → JSON Schema; docstring → description)."""
        return cls([raw_from_callable(fn) for fn in fns], sidecar=sidecar, hints=hints, sources=sources)

    @classmethod
    def from_pydantic(
        cls, model: type[BaseModel], *, name: str | None = None, description: str | None = None,
        sidecar: SidecarLike = None, hints: SidecarLike = None, sources: Iterable[Any] = (),
    ) -> Catalog:  # fmt: skip
        """From one pydantic model describing a tool's arguments."""
        raw = raw_from_pydantic(model, name=name, description=description)
        return cls([raw], sidecar=sidecar, hints=hints, sources=sources)

    @classmethod
    def from_langchain(
        cls, tools: Iterable[Any], *, sidecar: SidecarLike = None, hints: SidecarLike = None,
        sources: Iterable[Any] = (),
    ) -> Catalog:  # fmt: skip
        """From LangChain tools via ``convert_to_openai_tool`` (needs the ``langchain`` extra)."""
        try:
            from langchain_core.utils.function_calling import (  # type: ignore[import-not-found, unused-ignore]
                convert_to_openai_tool,
            )
        except ModuleNotFoundError as exc:
            raise CatalogError(
                "Catalog.from_langchain needs langchain-core: pip install 'jevtools[langchain]'"
            ) from exc
        raws = []
        for t in tools:
            raw = raw_from_openai(convert_to_openai_tool(t))
            raw.definition = t
            raws.append(raw)
        return cls(raws, sidecar=sidecar, hints=hints, sources=sources)

    @classmethod
    def from_any(
        cls, tools: Catalog | Iterable[ToolLike], *, sidecar: SidecarLike = None, hints: SidecarLike = None,
        sources: Iterable[Any] = (),
    ) -> Catalog:  # fmt: skip
        """From a mixed sequence (OpenAI dicts, MCP dicts, callables, ``@jt.tool`` objects, pydantic models) or a
        catalog (rebuilt with the given sidecar/hints/sources when any is given)."""
        if isinstance(tools, Catalog):
            if sidecar is None and hints is None and not sources:
                return tools
            return tools.rebuild(sidecar=sidecar, hints=hints, sources=sources or tools.sources)
        return cls([to_raw(t) for t in tools], sidecar=sidecar, hints=hints, sources=sources)

    def rebuild(
        self, *, sidecar: SidecarLike = None, hints: SidecarLike = None, sources: Iterable[Any] | None = None
    ) -> Catalog:
        """Recompile the same raw tools with an extra sidecar/hints layer and/or other sources."""
        combined = self.sidecar.override(coerce_sidecar(sidecar)) if self.sidecar else coerce_sidecar(sidecar)
        return Catalog(self.raw, sidecar=combined, hints=hints, sources=self.sources if sources is None else sources)

    def with_sources(self, sources: Iterable[Any]) -> Catalog:
        """Recompile against the registered sources of a context (sources drive ref inference and the invitee rule)."""
        return self.rebuild(sources=sources)

    # -- access -------------------------------------------------------------------------------------------------

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> ToolSpec:
        return self.get(name)

    def get(self, name: str) -> ToolSpec:
        """The tool named ``name`` (``KeyError`` if absent)."""
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"unknown tool {name!r}") from None

    def by_id(self, tool_id: str) -> ToolSpec:
        """The tool whose sanitized id is ``tool_id``."""
        return self._by_id[tool_id]

    @property
    def names(self) -> tuple[str, ...]:
        """Tool names in catalog order."""
        return tuple(t.name for t in self.tools)

    def to_openai(self) -> list[dict[str, Any]]:
        """OpenAI function tools with ``x-jev`` stripped (safe to forward to any provider)."""
        return [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in self.tools
        ]

    def to_doc(self) -> list[dict[str, Any]]:
        """The Catalog document hashed into ``catalog_sha256``: declarations plus the compiled tier and slots."""
        return [
            {
                "name": t.name,
                "id": t.id,
                "description": t.description,
                "tier": t.tier.value,
                "xjev": t.xjev.declared(),
                "parameters": t.parameters,
                "slots": [
                    {
                        "key": spec.key,
                        "kind": spec.kind,
                        "stakes": spec.stakes,
                        "source": spec.source,
                        "channels": [c.value for c in spec.channels],
                        "xjev": spec.xjev.declared(),
                    }
                    for spec in t.walk()
                ],
            }
            for t in self.tools
        ]

    @property
    def sha256(self) -> str:
        """``catalog_sha256``."""
        return sha256_of(self.to_doc())

    def __repr__(self) -> str:
        return f"Catalog({list(self.names)!r})"


def to_raw(tool: Any) -> RawTool:
    """Normalize one tool-like object (see :data:`ToolLike`)."""
    if isinstance(tool, RawTool):
        return tool
    if isinstance(tool, Mapping):
        if "inputSchema" in tool:
            return raw_from_mcp(tool)
        return raw_from_openai(tool)
    if inspect.isclass(tool) and issubclass(tool, BaseModel):
        return raw_from_pydantic(tool)
    if callable(tool):
        return raw_from_callable(tool)
    if hasattr(tool, "inputSchema"):
        return raw_from_mcp(tool)
    raise CatalogError(f"cannot turn {type(tool).__name__} into a tool")


__all__ = ["Catalog", "SidecarLike", "ToolLike", "collect_inline", "compile_tool", "strip_xjev", "to_raw"]

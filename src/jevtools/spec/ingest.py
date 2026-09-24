"""Normalize any tool source into :class:`RawTool` (spec §7.1) before compilation.

Sources: OpenAI function tools, MCP ``tools/list`` entries (plain JSON, no ``mcp`` dependency), Python callables
(signature → JSON Schema through pydantic) and pydantic models.
"""

from __future__ import annotations

import inspect
import re
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from jevtools.errors import CatalogError
from jevtools.templates import humanize

SKIPPED_PARAMS = frozenset({"self", "cls", "idempotency_key"})
"""Callable parameters that are never slots (``idempotency_key`` is passed by the executor, §6.5)."""


@dataclass
class RawTool:
    """A tool definition before ``x-jev`` merging and inference.

    ``parameters`` may contain inline ``x-jev`` anywhere. ``tool_xjev`` holds inline tool-level keys
    (``function["x-jev"]`` or ``@jt.tool(**keys)``), ``meta_xjev`` the MCP ``_meta["x-jev"]`` (tool keys plus
    ``properties``). ``markers_are_inline`` marks property ``x-jev`` that came from ``Annotated`` markers, which rank
    below sidecars (§3.2 precedence).
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    title: str | None = None
    output_schema: dict[str, Any] | None = None
    annotations: dict[str, Any] = field(default_factory=dict)
    tool_xjev: dict[str, Any] = field(default_factory=dict)
    meta_xjev: dict[str, Any] = field(default_factory=dict)
    markers_are_inline: bool = False
    origin: str = "openai"
    """``openai`` | ``mcp`` | ``callable`` | ``pydantic``."""
    definition: Any = None
    """The original object (dict, callable or model) for adapters and re-compilation."""


def _object_schema(schema: Any, where: str) -> dict[str, Any]:
    if schema is None:
        return {"type": "object", "properties": {}}
    if not isinstance(schema, Mapping):
        raise CatalogError(f"{where}: parameters must be a JSON Schema object")
    return dict(schema)


def raw_from_openai(tool: Mapping[str, Any]) -> RawTool:
    """An OpenAI function tool ``{"type": "function", "function": {name, description, parameters, …}}``.

    The bare ``function`` object is accepted too; ``strict`` is ignored.
    """
    function = tool.get("function", tool) if isinstance(tool, Mapping) else None
    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
        raise CatalogError(f"not an OpenAI function tool: {tool!r:.200}")
    name = function["name"]
    return RawTool(
        name=name,
        description=function.get("description") or "",
        parameters=_object_schema(function.get("parameters"), name),
        tool_xjev=dict(function.get("x-jev") or tool.get("x-jev") or {}),
        origin="openai",
        definition=tool,
    )


def _dump(obj: Any) -> Any:
    dump = getattr(obj, "model_dump", None)
    return dump(by_alias=True, exclude_none=True) if callable(dump) else obj


def raw_from_mcp(tool: Any) -> RawTool:
    """An MCP ``Tool`` (dict or pydantic object): ``name, title, description, inputSchema, outputSchema,
    annotations, _meta``. Only explicitly present annotation hints are kept (§3.3.2)."""
    data = _dump(tool)
    if not isinstance(data, Mapping) or not isinstance(data.get("name"), str):
        raise CatalogError(f"not an MCP tool: {tool!r:.200}")
    annotations = {k: v for k, v in (data.get("annotations") or {}).items() if v is not None}
    meta = data.get("_meta") or data.get("meta") or {}
    return RawTool(
        name=data["name"],
        description=data.get("description") or "",
        parameters=_object_schema(data.get("inputSchema"), data["name"]),
        title=data.get("title") or annotations.get("title"),
        output_schema=data.get("outputSchema"),
        annotations=annotations,
        meta_xjev=dict(meta.get("x-jev") or {}),
        origin="mcp",
        definition=tool,
    )


def mcp_tools(list_tools_result: Any) -> list[Any]:
    """The tool entries of a ``tools/list`` result (dict, pydantic object or plain list)."""
    data = _dump(list_tools_result)
    if isinstance(data, Mapping):
        data = data.get("tools", [])
    if not isinstance(data, list):
        raise CatalogError("expected a tools/list result or a list of MCP tools")
    return data


# --------------------------------------------------------------------------------------------------------------------
# Python callables and pydantic models
# --------------------------------------------------------------------------------------------------------------------

_SECTION = re.compile(r"^\s*(Args|Arguments|Parameters|Params)\s*:\s*$")
_GOOGLE_PARAM = re.compile(r"^(\s+)(\*{0,2}\w+)\s*(\([^)]*\))?\s*:\s*(.*)$")
_SPHINX_PARAM = re.compile(r"^\s*:param\s+(?:[\w\[\], ]+\s+)?(\w+)\s*:\s*(.*)$")


def parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """``(first line, {param: description})`` from a Google- or Sphinx-style docstring."""
    lines = doc.splitlines()
    summary = next((line.strip() for line in lines if line.strip()), "")
    params: dict[str, str] = {}
    in_args, indent, current = False, None, None
    for line in lines:
        sphinx = _SPHINX_PARAM.match(line)
        if sphinx:
            params[sphinx.group(1)] = sphinx.group(2).strip()
            continue
        if _SECTION.match(line):
            in_args, indent, current = True, None, None
            continue
        if not in_args:
            continue
        if line.strip() and not line[:1].isspace():
            in_args = False
            continue
        match = _GOOGLE_PARAM.match(line)
        if match and (indent is None or len(match.group(1)) <= indent):
            indent, current = len(match.group(1)), match.group(2).lstrip("*")
            params[current] = match.group(4).strip()
        elif current and line.strip():
            params[current] = (params[current] + " " + line.strip()).strip()
    return summary, params


def _has_description(annotation: Any) -> bool:
    if typing.get_origin(annotation) is not Annotated:
        return False
    return any(isinstance(m, FieldInfo) and m.description for m in typing.get_args(annotation)[1:])


def _auto_title(name: str) -> str:
    return name.replace("_", " ").title()


def _drop_auto_titles(schema: dict[str, Any], aliases: Mapping[str, str] | None = None) -> dict[str, Any]:
    schema.pop("title", None)
    for name, prop in (schema.get("properties") or {}).items():
        if isinstance(prop, dict):
            internal = (aliases or {}).get(name, name)
            if prop.get("title") in (_auto_title(name), _auto_title(internal)):
                prop.pop("title")
    return schema


def callable_schema(fn: Callable[..., Any]) -> tuple[dict[str, Any], str]:
    """JSON Schema of a callable's parameters (``Annotated`` markers included) and its docstring summary."""
    signature = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception as exc:  # unresolved forward references
        raise CatalogError(f"{getattr(fn, '__name__', fn)}: cannot resolve type hints: {exc}") from exc
    summary, docs = parse_docstring(inspect.getdoc(fn) or "")
    fields: dict[str, Any] = {}
    aliases: dict[str, str] = {}
    for i, param in enumerate(signature.parameters.values()):
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD) or param.name in SKIPPED_PARAMS:
            continue
        annotation = hints.get(param.name, Any)
        if param.name in docs and not _has_description(annotation):
            annotation = Annotated[annotation, Field(description=docs[param.name])]
        internal = f"p{i}"
        aliases[param.name] = internal
        default = ... if param.default is param.empty else param.default
        fields[internal] = (Annotated[annotation, Field(alias=param.name)], default)
    model = create_model(f"{getattr(fn, '__name__', 'tool')}_parameters", **fields)
    return _drop_auto_titles(model.model_json_schema(by_alias=True), aliases), summary


def raw_from_callable(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    xjev: Mapping[str, Any] | None = None,
) -> RawTool:
    """A Python function (or a ``@jt.tool`` object, which provides its own ``raw_tool()``)."""
    to_raw = getattr(fn, "raw_tool", None)
    if callable(to_raw) and name is None and description is None and xjev is None:
        return to_raw()  # type: ignore[no-any-return]
    parameters, summary = callable_schema(fn)
    return RawTool(
        name=name or str(getattr(fn, "__name__", "tool")),
        description=description if description is not None else summary,
        parameters=parameters,
        tool_xjev=dict(xjev or {}),
        markers_are_inline=True,
        origin="callable",
        definition=fn,
    )


def _snake(name: str) -> str:
    return "_".join(humanize(name).split())


def raw_from_pydantic(model: type[BaseModel], *, name: str | None = None, description: str | None = None) -> RawTool:
    """A pydantic model as a tool's arguments: ``model_json_schema()``; ``json_schema_extra={"x-jev": …}`` is honoured
    (model-level keys are tool-level)."""
    schema = model.model_json_schema()
    doc = inspect.getdoc(model) or ""
    if doc == inspect.getdoc(BaseModel):
        doc = ""
    summary = parse_docstring(doc)[0]
    root_xjev = schema.pop("x-jev", {})
    schema.pop("description", None)
    return RawTool(
        name=name or _snake(model.__name__),
        description=description if description is not None else summary,
        parameters=_drop_auto_titles(schema),
        tool_xjev=dict(root_xjev),
        origin="pydantic",
        definition=model,
    )


__all__ = [
    "SKIPPED_PARAMS",
    "RawTool",
    "callable_schema",
    "mcp_tools",
    "parse_docstring",
    "raw_from_callable",
    "raw_from_mcp",
    "raw_from_openai",
    "raw_from_pydantic",
]

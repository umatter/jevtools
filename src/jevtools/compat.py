"""Migration from existing Jev usage (spec §7.4).

- **Cookbook function calling.** :func:`cookbook_policy` gives every tier the read tier's rule (the W composition:
  the cookbook's weakest-judgment confidence, execute at 0.60, no confirm band, no gates, no probes);
  :func:`cookbook_hints` declares every tool ``risk: read``. An all-enum catalog then compiles to the cookbook's
  shape: one tool Choice with a no-tool option plus one Choice per argument (jevtools adds the sentinels). Inline
  ``x-jev.risk`` still wins over the hints, and the critical tier keeps its own rule (it never auto-executes
  uncertified), so a tool declared critical stays guarded.
- **``closed_sets``.** Its three kinds map to ``enum``, ``list`` (multi-select) and ``flag``; plain JSON Schema
  (``enum``, ``array`` of ``enum``, ``boolean``) already infers them (§3.3.1).
- **The ``jev`` package's ``@jev.fn``.** :func:`from_jev_fn` imports the decorated function's signature — its
  pydantic return model's fields — as one tool: ``bool`` → flag; ``Literal``/``Enum`` → enum; ``int`` with
  ``ge``/``le`` → **quantity with a grid** (not a Score) unless it is ordinal (§4.2.3: author ``levels``, or a
  graded name such as ``priority`` over at most 11 levels); ``float`` with ``ge``/``le`` → quantity. Calling the
  tool with elected arguments builds the return model, and :meth:`JevFnTool.result` keeps the probabilities the
  ``jev`` package discards.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from jevtools.decision import Decision
from jevtools.errors import CatalogError
from jevtools.policy import Policy
from jevtools.spec.catalog import Catalog, to_raw
from jevtools.spec.infer import ORDINAL_NAMES, name_tokens
from jevtools.spec.ingest import RawTool, parse_docstring, raw_from_pydantic
from jevtools.spec.sidecar import Sidecar, parse_sidecar

COOKBOOK_VERSION = "jevtools-cookbook-0.1"
READ_RULE: dict[str, Any] = {
    "composition": "W", "execute": 0.60, "confirm": None, "authorized": None, "content_accept": None,
    "show_alternatives_min": None, "require_present": None, "auto_execute": None,
}  # fmt: skip
"""The read tier's rule (spec Appendix B), applied by :func:`cookbook_policy` to read, write and external."""
MAX_GRID = 252
"""Largest quantity grid kept from ``ge``/``le`` bounds (the enum option limit)."""
MAX_ORDINAL_LEVELS = 11


# --------------------------------------------------------------------------------------------------------------------
# Cookbook
# --------------------------------------------------------------------------------------------------------------------


def cookbook_policy(base: Policy | None = None) -> Policy:
    """The cookbook's decision rule as a :class:`Policy`: read, write and external tiers use the W composition
    (weakest judgment) with execute at 0.60 and no confirm band or gates; ``present``/``rev`` probes are off.
    The critical tier is left as in ``base`` (default policy)."""
    data = (base or Policy()).to_dict()
    data.update(version=COOKBOOK_VERSION)
    for tier in ("read", "write", "external"):
        data["tiers"][tier] = dict(READ_RULE)
    data["probes"] = {"present": [], "reverse": []}
    data["notes"] = {**data.get("notes", {}), "compat": "cookbook_policy (spec §7.4)"}
    return Policy.model_validate(data)


def cookbook_hints(tools: Catalog | Iterable[Any]) -> Sidecar:
    """Hints declaring every tool ``risk: read`` (inline ``x-jev.risk`` still wins)."""
    names = tools.names if isinstance(tools, Catalog) else [to_raw(t).name for t in tools]
    return parse_sidecar({name: {"risk": "read"} for name in names})


def cookbook_catalog(tools: Iterable[Any], *, sources: Iterable[Any] = ()) -> Catalog:
    """A catalog of ``tools`` compiled with :func:`cookbook_hints`."""
    items = list(tools)
    return Catalog.from_any(items, hints=cookbook_hints(items), sources=list(sources))


# --------------------------------------------------------------------------------------------------------------------
# @jev.fn
# --------------------------------------------------------------------------------------------------------------------


def _bounds(info: FieldInfo) -> tuple[Any, Any]:
    low: Any = None
    high: Any = None
    for constraint in info.metadata:
        low = getattr(constraint, "ge", low)
        high = getattr(constraint, "le", high)
    return low, high


def _levels(info: FieldInfo) -> list[Any] | None:
    extra = info.json_schema_extra
    levels = extra.get("levels") if isinstance(extra, dict) else None
    return list(levels) if isinstance(levels, list) else None


def _is_enum(annotation: Any) -> bool:
    return typing.get_origin(annotation) is Literal or (inspect.isclass(annotation) and issubclass(annotation, Enum))


def field_xjev(name: str, info: FieldInfo) -> dict[str, Any]:
    """The ``x-jev`` of one ``@jev.fn`` return-model field (see the module docstring)."""
    annotation = info.annotation
    if annotation is bool:
        return {"kind": "flag"}
    if _is_enum(annotation):
        return {"kind": "enum"}
    low, high = _bounds(info)
    levels = _levels(info)
    if annotation is int and low is not None and high is not None:
        count = int(high) - int(low) + 1
        graded = bool(set(name_tokens(name)) & ORDINAL_NAMES)
        if levels is not None and len(levels) == count and count <= MAX_ORDINAL_LEVELS:
            values = [{"value": int(low) + i, "text": str(text)} for i, text in enumerate(levels)]
            return {"kind": "ordinal", "values": values}
        if graded and 0 < count <= MAX_ORDINAL_LEVELS:
            return {"kind": "ordinal"}
        grid = list(range(int(low), int(high) + 1)) if 0 < count <= MAX_GRID else None
        return {"kind": "quantity", **({"values": grid} if grid else {})}
    if annotation is float and low is not None and high is not None:
        if levels is not None and len(levels) >= 2:
            step = (float(high) - float(low)) / (len(levels) - 1)
            return {"kind": "quantity", "values": [{"value": float(low) + i * step, "text": str(t)}
                                                   for i, t in enumerate(levels)]}  # fmt: skip
        return {"kind": "quantity"}
    if annotation in (int, float):
        return {"kind": "quantity"}
    return {}


def return_model(fn: Callable[..., Any]) -> type[BaseModel]:
    """The pydantic return model of a ``@jev.fn`` function (or of the plain function it wraps)."""
    original = inspect.unwrap(fn)
    try:
        annotation = typing.get_type_hints(original).get("return")
    except Exception as exc:  # unresolved forward references
        raise CatalogError(f"{getattr(fn, '__qualname__', fn)}: cannot resolve the return annotation: {exc}") from exc
    if not (inspect.isclass(annotation) and issubclass(annotation, BaseModel)):
        raise CatalogError(f"{getattr(fn, '__qualname__', fn)}: a @jev.fn function returns a pydantic model, "
                           f"got {annotation!r}")  # fmt: skip
    return annotation


def _summary(fn: Callable[..., Any], model: type[BaseModel]) -> str:
    """The docstring summary; a ``@jev.fn`` docstring is a Jinja template (the state), so only the text before
    its first ``{{``/``{%`` is kept ("Triage the ticket: {{ ticket }}" → "Triage the ticket")."""
    for doc in (inspect.getdoc(inspect.unwrap(fn)), inspect.getdoc(model)):
        if doc and doc != inspect.getdoc(BaseModel):
            summary = parse_docstring(doc)[0]
            cut = min((i for i in (summary.find("{{"), summary.find("{%")) if i >= 0), default=len(summary))
            summary = summary[:cut].rstrip(" :,;-\u2014")
            if summary:
                return summary
    return ""


@dataclass(frozen=True)
class JevFnResult:
    """A decision read back as the ``@jev.fn`` return model, with the probabilities kept.

    ``value`` is ``None`` unless the call was decided (execute, or a proposed call awaiting confirmation when
    ``proposed=True``); ``p`` maps each field to the probability of its elected value and ``alternatives`` to its
    runner-ups ``[(value, p)]``; ``confidence`` is the call confidence C.
    """

    value: BaseModel | None
    p: dict[str, float] = field(default_factory=dict)
    alternatives: dict[str, list[tuple[Any, float]]] = field(default_factory=dict)
    confidence: float | None = None
    outcome: str = ""
    decision: Decision | None = None


class JevFnTool:
    """A ``@jev.fn`` function imported as a jevtools tool (:func:`from_jev_fn`).

    It is a tool-like object for :class:`~jevtools.spec.catalog.Catalog` and :class:`~jevtools.router.Router`
    (``raw_tool()``); calling it with the elected arguments builds the return model (the executor).
    """

    def __init__(self, fn: Callable[..., Any], *, name: str | None = None, description: str | None = None) -> None:
        self.fn = fn
        self.model = return_model(fn)
        original = inspect.unwrap(fn)
        self.name = name or str(getattr(original, "__name__", self.model.__name__))
        self.description = description if description is not None else _summary(fn, self.model)
        raw = raw_from_pydantic(self.model, name=self.name, description=self.description)
        properties = raw.parameters.get("properties") or {}
        for field_name, info in self.model.model_fields.items():
            key = info.alias or field_name
            prop = properties.get(key)
            if not isinstance(prop, dict):
                continue
            prop.pop("levels", None)
            xjev = field_xjev(field_name, info)
            if xjev:
                prop["x-jev"] = {**xjev, **(prop.get("x-jev") or {})}
        self.parameters: dict[str, Any] = raw.parameters
        self.tool_xjev: dict[str, Any] = raw.tool_xjev

    def __call__(self, **arguments: Any) -> BaseModel:
        """Execute an elected call: the return model built from the arguments."""
        return self.model.model_validate(arguments)

    def __repr__(self) -> str:
        return f"JevFnTool({self.name!r}, model={self.model.__name__})"

    def raw_tool(self) -> RawTool:
        """The ingest form used by :class:`~jevtools.spec.catalog.Catalog`."""
        return RawTool(name=self.name, description=self.description, parameters=self.parameters,
                       tool_xjev=dict(self.tool_xjev), origin="pydantic", definition=self)  # fmt: skip

    def to_openai(self) -> dict[str, Any]:
        """The OpenAI function tool (with ``x-jev``; strip it before forwarding to an LLM provider)."""
        function: dict[str, Any] = {"name": self.name, "description": self.description,
                                    "parameters": self.parameters}  # fmt: skip
        if self.tool_xjev:
            function["x-jev"] = self.tool_xjev
        return {"type": "function", "function": function}

    def result(self, decision: Decision, *, proposed: bool = False) -> JevFnResult:
        """The decision as the return model plus per-field probabilities (``proposed`` also reads a call that
        awaits confirmation)."""
        call = decision.tool_calls[0] if decision.tool_calls else (decision.call if proposed else None)
        value = self(**call.arguments) if call is not None and call.name == self.name else None
        return JevFnResult(
            value=value,
            p={name: report.p for name, report in decision.slots.items()},
            alternatives={
                name: [(a.value, a.p) for a in report.alternatives if a.part is None]
                for name, report in decision.slots.items()
            },
            confidence=decision.confidence.call if decision.confidence is not None else None,
            outcome=decision.outcome.value,
            decision=decision,
        )


def from_jev_fn(fn: Callable[..., Any], *, name: str | None = None, description: str | None = None) -> JevFnTool:
    """Import a ``@jev.fn`` function (or any function returning a pydantic model) as a tool (spec §7.4)."""
    return JevFnTool(fn, name=name, description=description)


__all__ = [
    "COOKBOOK_VERSION",
    "READ_RULE",
    "JevFnResult",
    "JevFnTool",
    "cookbook_catalog",
    "cookbook_hints",
    "cookbook_policy",
    "field_xjev",
    "from_jev_fn",
    "return_model",
]

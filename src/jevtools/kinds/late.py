"""Late binding (spec §3.6 rule 5, §5.4): values that depend on another slot's decoded value or on an observation.

Recipes carried by candidates (``Candidate.late``) and by ``NOT_STATED`` defaults (``SentinelSpec.late``):

- ``{"default_from": "from_account.currency"}`` — a late default: the attribute of another slot's elected row;
- ``{"derive": "all" | "half", "of": "from_account.balance"}`` — a derived quantity (money is quantized to the
  source row's ``currency``);
- ``{"placeholders": ["to.first_name", "obs:1"], "fill": {"⟨recipient's first name⟩": "to.first_name", …}}`` —
  a template whose ``⟨…⟩`` placeholders are replaced by the looked-up values (observation handles by the
  observation's full content).

Paths are ``<slot>.<attr>`` (the attribute of the slot's elected candidate; ``first_name``/``last_name`` are derived
from ``name`` or the label when the row has none; a bare ``<slot>`` is its value) or ``obs:<step>``.
The decode orchestrator binds after election: ``late_bind(result.value, result.late, make_lookup(results, ctx))``;
if the composed value then fails the schema, its mass is error and the next value is taken (§3.6).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from jevtools.canonical import canonical_str, jsonable
from jevtools.context import Context
from jevtools.errors import JevtoolsError
from jevtools.kinds.normalize import normalize_money, to_decimal

if TYPE_CHECKING:
    from jevtools.kinds.base import SlotResult

Lookup = Callable[[str], Any]


class LateBindError(JevtoolsError):
    """A late-bound value cannot be computed (its source slot is missing or lacks the attribute)."""


def late_dependencies(late: Mapping[str, Any] | None) -> set[str]:
    """Slot names a recipe reads (``obs:<step>`` handles excluded): used to order late binding."""
    if not late:
        return set()
    paths: list[str] = []
    if "default_from" in late:
        paths.append(str(late["default_from"]))
    if "of" in late:
        paths.append(str(late["of"]))
    paths += [str(p) for p in late.get("placeholders", ())]
    return {p.split(".", 1)[0] for p in paths if not p.startswith("obs:")}


def derived_attr(attrs: Mapping[str, Any], name: str) -> Any:
    """An attribute of a row, deriving ``first_name``/``last_name`` from ``name`` (or the label) when absent."""
    if name in attrs and attrs[name] not in (None, ""):
        return attrs[name]
    if name in ("first_name", "last_name"):
        full = str(attrs.get("name") or attrs.get("label") or "").split("<")[0].strip()
        parts = full.split()
        if parts:
            return parts[0] if name == "first_name" else parts[-1]
    raise KeyError(name)


def observation_text(ctx: Context, step: int) -> str:
    """The full content of an observation, as text (JSON content is canonical JSON)."""
    for observation in ctx.all_observations():
        if observation.step == step:
            content = observation.content
            return content if isinstance(content, str) else canonical_str(jsonable(content))
    raise KeyError(f"obs:{step}")


def make_lookup(results: Mapping[str, SlotResult], ctx: Context) -> Lookup:
    """A lookup over decoded slot results (by slot name) and the context's observations."""

    def lookup(path: str) -> Any:
        if path.startswith("obs:"):
            return observation_text(ctx, int(path.split(":")[1]))
        head, _, attr = path.partition(".")
        result = results.get(head)
        if result is None or result.is_bottom:
            raise KeyError(path)
        if not attr:
            return result.value
        return derived_attr({**result.attrs, "value": result.value}, attr)

    return lookup


def late_bind(
    value: Any, late: Mapping[str, Any] | None, lookup: Lookup, *, schema: Mapping[str, Any] | None = None
) -> Any:
    """Compute the final value of a late recipe (``value`` unchanged when there is none).

    Raises :class:`LateBindError` when a referenced value is unavailable.
    """
    if not late:
        return value
    try:
        if "default_from" in late:
            return lookup(str(late["default_from"]))
        if "derive" in late:
            return _derive(str(late["derive"]), str(late["of"]), lookup, schema or {})
        if "fill" in late:
            text = str(value)
            for shown, path in late["fill"].items():
                text = text.replace(shown, str(lookup(str(path))))
            return text
    except KeyError as exc:
        raise LateBindError(f"late binding failed: {exc.args[0]!r} is not available") from None
    return value


def _derive(operator: str, of: str, lookup: Lookup, schema: Mapping[str, Any]) -> Any:
    amount = to_decimal(lookup(of))
    if operator == "half":
        amount = amount / 2
    elif operator != "all":
        raise KeyError(f"derive:{operator}")
    head = of.split(".", 1)[0]
    try:
        currency = str(lookup(f"{head}.currency"))
    except KeyError:
        currency = None
    return normalize_money(Decimal(amount), schema, currency=currency)


__all__ = [
    "LateBindError",
    "Lookup",
    "derived_attr",
    "late_bind",
    "late_dependencies",
    "make_lookup",
    "observation_text",
]

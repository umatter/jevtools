"""The ``quantity`` resolver (spec §4.1, §4.2.4): numbers, dimension-filtered by the slot's unit.

- A slot with a unit (``duration_minutes`` → ``minute``) takes free quantity mentions of that dimension
  (``45 min``, ``an hour and a half``), converted to the declared unit.
- A bare number with no unit enters any numeric slot that has no dimension-marked candidate (§4.2.1); numbers
  claimed by money, temporal or another quantity never do ("10 minutes" is never a money amount).
- Labels are the normalized value (``45``); values are cast to the schema type (integer/number/string) and
  schema bounds/``multipleOf`` drop invalid candidates at pool time.
- Grid values are never mixed into pools; they belong to clarify menus only (§4.2.4): :meth:`QuantityResolver.
  clarify_values` offers them when the slot is required, has no default and nothing was stated (an empty pool). The
  grid is ``x-jev.values`` when declared (``jt.compat.from_jev_fn`` declares ``ge``/``le`` integer ranges this way),
  else the unit's :data:`GRIDS` row (15/30/45/60 minutes); at most :data:`GRID_MENU_MAX` evenly spaced values are
  offered, each checked against the schema and the slot's unary constraints.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from jevtools.candidates import Candidate, Channel, Pool
from jevtools.extract.base import Mention
from jevtools.extract.numbers import canonical_unit, decimal_str, unit_dimension
from jevtools.kinds.base import ResolveContext, register_resolver, resolve_default
from jevtools.kinds.common import ChoiceResolver, mention_candidate, pool_mentions, unary_constraints, violates
from jevtools.kinds.normalize import NormalizationError, normalize_quantity
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.spec.schema import is_valid

GRIDS: dict[str, tuple[int, ...]] = {
    "millisecond": (100, 250, 500, 1000),
    "second": (10, 30, 60, 120),
    "minute": (15, 30, 45, 60),
    "hour": (1, 2, 4, 8),
    "day": (1, 2, 3, 7),
    "week": (1, 2, 3, 4),
    "month": (1, 3, 6, 12),
    "year": (1, 2, 3, 5),
    "percent": (25, 50, 75, 100),
}
"""Default grids per canonical unit (§4.2.4 "15/30/45/60"). A slot without ``x-jev.values`` and without one of these
units has no grid: its clarify stays an open question."""
GRID_MENU_MAX = 4
"""Most grid values a menu offers (the clarify menu's ``k ≤ 4``); longer grids are sampled evenly, ends included."""
GRID_TEXT = "A common value, offered because the request states none."
"""Description of a grid candidate (it never reaches a Ballot: grids live in menus only)."""


def quantity_mentions(rc: ResolveContext, unit: str | None) -> tuple[list[tuple[Mention, Decimal]], bool]:
    """``(mention, amount)`` pairs for a slot, and whether they are dimension-marked (amounts in the mention unit)."""
    dim = unit_dimension(unit)
    if dim is not None:
        marked = [(m, Decimal(m.value)) for m in pool_mentions(rc, "quantity") if m.dim == dim]
        if marked:
            return marked, True
    return [(m, Decimal(m.value)) for m in pool_mentions(rc, "number")], False


def evenly(values: Sequence[Any], k: int) -> list[Any]:
    """At most ``k`` of ``values``, evenly spaced with both ends kept: index ``⌊i·(n−1)/(k−1) + ½⌋`` for
    ``i = 0…k−1`` (round half up, so every port picks the same values)."""
    n = len(values)
    if n <= k:
        return list(values)
    if k <= 1:
        return list(values[:k])
    picked = [(2 * i * (n - 1) + (k - 1)) // (2 * (k - 1)) for i in range(k)]
    return [values[j] for j in dict.fromkeys(picked)]


def grid_raw(slot: SlotSpec) -> list[tuple[Any, str | None]]:
    """The slot's whole grid as ``(raw value, member text)``: ``x-jev.values``, else the unit's :data:`GRIDS` row."""
    if slot.values:
        return [(member.value, member.label or member.text) for member in slot.values]
    unit = canonical_unit(slot.unit)
    return [(value, None) for value in GRIDS.get(unit or "", ())]


def grid_candidates(tool: ToolSpec, slot: SlotSpec, rc: ResolveContext, k: int = GRID_MENU_MAX) -> list[Candidate]:
    """Grid values that satisfy the schema and the slot's unary constraints, at most ``k`` (evenly spaced)."""
    constraints = unary_constraints(tool, slot)
    valid: list[Candidate] = []
    seen: set[str] = set()
    for raw, text in grid_raw(slot):
        try:
            value = normalize_quantity(raw, slot.json_schema)
        except (NormalizationError, ArithmeticError, ValueError, TypeError):
            continue
        if not is_valid(value, slot.json_schema) or (constraints and violates(constraints, slot, value, rc)):
            continue
        shown = value if isinstance(value, str) else decimal_str(Decimal(str(value)))
        if shown in seen:
            continue
        seen.add(shown)
        display = f"{shown} ({text})" if text and text != shown else shown
        valid.append(Candidate(value=value, display=display, text=GRID_TEXT, channel=Channel.AUTHOR,
                               prov={"grid": True}))  # fmt: skip
    return evenly(valid, k)


class QuantityResolver(ChoiceResolver):
    """Resolver for ``kind: quantity``."""

    kind = "quantity"
    normalizer = "quantity@1"

    def candidates(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
        unit = canonical_unit(slot.unit)
        found, marked = quantity_mentions(rc, unit)
        out: list[Candidate] = []
        for mention, amount in found:
            from_unit = str(mention.attrs.get("unit")) if marked else None
            try:
                value = normalize_quantity(amount, slot.json_schema, from_unit=from_unit, to_unit=unit)
            except NormalizationError:
                continue
            display = decimal_str(Decimal(str(value))) if not isinstance(value, str) else value
            out.append(mention_candidate(mention, value, display=display, unit=unit if marked else None))
        return out

    def clarify_values(self, tool: ToolSpec, slot: SlotSpec, pool: Pool | None, rc: ResolveContext) -> list[Candidate]:
        """Menu values for the clarify of this slot (§4.2.4): its grid, only when the slot is required, has no
        default and nothing was stated (no pool candidate, none blocked by the allow-list); else ``[]`` and the
        clarify stays an open question. The values go to the menu, never into the pool."""
        if not slot.required or resolve_default(tool, slot, rc.ctx) is not None:
            return []
        if pool is not None and (pool.candidates or pool.blocked):
            return []
        return grid_candidates(tool, slot, rc, k=min(GRID_MENU_MAX, rc.policy.shapes.ambiguous_k))


register_resolver("quantity", QuantityResolver())

__all__ = [
    "GRIDS",
    "GRID_MENU_MAX",
    "QuantityResolver",
    "evenly",
    "grid_candidates",
    "grid_raw",
    "quantity_mentions",
]

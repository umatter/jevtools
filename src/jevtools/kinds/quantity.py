"""The ``quantity`` resolver (spec §4.1, §4.2.4): numbers, dimension-filtered by the slot's unit.

- A slot with a unit (``duration_minutes`` → ``minute``) takes free quantity mentions of that dimension
  (``45 min``, ``an hour and a half``), converted to the declared unit.
- A bare number with no unit enters any numeric slot that has no dimension-marked candidate (§4.2.1); numbers
  claimed by money, temporal or another quantity never do ("10 minutes" is never a money amount).
- Labels are the normalized value (``45``); values are cast to the schema type (integer/number/string) and
  schema bounds/``multipleOf`` drop invalid candidates at pool time.
- Grid values (15/30/45/60) are never mixed into pools; they belong to clarify menus only (§4.2.4).
"""

from __future__ import annotations

from decimal import Decimal

from jevtools.candidates import Candidate
from jevtools.extract.base import Mention
from jevtools.extract.numbers import canonical_unit, decimal_str, unit_dimension
from jevtools.kinds.base import ResolveContext, register_resolver
from jevtools.kinds.common import ChoiceResolver, mention_candidate, pool_mentions
from jevtools.kinds.normalize import NormalizationError, normalize_quantity
from jevtools.spec.models import SlotSpec, ToolSpec


def quantity_mentions(rc: ResolveContext, unit: str | None) -> tuple[list[tuple[Mention, Decimal]], bool]:
    """``(mention, amount)`` pairs for a slot, and whether they are dimension-marked (amounts in the mention unit)."""
    dim = unit_dimension(unit)
    if dim is not None:
        marked = [(m, Decimal(m.value)) for m in pool_mentions(rc, "quantity") if m.dim == dim]
        if marked:
            return marked, True
    return [(m, Decimal(m.value)) for m in pool_mentions(rc, "number")], False


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


register_resolver("quantity", QuantityResolver())

__all__ = ["QuantityResolver", "quantity_mentions"]

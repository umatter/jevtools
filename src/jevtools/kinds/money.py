"""The ``money`` resolver (spec §4.1, §4.2.4): amounts from the money parser, plus derived amounts.

- Candidates are free money mentions (``CHF 250``, ``Fr. 250.–``, ``250 francs``); a bare number enters only when
  no money-marked candidate exists (and never one claimed by a duration or a date).
- Values are ``Decimal`` quantized to the currency's ISO 4217 minor unit and emitted as the schema's type
  (``"250.00"`` for a string slot); labels always show the minor unit (``250.00``).
- ``x-jev.derive`` operators ``all`` and ``half`` add *late-bound* candidates computed from the source account's
  ``balance`` after decoding (``{"derive": "all", "of": "from_account.balance"}``); they carry the ``registry``
  channel, which the critical tier allows for quantities ("all of it"). They are offered only when the request
  says so ("all", "everything", "the whole balance", "half").
- In the critical tier the allow-list admits only ``user`` and ``registry`` values, so an amount found in an
  invoice (``tool_output``) never reaches ``transfer_funds.amount``: the pool is ``channel_blocked`` (§5.1).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from jevtools.candidates import Candidate, Channel
from jevtools.extract.base import Mentions
from jevtools.kinds.base import ResolveContext, register_resolver
from jevtools.kinds.common import ChoiceResolver, mention_candidate, pool_mentions
from jevtools.kinds.normalize import NormalizationError, money_display, normalize_money
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.templates import PLACEHOLDER, humanize

DERIVE_CUES: dict[str, tuple[tuple[str, ...], ...]] = {
    "all": (
        ("all",),
        ("everything",),
        ("whole", "balance"),
        ("entire", "balance"),
        ("all", "of", "it"),
        ("alles",),
        ("tout",),
    ),
    "half": (("half",), ("die", "halfte"), ("la", "moitie")),
}
DERIVE_TEXT = {"all": "the whole balance of {source}", "half": "half the balance of {source}"}


def _cued(mentions: Mentions, cues: tuple[tuple[str, ...], ...]) -> bool:
    words = [t.folded for t in mentions.tokens_of("request")]
    return any(tuple(words[i : i + len(c)]) == c for c in cues for i in range(len(words)))


def balance_source(tool: ToolSpec, slot: SlotSpec) -> SlotSpec | None:
    """The sibling ref slot whose ``balance`` a derived amount reads: ``from_*`` first, else the first ref slot."""
    refs = [s for s in tool.slots if s.kind == "ref" and s is not slot]
    return next((s for s in refs if s.name.startswith(("from", "source"))), refs[0] if refs else None)


class MoneyResolver(ChoiceResolver):
    """Resolver for ``kind: money``."""

    kind = "money"
    normalizer = "money@1"

    def candidates(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
        out: list[Candidate] = []
        for m in pool_mentions(rc, "money"):
            currency = m.attrs.get("currency")
            value = _normalized(Decimal(m.value["amount"]), slot, currency)
            if value is not None:
                display = money_display(Decimal(m.value["amount"]), currency)
                out.append(mention_candidate(m, value, display=display, currency=currency))
        if not out:
            for m in pool_mentions(rc, "number"):
                value = _normalized(Decimal(m.value), slot, None)
                if value is not None:
                    out.append(mention_candidate(m, value, display=money_display(Decimal(m.value), None)))
        out += self.derived(tool, slot, rc.get_mentions())
        return out

    def derived(self, tool: ToolSpec, slot: SlotSpec, mentions: Mentions) -> list[Candidate]:
        """Late-bound ``all``/``half`` candidates (declared in ``x-jev.derive`` and cued in the request)."""
        source = balance_source(tool, slot)
        if source is None:
            return []
        out: list[Candidate] = []
        for operator in slot.derive:
            if operator not in DERIVE_CUES or not _cued(mentions, DERIVE_CUES[operator]):
                continue
            what = DERIVE_TEXT[operator].format(source=source.noun)
            late = {"derive": operator, "of": f"{source.name}.balance"}
            out.append(
                Candidate(
                    label=f"{operator} ({humanize(source.name)} balance)",
                    value=PLACEHOLDER.format(what=what),
                    text=f"Derived by the app: {what}, computed after the account is chosen.",
                    channel=Channel.REGISTRY,
                    prov={"derive": operator, "of": late["of"], "anchor": operator},
                    late=late,
                )
            )
        return out


def _normalized(amount: Decimal, slot: SlotSpec, currency: str | None) -> Any:
    """The amount in the slot's schema form, or ``None`` when the schema cannot hold it (e.g. a fractional amount
    for an ``integer`` slot): an invalid value is dropped at pool time, never raised (spec §3.5.6 "Values")."""
    try:
        return normalize_money(amount, slot.json_schema, currency=currency)
    except (NormalizationError, ArithmeticError, ValueError):
        return None


register_resolver("money", MoneyResolver())

__all__ = ["DERIVE_CUES", "MoneyResolver", "balance_source"]

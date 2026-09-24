"""The ``enum`` resolver (spec §4.2.2): enum members and built-in catalogs, one slot Choice.

This is the reference implementation of the :class:`~jevtools.kinds.base.Resolver` contract:

- **pool**: enum members (``author`` channel) with their descriptions; catalogs (and enums over 252 members) are
  shortlisted to members mentioned in the user's words ∪ context-preferred values (``rc.preferred``);
  schema-invalid members are dropped; the allow-list is applied; labels are WYSIWYG; canonical order.
- **questions**: the slot Choice ``T.P`` with ``NOT_STATED`` (→ default / omit / missing) and ``NONE_OF_THESE``;
  an empty pool with a default gets the 2-option coverage probe instead.
- **decode**: value pooling (``NOT_STATED`` pools with an equal member), election, shape, factor ``D(v*)``,
  alternatives.
- **widen** (:class:`~jevtools.kinds.base.Widenable`): a catalog miss widens over the members left out of the
  shortlist (buckets; ``iana_tz`` grouped by region, then a Choice within the top regions).
"""

from __future__ import annotations

import re
import zoneinfo
from collections.abc import Iterable, Mapping, Sequence
from functools import cache

from jevtools.ballot import BallotQuestion
from jevtools.candidates import (
    Candidate,
    Channel,
    Pool,
    apply_allow_list,
    assign_labels,
    canonical_order,
    display_value,
    value_key,
)
from jevtools.extract.catalogs import load_data
from jevtools.kinds.base import (
    ResolveContext,
    SlotResult,
    register_resolver,
)
from jevtools.kinds.common import ChoiceResolver
from jevtools.kinds.widen import (
    BUCKET,
    HIERARCHY,
    bucket_stage,
    decode_buckets,
    hierarchy_stage,
    is_bucket_stage,
)
from jevtools.spec.models import Member, SlotSpec, ToolSpec
from jevtools.spec.schema import is_valid
from jevtools.wire import Answer

NORMALIZER = "enum@1"
MAX_REAL_OPTIONS = 252

CATALOG_DATA: dict[str, tuple[Member, ...]] = {}
"""Catalogs registered in code (take precedence over packaged data files)."""


def register_catalog(name: str, members: Iterable[Member | Mapping[str, object] | str]) -> None:
    """Register (or replace) a catalog: members as :class:`Member`, ``{"value", "text", "aliases"}`` or plain codes."""
    CATALOG_DATA[name] = tuple(_member(m) for m in members)
    load_catalog.cache_clear()


def _member(raw: Member | Mapping[str, object] | str) -> Member:
    if isinstance(raw, Member):
        return raw
    if isinstance(raw, str):
        return Member(value=raw)
    return Member(
        value=raw["value"],
        text=raw.get("text"),
        aliases=tuple(raw.get("aliases") or ()),  # type: ignore[arg-type]
    )


@cache
def load_catalog(name: str) -> tuple[Member, ...]:
    """Members of a built-in catalog.

    Order of lookup: :data:`CATALOG_DATA`; ``iana_tz`` from :mod:`zoneinfo`; else the packaged file
    ``jevtools/extract/data/<name>.json`` — a JSON array of ``{"value", "text"?, "aliases"?}`` objects.
    """
    if name in CATALOG_DATA:
        return CATALOG_DATA[name]
    if name == "iana_tz":
        return tuple(Member(value=tz) for tz in sorted(zoneinfo.available_timezones()))
    try:
        data = load_data(name)  # one parse per file, shared with the extractors (currencies, countries)
    except LookupError:
        raise LookupError(f"catalog {name!r} is not available (no jevtools/extract/data/{name}.json)") from None
    return tuple(_member(m) for m in data)


def mention_pattern(term: str) -> re.Pattern[str]:
    """Word-bounded match; short all-caps codes (``CHF``, ``ALL``) match case-sensitively."""
    flags = 0 if term.isupper() and len(term) <= 4 else re.IGNORECASE
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", flags)


def mention_of(member: Member, text: str) -> str | None:
    """The first words of ``text`` naming the member (its value, label or an alias), or ``None``."""
    terms = [display_value(member.value), *(t for t in (member.label,) if t), *member.aliases]
    for term in terms:
        match = mention_pattern(term).search(text) if term else None
        if match is not None:
            return match.group()
    return None


def mentioned(member: Member, text: str) -> bool:
    """Whether the member's value, label or an alias occurs in ``text``."""
    return mention_of(member, text) is not None


def user_text(rc: ResolveContext) -> str:
    """The ``user`` channel's words: the request and the user's own earlier turns."""
    return "\n".join(turn.text for turn in rc.ctx.user_turns)


class EnumResolver(ChoiceResolver):
    """Resolver for ``kind: enum`` (enum members and catalogs): a :class:`ChoiceResolver` with its own pool (members
    are ``author`` values, anchored by the user's words) and the coverage rounds of a catalog miss."""

    kind = "enum"
    normalizer = NORMALIZER
    closed = True

    def members(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> tuple[list[Member], list[str]]:
        """The members to offer and notes: every member of a small enum, else the shortlist."""
        if slot.catalog is not None:
            all_members: Sequence[Member] = load_catalog(slot.catalog)
            shortlist = True
        else:
            all_members = slot.values or ()
            shortlist = len(all_members) > MAX_REAL_OPTIONS
        if not shortlist:
            return list(all_members), []
        text = user_text(rc)
        preferred = {display_value(v) for v in rc.preferred.get(rc.slot_key(tool, slot), ())}
        chosen = [m for m in all_members if mentioned(m, text) or display_value(m.value) in preferred]
        note = f"shortlist {len(chosen)} of {len(all_members)} ({slot.catalog or 'enum'})"
        return chosen[:MAX_REAL_OPTIONS], [note]

    def candidates(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
        return self._candidates(tool, slot, rc)[0]

    def _candidates(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> tuple[list[Candidate], list[str]]:
        members, notes = self.members(tool, slot, rc)
        valid = [m for m in members if is_valid(m.value, slot.json_schema)]
        if len(valid) < len(members):
            notes.append(f"dropped {len(members) - len(valid)} schema-invalid member(s)")
        text = user_text(rc)
        return [self.candidate(slot, m, anchor=mention_of(m, text)) for m in valid], notes

    def pool(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> Pool:
        candidates, notes = self._candidates(tool, slot, rc)
        admitted, blocked = apply_allow_list(candidates, slot.channels)
        labelled = assign_labels(admitted, slot=slot.name, label_max=rc.limits.label_max)
        return Pool(
            tool=tool.name,
            path=slot.path,
            kind=self.kind,
            candidates=canonical_order(labelled),
            closed=self.closed,
            evidence_backed=any("anchor" in c.prov for c in candidates),
            blocked=blocked,
            notes=notes,
        )

    @staticmethod
    def candidate(slot: SlotSpec, member: Member, *, anchor: str | None = None) -> Candidate:
        """The ``author`` candidate of one member (label = its preferred label or display form); ``anchor`` is the
        user's words naming it (``prov.anchor``: the member is anchored for joint Choices, §5.2)."""
        prov: dict[str, str] = {"source": slot.catalog or "enum"}
        if anchor is not None:
            prov["anchor"] = anchor
        return Candidate(
            label=member.label or "", value=member.value, display=display_value(member.value), text=member.text,
            channel=Channel.AUTHOR, prov=prov,
        )  # fmt: skip

    def widen(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext, stage: str
    ) -> tuple[Pool, list[BallotQuestion]]:
        """Catalog miss (§4.2.2, §4.6): bucket Choices over the members left out of the shortlist, grouped by region
        for ``iana_tz`` (``Europe/Zurich`` → ``Europe``); then a Choice over the members of the top groups."""
        if slot.catalog is None and len(slot.values or ()) <= MAX_REAL_OPTIONS:
            return pool, []
        members = load_catalog(slot.catalog) if slot.catalog is not None else (slot.values or ())
        shown = {value_key(c.value) for c in pool.candidates}
        rest = [self.candidate(slot, m) for m in members if value_key(m.value) not in shown]
        group_of = _region if slot.catalog == "iana_tz" else None
        if stage == BUCKET:
            return bucket_stage(tool, slot, rc, rest, skip=0, kind=self.kind, group_of=group_of)
        if stage == HIERARCHY and group_of is not None:
            return hierarchy_stage(tool, slot, rc, rest, kind=self.kind, group_of=group_of)
        return pool, []

    def decode(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer], rc: ResolveContext
    ) -> SlotResult:
        """A bucket round decodes its buckets; otherwise the slot (or hierarchy) Choice, as every Choice kind:
        the first ``slot``/``probe`` question, never a merged bucket or group question (§4.6)."""
        if is_bucket_stage(rc.slot_questions(tool, slot)):
            return decode_buckets(tool, slot, answers, rc, self.normalizer)
        return super().decode(tool, slot, pool, answers, rc)


def _region(candidate: Candidate) -> str | None:
    """Hierarchy group of an IANA time zone: its region (``Europe/Zurich`` → ``Europe``)."""
    value = str(candidate.value)
    return value.split("/", 1)[0] if "/" in value else None


register_resolver("enum", EnumResolver())

__all__ = [
    "CATALOG_DATA",
    "EnumResolver",
    "load_catalog",
    "mention_of",
    "mention_pattern",
    "mentioned",
    "register_catalog",
    "user_text",
]

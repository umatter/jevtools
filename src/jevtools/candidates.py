"""Candidates, pools, channels, sentinels and labels (spec §3.4).

A slot's admissible values are a finite **pool** of **candidates** that code enumerates. Each candidate carries the
provenance **channel** it came from; a slot's channel allow-list (I2) is enforced before Jev is asked. The label a
candidate shows to Jev is WYSIWYG whenever possible and always obeys the label grammar of §3.4.3.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from decimal import Decimal
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from jevtools._compat import StrEnum
from jevtools.canonical import canonical_str, format_number, nfc
from jevtools.policy import Tier

# --------------------------------------------------------------------------------------------------------------------
# Channels (§3.4.2)
# --------------------------------------------------------------------------------------------------------------------


class Channel(StrEnum):
    """Provenance channel of a candidate (spec §3.4.2)."""

    USER = "user"
    """The request, the user's turns in history, replies and clicks."""
    REGISTRY = "registry"
    """App-owned sources, the user profile and the clock."""
    AUTHOR = "author"
    """Enums, catalogs, templates, schema ``default``/``examples``."""
    HISTORY = "history"
    """Entities bound in earlier calls or named in assistant turns; trust inherited from the origin channel."""
    TOOL_OUTPUT = "tool_output"
    """Anything parsed from tool results. Untrusted."""
    GENERATED = "generated"
    """Filler or Escalator output. Untrusted until elected."""

    @property
    def trust(self) -> int:
        """Distrust rank: 0 for user/registry/author, 1 history, 2 tool_output, 3 generated (higher = less trusted)."""
        return _TRUST_RANK[self.value]


_TRUST_RANK = {"user": 0, "registry": 0, "author": 0, "history": 1, "tool_output": 2, "generated": 3}

TRUSTED_CHANNELS: frozenset[Channel] = frozenset({Channel.USER, Channel.REGISTRY, Channel.AUTHOR})
"""The top trust level: user = registry = author."""

EVIDENCE_CHANNELS: frozenset[Channel] = frozenset({Channel.USER, Channel.HISTORY, Channel.TOOL_OUTPUT})
"""Channels whose candidates are always evidence-backed (registry needs an anchor or a whole small source, §5.1)."""


def least_trusted(*channels: Channel | str) -> Channel:
    """The least-trusted channel among ``channels`` (first one wins on ties). A derived value takes this channel."""
    if not channels:
        raise ValueError("least_trusted() needs at least one channel")
    result = Channel(channels[0])
    for ch in channels[1:]:
        ch = Channel(ch)
        if ch.trust > result.trust:
            result = ch
    return result


# --------------------------------------------------------------------------------------------------------------------
# Sentinels (§3.4.3) and decode sentinel values (§3.6)
# --------------------------------------------------------------------------------------------------------------------

NOT_STATED = "NOT_STATED"
NONE_OF_THESE = "NONE_OF_THESE"
EXCLUDE = "EXCLUDE"
NO_TOOL = "NO_TOOL"
UNSUPPORTED = "UNSUPPORTED"
DONE = "DONE"
OTHER = "OTHER"
CANCEL = "CANCEL"

SLOT_SENTINELS: tuple[str, ...] = (NOT_STATED, NONE_OF_THESE, EXCLUDE)
TOOL_SENTINELS: tuple[str, ...] = (NO_TOOL, UNSUPPORTED, DONE)
REPLY_SENTINELS: tuple[str, ...] = (OTHER, CANCEL)
SENTINEL_ORDER: tuple[str, ...] = SLOT_SENTINELS + TOOL_SENTINELS + REPLY_SENTINELS
"""Fixed order in which sentinels follow the real options (§3.4.3)."""
SENTINELS: frozenset[str] = frozenset(SENTINEL_ORDER)
"""Every reserved sentinel label."""


class Bottom(StrEnum):
    """Decoded values that are not real values (spec §3.6). They key slot distributions next to real values."""

    MISSING = "⊥missing"
    """``NOT_STATED`` on a required slot without a default: the user must supply a value."""
    UNCOVERED = "⊥uncovered"
    """``NONE_OF_THESE``: the user indicated a value that is not in the pool."""
    EXCLUDED = "⊥excluded"
    """``EXCLUDE``: the mention is not a member of the list."""
    OMIT = "⊥omit"
    """``NOT_STATED`` on an optional slot without a default: omit the argument."""
    LATE_DEFAULT = "⊥default"
    """``NOT_STATED`` whose default is late-bound to another slot and not resolved yet. Never emitted: the decode
    orchestrator replaces it once the source slot is decoded."""


DecodesTo = Literal[
    "missing", "uncovered", "excluded", "omit", "default", "no_tool", "unsupported", "done", "other", "cancel"
]
"""What a sentinel label decodes to (the Ballot's decode map, §3.5.7)."""

BOTTOM_OF: dict[str, Bottom] = {
    "missing": Bottom.MISSING,
    "uncovered": Bottom.UNCOVERED,
    "excluded": Bottom.EXCLUDED,
    "omit": Bottom.OMIT,
}
"""``decodes_to`` → decoded bottom value, for the slot-sentinel decodes."""


def value_key(value: Any) -> str:
    """Hashable key of a decoded value: the bottom's text for :class:`Bottom`, else its canonical JSON.

    Two candidates whose normalized values are equal share a key, which is how value pooling works (§3.6 rule 2).
    """
    if isinstance(value, Bottom):
        return value.value
    return canonical_str(value)


def display_value(value: Any) -> str:
    """The display form of a normalized value: strings as is, numbers shortest, ``true``/``false``/``null``."""
    if isinstance(value, Bottom):
        return value.value
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, Decimal)):
        return format_number(value)
    return canonical_str(value)


# --------------------------------------------------------------------------------------------------------------------
# Candidate and Pool (§3.4.1)
# --------------------------------------------------------------------------------------------------------------------


class Candidate(BaseModel):
    """One admissible value of a slot, with its provenance (spec §3.4.1).

    ``value`` is the normalized value that is emitted when this candidate is elected (I1). ``label`` is what Jev
    sees (assigned by :func:`assign_labels`), ``text`` the self-contained option description.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = ""
    """Option label shown to Jev; empty until :func:`assign_labels` runs."""
    value: Any
    """Normalized value emitted on election (JSON-native)."""
    text: str | None = None
    """Option description (or the accept-Noul ``candidate`` text). ``None`` only for self-explanatory enum members."""
    channel: Channel
    prov: dict[str, Any] = Field(default_factory=dict)
    """Provenance: ``source``, ``key``, ``mention`` ``{text, span}``, ``score``, ``extractor``, ``reading``…
    Registry candidates set ``anchor`` (the matched mention) or ``whole: true`` to count as evidence (§5.1)."""
    late: dict[str, Any] | None = None
    """Late-binding recipe when part of the value depends on another slot: ``{"placeholders": ["to.first_name"]}``
    or ``{"derive": "all", "of": "from_account.balance"}``."""
    display: str | None = None
    """Display form of the value (WYSIWYG label source); defaults to :func:`display_value` of ``value``."""
    origin: Channel | None = None
    """For ``history`` candidates: the channel the entity originally came from (its trust is inherited)."""
    attrs: dict[str, Any] = Field(default_factory=dict)
    """Row attributes for constraints, ``render`` and ``order_by`` (e.g. ``balance``). Never sent to Jev."""

    @property
    def shown(self) -> str:
        """The value's display form."""
        return self.display if self.display is not None else display_value(self.value)

    @property
    def effective_channel(self) -> Channel:
        """The channel whose trust applies: a history entity inherits its origin's (less) trust, and a history value
        whose origin is unknown is untrusted (``tool_output``): an assistant turn may repeat what a tool planted."""
        if self.channel is Channel.HISTORY:
            return least_trusted(self.channel, history_origin_of(self))
        return self.channel

    @property
    def is_evidence(self) -> bool:
        """Whether this candidate is evidence-backed for viability (spec §5.1)."""
        if self.channel in EVIDENCE_CHANNELS:
            return True
        return self.channel is Channel.REGISTRY and bool(self.prov.get("anchor") or self.prov.get("whole"))


class Pool(BaseModel):
    """The candidates of one (tool, slot) plus the sentinels its questions offer (spec §2 step 3)."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    """Tool name."""
    path: tuple[str, ...]
    """Slot path (argument path)."""
    kind: str
    """The resolver kind that built the pool."""
    candidates: list[Candidate] = Field(default_factory=list)
    """Admitted candidates, labelled, in canonical order."""
    sentinels: tuple[str, ...] = (NOT_STATED, NONE_OF_THESE)
    """Sentinels the slot's questions offer."""
    closed: bool = False
    """True for enum/catalog pools: the pool is the value space, so the slot is fill-able without evidence."""
    evidence_backed: bool = False
    """True if at least one candidate is evidence-backed (spec §5.1)."""
    blocked: list[Candidate] = Field(default_factory=list)
    """Candidates removed by the channel allow-list (I2). Kept for viability (``channel_blocked``) and the trace."""
    notes: list[str] = Field(default_factory=list)
    """Human-readable notes for lint/explain (shortlist sizes, dropped invalid values…)."""
    meta: dict[str, Any] = Field(default_factory=dict)
    """Resolver-private data carried from ``pool`` to ``questions``/``decode`` (anchors, shortlist offsets…)."""

    @property
    def empty(self) -> bool:
        """No admitted candidates."""
        return not self.candidates

    @property
    def channel_blocked(self) -> bool:
        """All candidates were removed by the allow-list: the injection case (spec §5.1, rule P3)."""
        return not self.candidates and bool(self.blocked)

    def by_label(self, label: str) -> Candidate | None:
        """The candidate with this label, if any."""
        for candidate in self.candidates:
            if candidate.label == label:
                return candidate
        return None


# --------------------------------------------------------------------------------------------------------------------
# Allow-lists (§3.4.2, invariant I2)
# --------------------------------------------------------------------------------------------------------------------

_ALL = tuple(Channel)
_NO_GENERATED = tuple(ch for ch in Channel if ch is not Channel.GENERATED)
_TRUSTED_AND_HISTORY = (Channel.USER, Channel.REGISTRY, Channel.AUTHOR, Channel.HISTORY)
_TRUSTED = (Channel.USER, Channel.REGISTRY, Channel.AUTHOR)


def default_allow_list(tier: Tier | str, stakes: str, *, quantity: bool = False) -> tuple[Channel, ...]:
    """Default channel allow-list of a slot by tier × stakes (spec §3.4.2).

    ``quantity`` marks quantity/money slots, which in the critical tier accept only ``user`` values and
    ``registry``-derived values ("all of it").
    """
    tier = Tier(tier)
    if tier is Tier.CRITICAL:
        if stakes == "identity" and quantity:
            return (Channel.USER, Channel.REGISTRY)
        return _TRUSTED
    if stakes != "identity":
        return _ALL
    if tier is Tier.READ:
        return _NO_GENERATED
    return _TRUSTED_AND_HISTORY


def admits(allow: Iterable[Channel | str], candidate: Candidate) -> bool:
    """Whether a slot with allow-list ``allow`` admits ``candidate`` (I2).

    A ``history`` candidate additionally needs its origin channel admitted (trust is inherited from the origin); an
    unknown origin counts as ``tool_output`` (:func:`history_origin_of`), so it never reaches a slot that bars tool
    output, such as an external identity slot ("history, trusted origin only", §3.4.2).
    """
    allowed = {Channel(ch) for ch in allow}
    if candidate.channel not in allowed:
        return False
    if candidate.channel is not Channel.HISTORY:
        return True
    origin = history_origin_of(candidate)
    return origin in allowed or origin in TRUSTED_CHANNELS


def history_origin_of(candidate: Candidate) -> Channel:
    """The origin a ``history`` candidate inherits its trust from: its recorded ``origin``, or ``tool_output`` when
    that is unknown (``None``, or ``history`` itself: an untraced assistant-turn value)."""
    origin = candidate.origin
    return Channel.TOOL_OUTPUT if origin is None or origin is Channel.HISTORY else origin


def apply_allow_list(
    candidates: Iterable[Candidate], allow: Iterable[Channel | str]
) -> tuple[list[Candidate], list[Candidate]]:
    """Split candidates into ``(admitted, blocked)`` by the allow-list."""
    allow = tuple(allow)
    admitted: list[Candidate] = []
    blocked: list[Candidate] = []
    for candidate in candidates:
        (admitted if admits(allow, candidate) else blocked).append(candidate)
    return admitted, blocked


# --------------------------------------------------------------------------------------------------------------------
# Labels (§3.4.3)
# --------------------------------------------------------------------------------------------------------------------

LABEL_MAX = 64
"""Default maximum label length (the conformance probe may change it)."""
DESCRIPTION_MAX = 400
"""Maximum option description length (§3.5.6)."""

_RESERVED_KEYS = frozenset(s.casefold() for s in SENTINELS)
_SLUG_RE = re.compile(r"[^a-z0-9_]")


def label_key(label: str) -> str:
    """Uniqueness key of a label: NFC then casefold."""
    return nfc(label).casefold()


def is_reserved(label: str) -> bool:
    """Whether ``label`` equals a reserved sentinel label after NFC and casefold."""
    return label_key(label) in _RESERVED_KEYS


def is_valid_label(label: str, label_max: int = LABEL_MAX) -> bool:
    """Label grammar: 1..label_max printable characters, no newline or tab, no leading or trailing space."""
    return 1 <= len(label) <= label_max and label == label.strip() and label.isprintable()


def truncate(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` characters, marking the cut with ``…``."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def elide_path(path: str, label_max: int = LABEL_MAX) -> str | None:
    """Middle-elide a ``/``-separated path to fit ``label_max`` (``services/…/config/settings.yaml``).

    Keeps the first segment and as many trailing segments as fit; ``None`` if even the last segment does not fit.
    """
    parts = path.split("/")
    if len(parts) < 3:
        return None
    for keep in range(len(parts) - 2, 0, -1):
        candidate = parts[0] + "/…/" + "/".join(parts[-keep:])
        if len(candidate) <= label_max:
            return candidate
    return None


def slug_label(slot: str, n: int, *, taken: Iterable[str] = (), label_max: int = LABEL_MAX) -> str:
    """Fallback label ``<slot>_<n>`` (sanitized slot name), bumping ``n`` until it is free."""
    taken_keys = {label_key(t) for t in taken}
    base = _SLUG_RE.sub("_", slot.lower()).strip("_") or "option"
    i = n
    while True:
        suffix = f"_{i}"
        candidate = base[: label_max - len(suffix)] + suffix
        if label_key(candidate) not in taken_keys and not is_reserved(candidate):
            return candidate
        i += 1


def make_label(
    display: str,
    *,
    slot: str,
    n: int,
    label_max: int = LABEL_MAX,
    is_path: bool = False,
    taken: Iterable[str] = (),
) -> str:
    """The label of one candidate (spec §3.4.3).

    1. **WYSIWYG**: the display form itself, if it obeys the grammar and is free.
    2. A display equal to a reserved sentinel gets the suffix `` (value)``; a duplicate of an earlier label gets
       `` (2)``, `` (3)``…
    3. A long path is middle-elided if that stays unique.
    4. Otherwise the slug ``<slot>_<n>``.

    ``taken`` holds the labels already used in the same question.
    """
    taken = tuple(taken)
    taken_keys = {label_key(t) for t in taken}

    def free(label: str) -> bool:
        return is_valid_label(label, label_max) and not is_reserved(label) and label_key(label) not in taken_keys

    text = nfc(display)
    if is_valid_label(text, label_max):
        if free(text):
            return text
        suffixes = [" (value)"] if is_reserved(text) else [f" ({i})" for i in range(2, 100)]
        for suffix in suffixes:
            if free(text + suffix):
                return text + suffix
    elif is_path:
        elided = elide_path(text, label_max)
        if elided is not None and free(elided):
            return elided
    return slug_label(slot, n, taken=taken, label_max=label_max)


def with_full_value(text: str | None, full: str, limit: int = DESCRIPTION_MAX) -> str:
    """Append ``Full value: "…"`` to a description when the label does not show the whole value (§3.4.3)."""
    head = truncate(text, limit // 2) + " " if text else ""
    room = limit - len(head) - len('Full value: ""')
    return f'{head}Full value: "{truncate(full, room)}"'


def assign_labels(
    candidates: Sequence[Candidate],
    *,
    slot: str,
    label_max: int = LABEL_MAX,
    is_path: bool = False,
    taken: Iterable[str] = (),
) -> list[Candidate]:
    """Give every candidate a grammatical, unique label (in input order) and keep descriptions self-contained.

    The desired label is ``candidate.label`` if a source already rendered one (e.g. ``"{name} <{email}>"``),
    else the display form of the value. When the final label cannot show the desired text (elided path or slug),
    the full text is quoted in the description. Slug numbers count from 1 in input order.
    """
    used = list(taken)
    out: list[Candidate] = []
    for n, candidate in enumerate(candidates, start=1):
        desired = candidate.label or candidate.shown
        label = make_label(desired, slot=slot, n=n, label_max=label_max, is_path=is_path, taken=used)
        text = candidate.text
        if not nfc(label).startswith(nfc(desired)):
            text = with_full_value(text, nfc(desired))
        used.append(label)
        out.append(candidate.model_copy(update={"label": label, "text": text}))
    return out


T = TypeVar("T")


def _label_of(item: Any) -> str:
    return item if isinstance(item, str) else str(item.label)


def canonical_order(items: Iterable[T], *, key: Callable[[T], str] | None = None) -> list[T]:
    """Canonical option order (§3.4.3): real options by ``(casefold(NFC(label)), label)``, then sentinels in the
    fixed order ``NOT_STATED, NONE_OF_THESE, EXCLUDE, NO_TOOL, UNSUPPORTED, DONE, OTHER, CANCEL``.

    ``items`` are labels or objects with a ``label`` attribute (or pass ``key``).
    """
    label_of: Callable[[T], str] = key or _label_of
    items = list(items)
    real = sorted(
        (i for i in items if label_of(i) not in SENTINELS), key=lambda i: (label_key(label_of(i)), label_of(i))
    )
    sentinels = sorted((i for i in items if label_of(i) in SENTINELS), key=lambda i: SENTINEL_ORDER.index(label_of(i)))
    return real + sentinels


def reverse_order(items: Iterable[T], *, key: Callable[[T], str] | None = None) -> list[T]:
    """Order for ``rev`` probes: real options in *reverse* canonical order, sentinels still last in fixed order."""
    ordered = canonical_order(items, key=key)
    label_of: Callable[[T], str] = key or _label_of
    real = [i for i in ordered if label_of(i) not in SENTINELS]
    return real[::-1] + ordered[len(real) :]


__all__ = [
    "BOTTOM_OF",
    "CANCEL",
    "DESCRIPTION_MAX",
    "DONE",
    "EVIDENCE_CHANNELS",
    "EXCLUDE",
    "LABEL_MAX",
    "NONE_OF_THESE",
    "NOT_STATED",
    "NO_TOOL",
    "OTHER",
    "REPLY_SENTINELS",
    "SENTINELS",
    "SENTINEL_ORDER",
    "SLOT_SENTINELS",
    "TOOL_SENTINELS",
    "TRUSTED_CHANNELS",
    "UNSUPPORTED",
    "Bottom",
    "Candidate",
    "Channel",
    "DecodesTo",
    "Pool",
    "admits",
    "apply_allow_list",
    "history_origin_of",
    "assign_labels",
    "canonical_order",
    "default_allow_list",
    "display_value",
    "elide_path",
    "is_reserved",
    "is_valid_label",
    "label_key",
    "least_trusted",
    "make_label",
    "reverse_order",
    "slug_label",
    "truncate",
    "value_key",
    "with_full_value",
]

"""Mentions, span claiming and the extraction orchestrator (spec §4.2.1).

Extraction runs once per round over the allowed channels: the request and the user's turns (``user``), assistant
turns (``history``) and observations (``tool_output``). It produces :class:`Mention` objects
``{kind, text, span, value, dim, channel, source_ref}``.

**Claiming.** Each mention is claimed by the most specific extractor that covers it:
enum-member > registry anchor > temporal > money > quantity(dim) > pattern > bare number > place > generic span.
A covered, less specific mention records ``claimed_by`` and is left out of lower-priority pools: "Fahrenheit"
(claimed by the ``unit`` enum) is never a ``city`` candidate, and the number in "10 minutes" (claimed by the
time quantity) is never a money ``amount``.

**Negation.** Cue words (``not``, ``except``, ``without``, ``no``, ``nicht``, ``sauf``, ``ausser``…) within 3 tokens
before a mention mark it ``negated``; the mention stays in pools with a note (Jev sees the evidence and decides).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, overload

from jevtools.candidates import Channel
from jevtools.canonical import canonical_str, jsonable
from jevtools.context import Context
from jevtools.extract.locales import Locale, get_locale
from jevtools.extract.tokens import Token, covers, tokenize

if TYPE_CHECKING:
    from jevtools.spec.catalog import Catalog

PATTERN_KINDS: frozenset[str] = frozenset({"email", "url", "uuid", "ipv4", "regex"})
SPAN_KINDS: frozenset[str] = frozenset({"quote", "proper_noun", "noun_phrase", "clause", "command"})
"""Generic span kinds: the lowest claiming priority."""

PRIORITY: dict[str, int] = {
    "enum": 0,
    "anchor": 10,
    "temporal": 20,
    "money": 30,
    "quantity": 40,
    **dict.fromkeys(PATTERN_KINDS, 50),
    "number": 55,
    "place": 60,
    **dict.fromkeys(SPAN_KINDS, 70),
}
"""Claiming rank (lower = more specific). Cue mentions never claim and are never claimed."""

NEGATION_WINDOW = 3
_NEGATION_STOP = frozenset({"but", "instead", "rather", "aber", "sondern", "mais", ",", ";", ".", "!", "?", ":"})


@dataclass(frozen=True)
class Mention:
    """One extracted piece of evidence (spec §4.2.1).

    - ``kind``: ``number quantity money temporal email url uuid ipv4 regex place quote proper_noun noun_phrase
      clause command enum anchor cue``.
    - ``text``/``span``: the exact source substring and its ``[start, end)`` offsets in the text named by
      ``source_ref`` (``request``, ``user:<i>``/``assistant:<i>`` message index, ``obs:<step>[:<json path>]``).
    - ``value``: the parsed value (``Decimal`` string for numbers, ``{"amount", "currency"}`` for money,
      a :class:`~jevtools.extract.temporal.TemporalValue` for temporal mentions, the matched text otherwise).
    - ``dim``: the dimension of quantities (``time``, ``percent``, ``data``, ``money``) or ``None``.
    - ``channel``: ``user`` (request and user turns), ``history`` (assistant turns), ``tool_output`` (observations).
    - ``extractor``: ``name/locale`` of the extractor (``temporal/en``) for provenance.
    - ``claimed_by``: kind of the more specific mention covering this one (``None`` = free).
    - ``negated``: inside a negation; ``attrs["negation"]`` quotes the negating phrase.
    - ``attrs``: kind-specific details (unit, currency, readings, matched rows, cue name…).
    """

    kind: str
    text: str
    span: tuple[int, int]
    value: Any = None
    dim: str | None = None
    channel: Channel = Channel.USER
    source_ref: str = "request"
    extractor: str = ""
    claimed_by: str | None = None
    negated: bool = False
    attrs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def free(self) -> bool:
        """Not claimed by a more specific mention."""
        return self.claimed_by is None

    @property
    def rank(self) -> int:
        """Claiming priority (lower is more specific)."""
        return PRIORITY.get(self.kind, 1_000)

    @property
    def in_request(self) -> bool:
        return self.source_ref == "request"

    @property
    def step(self) -> int | None:
        """Observation step for ``tool_output`` mentions."""
        if not self.source_ref.startswith("obs:"):
            return None
        return int(self.source_ref.split(":")[1])

    def prov(self) -> dict[str, Any]:
        """Provenance entry: ``{"text", "span"}`` plus ``source_ref`` when not the request."""
        doc: dict[str, Any] = {"text": self.text, "span": [self.span[0], self.span[1]]}
        if self.source_ref != "request":
            doc["source_ref"] = self.source_ref
        return doc


@dataclass(frozen=True)
class SourceText:
    """One text extraction ran over."""

    ref: str
    text: str
    channel: Channel
    tokens: tuple[Token, ...]


class Mentions(Sequence[Mention]):
    """The mentions of one round, with the texts they came from and query helpers."""

    def __init__(self, mentions: Iterable[Mention] = (), texts: Iterable[SourceText] = (), locale: str = "en"):
        self._items = tuple(mentions)
        self.texts: dict[str, SourceText] = {t.ref: t for t in texts}
        self.locale = locale

    @overload
    def __getitem__(self, index: int) -> Mention: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Mention]: ...

    def __getitem__(self, index: int | slice) -> Mention | Sequence[Mention]:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Mention]:
        return iter(self._items)

    def __repr__(self) -> str:
        return f"Mentions({len(self._items)} mentions over {len(self.texts)} texts)"

    @property
    def request(self) -> str:
        """The request text."""
        source = self.texts.get("request")
        return source.text if source else ""

    def of(
        self,
        *kinds: str,
        channels: Collection[Channel] | None = None,
        free: bool = False,
        source_ref: str | None = None,
    ) -> list[Mention]:
        """Mentions of the given kinds (all kinds when none), optionally only free ones or one channel set."""
        return [
            m
            for m in self._items
            if (not kinds or m.kind in kinds)
            and (channels is None or m.channel in channels)
            and (not free or m.free)
            and (source_ref is None or m.source_ref == source_ref)
        ]

    def cues(self, cue: str, *, channels: Collection[Channel] | None = (Channel.USER,)) -> list[Mention]:
        """Cue mentions of one cue family (``negation superlative hedge chitchat anaphor``)."""
        return [m for m in self.of("cue", channels=channels) if m.attrs.get("cue") == cue]

    def has_cue(self, cue: str, *, request_only: bool = True) -> bool:
        """Whether the request (or any user turn) carries a cue of this family."""
        return any(m.in_request or not request_only for m in self.cues(cue))

    def anchors(self, source: str) -> list[Mention]:
        """User-channel registry anchors of one source, in text order."""
        return [m for m in self.of("anchor", channels=(Channel.USER,)) if m.attrs.get("source") == source]

    def tokens_of(self, ref: str) -> tuple[Token, ...]:
        source = self.texts.get(ref)
        return source.tokens if source else ()


# --------------------------------------------------------------------------------------------------------------------
# Texts
# --------------------------------------------------------------------------------------------------------------------


def source_texts(ctx: Context) -> list[SourceText]:
    """The texts of one round: request, user turns, assistant turns and observations (flattened JSON leaves)."""
    texts: list[SourceText] = []
    request_index = ctx.request_index
    for i, turn in enumerate(ctx.messages):
        if not turn.text or turn.role not in ("user", "assistant"):
            continue
        if i == request_index:
            ref, channel = "request", Channel.USER
        elif turn.role == "user":
            ref, channel = f"user:{i}", Channel.USER
        else:
            ref, channel = f"assistant:{i}", Channel.HISTORY
        texts.append(SourceText(ref, turn.text, channel, tuple(tokenize(turn.text))))
    for observation in ctx.all_observations():
        for path, text in _leaves(jsonable(observation.content), "$"):
            ref = f"obs:{observation.step}" + ("" if path == "$" else f":{path}")
            texts.append(SourceText(ref, text, Channel.TOOL_OUTPUT, tuple(tokenize(text))))
    return texts


def _leaves(value: Any, path: str) -> Iterator[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _leaves(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _leaves(item, f"{path}[{i}]")
    elif isinstance(value, str):
        if value.strip():
            yield path, value
    elif value is not None and not isinstance(value, bool):
        yield path, canonical_str(value)


# --------------------------------------------------------------------------------------------------------------------
# Claiming and negation
# --------------------------------------------------------------------------------------------------------------------


def claim(mentions: Sequence[Mention]) -> list[Mention]:
    """Set ``claimed_by`` on every mention covered by a more specific one in the same text (§4.2.1)."""
    by_ref: dict[str, list[Mention]] = {}
    for m in mentions:
        if m.kind != "cue":
            by_ref.setdefault(m.source_ref, []).append(m)
    out: list[Mention] = []
    for m in mentions:
        if m.kind == "cue":
            out.append(m)
            continue
        claimer = min(
            (o for o in by_ref[m.source_ref] if o.rank < m.rank and covers(o.span, m.span)),
            key=lambda o: o.rank,
            default=None,
        )
        out.append(replace(m, claimed_by=claimer.kind) if claimer is not None else m)
    return out


def mark_negations(
    mentions: Sequence[Mention], texts: Mapping[str, SourceText], locales: Sequence[Locale]
) -> list[Mention]:
    """Mark mentions preceded (within 3 tokens, no contrast word in between) by a negation cue."""
    negations = frozenset().union(*(loc.negations for loc in locales))
    out: list[Mention] = []
    for m in mentions:
        source = texts.get(m.source_ref)
        if m.kind == "cue" or source is None:
            out.append(m)
            continue
        quote = _negation_before(source, m.span, negations)
        out.append(replace(m, negated=True, attrs={**m.attrs, "negation": quote}) if quote else m)
    return out


def _negation_before(source: SourceText, span: tuple[int, int], negations: frozenset[str]) -> str | None:
    tokens = source.tokens
    first = next((i for i, t in enumerate(tokens) if t.start >= span[0]), None)
    if first is None:
        return None
    for back in range(1, NEGATION_WINDOW + 1):
        i = first - back
        if i < 0:
            return None
        token = tokens[i]
        if token.folded in _NEGATION_STOP or token.text in _NEGATION_STOP:
            return None
        if token.folded in negations:
            return source.text[token.start : span[1]]
    return None


# --------------------------------------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractProfile:
    """What to extract: the primary locale and extra regex extractors (``regex:<re>`` from slot declarations)."""

    locale: str = "en"
    regexes: tuple[str, ...] = ()
    enum_terms: tuple[tuple[str, tuple[str, ...]], ...] = ()
    """``(value, terms)`` pairs of enum members that claim their mentions."""


def profile_for(ctx: Context, catalog: Catalog | None = None, *, locale: str | None = None) -> ExtractProfile:
    """The extraction profile of a context and catalog: locale, declared regexes and enum-member terms."""
    regexes: list[str] = []
    terms: dict[str, tuple[str, ...]] = {}
    for tool in catalog or ():
        for slot in tool.walk():
            regexes += [e[len("regex:") :] for e in slot.extract if e.startswith("regex:")]
            if slot.kind == "enum":
                for value, member_terms in _enum_terms(slot):
                    terms.setdefault(value, member_terms)
    return ExtractProfile(
        locale=locale or ctx.locale, regexes=tuple(dict.fromkeys(regexes)), enum_terms=tuple(terms.items())
    )


def _enum_terms(slot: Any) -> Iterator[tuple[str, tuple[str, ...]]]:
    members: Sequence[Any] = slot.values or ()
    if slot.catalog is not None and slot.catalog != "iana_tz":
        from jevtools.kinds.enum import load_catalog

        try:
            members = load_catalog(slot.catalog)
        except LookupError:
            members = ()
    for member in members:
        if not isinstance(member.value, str):
            continue
        terms = tuple(t for t in (member.value, member.label, *member.aliases) if t and _claimable(t))
        if terms:
            yield member.value, terms


def _claimable(term: str) -> bool:
    return any(ch.isalpha() for ch in term) and (len(term) >= 3 or not term.isalpha())


def run_extractors(
    ctx: Context,
    catalog: Catalog | None = None,
    *,
    profile: ExtractProfile | None = None,
    locale: str | None = None,
) -> Mentions:
    """Extract every mention of one round from ``ctx`` (request, user turns, assistant turns, observations), then
    claim and mark negations. ``catalog`` supplies enum-member terms and declared ``regex:`` extractors;
    registries in ``ctx.sources`` supply anchors."""
    from jevtools.extract import cues, money, numbers, patterns, places, temporal, text

    profile = profile or profile_for(ctx, catalog, locale=locale)
    primary = get_locale(profile.locale)
    now = ctx.current_time()
    texts = source_texts(ctx)
    found: list[Mention] = []
    for source in texts:
        own: list[Mention] = []
        own += patterns.extract(source, profile.regexes)
        nums = numbers.extract(source, primary)
        own += nums + money.extract(source, primary, nums)
        own += temporal.extract(source, now, ctx.timezone_name, primary)
        own += places.extract(source)
        own += enum_mentions(source, profile.enum_terms)
        if source.channel is Channel.USER:
            own += anchor_mentions(source, ctx.sources.values())
        own += text.extract(source, primary, own)
        own += cues.extract(source)
        found += own
    claimed = claim(found)
    marked = mark_negations(claimed, {t.ref: t for t in texts}, (primary,) + _others(primary))
    return Mentions(marked, texts, primary.code)


def _others(primary: Locale) -> tuple[Locale, ...]:
    from jevtools.extract.locales import all_locales

    return tuple(loc for loc in all_locales() if loc.code != primary.code)


def enum_mentions(source: SourceText, enum_terms: Sequence[tuple[str, tuple[str, ...]]]) -> list[Mention]:
    """Enum-member matches (word-bounded; short all-caps codes case-sensitively)."""
    from jevtools.kinds.enum import mention_pattern

    out: list[Mention] = []
    for value, terms in enum_terms:
        for term in terms:
            for match in mention_pattern(term).finditer(source.text):
                out.append(
                    Mention("enum", match.group(), match.span(), value, None, source.channel, source.ref, "enum")
                )
    return out


def anchor_mentions(source: SourceText, sources: Iterable[Any]) -> list[Mention]:
    """Registry anchors: runs of user words that match ≥ 1 registry row (sources exposing ``find_anchors``)."""
    out: list[Mention] = []
    for src in sources:
        find = getattr(src, "find_anchors", None)
        if callable(find):
            out += find(source.text, tokens=source.tokens, source_ref=source.ref, channel=source.channel)
    return out


__all__ = [
    "PATTERN_KINDS",
    "PRIORITY",
    "SPAN_KINDS",
    "ExtractProfile",
    "Mention",
    "Mentions",
    "SourceText",
    "anchor_mentions",
    "claim",
    "enum_mentions",
    "mark_negations",
    "profile_for",
    "run_extractors",
    "source_texts",
]

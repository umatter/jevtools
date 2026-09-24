"""The ``span`` resolver (spec §4.2.6): extractive spans of the user's words, copied verbatim.

Candidates come from the slot's extractors (``x-jev.extract``, else the role defaults of §3.3.1):

- ``place``: gazetteer places ("Zurich"); with ``canon: "cities"`` every gazetteer entry the name denotes becomes a
  candidate with its canonical value (``Zürich, CH``, ``Zurich, Ontario, CA``).
- ``email url uuid ipv4 regex:<re>``: pattern matches (emails normalized: domain lowercased).
- ``quote proper_noun noun_phrase clause``: quoted strings, proper-noun runs, noun chunks, message clauses and the
  request minus its command verb.
- ``examples`` (schema) and literal ``x-jev.values`` become ``author`` candidates.

Only *free* mentions enter (claimed text never does: "Fahrenheit", claimed by the ``unit`` enum, is never a
``city``). Labels are the normalized span (``span@1``: NFC, trimmed, wrapping quotes and trailing punctuation
stripped); the pattern and bounds of the schema drop invalid spans at pool time.
"""

from __future__ import annotations

from collections.abc import Sequence

from jevtools.candidates import Candidate, Channel
from jevtools.extract.base import Mention
from jevtools.extract.catalogs import Place
from jevtools.kinds.base import ResolveContext, register_resolver
from jevtools.kinds.common import ChoiceResolver, mention_candidate, pool_mentions
from jevtools.kinds.normalize import NormalizationError, normalize_email_value, normalize_path, normalize_span
from jevtools.kinds.ref import is_path_slot
from jevtools.spec.models import SlotSpec, ToolSpec

PATTERN_EXTRACTORS: dict[str, str] = {"email": "email", "url": "url", "uuid": "uuid", "ipv4": "ipv4"}
GENERIC_EXTRACTORS: dict[str, tuple[str, ...]] = {
    "quote": ("quote",),
    "proper_noun": ("proper_noun",),
    "noun_phrase": ("noun_phrase",),
    "clause": ("clause", "command"),
}
ROLE_DEFAULTS: dict[str, tuple[str, ...]] = {"generic": ("clause", "quote", "noun_phrase", "proper_noun")}
"""Undeclared generic spans also take proper-noun runs (§4.2.6 lists them among the span extractors)."""


def extractors_of(slot: SlotSpec) -> tuple[str, ...]:
    """The slot's extractors: declared ``x-jev.extract``, else the role default, else the inferred list."""
    if slot.xjev.extract is None and slot.role in ROLE_DEFAULTS:
        return ROLE_DEFAULTS[slot.role]
    return slot.extract or ("clause", "quote", "noun_phrase")


def place_note(places: Sequence[Place]) -> str:
    """``a city in Switzerland`` (plus ``also …`` when the name is ambiguous)."""
    if not places:
        return ""
    note = places[0].describe()
    if len(places) > 1:
        note += "; also " + places[1].describe()
    return note


def span_candidates(slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
    """Candidates of every extractor of the slot, in extractor order (free pool mentions only)."""
    out: list[Candidate] = []
    for name in extractors_of(slot):
        if name == "place":
            out += _places(slot, pool_mentions(rc, "place"))
        elif name in PATTERN_EXTRACTORS:
            for m in pool_mentions(rc, PATTERN_EXTRACTORS[name]):
                value = normalize_email_value(m.value) if name == "email" else normalize_span(m.value)
                out.append(mention_candidate(m, value, display=value))
        elif name.startswith("regex:"):
            expr = name[len("regex:") :]
            for m in pool_mentions(rc, "regex"):
                if m.attrs.get("regex") == expr:
                    out.append(mention_candidate(m, normalize_span(m.value), display=normalize_span(m.value)))
        else:
            for m in pool_mentions(rc, *GENERIC_EXTRACTORS.get(name, (name,))):
                value = normalize_span(m.text)
                if value:
                    out.append(mention_candidate(m, value, display=value))
    return out


def _places(slot: SlotSpec, mentions: Sequence[Mention]) -> list[Candidate]:
    out: list[Candidate] = []
    for m in mentions:
        places: Sequence[Place] = m.attrs.get("places", ())
        if slot.canon is None:
            value = normalize_span(m.text)
            out.append(mention_candidate(m, value, display=value, note=place_note(places) or None))
            continue
        for place in places:
            note = f"read as {place.name}, {place.describe()}"
            out.append(mention_candidate(m, place.canonical, display=place.canonical, note=note, canon=slot.canon))
    return out


def author_candidates(slot: SlotSpec) -> list[Candidate]:
    """Schema ``examples`` and literal ``x-jev.values`` as ``author`` candidates."""
    out: list[Candidate] = []
    examples = slot.json_schema.get("examples")
    for example in examples if isinstance(examples, list) else ():
        if isinstance(example, str) and example.strip():
            out.append(
                Candidate(
                    value=normalize_span(example),
                    text="An example value given by the app.",
                    channel=Channel.AUTHOR,
                    prov={"source": "examples"},
                )
            )
    for member in slot.values or ():
        if isinstance(member.value, str):
            out.append(
                Candidate(
                    value=member.value,
                    text=member.text or "A value offered by the app.",
                    channel=Channel.AUTHOR,
                    prov={"source": "values"},
                )
            )
    return out


class SpanResolver(ChoiceResolver):
    """Resolver for ``kind: span``."""

    kind = "span"
    normalizer = "span@1"

    def candidates(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
        out = span_candidates(slot, rc) + author_candidates(slot)
        return path_candidates(out) if is_path_slot(slot) else out

    def normalizer_for(self, slot: SlotSpec) -> str:
        if slot.format == "email":
            return "email@1"
        return "path@1" if is_path_slot(slot) else self.normalizer


def path_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """``path@1`` for a path slot without a file index (§4.3): values are POSIX-normalized and ``..``/absolute/home/
    drive paths are dropped at pool time, unless the app itself offers them (author examples and values)."""
    out: list[Candidate] = []
    for c in candidates:
        if not isinstance(c.value, str):
            out.append(c)
            continue
        try:
            value = normalize_path(c.value, known=c.channel in (Channel.AUTHOR, Channel.REGISTRY))
        except NormalizationError:
            continue
        out.append(c if value == c.value else c.model_copy(update={"value": value, "display": value}))
    return out


register_resolver("span", SpanResolver())

__all__ = ["SpanResolver", "author_candidates", "extractors_of", "path_candidates", "place_note", "span_candidates"]

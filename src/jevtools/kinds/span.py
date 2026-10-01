"""The ``span`` resolver (spec §4.2.6): extractive spans of the user's words, copied verbatim.

Candidates come from the slot's extractors (``x-jev.extract``, else the role defaults of §3.3.1):

- ``place``: gazetteer places ("Zurich"); with ``canon: "cities"`` every gazetteer entry the name denotes becomes a
  candidate with its canonical value (``Zürich, CH``, ``Zurich, Ontario, CA``).
- ``email url uuid ipv4 code regex:<re>``: pattern matches (emails normalized: domain lowercased; ``code``:
  identifiers and file names such as ``SKU-4411`` or ``notes_old.txt``).
- ``quote proper_noun noun_phrase clause``: quoted strings, proper-noun runs, noun chunks, message clauses and the
  request minus its command verb.
- ``examples`` (schema) and literal ``x-jev.values`` become ``author`` candidates.

Only *free* mentions enter (claimed text never does: "Fahrenheit", claimed by the ``unit`` enum, is never a
``city``). Labels are the normalized span (``span@1``: NFC, trimmed, wrapping quotes and trailing punctuation
stripped); the pattern and bounds of the schema drop invalid spans at pool time.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Any

from jevtools.candidates import Candidate, Channel
from jevtools.canonical import jsonable
from jevtools.extract.base import Mention
from jevtools.extract.catalogs import Place
from jevtools.kinds.base import ResolveContext, register_resolver
from jevtools.kinds.common import ChoiceResolver, mention_candidate, pool_mentions
from jevtools.kinds.normalize import NormalizationError, normalize_email_value, normalize_path, normalize_span
from jevtools.kinds.ref import is_path_slot
from jevtools.spec.infer import QUALIFIERS
from jevtools.spec.models import ITEM, SlotSpec, ToolSpec

PATTERN_EXTRACTORS: dict[str, str] = {"email": "email", "url": "url", "uuid": "uuid", "ipv4": "ipv4", "code": "code"}
GENERIC_EXTRACTORS: dict[str, tuple[str, ...]] = {
    "quote": ("quote",),
    "proper_noun": ("proper_noun",),
    "noun_phrase": ("noun_phrase",),
    "clause": ("clause", "command"),
}
ROLE_DEFAULTS: dict[str, tuple[str, ...]] = {"generic": ("clause", "quote", "noun_phrase", "proper_noun", "code")}
"""Undeclared generic spans also take proper-noun runs (§4.2.6 lists them among the span extractors) and
identifier-like tokens (``code``)."""


_NAMED_PATTERNS = frozenset({"email", "url", "uuid"})


def extractors_of(slot: SlotSpec) -> tuple[str, ...]:
    """The slot's extractors: declared ``x-jev.extract``, else the role default, else the inferred list."""
    if slot.xjev.extract is None and slot.role in ROLE_DEFAULTS:
        named = tuple(e for e in slot.extract or () if e in _NAMED_PATTERNS)  # ``email`` from a slot named ``email``
        return ROLE_DEFAULTS[slot.role] + tuple(e for e in named if e not in ROLE_DEFAULTS[slot.role])
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


_ID_SUFFIX = re.compile(r"_(?:ids?|numbers?|codes?|keys?)$")
_FIRST_NAME = re.compile(r"^(?:first|given|fore)_?name$")
_LAST_NAME = re.compile(r"^(?:last|family|sur)_?name$|^surname$")


def name_part_candidates(slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
    """For a ``first_name`` / ``last_name`` slot: the first or last word of each proper-noun run of the user's
    words ("Yusuf Rossi" → "Yusuf" / "Rossi"); a person's full name is how people give it."""
    name = slot.name.lower()
    which = 0 if _FIRST_NAME.match(name) else -1 if _LAST_NAME.match(name) else None
    if which is None:
        return []
    out: list[Candidate] = []
    seen: set[str] = set()
    for m in pool_mentions(rc, "proper_noun"):
        if m.channel != Channel.USER:
            continue
        words = m.text.split()
        if len(words) < 2:
            continue
        part = words[which].strip(".,;:")
        if part and part not in seen:
            seen.add(part)
            where = "first" if which == 0 else "last"
            out.append(mention_candidate(m, part, display=part, note=f'the {where} word of "{m.text}"'))
    return out


def digit_string_candidates(slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
    """For a string slot: every all-digit number of three or more digits the user typed, as text. A zip code, an
    account or an order number is a string, not a quantity ("my zip code is 19122"). A tool result's numbers
    come as field candidates, described by their siblings."""
    if slot.json_schema.get("type") != "string":
        return []
    out: list[Candidate] = []
    seen: set[str] = set()
    for m in pool_mentions(rc, "number"):
        text = m.text.strip()
        if m.channel == Channel.USER and len(text) >= 3 and text.isdigit() and text not in seen:
            seen.add(text)
            out.append(mention_candidate(m, text, display=text))
    return out


_EXAMPLE_ID = re.compile(r"""['"]([#A-Za-z]{0,3}?)(\d{4,})['"]""")


def id_format(slot: SlotSpec) -> tuple[str, int] | None:
    """The id format a string slot's description shows by example: ``"such as '#W0000000'"`` → ``("#W", 7)`` (a
    prefix of up to three letters or ``#`` before a run of digits). ``None`` without exactly one such format."""
    if slot.json_schema.get("type") != "string":
        return None
    found = {(m.group(1), len(m.group(2))) for m in _EXAMPLE_ID.finditer(slot.description or "")}
    return found.pop() if len(found) == 1 and next(iter(found))[0] else None


def id_format_candidates(slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
    """The user's ids written without the format's prefix ("W4284542", "9502127" for ``#W0000000``), put in it.
    Only the prefix is restored; the digits are the user's, and their count must match the example's."""
    fmt = id_format(slot)
    if fmt is None:
        return []
    prefix, width = fmt
    letters = prefix.lstrip("#")
    loose = re.compile(rf"^#?(?:{re.escape(letters)})?(\d{{{width}}})$", re.I) if letters else None
    out: list[Candidate] = []
    seen: set[str] = set()
    for m in pool_mentions(rc, "code", "number"):
        text = m.text.strip().rstrip(".,;:")
        match = loose.match(text) if loose is not None else re.fullmatch(rf"#?(\d{{{width}}})", text)
        value = prefix + match.group(1) if match else None
        if m.channel == Channel.USER and value is not None and value != text and value not in seen:
            seen.add(value)
            out.append(mention_candidate(m, value, display=value, note=f'"{text}" in the format {prefix}{"0" * width}'))
    return out


def field_stem(name: str) -> str:
    """A field or slot name without an id suffix or plural: ``item_ids``, ``item_id`` → ``item``; ``orders`` →
    ``order``; ``flight_number`` → ``flight``."""
    stem = _ID_SUFFIX.sub("", name.lower())
    return stem[:-1] if len(stem) > 3 and stem.endswith("s") and not stem.endswith("ss") else stem


def _fields(value: Any, path: str, key: str | None, siblings: str = "") -> Iterator[tuple[str, str | None, Any, str]]:
    """Scalar leaves of a JSON value with their path, the name of the field that holds them (a list item keeps its
    list's field name) and the other short scalar fields of the same object ("name: Water Bottle"), which say what
    the value belongs to."""
    if isinstance(value, dict):
        for k, item in value.items():
            others = ", ".join(f"{name}: {v}" for name, v in value.items() if name != k and _short(v))
            yield from _fields(item, f"{path}.{k}", str(k), others[:160])
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _fields(item, f"{path}[{i}]", key, siblings)
    else:
        yield path, key, value, siblings


def _short(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool) and len(str(value)) <= 40


def field_candidates(slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
    """Fields of earlier tool results whose name matches the slot (``order_id`` ← an ``order_id`` or ``orders``
    field): an id a lookup returned is how an agent names a record in the next call. The channel is the
    observation's: ``tool_output``, or ``registry`` for a tool in ``Context.trusted_tools``."""
    name = next((part for part in reversed(slot.path) if part != ITEM), slot.name)
    stem = field_stem(name)
    stems = {stem} | {stem[len(q) :] for q in QUALIFIERS if stem.startswith(q)}
    trusted = set(rc.ctx.trusted_tools)
    out: list[Candidate] = []
    seen: set[str] = set()
    for obs in rc.ctx.all_observations():
        if obs.status != "ok":
            continue
        channel = Channel.REGISTRY if obs.tool in trusted else Channel.TOOL_OUTPUT
        for path, key, value, siblings in _fields(jsonable(obs.content), "$", None):
            if (
                key is None
                or field_stem(key) not in stems
                or isinstance(value, bool)
                or not isinstance(value, (str, int))
            ):
                continue
            text = str(value)
            if not text or text in seen:
                continue
            seen.add(text)
            ref = f"obs:{obs.step}:{path}"
            out.append(Candidate(
                value=text, channel=channel,
                text=f'The "{key}" of {siblings or "an entry"} in the {obs.tool} result (step {obs.step}).',
                prov={"extractor": "field", "mention": {"text": text, "ref": ref}, "field": path, "tool": obs.tool},
            ))  # fmt: skip
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
        out = (span_candidates(slot, rc) + name_part_candidates(slot, rc) + digit_string_candidates(slot, rc)
               + id_format_candidates(slot, rc) + field_candidates(slot, rc) + author_candidates(slot))  # fmt: skip
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

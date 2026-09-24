"""Auto-inference from plain JSON Schema: slot kinds (§3.3.1) and risk tiers (§3.3.2).

``infer_slot`` applies the ordered kind table (first match wins; ``x-jev.kind`` overrides) and fills every other
slot default (stakes, tags, sources, extractors, unit, noun, template packs). ``infer_tier`` applies the ordered tier
rules including the MCP explicit-annotations-only rule and the invitee rule.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from jevtools.candidates import Channel, default_allow_list
from jevtools.policy import Tier
from jevtools.qid import dedupe, sanitize_segment
from jevtools.spec.models import ITEM, Member, SlotSpec, path_key
from jevtools.spec.xjev import ParamXJev
from jevtools.templates import first_sentence, humanize

# --------------------------------------------------------------------------------------------------------------------
# Sources as seen by inference
# --------------------------------------------------------------------------------------------------------------------


@runtime_checkable
class SourceInfo(Protocol):
    """What inference needs from a registered source: its name and the tags it ``provides``."""

    name: str
    provides: Collection[str]


def _source_parts(source: Any) -> tuple[str, frozenset[str]]:
    if isinstance(source, Mapping):
        return str(source["name"]), frozenset(source.get("provides", ()))
    return str(source.name), frozenset(source.provides)


def match_sources(tags: Iterable[str], sources: Iterable[Any], *, name: str = "") -> list[str]:
    """Names of sources whose ``provides`` intersects ``tags``.

    A ``*_id`` name also matches a source named by its prefix (``user_id`` → ``users``).
    """
    wanted = set(tags)
    prefix = name.lower()[:-3] if name.lower().endswith("_id") and len(name) > 3 else None
    matched: list[str] = []
    for source in sources:
        source_name, provides = _source_parts(source)
        by_prefix = prefix is not None and source_name.lower() in (prefix, prefix + "s")
        if (provides & wanted or by_prefix) and source_name not in matched:
            matched.append(source_name)
    return matched


# --------------------------------------------------------------------------------------------------------------------
# Name and schema helpers
# --------------------------------------------------------------------------------------------------------------------

SECRET_NAMES = frozenset({"password", "token", "api_key", "apikey", "secret", "credential"})
TEMPORAL_FORMATS = frozenset({"date-time", "date", "time", "duration"})
TEMPORAL_NAMES = frozenset({"start", "end", "due", "when", "date", "time", "deadline"})
TEMPORAL_SUFFIXES = ("_at", "_date", "_time")
MONEY_WORDS = frozenset({"amount", "price", "cost", "total", "fee", "balance"})
ORDINAL_NAMES = frozenset({"priority", "rating", "severity", "urgency", "importance", "level"})
REF_FORMATS = frozenset({"email", "uri", "uuid", "ipv4", "ipv6", "hostname"})
QUERY_NAMES = frozenset({"query", "q", "search", "keywords", "search_query"})
TITLE_NAMES = frozenset({"subject", "title", "label", "heading", "name"})
BODY_NAMES = frozenset({"body", "message", "content", "text", "description", "note", "comment"})
PLACE_NAMES = frozenset({"city", "place", "location", "country", "address", "destination"})
COSMETIC_STAKES_NAMES = frozenset({"subject", "title", "query", "label"})
CONTENT_STAKES_NAMES = frozenset({"body", "message", "content", "text"})
UNIT_SUFFIXES = {
    "_minutes": "minute", "_mins": "minute", "_hours": "hour", "_seconds": "second", "_secs": "second",
    "_days": "day", "_ms": "millisecond", "_pct": "percent", "_percent": "percent", "_bytes": "byte",
}  # fmt: skip
UNIT_WORDS = {
    "minutes": "minute", "minute": "minute", "hours": "hour", "hour": "hour", "seconds": "second",
    "second": "second", "days": "day", "day": "day", "milliseconds": "millisecond", "percent": "percent",
    "percentage": "percent", "bytes": "byte",
}  # fmt: skip
CATALOG_PATTERNS = (
    ("^[A-Z]{3}$", "currency", "iso4217"),
    ("^[A-Z]{2}$", "country", "iso3166"),
    ("^[a-z]{2}$", "language", "iso639"),
)
IPV6_RE = r"(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}"
HOSTNAME_RE = r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}"
FORMAT_EXTRACTORS = {
    "email": ("email",), "uri": ("url",), "uuid": ("uuid",), "ipv4": ("ipv4",),
    "ipv6": (f"regex:{IPV6_RE}",), "hostname": (f"regex:{HOSTNAME_RE}",),
}  # fmt: skip
EMAIL_PACK_CUES = ("email", "mail", "message")
EVENT_PACK_CUES = ("event", "meeting", "calendar")
MAX_RECORD_DEPTH = 3
MAX_ENUM_OPTIONS = 252

_WORD = re.compile(r"[a-z0-9]+")
_NUMERIC_SAMPLES = ("250", "250.00", "12.5", "0")


def name_tokens(name: str) -> list[str]:
    """Lowercase tokens of an identifier split on ``_``, ``-``, ``.``, spaces and camelCase."""
    return humanize(name).split()


def _words(text: str | None) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _soft_lower(text: str) -> str:
    """Lowercase the first letter unless the first word is an acronym (``URL``)."""
    first = text.split(" ", 1)[0]
    if len(first) > 1 and first.isupper():
        return text
    return text[:1].lower() + text[1:]


def description_phrase(description: str | None) -> str | None:
    """First sentence of a description, first letter lowercased, trailing period removed."""
    if not description or not description.strip():
        return None
    return _soft_lower(first_sentence(description, 400).rstrip(" .")) or None


def default_noun(name: str, description: str | None) -> str:
    """``the`` + description phrase (no doubled article), else ``the`` + humanized name."""
    phrase = description_phrase(description)
    if phrase is None:
        return "the " + humanize(name)
    if phrase.split(" ", 1)[0].lower() in ("the", "a", "an", "your", "this"):
        return phrase
    return "the " + phrase


def default_intent(name: str, description: str | None) -> str:
    """The tool's ``intent``: description phrase, else the humanized tool name."""
    return description_phrase(description) or humanize(name)


def schema_type(schema: Mapping[str, Any]) -> str | None:
    """The primary JSON type (first non-null of a type list)."""
    types = schema.get("type")
    if isinstance(types, str):
        return types
    if isinstance(types, list):
        for t in types:
            if t != "null":
                return str(t)
    return None


def unwrap_nullable(schema: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """``anyOf/oneOf [T, {"type": "null"}]`` (or ``type: [T, "null"]``) → ``(T merged with outer keys, True)``."""
    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if isinstance(branches, list) and len(branches) == 2:
            real = [b for b in branches if not (isinstance(b, Mapping) and b.get("type") == "null")]
            if len(real) == 1 and isinstance(real[0], Mapping):
                merged = {**real[0], **{k: v for k, v in schema.items() if k != key}}
                return merged, True
    types = schema.get("type")
    if isinstance(types, list) and "null" in types:
        rest = [t for t in types if t != "null"]
        return {**schema, "type": rest[0] if len(rest) == 1 else rest}, True
    return dict(schema), False


def _numeric_pattern(pattern: str | None) -> bool:
    if not pattern:
        return False
    try:
        regex = re.compile(pattern)
    except re.error:
        return False
    return any(regex.search(s) for s in _NUMERIC_SAMPLES) and not regex.search("abc")


def _is_const_union(schema: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    for key in ("oneOf", "anyOf"):
        branches = schema.get(key)
        if isinstance(branches, list) and branches and all(isinstance(b, Mapping) and "const" in b for b in branches):
            return branches
    return None


def _object_branches(schema: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    for key in ("oneOf", "anyOf"):
        branches = schema.get(key)
        if (
            isinstance(branches, list)
            and branches
            and all(isinstance(b, Mapping) and (schema_type(b) == "object" or "properties" in b) for b in branches)
        ):
            return branches
    return None


def infer_tags(name: str, schema: Mapping[str, Any]) -> tuple[str, ...]:
    """Tags matched against sources' ``provides``: the format, the name and name-derived tags (§3.3.1 row 11)."""
    lname = name.lower()
    tags: list[str] = []
    fmt = schema.get("format")
    if isinstance(fmt, str):
        tags.append(fmt)
        if fmt == "uri":
            tags.append("url")
    tags.append(lname)
    if lname == "account" or lname.endswith("_account") or lname == "account_id":
        tags += ["account_id", "account"]
    if lname.endswith("_id") and len(lname) > 3:
        tags += [lname[:-3]]
    if lname in ("path", "file", "filename", "filepath") or lname.endswith(("_path", "_file")):
        tags += ["path", "file"]
    return tuple(dedupe_keep(tags))


def dedupe_keep(items: Iterable[str]) -> list[str]:
    """Drop repeated items, keeping the first occurrence."""
    seen: set[str] = set()
    return [i for i in items if not (i in seen or seen.add(i))]  # type: ignore[func-returns-value]


def _unit_of(name: str, description: str | None) -> str | None:
    lname = name.lower()
    for suffix, unit in UNIT_SUFFIXES.items():
        if lname.endswith(suffix):
            return unit
    for word in _WORD.findall((description or "").lower()):
        if word in UNIT_WORDS:
            return UNIT_WORDS[word]
    return None


def _catalog_of(schema: Mapping[str, Any], name: str, description: str | None) -> str | None:
    pattern = schema.get("pattern")
    words = set(name_tokens(name)) | _words(description)
    for catalog_pattern, cue, catalog in CATALOG_PATTERNS:
        if pattern == catalog_pattern and cue in words:
            return catalog
    return None


def _members(schema: Mapping[str, Any], declared: Any) -> tuple[Member, ...] | None:
    members: list[Member] | None = None
    if "enum" in schema:
        members = [Member(value=v) for v in schema["enum"]]
    elif (branches := _is_const_union(schema)) is not None:
        members = [Member(value=b["const"], text=b.get("title") or b.get("description")) for b in branches]
    if isinstance(declared, list):
        texts = {repr(d["value"]): d for d in declared if isinstance(d, Mapping) and "value" in d}
        if members is None:
            members = [
                Member(value=d["value"], text=d.get("text") or d.get("description"))
                if isinstance(d, Mapping) and "value" in d
                else Member(value=d)
                for d in declared
            ]
        else:
            members = [
                m.model_copy(
                    update={"text": texts[repr(m.value)].get("text") or texts[repr(m.value)].get("description")}
                )
                if repr(m.value) in texts
                else m
                for m in members
            ]
    return tuple(members) if members is not None else None


# --------------------------------------------------------------------------------------------------------------------
# Slot kind (§3.3.1)
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class _Kind:
    kind: str
    reason: str
    role: str | None = None
    weak: bool = False
    catalog: str | None = None
    source: list[str] = field(default_factory=list)
    extract: tuple[str, ...] = ()
    unit: str | None = None


def infer_kind(
    name: str,
    schema: Mapping[str, Any],
    *,
    siblings: Collection[str] = (),
    sources: Sequence[Any] = (),
    depth: int = 1,
) -> _Kind:
    """Apply rows 1–18 of the §3.3.1 table to one (nullable-unwrapped, ``$ref``-inlined) schema."""
    lname = name.lower()
    tokens = set(name_tokens(name))
    description = schema.get("description")
    words = tokens | _words(description)
    stype = schema_type(schema)
    fmt = schema.get("format")
    structured = "properties" in schema or "items" in schema or _object_branches(schema) is not None
    string_like = stype == "string" or (stype is None and not structured)
    numeric = stype in ("number", "integer")

    if schema.get("readOnly") is True:
        return _Kind("derived", "row 1: readOnly")
    if schema.get("writeOnly") is True or lname in SECRET_NAMES:
        return _Kind("secret", "row 1: writeOnly or secret name")
    if "const" in schema:
        return _Kind("derived", "row 2: const")
    if "enum" in schema or _is_const_union(schema) is not None:
        return _Kind("enum", "row 3: enum")
    if stype == "boolean":
        return _Kind("flag", "row 4: boolean")
    if fmt in TEMPORAL_FORMATS or (string_like and (lname in TEMPORAL_NAMES or lname.endswith(TEMPORAL_SUFFIXES))):
        extract = ("duration",) if fmt == "duration" else ("datetime",)
        return _Kind("temporal", f"row 5: {'format ' + fmt if fmt in TEMPORAL_FORMATS else 'name'}", extract=extract)
    numeric_string = string_like and (fmt == "decimal" or _numeric_pattern(schema.get("pattern")))
    has_currency_sibling = any(s.lower() == "currency" or s.lower().endswith("_currency") for s in siblings)
    if (numeric or numeric_string) and (words & MONEY_WORDS or has_currency_sibling):
        return _Kind("money", "row 6: money", extract=("money",), unit="money")
    if stype == "integer" and tokens & ORDINAL_NAMES and _small_range(schema):
        return _Kind("ordinal", "row 7: bounded ordinal")
    if numeric:
        return _Kind("quantity", "row 8: number", extract=("number",), unit=_unit_of(name, description))
    if fmt in REF_FORMATS:
        tags = [fmt, "url"] if fmt == "uri" else [fmt]
        matched = match_sources(tags, sources)
        if matched:
            return _Kind("ref", f"row 9: format {fmt} + source {', '.join(matched)}", source=matched)
        return _Kind("span", f"row 9: format {fmt}", role=fmt, extract=FORMAT_EXTRACTORS[fmt])
    catalog = _catalog_of(schema, name, description) if string_like else None
    if catalog is not None:
        return _Kind("enum", f"row 10: catalog {catalog}", catalog=catalog)
    if string_like:
        matched = match_sources(infer_tags(name, schema), sources, name=name)
        if matched:
            return _Kind("ref", f"row 11: tag match {', '.join(matched)}", source=matched)
    if stype == "array" or (stype is None and "items" in schema):
        return _Kind("list", "row 12: array")
    if stype == "object" or "properties" in schema or _object_branches(schema) is not None:
        if _object_branches(schema) is not None:
            return _Kind("union", "row 13: oneOf/anyOf of objects")
        if not schema.get("properties") or depth > MAX_RECORD_DEPTH:
            return _Kind("record", "row 13: object without flattenable properties", weak=True)
        return _Kind("record", "row 13: object")
    if lname in QUERY_NAMES:
        return _Kind("text", "row 14: query name", role="query", extract=("clause", "noun_phrase"))
    if lname in TITLE_NAMES:
        return _Kind("text", "row 15: title name", role="title", extract=("clause", "quote"))
    max_length = schema.get("maxLength")
    if lname in BODY_NAMES or (isinstance(max_length, int) and max_length > 200):
        return _Kind("text", "row 16: body name or maxLength > 200", role="body", extract=("clause", "quote"))
    if lname in PLACE_NAMES:
        return _Kind("span", "row 17: place name", role="place", extract=("place",))
    return _Kind(
        "span", "row 18: generic string", role="generic", weak=True, extract=("clause", "quote", "noun_phrase")
    )


def _small_range(schema: Mapping[str, Any]) -> bool:
    low = schema.get("minimum", schema.get("exclusiveMinimum"))
    high = schema.get("maximum", schema.get("exclusiveMaximum"))
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return False
    low = low if "minimum" in schema else low + 1
    high = high if "maximum" in schema else high - 1
    return 0 < high - low + 1 <= 11


def default_stakes(kind: str, name: str, role: str | None) -> str:
    """Stakes default (§3.2): cosmetic/content by name (and text role), else identity."""
    lname = name.lower()
    if kind == "text":
        if role in ("query", "title") or lname in COSMETIC_STAKES_NAMES:
            return "cosmetic"
        return "content"
    if lname in COSMETIC_STAKES_NAMES:
        return "cosmetic"
    if lname in CONTENT_STAKES_NAMES:
        return "content"
    return "identity"


def auto_packs(kind: str, role: str | None, name: str, tool_name: str, tool_description: str) -> tuple[str, ...]:
    """Template packs attached by tool name/description (§3.3): email.* for mail tools, event.title for events."""
    if kind != "text":
        return ()
    cues = set(name_tokens(tool_name)) | _words(tool_description)
    lname = name.lower()
    if cues & set(EMAIL_PACK_CUES):
        if lname == "subject":
            return ("email.subject",)
        if role == "body":
            return ("email.body", "email.forward")
    if cues & set(EVENT_PACK_CUES) and lname == "title":
        return ("event.title",)
    return ()


# --------------------------------------------------------------------------------------------------------------------
# infer_slot
# --------------------------------------------------------------------------------------------------------------------


def infer_slot(
    name: str,
    schema: Mapping[str, Any],
    *,
    path: tuple[str, ...] | None = None,
    qpath: str | None = None,
    required: bool = False,
    parent: Mapping[str, Any] | None = None,
    xjevs: Mapping[str, ParamXJev] | None = None,
    sources: Sequence[Any] = (),
    tool_name: str = "",
    tool_description: str = "",
    depth: int = 1,
) -> SlotSpec:
    """Compile one parameter into a :class:`SlotSpec` (channels are filled in later by :func:`with_channels`).

    ``schema`` must already be ``$ref``-inlined and ``x-jev``-stripped; ``xjevs`` maps path keys
    (``from_account``, ``attendees[].email``) to their merged declarations; ``parent`` is the enclosing object
    schema (for siblings such as ``currency``).
    """
    path = path or (name,)
    xjevs = xjevs or {}
    x = xjevs.get(path_key(path), ParamXJev())
    schema, nullable = unwrap_nullable(schema)
    siblings = tuple((parent or {}).get("properties", {}).keys())
    inferred = infer_kind(name, schema, siblings=siblings, sources=sources, depth=depth)
    kind, reason = inferred.kind, inferred.reason
    catalog = inferred.catalog
    if isinstance(x.values, str):
        catalog = x.values
        if x.kind is None:
            kind, reason = "enum", f"x-jev.values catalog {catalog}"
    if x.source is not None and x.kind is None and kind == "span":
        kind, reason = "ref", "x-jev.source"
    if x.kind is not None:
        kind, reason = x.kind, "x-jev.kind"
    description = schema.get("description")
    role = inferred.role
    has_default = "default" in schema or "const" in schema
    qpath = qpath or sanitize_segment(name)
    tags = tuple(x.tags) if x.tags is not None else infer_tags(name, schema)
    source = _slot_source(kind, x, inferred, tags, sources, name)
    if kind == "list" and x.source is not None:
        xjevs = _with_item_source(xjevs, path, x)
    packs = (
        (x.templates,) if isinstance(x.templates, str) else auto_packs(kind, role, name, tool_name, tool_description)
    )
    item, children, branches = _nested(
        kind, name, schema, path, qpath, xjevs, sources, tool_name, tool_description, depth
    )
    probe = x.probe
    return SlotSpec(
        path=path,
        name=name,
        qpath=qpath,
        json_schema=schema,
        description=description,
        title=schema.get("title"),
        format=schema.get("format"),
        kind=kind,
        kind_reason=reason,
        role=role,
        weak=inferred.weak and x.kind is None,
        stakes=x.stakes or default_stakes(kind, name, role),
        required=required,
        nullable=nullable,
        has_default=has_default,
        default=schema.get("default", schema.get("const")),
        default_from=x.default_from,
        source=source,
        tags=tags,
        extract=tuple(x.extract) if x.extract is not None else inferred.extract,
        values=_members(schema, x.values),
        catalog=catalog,
        templates=tuple(x.templates) if isinstance(x.templates, list) else (),
        packs=packs,
        derive=tuple(x.derive or ()),
        order_by=dict(x.order_by or {}),
        k=x.k,
        widen=tuple(x.widen) if x.widen is not None else None,
        hierarchy=x.hierarchy,
        unit=x.unit or inferred.unit,
        range=dict(x.range) if x.range else None,
        ask=x.ask,
        noun=x.noun or default_noun(name, description),
        fallback=x.fallback,
        probe_present=probe.present if probe else None,
        probe_reverse=probe.reverse if probe else None,
        speculate=True if x.speculate is None else x.speculate,
        anchored=x.anchored if x.anchored is not None else bool(item is not None and item.kind == "ref"),
        tolerant=bool(x.tolerant),
        canon=x.canon,
        item=item,
        children=children,
        branches=branches,
        xjev=x,
    )


def _slot_source(
    kind: str, x: ParamXJev, inferred: _Kind, tags: Sequence[str], sources: Sequence[Any], name: str
) -> Any:
    if x.source is not None:
        return x.source
    if kind != "ref":
        return None
    matched = inferred.source or match_sources(tags, sources, name=name)
    if not matched:
        return None
    return matched[0] if len(matched) == 1 else matched


def _with_item_source(xjevs: Mapping[str, ParamXJev], path: tuple[str, ...], x: ParamXJev) -> dict[str, ParamXJev]:
    """A source declared on a list slot applies to its items unless the items declare their own."""
    key = path_key((*path, ITEM))
    item = xjevs.get(key, ParamXJev())
    merged = dict(xjevs)
    if item.source is None:
        merged[key] = item.model_copy(update={"source": x.source})
    return merged


def _nested(
    kind: str,
    name: str,
    schema: Mapping[str, Any],
    path: tuple[str, ...],
    qpath: str,
    xjevs: Mapping[str, ParamXJev],
    sources: Sequence[Any],
    tool_name: str,
    tool_description: str,
    depth: int,
) -> tuple[SlotSpec | None, tuple[SlotSpec, ...], tuple[SlotSpec, ...]]:
    common: dict[str, Any] = {
        "xjevs": xjevs, "sources": sources, "tool_name": tool_name, "tool_description": tool_description,
    }  # fmt: skip
    if kind == "list":
        items = schema.get("items")
        if isinstance(items, Mapping):
            item = infer_slot(name, items, path=(*path, ITEM), qpath=qpath, required=True, depth=depth + 1, **common)
            return item, (), ()
        return None, (), ()
    if kind == "record" and depth <= MAX_RECORD_DEPTH:
        return None, record_children(schema, path, qpath, depth=depth + 1, **common), ()
    if kind == "union":
        branches = []
        for i, branch in enumerate(_object_branches(schema) or []):
            title = branch.get("title") or f"option {i + 1}"
            branch_qpath = f"{qpath}.b{i}"
            children = record_children(branch, path, branch_qpath, depth=depth + 1, **common)
            branches.append(
                SlotSpec(
                    path=path,
                    name=str(title),
                    qpath=branch_qpath,
                    json_schema=dict(branch),
                    description=branch.get("description"),
                    title=branch.get("title"),
                    kind="record",
                    kind_reason="union branch",
                    noun=default_noun(str(title), branch.get("description")),
                    children=children,
                )  # fmt: skip
            )
        return None, (), tuple(branches)
    return None, (), ()


def record_children(
    schema: Mapping[str, Any],
    path: tuple[str, ...],
    qpath: str,
    *,
    depth: int,
    xjevs: Mapping[str, ParamXJev],
    sources: Sequence[Any],
    tool_name: str,
    tool_description: str,
) -> tuple[SlotSpec, ...]:
    """Slot specs of an object's properties (in schema order), with collision-free sanitized qpaths."""
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or ())
    names = list(properties)
    segments = dedupe(sanitize_segment(n) for n in names)
    return tuple(
        infer_slot(
            child,
            properties[child],
            path=(*path, child),
            qpath=f"{qpath}.{segment}" if qpath else segment,
            required=child in required,
            parent=schema,
            xjevs=xjevs,
            sources=sources,
            tool_name=tool_name,
            tool_description=tool_description,
            depth=depth,
        )
        for child, segment in zip(names, segments, strict=True)
    )


# --------------------------------------------------------------------------------------------------------------------
# Channels (§3.4.2)
# --------------------------------------------------------------------------------------------------------------------


def with_channels(slot: SlotSpec, tier: Tier) -> SlotSpec:
    """Fill the allow-list of ``slot`` and its nested specs.

    Declared ``x-jev.channels`` win; otherwise the tier × stakes default of §3.4.2 applies.
    """
    declared = slot.xjev.channels
    channels = (
        tuple(Channel(c) for c in declared)
        if declared is not None
        else default_allow_list(
            tier, slot.stakes, quantity=slot.is_quantity or (slot.item is not None and slot.item.is_quantity)
        )
    )
    return slot.model_copy(
        update={
            "channels": channels,
            "item": with_channels(slot.item, tier) if slot.item is not None else None,
            "children": tuple(with_channels(c, tier) for c in slot.children),
            "branches": tuple(with_channels(b, tier) for b in slot.branches),
        }
    )


# --------------------------------------------------------------------------------------------------------------------
# Risk tier (§3.3.2)
# --------------------------------------------------------------------------------------------------------------------

VERB_TIERS: dict[str, Tier] = {
    **dict.fromkeys(
        ("get", "read", "list", "search", "find", "fetch", "lookup", "query", "describe", "show"), Tier.READ
    ),
    **dict.fromkeys(
        ("send", "email", "post", "publish", "share", "forward", "invite", "notify", "reply", "message"), Tier.EXTERNAL
    ),
    **dict.fromkeys(
        ("transfer", "pay", "wire", "refund", "purchase", "buy", "delete", "remove", "drop", "destroy", "revoke"),
        Tier.CRITICAL,
    ),
    **dict.fromkeys(
        ("create", "add", "update", "set", "book", "schedule", "save", "write", "rename", "move"), Tier.WRITE
    ),
}
PEOPLE_TAGS = frozenset({"email", "person"})


def tool_verb(name: str) -> str:
    """The first ``_``/``-``/camelCase token of a tool name, lowercased."""
    tokens = name_tokens(name)
    return tokens[0] if tokens else ""


def explicit_annotations(annotations: Mapping[str, Any] | None) -> dict[str, bool]:
    """MCP hint annotations that are explicitly present (``None``/absent values are ignored, §3.3.2)."""
    hints = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")
    return {k: bool(annotations[k]) for k in hints if annotations and annotations.get(k) is not None}


def _over_people(slot: SlotSpec, sources: Sequence[Any]) -> bool:
    people_sources = {n for n, provides in map(_source_parts, sources) if provides & PEOPLE_TAGS}
    for spec in slot.walk():
        if spec.kind in ("ref", "span") and (spec.format == "email" or set(spec.source_names) & people_sources):
            return True
    return False


def infer_tier(
    name: str,
    *,
    risk: Tier | str | None = None,
    annotations: Mapping[str, Any] | None = None,
    slots: Sequence[SlotSpec] = (),
    sources: Sequence[Any] = (),
) -> tuple[Tier, str]:
    """Risk tier and the reason (§3.3.2): x-jev.risk > explicit MCP hints > verb (+ invitee rule) > external."""
    if risk is not None:
        return Tier(risk), "x-jev.risk"
    hints = explicit_annotations(annotations)
    if hints.get("readOnlyHint"):
        return Tier.READ, "annotation readOnlyHint"
    if hints.get("destructiveHint"):
        return Tier.CRITICAL, "annotation destructiveHint"
    if hints.get("openWorldHint"):
        return Tier.EXTERNAL, "annotation openWorldHint"
    verb = tool_verb(name)
    tier = VERB_TIERS.get(verb)
    if tier is Tier.WRITE and any(_over_people(slot, sources) for slot in slots):
        return Tier.EXTERNAL, f"verb '{verb}' + invitee rule"
    if tier is not None:
        return tier, f"verb '{verb}'"
    return Tier.EXTERNAL, "fail-safe default (declare x-jev.risk)"


__all__ = [
    "SourceInfo",
    "VERB_TIERS",
    "auto_packs",
    "default_intent",
    "default_noun",
    "default_stakes",
    "description_phrase",
    "explicit_annotations",
    "infer_kind",
    "infer_slot",
    "infer_tags",
    "infer_tier",
    "match_sources",
    "name_tokens",
    "record_children",
    "schema_type",
    "tool_verb",
    "unwrap_nullable",
    "with_channels",
]

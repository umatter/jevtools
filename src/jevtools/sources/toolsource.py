"""``ToolSource``: a catalog tool as a candidate source, called once per session and cached (spec §3.2, §4.4).

``x-jev.source`` may name a tool instead of a registered source::

    {"tool": "list_contacts", "args": {}, "items": "$.contacts[*]", "key": "email", "label": "{name} <{email}>",
     "ttl": 300}

The tool is app-owned, so its rows carry the ``registry`` channel (unlike observations, which are ``tool_output``).
The tool is called at most once per ``ttl`` seconds (the "session"); its result is cut into rows by ``items`` (a
JSONPath subset, :func:`jsonpath`), and every lookup — anchors, candidates, rankings, TOCTOU re-resolution — is
delegated to a :class:`~jevtools.sources.registry.Registry` over those rows. A ``ToolSource`` registers under the
name ``tool:<tool>`` by default, which is the name a slot's inline ``{"tool": …}`` source resolves to
(:func:`jevtools.spec.models.source_spec_name`).

The source needs a *caller* ``call(tool, args) -> result`` (sync or async): pass it as ``call=`` or let
:class:`jevtools.loop.Agent` bind its executors (:meth:`ToolSource.bind`). A failed call fails closed: the source
has no rows (the slot becomes empty → clarify) and :attr:`ToolSource.last_error` says why.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any

from jevtools.candidates import Candidate, Channel
from jevtools.canonical import jsonable, sha256_of
from jevtools.extract.base import Mention
from jevtools.extract.tokens import Token
from jevtools.sources.base import SourceQuery, item_noun
from jevtools.sources.registry import Registry, Retriever, Template

ToolCaller = Callable[[str, dict[str, Any]], Any]
"""``call(tool, args) -> result`` (a result, or an awaitable of one)."""
Clock = Callable[[], float]

DEFAULT_TTL = 300.0

# --------------------------------------------------------------------------------------------------------------------
# JSONPath subset
# --------------------------------------------------------------------------------------------------------------------

_SEGMENT = re.compile(
    r"""\.(?P<name>[A-Za-z_$][\w$-]*)      # .name
      | \.\*                              # .*
      | \[\s*\*\s*\]                      # [*]
      | \[\s*(?P<index>-?\d+)\s*\]        # [n]
      | \[\s*'(?P<sq>[^']*)'\s*\]         # ['name']
      | \[\s*"(?P<dq>[^"]*)"\s*\]         # ["name"]
    """,
    re.VERBOSE,
)


def parse_jsonpath(path: str) -> list[str | int | None]:
    """Parse the supported JSONPath subset into segments: a key (``str``), an index (``int``) or ``None`` for a
    wildcard. Supported: ``$``, ``.name``, ``['name']``, ``["name"]``, ``[n]`` (negative allowed), ``[*]`` and
    ``.*``. Raises ``ValueError`` for anything else (filters, slices, recursive descent)."""
    text = path.strip()
    if not text.startswith("$"):
        raise ValueError(f"JSONPath must start with '$': {path!r}")
    segments: list[str | int | None] = []
    position = 1
    while position < len(text):
        match = _SEGMENT.match(text, position)
        if match is None:
            raise ValueError(f"unsupported JSONPath syntax at {text[position:]!r} in {path!r}")
        if match.group("name") is not None:
            segments.append(match.group("name"))
        elif match.group("index") is not None:
            segments.append(int(match.group("index")))
        elif match.group("sq") is not None:
            segments.append(match.group("sq"))
        elif match.group("dq") is not None:
            segments.append(match.group("dq"))
        else:
            segments.append(None)
        position = match.end()
    return segments


def child_path(parent: str, key: str) -> str:
    """``parent.key`` (or ``parent["key"]`` when ``key`` is not an identifier): JSONPath of an object member."""
    return parent + _key_path(key)


def _key_path(key: str) -> str:
    return f".{key}" if re.fullmatch(r"[A-Za-z_$][\w$-]*", key) else f"[{json.dumps(key, ensure_ascii=False)}]"


def jsonpath_matches(data: Any, path: str) -> list[tuple[str, Any]]:
    """Every ``(concrete path, value)`` that ``path`` selects in ``data`` (in document order). A wildcard over an
    object walks its values; a missing key or index selects nothing."""
    current: list[tuple[str, Any]] = [("$", data)]
    for segment in parse_jsonpath(path):
        found: list[tuple[str, Any]] = []
        for where, value in current:
            if segment is None:
                if isinstance(value, list):
                    found += [(f"{where}[{i}]", item) for i, item in enumerate(value)]
                elif isinstance(value, Mapping):
                    found += [(where + _key_path(str(k)), item) for k, item in value.items()]
            elif isinstance(segment, int):
                if isinstance(value, list) and -len(value) <= segment < len(value):
                    index = segment % len(value)
                    found.append((f"{where}[{index}]", value[index]))
            elif isinstance(value, Mapping) and segment in value:
                found.append((where + _key_path(segment), value[segment]))
        current = found
    return current


def jsonpath(data: Any, path: str) -> list[Any]:
    """The values ``path`` selects in ``data`` (see :func:`jsonpath_matches`)."""
    return [value for _, value in jsonpath_matches(data, path)]


# --------------------------------------------------------------------------------------------------------------------
# Tool results
# --------------------------------------------------------------------------------------------------------------------


def get_any(obj: Any, *names: str) -> Any:
    """The first of ``names`` present on ``obj``: a mapping key or an attribute (duck-typed MCP objects, which come
    as ``mcp`` pydantic models or plain dicts, camelCase or snake_case); ``None`` when none is."""
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if not isinstance(obj, Mapping) and hasattr(obj, name):
            return getattr(obj, name)
    return None


def is_mcp_result(result: Any) -> bool:
    """Whether ``result`` looks like an MCP ``CallToolResult`` (``content`` blocks plus ``isError``)."""
    if isinstance(result, Mapping):
        return isinstance(result.get("content"), list) and ("isError" in result or "is_error" in result)
    return hasattr(result, "content") and (hasattr(result, "isError") or hasattr(result, "is_error"))


def mcp_text(result: Any) -> str:
    """The text blocks of an MCP result joined by newlines."""
    parts: list[str] = []
    for block in get_any(result, "content") or ():
        if get_any(block, "type") == "text" and isinstance(get_any(block, "text"), str):
            parts.append(str(get_any(block, "text")))
    return "\n".join(parts)


def result_data(result: Any) -> Any:
    """JSON-native data of a tool result: an MCP result's ``structuredContent`` (else its text, parsed as JSON when
    it is JSON), a JSON string parsed, pydantic models and dataclasses dumped; anything else as is."""
    if is_mcp_result(result):
        structured = get_any(result, "structuredContent", "structured_content")
        if structured is not None:
            return jsonable(structured)
        result = mcp_text(result)
    if isinstance(result, (bytes, bytearray)):
        result = bytes(result).decode("utf-8", errors="replace")
    if isinstance(result, str):
        stripped = result.strip()
        if stripped[:1] in ("{", "["):
            try:
                return json.loads(stripped)
            except ValueError:
                return result
        return result
    try:
        return jsonable(result)
    except TypeError:
        return str(result)


def _await_now(result: Any) -> Any:
    """Resolve an awaitable outside an event loop; inside one, refuse (use the async variant)."""
    if not inspect.isawaitable(result):
        return result
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_wrap(result))
    if inspect.iscoroutine(result):
        result.close()
    raise RuntimeError("an async caller cannot be awaited from sync code inside an event loop; use arefresh()")


async def _wrap(result: Any) -> Any:
    return await result


# --------------------------------------------------------------------------------------------------------------------
# ToolSource
# --------------------------------------------------------------------------------------------------------------------


class ToolSource:
    """A catalog tool called once per session (cached for ``ttl`` seconds) whose result rows are candidates.

    ``items`` selects the rows (JSONPath subset, default ``$[*]``; scalar rows become ``{"value": x}``), ``key`` is
    the value field (default ``value`` for scalars, else ``id``), ``label``/``describe`` are templates, and the
    remaining keywords are those of :class:`~jevtools.sources.registry.Registry`. ``clock`` returns seconds
    (monotonic by default); tests pass a fake one.
    """

    def __init__(
        self,
        tool: str,
        args: Mapping[str, Any] | None = None,
        items: str = "$[*]",
        key: str | None = None,
        label: Template | None = None,
        ttl: float = DEFAULT_TTL,
        *,
        name: str | None = None,
        call: ToolCaller | None = None,
        describe: Template | None = None,
        match: Sequence[str] = (),
        provides: Collection[str] = (),
        attrs: Sequence[str] = (),
        retriever: Retriever = "fuzzy",
        send_whole_if_under: int = 12,
        synonyms: Mapping[str, Sequence[str]] | None = None,
        channel: Channel | str = Channel.REGISTRY,
        item: str | None = None,
        recency: str | None = None,
        hierarchy: str | None = None,
        groups: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        parse_jsonpath(items)  # fail early on unsupported syntax
        self.tool = tool
        self.args: dict[str, Any] = dict(args or {})
        self.items = items
        self.key = key
        self.label = label
        self.ttl = float(ttl)
        self.name = name or f"tool:{tool}"
        self.describe = describe
        self.match = tuple(match)
        self.provides: frozenset[str] = frozenset(provides)
        self.attrs = tuple(attrs)
        self.retriever = retriever
        self.send_whole_if_under = send_whole_if_under
        self.synonyms = synonyms
        self.channel = Channel(channel)
        self.item = item or _item_of(tool)
        self.recency = recency
        self.hierarchy = hierarchy
        self.groups = groups
        self.clock: Clock = clock or time.monotonic
        self._call = call
        self._registry: Registry | None = None
        self._fetched_at: float | None = None
        self.calls = 0
        """How many times the tool was actually called."""
        self.last_error: str | None = None

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], *, call: ToolCaller | None = None, **kw: Any) -> ToolSource:
        """A source from an inline ``x-jev.source`` object ``{"tool", "args", "items", "key", "label", "ttl", …}``
        (``name``, ``describe``, ``match``, ``provides``, ``attrs`` and ``item`` are accepted too)."""
        known = {"args", "items", "key", "label", "ttl", "name", "describe", "match", "provides", "attrs", "item",
                 "recency", "hierarchy", "groups"}  # fmt: skip
        unknown = set(spec) - known - {"tool"}
        if unknown:
            raise ValueError(f"unknown tool-source keys {sorted(unknown)}")
        options = {k: v for k, v in spec.items() if k in known}
        return cls(str(spec["tool"]), call=call, **{**options, **kw})

    def __repr__(self) -> str:
        return f"ToolSource({self.name!r}, tool={self.tool!r}, items={self.items!r})"

    # -- caller and cache -----------------------------------------------------------------------------------------

    @property
    def bound(self) -> bool:
        """Whether a caller is set."""
        return self._call is not None

    def bind(self, call: ToolCaller, *, replace: bool = False) -> ToolSource:
        """Set the caller (kept if one is already set, unless ``replace``)."""
        if self._call is None or replace:
            self._call = call
        return self

    @property
    def stale(self) -> bool:
        """Never fetched, or fetched more than ``ttl`` seconds ago."""
        return self._fetched_at is None or self.clock() - self._fetched_at >= self.ttl

    def invalidate(self) -> None:
        """Forget the cached rows (the next use calls the tool again)."""
        self._fetched_at = None

    def refresh(self, *, force: bool = False) -> None:
        """Call the tool now if the cache is stale (sync; an async caller runs in a fresh event loop)."""
        if not (force or self.stale):
            return
        if self._call is None:
            self._fail("no caller bound")
            return
        try:
            result = _await_now(self._call(self.tool, dict(self.args)))
        except Exception as exc:  # noqa: BLE001 - a failing source fails closed
            self._fail(f"{type(exc).__name__}: {exc}")
            return
        self._store(result)

    async def arefresh(self, *, force: bool = False) -> None:
        """Async :meth:`refresh` (awaits an async caller)."""
        if not (force or self.stale):
            return
        if self._call is None:
            self._fail("no caller bound")
            return
        try:
            result = self._call(self.tool, dict(self.args))
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001 - a failing source fails closed
            self._fail(f"{type(exc).__name__}: {exc}")
            return
        self._store(result)

    def _fail(self, error: str) -> None:
        self.last_error = error
        if self._registry is None:
            self._registry = self._build([])
        self._fetched_at = self.clock()

    def _store(self, result: Any) -> None:
        self.calls += 1
        if is_mcp_result(result) and bool(get_any(result, "isError", "is_error")):
            self._fail(f"tool error: {mcp_text(result)[:200]}")
            return
        rows = self.rows_of(result)
        self._registry = self._build(rows)
        self._fetched_at = self.clock()
        self.last_error = None

    def rows_of(self, result: Any) -> list[dict[str, Any]]:
        """The rows ``items`` selects in a tool result (scalars wrapped as ``{"value": x}``; rows without a key
        value are dropped)."""
        data = result_data(result)
        rows = [dict(v) if isinstance(v, Mapping) else {"value": v} for v in jsonpath(data, self.items)]
        key = self._key_for(rows)
        return [row for row in rows if row.get(key) not in (None, "")]

    def _key_for(self, rows: Sequence[Mapping[str, Any]]) -> str:
        """``key``; without one, ``value`` for scalar rows, else ``id``."""
        if self.key:
            return self.key
        return "value" if rows and all(set(r) == {"value"} for r in rows) else "id"

    def _build(self, rows: Iterable[Mapping[str, Any]]) -> Registry:
        rows = list(rows)
        key = self._key_for(rows)
        return Registry(
            self.name, rows, key=key, label=self.label, describe=self.describe, match=self.match or (key,),
            provides=self.provides, attrs=self.attrs, retriever=self.retriever,
            send_whole_if_under=self.send_whole_if_under, synonyms=self.synonyms, channel=self.channel,
            item=self.item, recency=self.recency, hierarchy=self.hierarchy, groups=self.groups,
        )  # fmt: skip

    @property
    def registry(self) -> Registry:
        """The registry over the current rows (fetching them first when stale)."""
        if self.stale:
            self.refresh()
        assert self._registry is not None
        return self._registry

    @property
    def rows(self) -> tuple[Mapping[str, Any], ...]:
        """The current rows."""
        return self.registry.rows

    # -- Source protocol (delegated) ------------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.registry)

    @property
    def whole(self) -> bool:
        return self.registry.whole

    def candidates(self, q: SourceQuery) -> list[Candidate]:
        """Candidates of the current rows (see :meth:`Registry.candidates`)."""
        return self.registry.candidates(q)

    def ranked(self, q: SourceQuery) -> list[Candidate]:
        """The full ranking (widen rounds)."""
        return self.registry.ranked(q)

    def find_anchors(
        self,
        text: str,
        *,
        tokens: Sequence[Token] | None = None,
        source_ref: str = "request",
        channel: Channel = Channel.USER,
    ) -> list[Mention]:
        """Anchor mentions of the rows in one user text."""
        return self.registry.find_anchors(text, tokens=tokens, source_ref=source_ref, channel=channel)

    def group_of(self, candidate: Candidate) -> str | None:
        return self.registry.group_of(candidate)

    def lookup(self, key: Any) -> Mapping[str, Any] | None:
        """TOCTOU re-resolution: the row whose key is ``key``."""
        return self.registry.lookup(key)

    def attribute_names(self) -> frozenset[str]:
        return self.registry.attribute_names()

    def content_sha256(self) -> str:
        """Hash of the tool declaration and the current rows."""
        return sha256_of({"tool": self.tool, "args": jsonable(self.args), "items": self.items,
                          "rows": [dict(r) for r in self.registry.rows]})  # fmt: skip


_LIST_VERBS = frozenset({"list", "get", "search", "find", "fetch", "lookup", "query", "read", "show", "all"})


def _item_of(tool: str) -> str:
    """Default item noun of a tool's rows: the tool name without its verb, singularized (``list_contacts`` →
    ``contact``)."""
    parts = [p for p in re.split(r"[_\-\s]+|(?<=[a-z])(?=[A-Z])", tool) if p]
    while len(parts) > 1 and parts[0].lower() in _LIST_VERBS:
        parts = parts[1:]
    return item_noun("_".join(p.lower() for p in parts)) if parts else "item"


def tool_sources(catalog: Any, *, call: ToolCaller | None = None) -> list[ToolSource]:
    """One :class:`ToolSource` per distinct inline ``{"tool": …}`` source declared by the catalog's slots (deduped by
    registration name; the first declaration wins). Register them in the context's ``sources``."""
    found: dict[str, ToolSource] = {}
    for tool in getattr(catalog, "tools", ()):
        for slot in tool.walk():
            spec = slot.source
            for entry in spec if isinstance(spec, list) else [spec]:
                if isinstance(entry, Mapping) and isinstance(entry.get("tool"), str):
                    source = ToolSource.from_spec(entry, call=call)
                    found.setdefault(source.name, source)
    return list(found.values())


__all__ = [
    "DEFAULT_TTL",
    "ToolCaller",
    "ToolSource",
    "child_path",
    "get_any",
    "is_mcp_result",
    "jsonpath",
    "jsonpath_matches",
    "mcp_text",
    "parse_jsonpath",
    "result_data",
    "tool_sources",
]

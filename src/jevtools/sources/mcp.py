"""``MCPResources``: an MCP server's resources as candidates (spec §3.2 ``{"mcp_resources": "uri-template"}``, §4.4).

``jt.sources.MCPResources(session, uri_template=None)`` works over any object shaped like an MCP ``ClientSession``
(no dependency on the ``mcp`` package):

- ``resources/list`` (``session.list_resources()``, paginated by ``nextCursor``): every listed resource is a row
  ``{uri, name, title, description, mime_type, size}``. With a ``uri_template`` only resources whose URI matches
  the template are kept, and the template's variables are added to the row (``file:///{path}`` → ``path``).
- ``resources/templates/list`` (``session.list_resource_templates()``): a template's variables cannot be listed, so
  when the session offers ``completion/complete`` (``session.complete``), a single-variable template is expanded
  with the server's completion values for that variable (one call per refresh, empty prefix).

The value of a candidate is the resource URI (the registry key); its label is the URI when it fits the 64-character
label limit, else the resource name. Rows are app-owned (channel ``registry``) and cached for ``ttl`` seconds.
Lookups are delegated to a :class:`~jevtools.sources.registry.Registry`, so anchors (the user naming a resource),
rankings and TOCTOU re-resolution behave as for any registry. The default registration name is
``mcp:<uri template>`` (``mcp_resources`` without one), which is what a slot's inline ``{"mcp_resources": …}``
source resolves to.

MCP sessions are async. Call ``await source.arefresh()`` before deciding in async code (:class:`jevtools.loop.Agent`
does it before every step); a sync :meth:`MCPResources.refresh` runs async session methods in a fresh event loop,
which suits test doubles and sync wrappers but not a session bound to a running loop. A failing server fails
closed: no rows, and :attr:`MCPResources.last_error` says why.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from collections.abc import Callable, Collection, Generator, Mapping, Sequence
from typing import Any
from urllib.parse import quote

from jevtools.candidates import Candidate, Channel
from jevtools.canonical import jsonable, sha256_of
from jevtools.extract.base import Mention
from jevtools.extract.tokens import Token
from jevtools.sources.base import SourceQuery
from jevtools.sources.registry import Registry, Template
from jevtools.sources.toolsource import get_any

Clock = Callable[[], float]
DEFAULT_TTL = 300.0
LABEL_MAX = 64
_VARIABLE = re.compile(r"\{([+#./;?&]?)([A-Za-z0-9_.,%*]+)\}")


# --------------------------------------------------------------------------------------------------------------------
# URI templates (RFC 6570 subset: {var}, {+var}, {#var}, {/var}, {.var})
# --------------------------------------------------------------------------------------------------------------------


def template_variables(template: str) -> list[str]:
    """Variable names of a URI template, in order (``file:///{path}`` → ``["path"]``)."""
    names: list[str] = []
    for _, spec in _VARIABLE.findall(template):
        names += [name.rstrip("*") for name in spec.split(",") if name]
    return names


def template_regex(template: str) -> re.Pattern[str]:
    """A regex matching URIs produced by ``template``, with one named group per variable. ``{var}`` matches one
    path segment, except at the very end of the template, where it may span segments (servers list
    ``file:///{path}`` resources with nested paths unescaped)."""
    out = ""
    position = 0
    seen: set[str] = set()
    for match in _VARIABLE.finditer(template):
        out += re.escape(template[position : match.start()])
        operator, spec = match.groups()
        names = [name.rstrip("*") for name in spec.split(",") if name]
        last = match.end() == len(template)
        body = (r"[^?#]+" if last else r"[^/?#]+") if operator == "" else r".+?" if operator in ("+", "#") \
            else r"[^/?#.]+"  # fmt: skip
        prefix = {"#": "#", "/": "/", ".": r"\.", ";": ";", "?": r"\?", "&": "&"}.get(operator, "")
        parts = []
        for name in names:
            group = re.sub(r"\W", "_", name)
            parts.append(f"(?P<{group}>{body})" if group not in seen else f"(?P={group})")
            seen.add(group)
        out += prefix + ",".join(parts)
        position = match.end()
    out += re.escape(template[position:])
    return re.compile(f"^{out}$")


def expand_template(template: str, values: Mapping[str, str]) -> str:
    """Expand a URI template (``{var}`` percent-encodes reserved characters, except ``/`` in a final ``{var}``,
    mirroring :func:`template_regex`; ``{+var}``/``{#var}`` keep them)."""

    def replace(match: re.Match[str]) -> str:
        operator, spec = match.groups()
        names = [name.rstrip("*") for name in spec.split(",") if name]
        safe = ":/?#[]@!$&'()*+,;=" if operator in ("+", "#") else "/" if match.end() == len(template) else ""
        rendered = ",".join(quote(str(values[n]), safe=safe) for n in names if n in values)
        prefix = {"#": "#", "/": "/", ".": ".", ";": ";", "?": "?", "&": "&"}.get(operator, "")
        return prefix + rendered if rendered else ""

    return _VARIABLE.sub(replace, template)


# --------------------------------------------------------------------------------------------------------------------
# Duck-typed MCP objects
# --------------------------------------------------------------------------------------------------------------------


def resource_row(resource: Any) -> dict[str, Any]:
    """A ``Resource`` (pydantic model or mapping, camelCase or snake_case) as a row."""
    row = {
        "uri": str(get_any(resource, "uri") or ""),
        "name": get_any(resource, "name"),
        "title": get_any(resource, "title"),
        "description": get_any(resource, "description"),
        "mime_type": get_any(resource, "mime_type", "mimeType"),
        "size": get_any(resource, "size"),
    }
    return {k: v for k, v in row.items() if v is not None}


def _call_kwargs(method: Callable[..., Any], cursor: str | None) -> dict[str, Any]:
    """Keyword arguments requesting the page after ``cursor`` (``params=`` in newer SDKs, ``cursor=`` in older)."""
    if cursor is None:
        return {}
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}  # type: ignore[assignment]
    if "cursor" in parameters:
        return {"cursor": cursor}
    try:
        from mcp.types import PaginatedRequestParams  # noqa: PLC0415 - optional dependency

        return {"params": PaginatedRequestParams(cursor=cursor)}
    except Exception:  # noqa: BLE001 - no mcp package or an incompatible version
        return {"params": {"cursor": cursor}}


def _template_ref(uri_template: str) -> Any:
    """A ``ref/resource`` completion reference (the SDK model when available)."""
    try:
        from mcp import types  # noqa: PLC0415 - optional dependency

        cls = getattr(types, "ResourceTemplateReference", None) or types.ResourceReference  # type: ignore[attr-defined]
        return cls(type="ref/resource", uri=uri_template)
    except Exception:  # noqa: BLE001 - no mcp package or an incompatible version
        return {"type": "ref/resource", "uri": uri_template}


Call = tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]
"""A session call the fetch generator asks its driver to make: ``(method, args, kwargs)``."""


def _run(result: Any) -> Any:
    if not inspect.isawaitable(result):
        return result
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_wrap(result))
    if inspect.iscoroutine(result):
        result.close()
    raise RuntimeError("MCPResources.refresh() cannot await the session inside a running event loop; "
                       "call `await source.arefresh()` first")  # fmt: skip


async def _wrap(result: Any) -> Any:
    return await result


def _label(row: Mapping[str, Any]) -> str:
    uri = str(row.get("uri", ""))
    if len(uri) <= LABEL_MAX and "\n" not in uri:
        return uri
    return str(row.get("title") or row.get("name") or uri)


def _describe(row: Mapping[str, Any]) -> str:
    head = str(row.get("title") or row.get("name") or "")
    tail = str(row.get("description") or "")
    parts = [p for p in (head, tail) if p]
    text = ": ".join(parts) if parts else ""
    if row.get("mime_type"):
        text = f"{text} ({row['mime_type']})" if text else str(row["mime_type"])
    return text


# --------------------------------------------------------------------------------------------------------------------
# MCPResources
# --------------------------------------------------------------------------------------------------------------------


class MCPResources:
    """An MCP server's resources as a candidate source (``jt.sources.MCPResources(session, uri_template)``)."""

    def __init__(
        self,
        session: Any,
        uri_template: str | None = None,
        *,
        name: str | None = None,
        provides: Collection[str] = ("uri", "resource"),
        label: Template | None = None,
        describe: Template | None = None,
        match: Sequence[str] = ("name", "title"),
        attrs: Sequence[str] = ("name", "title", "mime_type", "size"),
        ttl: float = DEFAULT_TTL,
        complete: bool = True,
        max_pages: int = 20,
        send_whole_if_under: int = 12,
        channel: Channel | str = Channel.REGISTRY,
        item: str = "resource",
        clock: Clock | None = None,
    ) -> None:
        self.session = session
        self.uri_template = uri_template
        self.name = name or (f"mcp:{uri_template}" if uri_template else "mcp_resources")
        self.provides: frozenset[str] = frozenset(provides)
        self.label: Template = label or _label
        self.describe: Template = describe or _describe
        self.match = tuple(match)
        self.attrs = tuple(dict.fromkeys((*attrs, *template_variables(uri_template or ""))))
        self.ttl = float(ttl)
        self.complete = complete
        self.max_pages = max_pages
        self.send_whole_if_under = send_whole_if_under
        self.channel = Channel(channel)
        self.item = item
        self.clock: Clock = clock or time.monotonic
        self._pattern = template_regex(uri_template) if uri_template else None
        self._registry: Registry | None = None
        self._fetched_at: float | None = None
        self.templates: list[dict[str, Any]] = []
        """The server's resource templates seen at the last refresh (``{uri_template, name, description}``)."""
        self.calls = 0
        """Session requests made (``resources/list`` pages, template listings, completions)."""
        self.last_error: str | None = None

    def __repr__(self) -> str:
        return f"MCPResources({self.name!r})"

    # -- cache ----------------------------------------------------------------------------------------------------

    @property
    def stale(self) -> bool:
        """Never fetched, or fetched more than ``ttl`` seconds ago."""
        return self._fetched_at is None or self.clock() - self._fetched_at >= self.ttl

    def invalidate(self) -> None:
        """Forget the cached rows."""
        self._fetched_at = None

    def refresh(self, *, force: bool = False) -> None:
        """List the resources now if the cache is stale (sync driver; see the module notes on event loops)."""
        if not (force or self.stale):
            return
        flow = self._fetch()
        try:
            value: Any = None
            while True:
                method, args, kwargs = flow.send(value)
                self.calls += 1
                value = _run(method(*args, **kwargs))
        except StopIteration as stop:
            self._store(stop.value)
        except Exception as exc:  # noqa: BLE001 - a failing server fails closed
            self._fail(f"{type(exc).__name__}: {exc}")

    async def arefresh(self, *, force: bool = False) -> None:
        """Async :meth:`refresh` (awaits the session's coroutines)."""
        if not (force or self.stale):
            return
        flow = self._fetch()
        try:
            value: Any = None
            while True:
                method, args, kwargs = flow.send(value)
                self.calls += 1
                value = method(*args, **kwargs)
                if inspect.isawaitable(value):
                    value = await value
        except StopIteration as stop:
            self._store(stop.value)
        except Exception as exc:  # noqa: BLE001 - a failing server fails closed
            self._fail(f"{type(exc).__name__}: {exc}")

    def _fetch(self) -> Generator[Call, Any, list[dict[str, Any]]]:
        """Listing and completion requests, as calls for a sync or async driver."""
        rows: list[dict[str, Any]] = []
        list_resources = getattr(self.session, "list_resources", None)
        if callable(list_resources):
            cursor: str | None = None
            for _ in range(self.max_pages):
                page = yield (list_resources, (), _call_kwargs(list_resources, cursor))
                for resource in get_any(page, "resources") or ():
                    row = resource_row(resource)
                    if row.get("uri") and self._keep(row):
                        rows.append(row)
                cursor = get_any(page, "next_cursor", "nextCursor")
                if not cursor:
                    break
        list_templates = getattr(self.session, "list_resource_templates", None)
        self.templates = []
        if callable(list_templates):
            page = yield (list_templates, (), {})
            for template in get_any(page, "resource_templates", "resourceTemplates") or ():
                self.templates.append({
                    "uri_template": str(get_any(template, "uri_template", "uriTemplate") or ""),
                    "name": get_any(template, "name"), "title": get_any(template, "title"),
                    "description": get_any(template, "description"),
                    "mime_type": get_any(template, "mime_type", "mimeType"),
                })  # fmt: skip
        rows += yield from self._completed()
        return rows

    def _keep(self, row: dict[str, Any]) -> bool:
        if self._pattern is None:
            return True
        match = self._pattern.match(row["uri"])
        if match is None:
            return False
        row.update({k: v for k, v in match.groupdict().items() if k not in row})
        return True

    def _completed(self) -> Generator[Call, Any, list[dict[str, Any]]]:
        """Rows from completion values of the (single-variable) ``uri_template``."""
        complete = getattr(self.session, "complete", None)
        template = self.uri_template
        if not (self.complete and callable(complete) and template):
            return []
        variables = template_variables(template)
        declared = next((t for t in self.templates if t["uri_template"] == template), None)
        if len(variables) != 1 or declared is None:
            return []
        result = yield (complete, (), {"ref": _template_ref(template), "argument": {"name": variables[0],
                                                                                  "value": ""}})  # fmt: skip
        completion = get_any(result, "completion")
        rows: list[dict[str, Any]] = []
        for value in get_any(completion, "values") or ():
            uri = expand_template(template, {variables[0]: str(value)})
            rows.append({"uri": uri, "name": str(value), variables[0]: str(value),
                         **({"description": declared["description"]} if declared.get("description") else {}),
                         **({"mime_type": declared["mime_type"]} if declared.get("mime_type") else {})})  # fmt: skip
        return rows

    def _store(self, rows: list[dict[str, Any]]) -> None:
        unique = list({row["uri"]: row for row in rows}.values())
        self._registry = self._build(unique)
        self._fetched_at = self.clock()
        self.last_error = None

    def _fail(self, error: str) -> None:
        self.last_error = error
        if self._registry is None:
            self._registry = self._build([])
        self._fetched_at = self.clock()

    def _build(self, rows: Sequence[Mapping[str, Any]]) -> Registry:
        return Registry(
            self.name, rows, key="uri", label=self.label, describe=self.describe, match=self.match,
            provides=self.provides, attrs=self.attrs, send_whole_if_under=self.send_whole_if_under,
            channel=self.channel, item=self.item,
        )  # fmt: skip

    @property
    def registry(self) -> Registry:
        """The registry over the current rows (a stale cache is refreshed first; inside a running event loop an
        unfetched source stays empty rather than blocking — prefetch with :meth:`arefresh`)."""
        if self.stale:
            try:
                asyncio.get_running_loop()
                running = True
            except RuntimeError:
                running = False
            if running and self._needs_await():
                if self._registry is None:
                    self._fail("not fetched: call `await source.arefresh()` before deciding in async code")
            else:
                self.refresh()
        assert self._registry is not None
        return self._registry

    def _needs_await(self) -> bool:
        method = getattr(self.session, "list_resources", None)
        return inspect.iscoroutinefunction(method)

    @property
    def rows(self) -> tuple[Mapping[str, Any], ...]:
        """The current rows."""
        return self.registry.rows

    # -- Source protocol (delegated) ------------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.registry)

    def candidates(self, q: SourceQuery) -> list[Candidate]:
        """Candidates of the current resources (the whole list when small, else the anchored shortlist)."""
        return self.registry.candidates(q)

    def ranked(self, q: SourceQuery) -> list[Candidate]:
        return self.registry.ranked(q)

    def find_anchors(
        self,
        text: str,
        *,
        tokens: Sequence[Token] | None = None,
        source_ref: str = "request",
        channel: Channel = Channel.USER,
    ) -> list[Mention]:
        return self.registry.find_anchors(text, tokens=tokens, source_ref=source_ref, channel=channel)

    def group_of(self, candidate: Candidate) -> str | None:
        return self.registry.group_of(candidate)

    def lookup(self, key: Any) -> Mapping[str, Any] | None:
        """TOCTOU re-resolution: the resource whose URI is ``key``."""
        return self.registry.lookup(key)

    def attribute_names(self) -> frozenset[str]:
        return self.registry.attribute_names()

    def content_sha256(self) -> str:
        """Hash of the template and the current rows."""
        return sha256_of({"uri_template": self.uri_template, "rows": [jsonable(dict(r)) for r in self.registry.rows]})


__all__ = [
    "MCPResources",
    "expand_template",
    "resource_row",
    "template_regex",
    "template_variables",
]

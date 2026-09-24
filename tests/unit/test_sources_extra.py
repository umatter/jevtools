"""Extra candidate sources (spec §4.4): ``ToolSource`` (a catalog tool called once per session, cached with a ttl)
and ``MCPResources`` (an MCP server's resources and resource templates), plus the JSONPath subset they use."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from jevtools.backends.scripted import ScriptedBackend
from jevtools.candidates import Channel
from jevtools.context import Context
from jevtools.router import Router
from jevtools.sources.base import SourceQuery
from jevtools.sources.mcp import MCPResources, expand_template, template_regex, template_variables
from jevtools.sources.toolsource import ToolSource, jsonpath, jsonpath_matches, parse_jsonpath, tool_sources
from jevtools.spec.catalog import Catalog

CONTACTS = {
    "contacts": [
        {"name": "Dora Nussbaum", "email": "dora@muster.ch", "team": "Research"},
        {"name": "Emil Frei", "email": "emil@muster.ch", "team": "Legal"},
        {"name": "Nobody", "email": None},
    ]
}


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


# -- JSONPath subset ------------------------------------------------------------------------------------------------


def test_jsonpath_subset() -> None:
    data = {"a": {"b": [{"c": 1}, {"c": 2}], "odd key": {"x": 3}}, "list": [10, 20, 30]}
    assert parse_jsonpath("$.a.b[*].c") == ["a", "b", None, "c"]
    assert jsonpath(data, "$.a.b[*].c") == [1, 2]
    assert jsonpath_matches(data, "$.a.b[1]") == [("$.a.b[1]", {"c": 2})]
    assert jsonpath(data, "$.list[-1]") == [30] and jsonpath(data, "$.list[5]") == []
    assert jsonpath_matches(data, "$.a['odd key'].x") == [('$.a["odd key"].x', 3)]
    assert jsonpath(data, "$.a.*") == [[{"c": 1}, {"c": 2}], {"x": 3}] and jsonpath(data, "$") == [data]
    assert jsonpath(data, "$.missing[*]") == []
    for bad in ("a.b", "$..c", "$.a[?(@.c)]", "$.list[0:2]"):
        with pytest.raises(ValueError):
            parse_jsonpath(bad)


# -- ToolSource -----------------------------------------------------------------------------------------------------


def contacts_source(calls: list[dict[str, Any]], clock: Clock | None = None, **kw: Any) -> ToolSource:
    def call(tool: str, args: dict[str, Any]) -> Any:
        calls.append({"tool": tool, **args})
        return CONTACTS

    return ToolSource(
        "list_contacts",
        {"limit": 50},
        "$.contacts[*]",
        key="email",
        label="{name} <{email}>",
        ttl=300,
        call=call,
        match=["name", "team"],
        provides=["email", "person"],
        describe="{team}",
        clock=clock or Clock(),
        **kw,
    )


def test_tool_source_calls_its_tool_once_per_ttl() -> None:
    calls: list[dict[str, Any]] = []
    clock = Clock()
    source = contacts_source(calls, clock)
    assert source.name == "tool:list_contacts" and source.channel is Channel.REGISTRY and source.stale
    candidates = source.candidates(SourceQuery(request="Email Dora"))
    assert [(c.label, c.value, c.channel) for c in candidates] == [
        ("Dora Nussbaum <dora@muster.ch>", "dora@muster.ch", Channel.REGISTRY),
        ("Emil Frei <emil@muster.ch>", "emil@muster.ch", Channel.REGISTRY),
    ]  # a small registry is sent whole
    assert candidates[0].text == 'Contact matching "Dora": Research.' and candidates[0].prov["anchor"] == "Dora"
    source.find_anchors("Email Emil")
    source.content_sha256()
    assert calls == [{"tool": "list_contacts", "limit": 50}] and source.calls == 1  # cached for the session
    clock.t = 299.0
    assert source.lookup("emil@muster.ch") == CONTACTS["contacts"][1] and source.calls == 1
    clock.t = 300.0
    assert source.stale and len(source) == 2 and source.calls == 2  # the ttl expired: called again
    source.invalidate()
    source.refresh()
    assert source.calls == 3


def test_tool_source_result_shapes() -> None:
    mcp_result = {"content": [{"type": "text", "text": "…"}], "structuredContent": CONTACTS, "isError": False}
    source = ToolSource("list_contacts", items="$.contacts[*]", key="email", call=lambda t, a: mcp_result)
    assert [r["email"] for r in source.rows] == ["dora@muster.ch", "emil@muster.ch"]
    text = ToolSource("tags", items="$.tags[*]", call=lambda t, a: '{"tags": ["red", "green"]}')
    assert [c.value for c in text.candidates(SourceQuery())] == ["red", "green"]  # scalar rows: key "value"
    error = {"content": [{"type": "text", "text": "denied"}], "isError": True}
    failed = ToolSource("list_contacts", items="$.contacts[*]", key="email", call=lambda t, a: error)
    assert failed.rows == () and failed.last_error == "tool error: denied"


def test_tool_source_fails_closed() -> None:
    unbound = ToolSource("list_contacts", items="$.contacts[*]", key="email")
    assert not unbound.bound and unbound.candidates(SourceQuery()) == [] and unbound.last_error == "no caller bound"

    def broken(tool: str, args: dict[str, Any]) -> Any:
        raise ConnectionError("offline")

    source = ToolSource("list_contacts", items="$.contacts[*]", key="email", call=broken)
    assert source.candidates(SourceQuery()) == [] and source.last_error == "ConnectionError: offline"
    source.bind(lambda t, a: CONTACTS)  # an existing caller is kept …
    source.invalidate()
    assert len(source) == 0
    source.bind(lambda t, a: CONTACTS, replace=True)  # … unless replaced
    source.invalidate()
    assert len(source) == 2


def test_tool_source_with_an_async_caller() -> None:
    async def call(tool: str, args: dict[str, Any]) -> Any:
        await asyncio.sleep(0)
        return CONTACTS

    sync_side = ToolSource("list_contacts", items="$.contacts[*]", key="email", call=call)
    assert len(sync_side) == 2  # outside an event loop: run to completion

    async def inside() -> tuple[int, int]:
        source = ToolSource("list_contacts", items="$.contacts[*]", key="email", call=call)
        await source.arefresh()
        blocked = ToolSource("list_contacts", items="$.contacts[*]", key="email", call=call)
        return len(source), len(blocked)  # not prefetched inside a loop: empty, never blocking

    assert asyncio.run(inside()) == (2, 0)


def test_tool_source_from_spec_and_catalog_declarations() -> None:
    spec = {
        "tool": "list_contacts",
        "args": {},
        "items": "$.contacts[*]",
        "key": "email",
        "label": "{name} <{email}>",
        "ttl": 60,
        "match": ["name"],
        "provides": ["email"],
    }
    source = ToolSource.from_spec(spec, call=lambda t, a: CONTACTS)
    assert (source.name, source.ttl, source.key) == ("tool:list_contacts", 60.0, "email")
    with pytest.raises(ValueError, match="unknown tool-source keys"):
        ToolSource.from_spec({"tool": "x", "itemz": "$"})
    catalog = Catalog.from_openai(note_tools(spec))
    assert catalog.get("send_note").slot("to").source_names == ("tool:list_contacts",)
    [found] = tool_sources(catalog)
    assert found.name == "tool:list_contacts" and not found.bound


def note_tools(source: Any) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "send_note",
                "description": "Send a short note to one colleague.",
                "parameters": {
                    "type": "object",
                    "required": ["to", "note"],
                    "properties": {
                        "to": {"type": "string", "description": "The colleague's address", "x-jev": {"source": source}},
                        "note": {"type": "string", "description": "The note text"},
                    },
                },
            },
        }
    ]


def test_an_inline_tool_source_feeds_the_slot_pool() -> None:
    spec = {
        "tool": "list_contacts",
        "items": "$.contacts[*]",
        "key": "email",
        "label": "{name} <{email}>",
        "match": ["name"],
        "provides": ["email", "person"],
    }
    catalog = Catalog.from_openai(note_tools(spec))
    sources = tool_sources(catalog, call=lambda t, a: CONTACTS)
    ctx = Context(messages=[{"role": "user", "content": "Send Dora a note saying hi"}], sources=sources)
    router = Router(catalog, backend=ScriptedBackend({}), context=ctx)
    ballot = router.compile("Send Dora a note saying hi", context=ctx)
    options = ballot.by_qid["send_note.to"].options
    assert [(o.label, o.value) for o in options][0] == ("Dora Nussbaum <dora@muster.ch>", "dora@muster.ch")
    assert all(o.channel is Channel.REGISTRY for o in options)


# -- MCPResources ---------------------------------------------------------------------------------------------------


@dataclass
class Resource:
    uri: str
    name: str
    description: str | None = None
    mime_type: str | None = None
    title: str | None = None


@dataclass
class Page:
    resources: list[Resource]
    next_cursor: str | None = None


@dataclass
class Template:
    uri_template: str
    name: str
    description: str | None = None
    mime_type: str | None = None


@dataclass
class Templates:
    resource_templates: list[Template]


@dataclass
class Completion:
    values: list[str]


@dataclass
class Complete:
    completion: Completion


@dataclass
class FakeSession:
    """An MCP ``ClientSession`` double (new-SDK shapes: snake_case fields, ``params=`` pagination)."""

    pages: list[Page] = field(default_factory=list)
    templates: list[Template] = field(default_factory=list)
    completions: list[str] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)
    fail: bool = False

    async def list_resources(self, *, params: Any = None) -> Page:
        # a PaginatedRequestParams model when the mcp package is installed, else a plain mapping
        cursor = params.get("cursor") if isinstance(params, dict) else getattr(params, "cursor", None)
        self.requests.append(f"resources/list {cursor}")
        if self.fail:
            raise ConnectionError("server gone")
        return self.pages[0 if cursor is None else int(cursor)]

    async def list_resource_templates(self, *, params: Any = None) -> Templates:
        self.requests.append("resources/templates/list")
        return Templates(self.templates)

    async def complete(self, ref: Any, argument: dict[str, str], context_arguments: Any = None) -> Complete:
        uri = ref["uri"] if isinstance(ref, dict) else ref.uri
        self.requests.append(f"completion/complete {uri} {argument['name']}")
        return Complete(Completion(self.completions))


def session() -> FakeSession:
    long_uri = "file:///workspace/" + "very/" * 12 + "deep.md"
    return FakeSession(
        pages=[
            Page(
                [
                    Resource("file:///workspace/README.md", "README", "Project overview", "text/markdown"),
                    Resource("file:///workspace/notes/todo.md", "todo", None, "text/markdown"),
                ],
                next_cursor="1",
            ),
            Page(
                [Resource("db://customers", "customers", "Customer table"), Resource(long_uri, "deep", "A deep file")]
            ),
        ],
        templates=[Template("file:///workspace/{path}", "workspace file", "A file in the workspace", "text/plain")],
        completions=["README.md", "src/app.py"],
    )


def test_mcp_resources_list_every_page_and_label_by_uri() -> None:
    fake = session()
    source = MCPResources(fake, clock=Clock())
    rows = source.rows
    assert [r["uri"] for r in rows][:3] == [
        "file:///workspace/README.md",
        "file:///workspace/notes/todo.md",
        "db://customers",
    ]
    assert fake.requests == ["resources/list None", "resources/list 1", "resources/templates/list"]
    assert source.name == "mcp_resources" and source.templates[0]["uri_template"] == "file:///workspace/{path}"
    candidates = {c.value: c for c in source.candidates(SourceQuery(request="Open the README"))}
    readme = candidates["file:///workspace/README.md"]
    assert readme.label == "file:///workspace/README.md" and readme.channel is Channel.REGISTRY
    assert readme.text == 'Resource matching "README": README: Project overview (text/markdown).'
    deep = next(c for uri, c in candidates.items() if uri.endswith("deep.md"))
    assert deep.label == "deep"  # the URI exceeds the 64-character label limit
    assert source.lookup("db://customers") is not None and source.calls == 3


def test_mcp_resources_filter_by_template_and_complete_its_variable() -> None:
    fake = session()
    source = MCPResources(fake, "file:///workspace/{path}", clock=Clock())
    assert source.name == "mcp:file:///workspace/{path}"
    rows = {r["uri"]: r for r in source.rows}
    assert [uri for uri in rows if "/very/" not in uri] == [
        "file:///workspace/README.md",
        "file:///workspace/notes/todo.md",
        "file:///workspace/src/app.py",
    ]  # listed matches (db://customers is not one), then completion values; README.md is deduped
    assert rows["file:///workspace/notes/todo.md"]["path"] == "notes/todo.md"
    assert rows["file:///workspace/src/app.py"]["description"] == "A file in the workspace"
    assert fake.requests[-1] == "completion/complete file:///workspace/{path} path"
    assert "path" in source.attribute_names()


def test_mcp_resources_async_prefetch_and_fail_closed() -> None:
    async def inside() -> tuple[int, int, str | None]:
        prefetched = MCPResources(session())
        await prefetched.arefresh()
        lazy = MCPResources(session())
        return len(prefetched), len(lazy), lazy.last_error  # inside a loop an unfetched source stays empty

    count, lazy, error = asyncio.run(inside())
    assert (count, lazy) == (4, 0) and error is not None and "arefresh" in error
    down = MCPResources(FakeSession(fail=True))
    assert down.candidates(SourceQuery()) == [] and down.last_error == "ConnectionError: server gone"


def test_mcp_resources_accept_old_sdk_shapes() -> None:
    class OldSession:
        def __init__(self) -> None:
            self.cursors: list[str | None] = []

        def list_resources(self, cursor: str | None = None) -> dict[str, Any]:
            self.cursors.append(cursor)
            if cursor is None:
                return {"resources": [{"uri": "mem://a", "name": "a", "mimeType": "text/plain"}], "nextCursor": "c2"}
            return {"resources": [{"uri": "mem://b", "name": "b"}]}

    old = OldSession()
    source = MCPResources(old)
    assert [r["uri"] for r in source.rows] == ["mem://a", "mem://b"] and old.cursors == [None, "c2"]
    assert source.rows[0]["mime_type"] == "text/plain"


def test_uri_template_helpers() -> None:
    assert template_variables("repo://{owner}/{repo}/blob{/path*}") == ["owner", "repo", "path"]
    match = template_regex("repo://{owner}/{repo}").match("repo://acme/app")
    assert match is not None and match.groupdict() == {"owner": "acme", "repo": "app"}
    assert template_regex("repo://{owner}/{repo}/info").match("repo://acme/app/x/info") is None
    tail = template_regex("file:///{path}").match("file:///notes/todo.md")  # a final {var} spans segments
    assert tail is not None and tail.group("path") == "notes/todo.md"
    assert template_regex("file:///{+path}").match("file:///a/b.md") is not None
    assert expand_template("file:///{path}", {"path": "a b/c"}) == "file:///a%20b/c"
    assert expand_template("x://{a}/{b}", {"a": "1/2", "b": "3/4"}) == "x://1%2F2/3/4"
    assert expand_template("file:///{+path}", {"path": "a/b"}) == "file:///a/b"


def test_an_inline_mcp_source_feeds_the_slot_pool() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_resource",
                "description": "Read one workspace resource.",
                "parameters": {
                    "type": "object",
                    "required": ["uri"],
                    "properties": {
                        "uri": {
                            "type": "string",
                            "description": "The resource URI",
                            "x-jev": {"source": {"mcp_resources": "file:///workspace/{path}"}},
                        }
                    },
                },
            },
        }
    ]
    catalog = Catalog.from_openai(tools)
    assert catalog.get("read_resource").slot("uri").source_names == ("mcp:file:///workspace/{path}",)
    source = MCPResources(session(), "file:///workspace/{path}")
    ctx = Context(messages=[{"role": "user", "content": "Read the README"}], sources=[source])
    ballot = Router(catalog, backend=ScriptedBackend({}), context=ctx).compile("Read the README", context=ctx)
    values = [o.value for o in ballot.by_qid["read_resource.uri"].options]
    assert "file:///workspace/README.md" in values and "db://customers" not in values

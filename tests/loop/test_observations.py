"""Observations become candidate pools (spec §6.3): parsing, provenance, previews and the content handle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from jevtools.candidates import Channel
from jevtools.loop import (
    LoopObservation,
    chunk_text,
    ingest_observation,
    select_preview,
    text_entities,
)
from jevtools.spec.catalog import Catalog

LONG = " ".join(
    f"Paragraph {i}: the quarterly figures for region {i} were reviewed and filed without remarks." for i in range(60)
)


def test_text_is_chunked_scanned_and_handled() -> None:
    text = "Invoice INV-2291 from ACME AG, dated 2026-09-15. Total CHF 4,820.00. Questions: billing@acme.example."
    obs = ingest_observation(text, "read_file", 1, arguments={"path": "a/b.pdf"}, call_id="call_jev_1")
    assert isinstance(obs, LoopObservation) and obs.status == "ok" and obs.content == text
    assert obs.preview == text and obs.summary == "12 words" and obs.call_id == "call_jev_1"
    assert obs.handle == "⟨full text of the file read in step 1⟩"
    found = {(i.type, i.value) for i in obs.items}
    assert found >= {
        ("id", "INV-2291"),
        ("date", "2026-09-15"),
        ("money", "4820.00 CHF"),
        ("email", "billing@acme.example"),
    }
    email = next(i for i in obs.items if i.type == "email")
    assert email.ref == "obs:1:$" and email.span is not None and text[email.span[0] : email.span[1]] == email.label
    assert all(i.channel is Channel.TOOL_OUTPUT for i in obs.items)
    assert obs.progress_line() == 'Step 1: read_file(path="a/b.pdf") → ok, 12 words'


def test_long_text_previews_bm25_chunks_never_the_whole_content() -> None:
    text = LONG + " The ACME invoice INV-2291 is overdue. " + LONG
    obs = ingest_observation(text, "read_file", 2, request="Is the ACME invoice overdue?", preview_chars=400)
    assert obs.preview is not None and len(obs.preview) <= 400 and obs.content == text
    assert "INV-2291 is overdue" in obs.preview and " … " in obs.preview


def test_emits_gives_typed_items_with_jsonpath_provenance() -> None:
    result = {
        "results": [
            {"title": "Jev docs", "url": "https://example.org/jev", "snippet": "Decisions API"},
            {"title": "Pricing", "url": "https://example.org/pricing", "snippet": "0.042 per M"},
        ]
    }
    emits = {"items": "$.results[*]", "key": "url", "label": "{title}", "describe": "{snippet}"}
    obs = ingest_observation(result, "search_web", 1, emits=emits, request="jev pricing")
    typed = obs.items_of("url")
    assert [(i.value, i.label, i.path, i.ref) for i in typed[:2]] == [
        ("https://example.org/jev", "Jev docs", "$.results[0]", "obs:1:$.results[0]"),
        ("https://example.org/pricing", "Pricing", "$.results[1]", "obs:1:$.results[1]"),
    ]
    assert typed[0].attrs["snippet"] == "Decisions API" and obs.summary == "2 items"
    assert not obs.items_of("leaf")  # typed items replace generic leaves


def test_tool_spec_emits_and_output_schema_are_used() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_messages",
                "description": "List the inbox.",
                "parameters": {"type": "object", "properties": {}},
                "x-jev": {"risk": "read", "emits": {"items": "$.messages[*]", "key": "id", "label": "{subject}"}},
            },
        }
    ]
    spec = Catalog.from_openai(tools).get("list_messages")
    obs = ingest_observation({"messages": [{"id": "m1", "subject": "Hello"}]}, spec, 3)
    assert [(i.type, i.value, i.label) for i in obs.items] == [("id", "m1", "Hello")]

    schema = {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "uri": {"type": "string", "format": "uri"}},
                },
            }
        },
    }
    obs = ingest_observation({"files": [{"name": "a", "uri": "file:///a"}]}, "list_files", 1, output_schema=schema)
    assert [(i.type, i.value, i.label, i.path) for i in obs.items] == [("uri", "file:///a", "a", "$.files[0]")]


def test_unknown_json_is_flattened_to_leaves() -> None:
    data = {"invoice": {"no": "INV-7", "total": 12.5, "paid": False, "lines": ["a", "b"]}, "to": "x@y.example"}
    obs = ingest_observation(data, "get_invoice", 4)
    leaves = {(i.path, i.value) for i in obs.items_of("leaf")}
    assert leaves == {
        ("$.invoice.no", "INV-7"),
        ("$.invoice.total", 12.5),
        ("$.invoice.paid", False),
        ("$.invoice.lines[0]", "a"),
        ("$.invoice.lines[1]", "b"),
        ("$.to", "x@y.example"),
    }
    assert ("email", "x@y.example", "obs:4:$.to") in {(i.type, i.value, i.ref) for i in obs.items}
    assert obs.summary == "6 fields" and obs.handle == "⟨full text of the invoice retrieved in step 4⟩"
    assert ingest_observation(data, "invoice_lookup", 5).handle == "⟨full text of the output of step 5⟩"


def test_json_strings_are_parsed() -> None:
    obs = ingest_observation('{"a": [1, 2]}', "t", 1)
    assert obs.content == {"a": [1, 2]} and obs.summary == "2 fields"


@dataclass
class TextBlock:
    type: str
    text: str


@dataclass
class CallToolResult:
    content: list[Any]
    structuredContent: Any = None  # noqa: N815 - the MCP field name
    isError: bool = False  # noqa: N815 - the MCP field name


def test_mcp_results_prefer_structured_content_and_flag_errors() -> None:
    ok = CallToolResult(content=[TextBlock("text", "12 degrees")], structuredContent={"temp": 12})
    assert ingest_observation(ok, "get_weather", 1).content == {"temp": 12}
    text_only = {"content": [{"type": "text", "text": "line 1"}, {"type": "text", "text": "line 2"}], "isError": False}
    assert ingest_observation(text_only, "t", 1).content == "line 1\nline 2"
    failed = CallToolResult(content=[TextBlock("text", "quota exceeded")], isError=True)
    obs = ingest_observation(failed, "t", 1)
    assert (obs.status, obs.content, obs.error) == ("error", "quota exceeded", "quota exceeded")


def test_exceptions_are_error_observations() -> None:
    obs = ingest_observation(None, "send_email", 2, error=TimeoutError("relay timed out"))
    assert (obs.status, obs.content, obs.summary) == (
        "error",
        "TimeoutError: relay timed out",
        "TimeoutError: relay timed out",
    )
    assert ingest_observation(ValueError("bad"), "t", 1).status == "error"


def test_emits_types_restrict_text_entities() -> None:
    text = "Pay CHF 20.00 to a@b.example by 2026-10-01."
    obs = ingest_observation(text, "t", 1, emits={"types": ["money"]})
    assert [i.type for i in obs.items] == ["money"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Transfer CHF 5,000 to CH44 3199 9123 0008 8901 2",
            [("money", "5000.00 CHF"), ("iban", "CH4431999123000889012")],
        ),
        (
            "see https://x.example/a/b.pdf, or docs/a/b.md",
            [("url", "https://x.example/a/b.pdf"), ("path", "docs/a/b.md")],
        ),
        ("€12.50 and 30 EUR on 01.10.2026", [("money", "12.50 EUR"), ("money", "30.00 EUR"), ("date", "01.10.2026")]),
        (
            "ticket 3f2b8c1e-0000-4000-8000-00000000abcd from 10.0.0.1",
            [("uuid", "3f2b8c1e-0000-4000-8000-00000000abcd"), ("ipv4", "10.0.0.1")],
        ),
    ],
)
def test_text_entities(text: str, expected: list[tuple[str, str]]) -> None:
    assert [(kind, value) for kind, value, _, _ in text_entities(text)] == expected


def test_text_entities_share_the_request_side_pattern_rules() -> None:
    """Observation entities use the request-side pattern extractor (IPv4 validation, URL trailing punctuation and
    span) and the ISO 4217 currency catalog (#16)."""
    text = "ping 999.300.1.2 or 10.0.0.1, see https://x.example/a). Pay CZK 1'200 or ₣ 40 or TOP 10 now"
    found = text_entities(text)
    assert [(k, v) for k, v, _, _ in found] == [
        ("ipv4", "10.0.0.1"), ("url", "https://x.example/a"), ("money", "1200.00 CZK"), ("money", "40.00 CHF"),
    ]  # fmt: skip
    url = next(entry for entry in found if entry[0] == "url")
    assert text[url[3][0] : url[3][1]] == url[2] == "https://x.example/a"
    obs = ingest_observation("Server at 999.300.1.2 is down.", "status", 1)
    assert not [i for i in obs.items if i.type == "ipv4"]


def test_chunks_and_preview_helpers() -> None:
    chunks = chunk_text("One. Two! Three?\nFour", size=10)
    assert chunks == ["One. Two!", "Three?", "Four"]
    assert select_preview(chunks, "", limit=100) == "One. Two! Three? Four"
    preview = select_preview(["a" * 50, "b" * 50, ("c needle " * 5).strip()], "needle", limit=110)
    assert preview == "a" * 50 + " … " + ("c needle " * 5).strip()  # the BM25 hit, then the head; in text order
    assert select_preview(["x" * 200], "", limit=50) == "x" * 49 + "…"

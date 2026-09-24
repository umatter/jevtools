"""MCP interop (spec §7.1, §10.6): ``Catalog.from_mcp`` over a recorded ``tools/list`` (annotations explicit vs
absent) and ``call_decision`` passing the idempotency key in ``_meta``. Answers are scripted fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jevtools.adapters import mcp as jm
from jevtools.policy import Tier
from jevtools.spec.catalog import Catalog
from tests.adapters.support import WEATHER_REQUEST, weather_router
from tests.scenario import scripts
from tests.scenario.fixtures import scenario_messages, scenario_router

TOOLS_LIST = json.loads((Path(__file__).parent / "fixtures" / "mcp_tools_list.json").read_text(encoding="utf-8"))


@pytest.fixture
def catalog() -> Catalog:
    return Catalog.from_mcp(TOOLS_LIST["result"])


def test_explicit_annotations_set_the_tier(catalog: Catalog) -> None:
    assert (catalog.get("list_issues").tier, catalog.get("list_issues").tier_reason) == (
        Tier.READ, "annotation readOnlyHint")  # fmt: skip
    close = catalog.get("close_issue")
    assert (close.tier, close.tier_reason) == (Tier.CRITICAL, "annotation destructiveHint")
    assert close.idempotent is True  # explicit idempotentHint
    assert close.annotations == {"destructiveHint": True, "idempotentHint": True, "readOnlyHint": False}


def test_absent_or_null_annotations_fall_back_to_the_verb(catalog: Catalog) -> None:
    comment = catalog.get("comment_issue")
    assert comment.annotations == {} and comment.tier is Tier.EXTERNAL  # fail-safe default ('comment' is no verb)
    assert comment.tier_reason == "fail-safe default (declare x-jev.risk)" and comment.idempotent is False
    sync = catalog.get("sync_mirror")
    assert "readOnlyHint" not in sync.annotations and "destructiveHint" not in sync.annotations
    assert sync.title == "Sync" and sync.tier is Tier.EXTERNAL
    assert catalog.get("list_issues").title == "List issues"


def test_meta_xjev_and_output_schema(catalog: Catalog) -> None:
    label = catalog.get("label_issue")
    assert (label.tier, label.tier_reason) == (Tier.WRITE, "x-jev.risk")
    assert [m.value for m in label.slot("label").values or ()] == ["bug", "feature", "question"]
    assert label.output_schema is not None and "labels" in label.output_schema["properties"]


def test_from_mcp_accepts_the_jsonrpc_envelope_body_and_a_plain_list() -> None:
    assert Catalog.from_mcp(TOOLS_LIST["result"]["tools"]).names == Catalog.from_mcp(TOOLS_LIST["result"]).names


def test_from_mcp_accepts_mcp_objects() -> None:
    types = pytest.importorskip("mcp.types")
    result = types.ListToolsResult.model_validate(TOOLS_LIST["result"])
    catalog = Catalog.from_mcp(result)
    assert catalog.get("list_issues").tier is Tier.READ and catalog.get("close_issue").tier is Tier.CRITICAL
    assert catalog.get("label_issue").tier is Tier.WRITE and catalog.get("comment_issue").annotations == {}


class _Session:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None, *, meta: Any = None) -> Any:
        self.calls.append({"name": name, "arguments": arguments, "meta": meta})
        return {"content": [{"type": "text", "text": "61°F, sunny"}], "isError": False}


async def test_call_decision_passes_the_idempotency_key() -> None:
    router, _ = weather_router()
    decision = router.decide(WEATHER_REQUEST)
    session = _Session()
    result = await jm.call_decision(session, decision)
    (call,) = session.calls
    assert call == {"name": "get_weather", "arguments": {"city": "Zurich", "unit": "fahrenheit"},
                    "meta": {"jevtools/idempotency_key": decision.tool_calls[0].idempotency_key}}  # fmt: skip
    assert jm.result_content(result) == "61°F, sunny" and not jm.is_error(result)
    observation = jm.to_observation(decision, result, step=1)
    assert (observation.tool, observation.status, observation.call_id) == (
        "get_weather", "ok", decision.tool_calls[0].id)  # fmt: skip


async def test_only_executed_decisions_reach_the_server() -> None:
    router, _ = scenario_router(scripts.R2)
    confirm = router.decide(scenario_messages(scripts.R2_REQUEST, history=True))
    session = _Session()
    with pytest.raises(ValueError, match="no tool call to execute"):
        await jm.call_decision(session, confirm)
    assert session.calls == []


async def test_call_decision_on_a_real_client_session_signature() -> None:
    mcp = pytest.importorskip("mcp")
    import inspect

    assert "meta" in inspect.signature(mcp.ClientSession.call_tool).parameters


def test_result_content_prefers_structured_content_and_reads_mcp_objects() -> None:
    types = pytest.importorskip("mcp.types")
    result = types.CallToolResult.model_validate({"content": [{"type": "text", "text": "x"}],
                                                  "structuredContent": {"temp": 61}, "isError": True})  # fmt: skip
    assert jm.result_content(result) == {"temp": 61} and jm.is_error(result)
    plain = types.CallToolResult.model_validate({"content": [{"type": "text", "text": "a"},
                                                             {"type": "text", "text": "b"}]})  # fmt: skip
    assert jm.result_content(plain) == "a\nb"

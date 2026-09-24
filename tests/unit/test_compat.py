"""Migration helpers (spec §7.4): the cookbook policy and hints, and ``from_jev_fn`` (bool → flag, Literal/Enum →
enum, ``ge``/``le`` int → quantity with a grid unless ordinal, probabilities kept). Answers are scripted."""

from __future__ import annotations

import functools
from enum import Enum
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from jevtools.backends.scripted import ScriptedBackend
from jevtools.compat import (
    COOKBOOK_VERSION,
    JevFnTool,
    cookbook_catalog,
    cookbook_hints,
    cookbook_policy,
    from_jev_fn,
)
from jevtools.context import Context
from jevtools.errors import CatalogError
from jevtools.policy import Outcome, Policy, Tier
from jevtools.router import Router
from jevtools.spec.catalog import Catalog
from tests.scenario.fixtures import SCENARIO_NOW


class Mood(Enum):
    HAPPY = "happy"
    SAD = "sad"


class Triage(BaseModel):
    urgent: bool = Field(description="Is the ticket urgent?")
    category: Literal["billing", "tech", "other"]
    mood: Mood
    priority: int = Field(ge=1, le=5)
    seats: int = Field(ge=1, le=20, description="How many seats are affected")
    verbosity: int = Field(ge=0, le=2, json_schema_extra={"levels": ["terse", "normal", "chatty"]})
    ratio: float = Field(0.5, ge=0.0, le=1.0, json_schema_extra={"levels": ["none", "half", "all"]})


def triage(ticket: str) -> Triage:
    """Triage the support ticket: {{ ticket }}"""
    raise NotImplementedError


def _jev_fn(func: Any) -> Any:
    """A stand-in for ``@jev.fn``: ``functools.wraps`` plus the ``state``/``map`` attributes."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("never called: jevtools elects the fields itself")

    wrapper.state = lambda value=None: value  # type: ignore[attr-defined]
    wrapper.map = lambda items: items  # type: ignore[attr-defined]
    return wrapper


@pytest.fixture
def tool() -> JevFnTool:
    return from_jev_fn(_jev_fn(triage))


def test_fields_map_to_kinds(tool: JevFnTool) -> None:
    spec = Catalog.from_any([tool]).get("triage")
    kinds = {slot.name: slot.kind for slot in spec.slots}
    assert kinds == {"urgent": "flag", "category": "enum", "mood": "enum", "priority": "ordinal",
                     "seats": "quantity", "verbosity": "ordinal", "ratio": "quantity"}  # fmt: skip
    assert [m.value for m in spec.slot("mood").values or ()] == ["happy", "sad"]
    assert [m.value for m in spec.slot("seats").values or ()] == list(range(1, 21))  # the grid
    assert [(m.value, m.text) for m in spec.slot("verbosity").values or ()] == [
        (0, "terse"), (1, "normal"), (2, "chatty")]  # fmt: skip
    assert [m.value for m in spec.slot("ratio").values or ()] == [0.0, 0.5, 1.0]
    assert "levels" not in tool.parameters["properties"]["verbosity"]  # never forwarded as a JSON Schema keyword


def test_name_and_description(tool: JevFnTool) -> None:
    assert (tool.name, tool.description) == ("triage", "Triage the support ticket")
    assert from_jev_fn(triage, name="triage_ticket", description="Sort a ticket.").name == "triage_ticket"


def test_a_non_model_return_is_rejected() -> None:
    def plain(x: str) -> dict[str, Any]:
        return {}

    with pytest.raises(CatalogError, match="pydantic model"):
        from_jev_fn(plain)


def test_the_tool_builds_the_model_and_keeps_probabilities(tool: JevFnTool) -> None:
    script = {
        "tool": {"triage": 0.95, "NO_TOOL": 0.04, "UNSUPPORTED": 0.01},
        "triage.urgent": 0.9,
        "triage.category": {"billing": 0.8, "tech": 0.15, "other": 0.03, "NOT_STATED": 0.01, "NONE_OF_THESE": 0.01},
        "triage.mood": {"sad": 0.85, "happy": 0.1, "NOT_STATED": 0.03, "NONE_OF_THESE": 0.02},
        "triage.priority": {2: 0.9, 1: 0.05, 3: 0.05},  # Score level indexes: level 2 is priority 3
        "triage.seats": {"3": 0.9, "NOT_STATED": 0.05, "NONE_OF_THESE": 0.05},
        "triage.verbosity": {0: 0.9, 1: 0.05, 2: 0.05},
        "triage.ratio": {"NOT_STATED": 0.9, "NONE_OF_THESE": 0.1},
    }
    router = Router([tool], backend=ScriptedBackend(script), policy=cookbook_policy(),
                    context=Context(now=SCENARIO_NOW))  # fmt: skip
    decision = router.decide("I was double charged, 3 seats are affected, please fix it now")
    result = tool.result(decision, proposed=True)
    assert result.decision is decision and result.outcome == decision.outcome.value
    assert result.p["category"] == decision.slots["category"].p >= 0.8 and result.p["mood"] >= 0.85
    assert [v for v, _ in result.alternatives["category"]][0] == "tech"  # runner-ups are kept, not discarded
    assert decision.call is not None and decision.call.arguments["seats"] == 3
    assert isinstance(result.value, Triage) and result.value.mood is Mood.SAD and result.value.priority == 3
    assert result.value.verbosity == 0 and result.confidence is not None
    built = tool(urgent=True, category="tech", mood="happy", priority=2, seats=4, verbosity=1, ratio=0.5)
    assert isinstance(built, Triage) and built.mood is Mood.HAPPY


def test_cookbook_policy() -> None:
    policy = cookbook_policy()
    assert policy.version == COOKBOOK_VERSION and policy.probes.present == () and policy.probes.reverse == ()
    for tier in ("read", "write", "external"):
        rule = policy.tier(tier)
        assert (rule.composition, rule.execute, rule.confirm, rule.authorized) == ("W", 0.60, None, None)
    assert policy.tiers.critical == Policy().tiers.critical  # critical keeps its own rule
    assert policy.sha256 != Policy().sha256


ENUM_TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "set_light", "description": "Switch a light on or off.",
     "parameters": {"type": "object", "required": ["room", "state"],
                    "properties": {"room": {"type": "string", "enum": ["kitchen", "office"]},
                                   "state": {"type": "string", "enum": ["on", "off"]}}}}},
    {"type": "function", "function": {"name": "wire_money", "description": "Wire money.",
     "parameters": {"type": "object", "properties": {"speed": {"type": "string", "enum": ["fast", "slow"]}}},
     "x-jev": {"risk": "critical"}}},
]  # fmt: skip


def test_cookbook_catalog_compiles_to_the_cookbook_shape() -> None:
    catalog = cookbook_catalog(ENUM_TOOLS)
    assert catalog.get("set_light").tier is Tier.READ  # 'set' would be write
    assert catalog.get("wire_money").tier is Tier.CRITICAL  # inline x-jev.risk wins over the hints
    assert cookbook_hints(catalog).entries == {"set_light": {"risk": "read"}, "wire_money": {"risk": "read"}}
    backend = ScriptedBackend({"tool": {"set_light": 0.9, "NO_TOOL": 0.1}, "set_light.room": "kitchen",
                               "set_light.state": "on"})  # fmt: skip
    router = Router(cookbook_catalog(ENUM_TOOLS[:1]), backend=backend, policy=cookbook_policy(),
                    context=Context(now=SCENARIO_NOW))  # fmt: skip
    decision = router.decide("Turn the kitchen light on")
    assert list(backend.requests[0].questions) == ["tool", "set_light.room", "set_light.state"]
    assert "NO_TOOL" in backend.requests[0].questions["tool"].criteria  # type: ignore[union-attr]
    assert decision.outcome is Outcome.EXECUTE and decision.confidence is not None
    assert decision.confidence.composition == "W"

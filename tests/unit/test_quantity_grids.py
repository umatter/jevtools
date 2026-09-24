"""Quantity grids (spec §4.2.4): grid values are offered only in the clarify menu of a required quantity slot without
a default that nothing stated, never mixed into a pool; ``jt.compat.from_jev_fn`` ``ge``/``le`` grids flow through
the same path. Answers are scripted (plumbing and policy, never Jev accuracy)."""

from __future__ import annotations

from typing import Any

import pytest

from jevtools.backends.scripted import ScriptedBackend
from jevtools.compat import cookbook_policy, from_jev_fn
from jevtools.context import Context
from jevtools.kinds.base import ResolveContext, get_resolver
from jevtools.kinds.quantity import GRIDS, evenly, grid_candidates
from jevtools.policy import Outcome
from jevtools.router import Router
from jevtools.spec.catalog import Catalog
from tests.scenario.fixtures import SCENARIO_NOW
from tests.unit.test_compat import _jev_fn, triage


def timer_tool(**xjev: Any) -> dict[str, Any]:
    """``set_timer(duration_minutes, label?)``: a required quantity (minutes, 1–180) without a default."""
    return {"type": "function", "function": {
        "name": "set_timer", "description": "Start a countdown timer.", "x-jev": {"risk": "external", **xjev},
        "parameters": {"type": "object", "required": ["duration_minutes"], "properties": {
            "duration_minutes": {"type": "integer", "minimum": 1, "maximum": 180,
                                 "description": "The timer length in minutes"},
            "label": {"type": "string", "description": "The timer label"}}}}}  # fmt: skip


def timer_router(script: dict[str, Any], **xjev: Any) -> tuple[Router, ScriptedBackend]:
    backend = ScriptedBackend(script)
    return Router([timer_tool(**xjev)], backend=backend, context=Context(now=SCENARIO_NOW)), backend


TIMER = {"tool": {"set_timer": 0.95, "NO_TOOL": 0.04, "UNSUPPORTED": 0.01}, "set_timer.authorized": 0.97,
         "set_timer.label.accept.*": 0.2}  # fmt: skip


def test_evenly_keeps_both_ends() -> None:
    assert evenly(list(range(1, 21)), 4) == [1, 7, 14, 20]
    assert evenly([0, 1, 2], 4) == [0, 1, 2] and evenly(list(range(11)), 4) == [0, 3, 7, 10]
    assert evenly([5, 6], 1) == [5]


def test_grid_values_are_unit_aware_and_schema_checked() -> None:
    tool = Catalog.from_openai([timer_tool()])["set_timer"]
    slot = tool.slot("duration_minutes")
    rc = ResolveContext(ctx=Context(now=SCENARIO_NOW))
    assert [c.value for c in grid_candidates(tool, slot, rc)] == list(GRIDS["minute"]) == [15, 30, 45, 60]
    assert all(c.prov == {"grid": True} and c.channel.value == "author" for c in grid_candidates(tool, slot, rc))
    short = timer_tool()
    short["function"]["parameters"]["properties"]["duration_minutes"]["maximum"] = 40
    small = Catalog.from_openai([short])["set_timer"]
    assert [c.value for c in grid_candidates(small, small.slot("duration_minutes"), rc)] == [15, 30]


def test_clarify_values_only_when_nothing_is_stated() -> None:
    router, _ = timer_router(TIMER)
    tool = router.catalog.get("set_timer")
    slot = tool.slot("duration_minutes")
    resolver = get_resolver("quantity")
    stated_ctx = Context(now=SCENARIO_NOW, messages="Set a timer for 20 minutes")
    stated = ResolveContext(ctx=stated_ctx, catalog=router.catalog)
    pool = resolver.pool(tool, slot, stated)
    assert [c.value for c in pool.candidates] == [20]  # the stated value, and no grid value next to it
    assert resolver.clarify_values(tool, slot, pool, stated) == []  # type: ignore[attr-defined]
    silent = ResolveContext(ctx=Context(now=SCENARIO_NOW, messages="Set a timer"), catalog=router.catalog)
    empty = resolver.pool(tool, slot, silent)
    assert empty.candidates == [] and len(resolver.clarify_values(tool, slot, empty, silent)) == 4  # type: ignore[attr-defined]
    defaulted = Catalog.from_openai([timer_tool()]).get("set_timer")
    label = defaulted.slot("label")  # optional: no grid
    assert get_resolver("quantity").clarify_values(defaulted, label, None, silent) == []  # type: ignore[attr-defined]


def test_missing_speculated_slot_gets_a_complete_call_grid_menu() -> None:
    router, backend = timer_router(TIMER, speculate="always")
    d = router.decide("Start a timer")
    assert (d.outcome, d.bottleneck.slot if d.bottleneck else None) == (Outcome.CLARIFY, "duration_minutes")
    assert d.prompt is not None and d.prompt.kind == "menu"
    assert d.prompt.text == "What should the timer length in minutes be?"
    texts = [o.text for o in d.prompt.options]
    assert texts[0] == "Start a countdown timer — duration 15 minutes" and texts[-1] == "Something else"
    assert "duration_minutes" not in str(backend.requests[0].questions)  # the grid never reaches the Ballot
    done = router.resume(d.pending_id or "", selection="pick:duration_minutes:2")  # a complete call: confirmation
    assert (done.outcome, done.call.arguments if done.call else None) == (Outcome.EXECUTE, {"duration_minutes": 45})
    assert len(backend.requests) == 1 and done.slots["duration_minutes"].channel == "user"  # no Jev call
    other = router.resume(d.pending_id or "", selection="other")
    assert other.prompt is not None and other.prompt.kind == "open"  # "Something else" opens the question


def test_from_jev_fn_grid_flows_through_the_menu() -> None:
    tool = from_jev_fn(_jev_fn(triage))
    script: dict[str, Any] = {
        "tool": {"triage": 0.95, "NO_TOOL": 0.04, "UNSUPPORTED": 0.01}, "triage.authorized": 0.97,
        "triage.urgent": 0.9, "triage.category": {"billing": 0.9, "NOT_STATED": 0.05, "NONE_OF_THESE": 0.05},
        "triage.mood": {"sad": 0.9, "NOT_STATED": 0.05, "NONE_OF_THESE": 0.05},
        "triage.priority": {2: 0.9, 1: 0.05, 3: 0.05}, "triage.verbosity": {0: 0.9, 1: 0.05, 2: 0.05},
        "triage.ratio": {"NOT_STATED": 0.9, "NONE_OF_THESE": 0.1},
    }  # fmt: skip
    backend = ScriptedBackend(script)
    router = Router([tool], backend=backend, policy=cookbook_policy(), context=Context(now=SCENARIO_NOW))
    d = router.decide("I was double charged, please fix it now")
    assert (d.outcome, d.rule) == (Outcome.CLARIFY, "P6.tool.not_speculated")  # seats: required, nothing stated
    assert d.prompt is not None and d.prompt.kind == "menu"
    assert [o.text for o in d.prompt.options] == ["1", "7", "14", "20", "Something else"]  # x-jev.values 1…20
    assert len(backend.requests) == 1 and set(backend.requests[0].questions) == {"tool"}
    picked = router.resume(d.pending_id or "", selection="pick:seats:1")
    assert len(backend.requests) == 2 and "tool" not in backend.requests[1].questions  # the tool is named
    assert picked.call is not None and picked.call.arguments["seats"] == 7
    assert picked.slots["seats"].p == 1.0 and picked.slots["seats"].channel == "user"


@pytest.mark.parametrize("unit", sorted(GRIDS))
def test_every_default_grid_is_increasing(unit: str) -> None:
    assert list(GRIDS[unit]) == sorted(set(GRIDS[unit])) and len(GRIDS[unit]) == 4

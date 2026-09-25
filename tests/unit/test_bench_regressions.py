"""Regressions found by the BFCL bench: values the ballot cannot carry are dropped at pool time, never raised."""

from __future__ import annotations

import jevtools as jt
from jevtools.kinds import get_resolver
from tests.kinds_support import custom


def test_fractional_amount_is_not_a_candidate_for_an_integer_money_slot() -> None:
    # "total" in the description makes this an integer money slot; 43.65 cannot be one, 80000 can
    area = {"type": "integer", "description": "The total solar panel area in square feet"}
    tool, rc = custom("solar", {"panelArea": area}, "Coordinates 43.65 and a total area of 80000 sq ft",
                      required=["panelArea"])  # fmt: skip
    slot = tool.slot("panelArea")
    assert slot.kind == "money"
    values = [c.value for c in get_resolver("money").pool(tool, slot, rc).candidates]
    assert 80000 in values and all(isinstance(v, int) for v in values)


def test_text_candidate_longer_than_the_accept_limit_is_not_nominated() -> None:
    query = {"type": "string", "description": "The search query"}
    request = "Search for " + "a very long question " * 250
    tool, rc = custom("search", {"query": query}, request, required=["query"])
    slot = tool.slot("query")
    pool = get_resolver(slot.kind).pool(tool, slot, rc)
    assert all(len(str(c.value)) <= rc.limits.accept_max for c in pool.candidates)
    search = {"type": "function", "function": {"name": "search", "description": "Search the web.", "parameters": {
        "type": "object", "properties": {"query": query}, "required": ["query"]}}}  # fmt: skip
    jt.Router([search], backend=jt.backends.ScriptedBackend({})).compile(request)  # the validator accepts the ballot

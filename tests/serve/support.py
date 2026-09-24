"""Helpers of the proxy tests: a scenario configuration (sources from the §13 fixtures) and scripted backends."""

from __future__ import annotations

from typing import Any

from jevtools.serve.config import ServeConfig

SCENARIO_CONTEXT: dict[str, Any] = {
    "now": "2026-09-24T14:05:00+02:00",
    "tz": "Europe/Zurich",
    "locale": "en-CH",
    "user": {"name": "Sam Muster", "home_city": "Zurich"},
}


def scenario_config(**fields: Any) -> ServeConfig:
    """The §13 context and sources (built by ``module:function`` factories of the scenario fixtures)."""
    data: dict[str, Any] = {
        "backend": "simulator",
        "context": dict(SCENARIO_CONTEXT),
        "sources": [
            {"name": "contacts", "type": "source", "function": "tests.scenario.fixtures:contacts"},
            {"name": "accounts", "type": "source", "function": "tests.scenario.fixtures:accounts"},
            {"name": "files", "type": "source", "function": "tests.scenario.fixtures:files"},
        ],
        **fields,
    }
    return ServeConfig.from_dict(data)


def tickets(query: Any) -> list[dict[str, str]]:
    """A ``provider`` source function (rows for any query)."""
    return [{"id": "T-1", "label": "Printer broken"}]


def make_simulator(**kw: Any) -> Any:
    """A ``module:factory`` backend."""
    from jevtools.backends.simulator import LexicalSimulator

    return LexicalSimulator(**kw)


__all__ = ["SCENARIO_CONTEXT", "make_simulator", "scenario_config", "tickets"]

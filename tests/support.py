"""Test helpers shared across test packages (import as ``tests.support``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.scenario.fixtures import SCENARIO_NOW

FIXTURES = Path(__file__).parent / "fixtures"
SCENARIO_SOURCES: list[dict[str, Any]] = [
    {"name": "contacts", "provides": ["email", "person"]},
    {"name": "accounts", "provides": ["account_id"]},
    {"name": "files", "provides": ["path", "file"]},
]
"""Source descriptors (name + provides) of the scenario registries, enough for catalog inference."""


def load_fixture(name: str) -> Any:
    """Parse a JSON file from ``tests/fixtures``."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


__all__ = ["FIXTURES", "SCENARIO_NOW", "SCENARIO_SOURCES", "load_fixture"]

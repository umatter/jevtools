"""Shared fixtures: the spec §13 scenario catalog and context, and the ``live`` marker gate."""

from __future__ import annotations

import os
from typing import Any

import pytest

from jevtools.context import Context
from jevtools.spec.catalog import Catalog
from tests.scenario.fixtures import scenario_context
from tests.support import SCENARIO_SOURCES, load_fixture


@pytest.fixture(autouse=True)
def _isolated_limits_cache(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Routers load the limits ``jevtools probe`` cached for their backend (§8.7); tests never read the developer's
    cache (``~/.cache/jevtools``)."""
    monkeypatch.setenv("JEVTOOLS_CACHE_DIR", str(tmp_path_factory.getbasetemp() / "jevtools-cache"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``@pytest.mark.live`` tests unless a Jev API key is configured."""
    if os.environ.get("TYPESAFE_API_KEY") or os.environ.get("OPENROUTER_API_KEY"):
        return
    skip = pytest.mark.skip(reason="live test: set TYPESAFE_API_KEY or OPENROUTER_API_KEY")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def scenario_tools() -> list[dict[str, Any]]:
    """The six OpenAI tools of spec §13.2."""
    return load_fixture("scenario_catalog.json")  # type: ignore[no-any-return]


@pytest.fixture
def scenario_catalog(scenario_tools: list[dict[str, Any]]) -> Catalog:
    """The §13.2 catalog compiled against the scenario sources (contacts, accounts, files)."""
    return Catalog.from_openai(scenario_tools, sources=SCENARIO_SOURCES)


@pytest.fixture
def ctx_default() -> Context:
    """The §13.1 context (no sources) with the R2 request and its history."""
    return scenario_context("Email Anna that I'll be 10 minutes late", history=True, sources=[])

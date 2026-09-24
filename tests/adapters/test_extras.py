"""The optional extras in ``pyproject.toml`` admit only versions the adapters run on.

The adapter tests use duck-typed fakes, so an extra's lower bound is never exercised by the suite; these tests pin the
minimum versions that were checked against the README snippets in scratch venvs (review of the docs-tests area):

- ``mcp``: :func:`jevtools.adapters.mcp.call_decision` passes ``meta={"jevtools/idempotency_key": …}`` to
  ``ClientSession.call_tool``, which accepts ``meta`` from mcp 1.19.0 on (1.10.0 to 1.18.0 raise ``TypeError``);
- ``pydantic-ai``: ``JevModel.request`` calls ``Model.prepare_request``, present from pydantic-ai-slim 1.0.13 on
  (1.0.0 to 1.0.12 raise ``AttributeError``);
- ``dev``: installs the ``openai`` SDK, so the ``wrap`` tests also run on the ``ChatCompletion`` shape users get.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement

from jevtools._compat import load_toml

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def extra(name: str) -> dict[str, Requirement]:
    """The requirements of one optional extra, by distribution name."""
    doc = load_toml(PYPROJECT.read_text(encoding="utf-8"))
    requirements = [Requirement(line) for line in doc["project"]["optional-dependencies"][name]]
    return {r.name: r for r in requirements}


@pytest.mark.parametrize(
    ("extra_name", "distribution", "last_broken", "first_working"),
    [
        ("mcp", "mcp", "1.18.0", "1.19.0"),  # ClientSession.call_tool(meta=…)
        ("pydantic-ai", "pydantic-ai-slim", "1.0.12", "1.0.13"),  # Model.prepare_request
    ],
)
def test_extra_lower_bounds_exclude_versions_the_adapters_crash_on(
    extra_name: str, distribution: str, last_broken: str, first_working: str
) -> None:
    specifier = extra(extra_name)[distribution].specifier
    assert not specifier.contains(last_broken), f"{distribution}=={last_broken} is admitted by {specifier}"
    assert specifier.contains(first_working), f"{distribution}=={first_working} is excluded by {specifier}"


def test_the_installed_pydantic_ai_has_prepare_request() -> None:
    models = pytest.importorskip("pydantic_ai.models")
    assert callable(getattr(models.Model, "prepare_request", None))


def test_dev_extra_installs_the_openai_sdk() -> None:
    assert "openai" in extra("dev")

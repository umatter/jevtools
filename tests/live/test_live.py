"""Live smoke tests (spec §10.7): real Jev endpoints, skipped without a key (the gate is in ``tests/conftest.py``).

Run them with ``python -m pytest -m live`` and ``TYPESAFE_API_KEY`` and/or ``OPENROUTER_API_KEY`` set. Each HTTP
backend runs when its own key is set: TypeSafe direct with ``TYPESAFE_API_KEY``, OpenRouter System One and OpenRouter
Decisions with ``OPENROUTER_API_KEY``; the others skip.

What is covered, per backend:

- the conformance probe (``jevtools.probe.run_probe``, 19 requests on a permissive backend; the limits file goes to
  a temporary path, never the developer's cache);
- the §13 cases R1–R5 and R7 in turn mode, and R6 as an agent loop over the fake demo workspace;
- a record → replay round trip through a cassette.

Only *invariants* are asserted: I1–I5 through ``jt.verify`` (values equal their options, channels were allowed, the
composition and the policy recompute, the ballot rebuilds), ``C ≤ W``, tool calls only on execute, schema-valid
arguments, no execute on the critical tier without a click, and nothing from the injected instruction in R6. Never
probabilities or outcomes: whatever Jev answers, these must hold. Cost per backend is well under one cent [I].
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import jevtools as jt
from jevtools.backends.cassette import Cassette
from jevtools.backends.http import HTTPBackend
from jevtools.context import Context
from jevtools.decision import Decision
from jevtools.demo import scenario, scripts
from jevtools.policy import Outcome, Tier
from jevtools.probe import run_probe
from jevtools.router import Router
from jevtools.trace import verify
from jevtools.validate import Limits

pytestmark = pytest.mark.live

BACKENDS: dict[str, tuple[str, Callable[..., HTTPBackend]]] = {
    "typesafe": ("TYPESAFE_API_KEY", HTTPBackend.typesafe),
    "openrouter_systemone": ("OPENROUTER_API_KEY", HTTPBackend.openrouter),
    "openrouter_decisions": ("OPENROUTER_API_KEY", HTTPBackend.openrouter_decisions),
}
"""The three HTTP backends (spec §8.2) and the key each needs."""

CASES: dict[str, tuple[str, bool]] = {
    "R1": (scripts.R1_REQUEST, False),
    "R2": (scripts.R2_REQUEST, True),
    "R2-no-history": (scripts.R2_REQUEST, False),
    "R3": (scripts.R3_REQUEST, False),
    "R4": (scripts.R4_REQUEST, False),
    "R5": (scripts.R5_REQUEST, False),
    "R7": (scripts.R7_REQUEST, False),
}
"""The §13 turn-mode cases: ``(request, with the R2 history)``."""


@pytest.fixture(scope="module", params=sorted(BACKENDS))
def backend(request: pytest.FixtureRequest) -> Iterator[HTTPBackend]:
    variable, factory = BACKENDS[request.param]
    key = os.environ.get(variable)
    if not key:
        pytest.skip(f"{request.param} needs {variable}")
    live = factory(key)
    yield live
    live.close()


def decide(backend: jt.backends.Backend, request: str, *, history: bool = False) -> tuple[Router, Decision, Context]:
    """One §13 decision in turn mode over the demo world, and the context ``verify`` rebuilds the ballot from."""
    messages = scenario.scenario_messages(request, history=history)
    router = scenario.demo_router(backend)
    decision = router.decide(messages)
    return router, decision, router.context_for(messages)


def assert_invariants(router: Router, d: Decision, ctx: Context) -> None:
    """I1–I5 and the tier rules, whatever Jev answered."""
    assert bool(d.tool_calls) == (d.outcome is Outcome.EXECUTE)  # only an execute carries tool calls
    if d.confidence is not None:
        assert d.confidence.call <= d.confidence.W + 1e-9  # I3: never above the weakest factor
    if d.outcome is Outcome.EXECUTE:
        jsonschema = pytest.importorskip("jsonschema")
        for call in d.tool_calls:
            tool = router.catalog.get(call.name)
            jsonschema.validate(call.arguments, tool.parameters)  # a schema-valid call
            if tool.tier is Tier.CRITICAL:
                assert d.trace.resumed_from is not None  # a critical call runs only after the user's click
    report = verify(d.trace, catalog=router.catalog, context=ctx)  # I1, I2, I3, policy; model out of the loop
    assert report.ok, report.failures


def test_conformance_probe(backend: HTTPBackend, tmp_path: Path) -> None:
    report = run_probe(backend, path=tmp_path / "limits.json")
    baseline = report.check("baseline")
    assert baseline is not None and baseline.ok
    names = [check.name for check in report.checks if check.name != "models"]  # one check per request
    assert report.calls == len(names) and {"smoke.e2", "smoke.e3", "smoke.e4"} <= set(names)
    assert Limits.from_file(tmp_path / "limits.json") == report.limits


@pytest.mark.parametrize("case", sorted(CASES))
def test_scenario_cases_keep_the_invariants(backend: HTTPBackend, case: str) -> None:
    request, history = CASES[case]
    router, d, ctx = decide(backend, request, history=history)
    assert d.usage.jev_calls >= 1
    assert_invariants(router, d, ctx)


def test_r6_loop_never_follows_the_injected_instruction(backend: HTTPBackend) -> None:
    workspace = scenario.Workspace()
    agent = jt.Agent(scenario.demo_router(backend), workspace.executors())
    result = agent.run([{"role": "user", "content": scenario.R6_REQUEST}])
    assert result.decisions
    for d in result.decisions:
        assert bool(d.tool_calls) == (d.outcome is Outcome.EXECUTE)
    assert workspace.transfers == []  # transfer_funds is critical: never without a click
    assert all(scenario.INJECTED_ADDRESS not in m["to"] for m in workspace.sent)
    assert all(scenario.INJECTED_IBAN not in str(arguments) for _, arguments in workspace.log)


def test_auto_picks_a_live_backend() -> None:
    backend = jt.backends.auto()
    if not isinstance(backend, HTTPBackend):
        pytest.skip(f"JEVTOOLS_BACKEND selects {backend.name}")
    router, d, ctx = decide(backend, scripts.R1_REQUEST)
    assert_invariants(router, d, ctx)


def test_a_recorded_cassette_replays_offline(backend: HTTPBackend, tmp_path: Path) -> None:
    path = tmp_path / "r1.jsonl"
    _, recorded, _ = decide(Cassette(path, "record", backend), scripts.R1_REQUEST)
    router, replayed, ctx = decide(Cassette(path, "replay"), scripts.R1_REQUEST)
    assert (replayed.outcome, replayed.rule, replayed.call) == (recorded.outcome, recorded.rule, recorded.call)
    assert _exchanges(replayed) == _exchanges(recorded)  # same requests, same answers (I4: keyed by the request)
    assert_invariants(router, replayed, ctx)


def _exchanges(d: Decision) -> list[tuple[str, str, str | None]]:
    """``(ballot, request, response)`` hashes of every Jev call of a decision (latency differs between runs)."""
    return [(r.ballot_sha256, c.request_sha256, c.response_sha256) for r in d.trace.rounds for c in r.calls]

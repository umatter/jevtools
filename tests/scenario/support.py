"""Helpers of the scenario tests: decide a §13 request through a scripted Router and check the result."""

from __future__ import annotations

import json
from typing import Any

from jevtools.backends.scripted import ScriptedBackend
from jevtools.context import Context, Mode
from jevtools.decision import Decision
from jevtools.router import Router
from jevtools.trace import verify
from tests.scenario import scripts
from tests.scenario.fixtures import scenario_messages, scenario_router


def decide(
    script: Any, request: str, *, history: bool = False, mode: Mode = "turn", **kw: Any
) -> tuple[Router, ScriptedBackend, Decision]:
    """A scripted Router over the §13 catalog and context, and its decision on ``request``."""
    router, backend = scenario_router(script, **kw)
    return router, backend, router.decide(scenario_messages(request, history=history), mode=mode)


def qids(backend: ScriptedBackend, call: int = 0) -> list[str]:
    """The question ids of one sent request, in plan order."""
    return list(backend.requests[call].questions)


def criteria(backend: ScriptedBackend, qid: str, call: int = 0) -> dict[str, Any]:
    """The criteria of a Choice in one sent request."""
    return scripts.criteria(backend.requests[call], qid)


def verified(
    decision: Decision, router: Router, request: str, *, history: bool = False, context: Context | None = None
) -> None:
    """``jt.verify`` passes on the decision's trace (with the catalog and the decision's context)."""
    ctx = (context or router.context).with_messages(scenario_messages(request, history=history))
    report = verify(decision.trace, catalog=router.catalog, context=ctx)
    assert report.ok, report.failures


def ordered_json(document: Any) -> str:
    """JSON with key order preserved (the spec's request listings are order-normative)."""
    return json.dumps(document, ensure_ascii=False)


__all__ = ["criteria", "decide", "ordered_json", "qids", "verified"]

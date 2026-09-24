"""Loop-test helpers: the §13 demo world's fake workspace and mailbox and the R6 scripts (§6.6), plus an Agent factory.

The workspace, the R6 invoice (with its injected instruction) and the R6 scripts are re-exported from
:mod:`jevtools.demo.scenario` and :mod:`jevtools.demo.scripts`, the same objects the examples and cassette fixtures use,
so the loop and injection tests cover the scenario the examples ship. Only :func:`r6_agent` and :func:`r6_messages`
live here.

All probabilities are illustrative [I]: scripted answers exercise plumbing and policy branches, never Jev accuracy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from jevtools.backends.scripted import ScriptedBackend
from jevtools.context import Context
from jevtools.demo.scenario import (
    FINANCE,
    INJECTED_ADDRESS,
    INJECTED_IBAN,
    INJECTION,
    INV_2291,
    INVOICE_TEXT,
    R6_REQUEST,
    Workspace,
)
from jevtools.demo.scripts import R6_MEMBERS, R6_STEP2, member_answers, observations_of, r6_script, r6_step1
from jevtools.loop import Agent, LoopBudget
from jevtools.router import Router
from jevtools.wire import DecisionRequest
from tests.scenario.fixtures import scenario_context, scenario_router


def r6_agent(
    script: Callable[[DecisionRequest], Mapping[str, Any]] | Mapping[str, Any],
    workspace: Workspace | None = None,
    *,
    context: Context | None = None,
    budget: LoopBudget | None = None,
    **kw: Any,
) -> tuple[Agent, Router, ScriptedBackend, Workspace]:
    """An Agent over the §13 scenario router with a fake workspace as executors."""
    ws = workspace or Workspace()
    ctx = context or scenario_context()
    router, backend = scenario_router(script, context=ctx, **kw)
    return Agent(router, ws.executors(), budget=budget), router, backend, ws


def r6_messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": R6_REQUEST}]


__all__ = [
    "FINANCE",
    "INJECTED_ADDRESS",
    "INJECTED_IBAN",
    "INJECTION",
    "INVOICE_TEXT",
    "INV_2291",
    "R6_MEMBERS",
    "R6_REQUEST",
    "R6_STEP2",
    "Workspace",
    "member_answers",
    "observations_of",
    "r6_agent",
    "r6_messages",
    "r6_script",
    "r6_step1",
]

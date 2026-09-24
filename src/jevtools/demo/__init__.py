"""A reusable demo world for examples, tutorials and tests: the spec §13 scenario (R1–R7).

**All data in this package is synthetic** (people, e-mail addresses, accounts, balances, IBANs, file paths, the invoice
text), and the scripted answers in :mod:`jevtools.demo.scripts` are the spec's illustrative [I] numbers: they exercise
plumbing and policy branches and are **never evidence about Jev's accuracy** (neither are the answers of the offline
``LexicalSimulator``).

- :mod:`jevtools.demo.scenario`: the world — ``now``, locale, user, the ``contacts``/``accounts``/``files`` sources, the
  six plain OpenAI tools of §13.2, :func:`~jevtools.demo.scenario.scenario_context`,
  :func:`~jevtools.demo.scenario.demo_router` / :func:`~jevtools.demo.scenario.scenario_router`, and a fake
  :class:`~jevtools.demo.scenario.Workspace` (executors for the agent loop);
- :mod:`jevtools.demo.scripts`: the §13.3 answer scripts (R1…R7, R2 without history, the R4 miss, the R6 steps).

Usage::

    from jevtools.demo import scenario, scripts

    router, backend = scenario.scenario_router(scripts.R2)
    decision = router.decide(scenario.scenario_messages(scripts.R2_REQUEST, history=True))
"""

from jevtools.demo import scenario, scripts
from jevtools.demo.scenario import (
    SCENARIO_NOW,
    SCENARIO_TOOLS,
    Workspace,
    default_sources,
    demo_router,
    scenario_catalog,
    scenario_context,
    scenario_messages,
    scenario_router,
    scenario_tools,
)

__all__ = [
    "SCENARIO_NOW",
    "SCENARIO_TOOLS",
    "Workspace",
    "default_sources",
    "demo_router",
    "scenario",
    "scenario_catalog",
    "scenario_context",
    "scenario_messages",
    "scenario_router",
    "scenario_tools",
    "scripts",
]

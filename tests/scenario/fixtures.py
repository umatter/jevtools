"""The spec §13 scenario at full size: a thin re-export of :mod:`jevtools.demo.scenario` (the synthetic demo world
the examples use), kept so ``tests.scenario.fixtures`` stays importable (including as ``module:function`` source
factories in proxy configurations).

``CATALOG_PATH`` is the §13.2 catalog as a plain JSON file (``tests/fixtures/scenario_catalog.json``);
``scenario_tools()`` returns the same tools from the demo module (a test pins that both agree).
"""

from __future__ import annotations

from pathlib import Path

from jevtools.demo.scenario import (
    ACCOUNT_ROWS,
    CONTACTS_SIZE,
    FILE_SYNONYMS,
    FILES_SIZE,
    LOCALE,
    NAMED_PATHS,
    R2_HISTORY,
    R6_NEAR_MISSES,
    SCENARIO_CONTACTS,
    SCENARIO_MODEL,
    SCENARIO_NOW,
    SCENARIO_TOOLS,
    USER,
    Workspace,
    account_rows,
    accounts,
    contact_rows,
    contacts,
    default_sources,
    demo_router,
    files,
    filler_contacts,
    generated_paths,
    scenario_catalog,
    scenario_context,
    scenario_messages,
    scenario_router,
    scenario_tools,
    workspace_paths,
)

CATALOG_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "scenario_catalog.json"

__all__ = [
    "ACCOUNT_ROWS",
    "CATALOG_PATH",
    "CONTACTS_SIZE",
    "FILES_SIZE",
    "FILE_SYNONYMS",
    "LOCALE",
    "NAMED_PATHS",
    "R2_HISTORY",
    "R6_NEAR_MISSES",
    "SCENARIO_CONTACTS",
    "SCENARIO_MODEL",
    "SCENARIO_NOW",
    "SCENARIO_TOOLS",
    "USER",
    "Workspace",
    "account_rows",
    "accounts",
    "contact_rows",
    "contacts",
    "default_sources",
    "demo_router",
    "files",
    "filler_contacts",
    "generated_paths",
    "scenario_catalog",
    "scenario_context",
    "scenario_messages",
    "scenario_router",
    "scenario_tools",
    "workspace_paths",
]

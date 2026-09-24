"""A small §13.1 workspace for resolver unit tests, built from the full-size fixtures' named rows.

``contacts`` holds the eight rows §13.1 names plus 12 fillers (so it is not sent whole); ``accounts`` has the four
rows (sent whole); ``files`` holds only the paths §13 names. End-to-end tests use ``tests.scenario.fixtures``.
"""

from __future__ import annotations

from typing import Any

from jevtools.context import Context
from jevtools.sources import FileIndex, Registry
from tests.scenario import fixtures
from tests.scenario.fixtures import ACCOUNT_ROWS, NAMED_PATHS, SCENARIO_CONTACTS

CONTACT_ROWS: list[dict[str, Any]] = [
    *(dict(row) for row in SCENARIO_CONTACTS),
    *(
        {"name": f"Filler Person{i}", "email": f"filler{i}@example.org", "notes": "rarely contacted",
         "last": "2024-01-01"}
        for i in range(1, 13)
    ),
]  # fmt: skip
FILE_PATHS: list[str] = list(NAMED_PATHS)

__all__ = ["ACCOUNT_ROWS", "CONTACT_ROWS", "FILE_PATHS", "accounts", "contacts", "files", "scenario_context"]


def contacts() -> Registry:
    return fixtures.contacts(CONTACT_ROWS)


def accounts() -> Registry:
    return fixtures.accounts()


def files() -> FileIndex:
    return fixtures.files(FILE_PATHS)


def scenario_context(request: str, *, history: bool = False, **kw: Any) -> Context:
    """The §13.1 context with the given request (after the R2 history turns when ``history``)."""
    return fixtures.scenario_context(request, history=history, sources=[contacts(), accounts(), files()], **kw)

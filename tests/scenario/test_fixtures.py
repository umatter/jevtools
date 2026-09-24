"""The §13.1 scenario fixtures: sizes, the named rows, and generated rows that never compete with them."""

from __future__ import annotations

import pytest

from jevtools.extract import run_extractors
from tests.scenario import scripts
from tests.scenario.fixtures import (
    ACCOUNT_ROWS,
    CONTACTS_SIZE,
    FILES_SIZE,
    NAMED_PATHS,
    SCENARIO_CONTACTS,
    SCENARIO_NOW,
    contact_rows,
    default_sources,
    scenario_catalog,
    scenario_context,
    workspace_paths,
)

NAMED_EMAILS = {row["email"] for row in SCENARIO_CONTACTS}


def test_sizes_and_named_rows() -> None:
    contacts, accounts, files = default_sources()
    rows = contact_rows()
    assert len(rows) == len(contacts) == CONTACTS_SIZE == 500 and len({r["email"] for r in rows}) == 500
    assert NAMED_EMAILS <= {r["email"] for r in rows}
    assert len(accounts) == len(ACCOUNT_ROWS) == 4 and accounts.whole
    paths = workspace_paths()
    assert len(files) == len(set(paths)) == FILES_SIZE == 3000 and set(NAMED_PATHS) <= set(paths)


def test_context_and_catalog() -> None:
    ctx = scenario_context(scripts.R1_REQUEST, history=True)
    assert ctx.current_time() == SCENARIO_NOW and ctx.locale == "en-CH" and ctx.user["home_city"] == "Zurich"
    assert ctx.request == scripts.R1_REQUEST and len(ctx.messages) == 3
    catalog = scenario_catalog()
    assert [t.name for t in catalog] == ["get_weather", "send_email", "create_event", "transfer_funds", "read_file",
                                         "search_web"]  # fmt: skip
    assert catalog.get("transfer_funds").tier.value == "critical"
    assert catalog.get("get_weather").slot("city").default_from == "user.home_city"


@pytest.mark.parametrize(
    "request_text",
    [scripts.R1_REQUEST, scripts.R2_REQUEST, scripts.R3_REQUEST, scripts.R4_REQUEST, scripts.R5_REQUEST,
     scripts.R7_REQUEST],
)  # fmt: skip
def test_generated_contacts_never_match_a_scenario_request(request_text: str) -> None:
    ctx = scenario_context(request_text)
    contacts = default_sources()[0]
    anchors = run_extractors(ctx, scenario_catalog()).anchors("contacts")
    matched = {contacts.rows[m.index]["email"] for anchor in anchors for m in anchor.attrs["matches"]}
    assert matched <= NAMED_EMAILS

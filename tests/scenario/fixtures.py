"""The spec §13 scenario at full size: context, sources, catalog and router builders.

Importable as ``tests.scenario.fixtures`` (the examples reuse it). Everything is deterministic:

- ``now`` = Thursday 2026-09-24 14:05 Europe/Zurich, locale ``en-CH``, user Sam Muster (home city Zurich);
- ``contacts``: a :class:`~jevtools.sources.Registry` of 500 rows — the eight rows named in §13.1 plus generated
  colleagues whose names, aliases and teams never match a §13 request (so the scenario pools stay as specified);
- ``accounts``: the four accounts of §13.1 (sent whole; balances are attributes, never sent to Jev);
- ``files``: a :class:`~jevtools.sources.FileIndex` of 3,000 workspace paths, including every path §13 names;
- the §13.2 catalog: plain OpenAI tools (``tests/fixtures/scenario_catalog.json``).
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from jevtools.backends.scripted import Script, ScriptedBackend
from jevtools.context import Context, Observation
from jevtools.router import Router
from jevtools.sources import FileIndex, Registry
from jevtools.spec.catalog import Catalog

SCENARIO_NOW = datetime(2026, 9, 24, 14, 5, tzinfo=ZoneInfo("Europe/Zurich"))
"""Thursday 2026-09-24 14:05 Europe/Zurich (UTC+02:00)."""
LOCALE = "en-CH"
USER: dict[str, str] = {"name": "Sam Muster", "home_city": "Zurich"}
CONTACTS_SIZE = 500
FILES_SIZE = 3000
CATALOG_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "scenario_catalog.json"

R2_HISTORY: list[dict[str, str]] = [
    {"role": "user", "content": "What's next on my calendar?"},
    {"role": "assistant", "content": "14:30 ACME quarterly review with Anna Keller."},
]
"""The two turns before R2 (a 14:30 review with Anna Keller)."""

# --------------------------------------------------------------------------------------------------------------------
# contacts
# --------------------------------------------------------------------------------------------------------------------

SCENARIO_CONTACTS: list[dict[str, Any]] = [
    {"name": "Anna Keller", "email": "anna.keller@acme.com", "team": "ACME",
     "notes": "Account Manager at ACME; last emailed 2 days ago", "last": "2026-09-22"},
    {"name": "Anna Rossi", "email": "anna.rossi@gmail.com",
     "notes": "personal contact; last emailed 3 weeks ago", "last": "2026-09-03"},
    {"name": "Annabel Frey", "email": "annabel.frey@muster.ch", "team": "Finance",
     "notes": "Finance, the user's own company; last emailed 5 months ago", "last": "2026-04-20"},
    {"name": "Bob Meier", "email": "bob.meier@muster.ch", "team": "Payments",
     "notes": "Payments team, the user's own company; 14 shared meetings in the last 90 days", "last": "2026-09-23"},
    {"name": "Robert Brown", "email": "rbrown@partner.io", "aliases": ["Bob"],
     "notes": "Partner Inc.; last met 4 months ago", "last": "2026-05-12"},
    {"name": "Carol Liu", "email": "carol.liu@muster.ch", "team": "Payments",
     "notes": "Payments team, the user's own company; 9 shared meetings in the last 90 days", "last": "2026-09-21"},
    {"name": "Caroline Weber", "email": "caroline.weber@muster.ch", "team": "Legal",
     "notes": "Legal, the user's own company; no shared meetings", "last": "2025-11-02"},
    {"name": "Finance Team", "email": "finance@muster.ch",
     "notes": "group mailbox of the Finance team", "last": "2026-09-01"},
]  # fmt: skip
"""The rows §13.1 names (their notes reproduce the option descriptions of §13.4 and §13.5)."""

_FIRST_NAMES = (
    "Liam", "Noah", "Mia", "Emma", "Luca", "Lea", "Elias", "Nora", "Jonas", "Sara", "David", "Julia", "Felix",
    "Laura", "Simon", "Lena", "Marco", "Elena", "Tim", "Eva", "Levin", "Fabio", "Nico", "Chiara", "Dario", "Yannick",
    "Leonie", "Samuel", "Livia", "Remo", "Urs", "Heidi", "Beat", "Ursula", "Hugo", "Ida", "Otto", "Paula",
    "Quentin", "Rita", "Stefan", "Tobias", "Vera", "Walter", "Xenia", "Yves", "Zoe", "Georg", "Ivo", "Mirco",
)  # fmt: skip
_LAST_NAMES = (
    "Ammann", "Baumann", "Brunner", "Egli", "Fischer", "Gerber", "Graf", "Huber", "Iten", "Jost", "Kunz", "Lehmann",
    "Moser", "Nussbaum", "Odermatt", "Pfister", "Quadri", "Roth", "Schmid", "Steiner", "Tanner", "Vogel", "Wyss",
    "Zbinden", "Zimmermann", "Hess", "Frei", "Bucher", "Suter", "Widmer",
)  # fmt: skip
_TEAMS = ("Engineering", "Sales", "Marketing", "Support", "Operations", "Research", "Design", "Logistics",
          "Procurement", "Security", "Facilities", "Training")  # fmt: skip
_DOMAINS = ("muster.ch", "example.com", "example.net")


def filler_contacts(n: int) -> list[dict[str, Any]]:
    """``n`` generated colleagues (deterministic). No name, alias or team overlaps the §13 requests' words."""
    rows: list[dict[str, Any]] = []
    for i, (last, first) in enumerate(itertools.product(_LAST_NAMES, _FIRST_NAMES)):
        if i >= n:
            break
        team = _TEAMS[i % len(_TEAMS)]
        domain = _DOMAINS[i % len(_DOMAINS)]
        company = ", the user's own company" if domain == "muster.ch" else ""
        months = 1 + i % 11
        rows.append({
            "name": f"{first} {last}",
            "email": f"{first.lower()}.{last.lower()}@{domain}",
            "team": team,
            "notes": f"{team}{company}; last emailed {months} months ago",
            "last": (SCENARIO_NOW.date() - timedelta(days=30 * months + i % 17)).isoformat(),
        })  # fmt: skip
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} distinct filler contacts can be generated")
    return rows


def contact_rows(size: int = CONTACTS_SIZE) -> list[dict[str, Any]]:
    """The §13.1 contacts: the named rows plus generated colleagues up to ``size`` rows."""
    return [dict(r) for r in SCENARIO_CONTACTS] + filler_contacts(size - len(SCENARIO_CONTACTS))


def contacts(rows: Sequence[Mapping[str, Any]] | None = None) -> Registry:
    """The ``contacts`` registry (key ``email``, label ``{name} <{email}>``, matched on name, aliases and team)."""
    return Registry(
        "contacts",
        contact_rows() if rows is None else rows,
        key="email",
        label="{name} <{email}>",
        describe="{notes}",
        match=["name", "aliases", "team"],
        provides=["email", "person"],
        attrs=["name", "team"],
        recency="last",
        hierarchy="team",
    )


# --------------------------------------------------------------------------------------------------------------------
# accounts
# --------------------------------------------------------------------------------------------------------------------

ACCOUNT_ROWS: list[dict[str, Any]] = [
    {"id": "acc_7731", "nickname": "Savings", "currency": "CHF", "iban_masked": "CH93…2957", "balance": 12500.0},
    {"id": "acc_2210", "nickname": "Checking", "currency": "CHF", "iban_masked": "CH56…1180", "balance": 3100.0},
    {"id": "acc_4410", "nickname": "Travel savings", "currency": "EUR", "iban_masked": "CH08…4410", "balance": 900.0},
    {"id": "acc_5102", "nickname": "Joint household", "currency": "CHF", "iban_masked": "CH12…5102", "balance": 4200.0},
]
"""The four accounts of §13.1."""


def account_rows(**balances: float) -> list[dict[str, Any]]:
    """The account rows, with balances overridden by id (``account_rows(acc_7731=100.0)``)."""
    return [{**row, "balance": balances.get(row["id"], row["balance"])} for row in ACCOUNT_ROWS]


def accounts(rows: Sequence[Mapping[str, Any]] | None = None) -> Registry:
    """The ``accounts`` registry (sent whole; ``balance`` is an attribute for constraints only)."""
    return Registry(
        "accounts",
        ACCOUNT_ROWS if rows is None else rows,
        key="id",
        label="{nickname} · {currency} · {iban_masked}",
        describe="{nickname} account in {currency}",
        match=["nickname"],
        provides=["account_id"],
        attrs=["nickname", "currency", "balance"],
    )


# --------------------------------------------------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------------------------------------------------

NAMED_PATHS: list[str] = [
    "services/payments/config/app.yaml",
    "services/payments/config/prod.yaml",
    "services/payments/src/handler.py",
    "services/payments/README.md",
    "services/billing/config/app.yaml",
    "services/auth/config/settings.toml",
    "finance/invoices/acme/2026-09-15_ACME_INV-2291.pdf",
    "finance/invoices/acme/2026-08-14_ACME_INV-2204.pdf",
    "finance/invoices/acme/2026-07-15_ACME_INV-2130.pdf",
    "finance/quotes/acme/2026-09-20_ACME_Q-118.pdf",
    "finance/invoices/outgoing/2026-09-18_INV-0412_to_ACME.pdf",
    "finance/invoices/globex/2026-09-01_GLOBEX_INV-77.pdf",
    "finance/reports/2026-Q2_summary.xlsx",
    "docs/onboarding/welcome.md",
    "docs/architecture/overview.md",
]
"""The paths §13 relies on (R4's payments config, R6's ACME invoices) and their near misses."""
FILE_SYNONYMS: dict[str, list[str]] = {
    "config": ["conf", "cfg", "settings", "values", "yaml", "toml", "ini", "env"],
    "payments": ["payment", "pay"],
}

_SERVICES = (
    "accounts", "analytics", "audit", "billing", "catalog", "checkout", "gateway", "inventory", "ledger", "mailer",
    "notifications", "orders", "pricing", "profiles", "reports", "search", "shipping", "users", "webhooks", "media",
    "scheduler", "exports", "imports", "loyalty", "reviews", "tax", "fraud", "identity", "sessions", "quotas",
)  # fmt: skip
_SERVICE_FILES = (
    "config/app.yaml", "config/prod.yaml", "config/staging.yaml", "src/main.py", "src/models.py", "src/routes.py",
    "src/client.py", "tests/test_main.py", "tests/test_models.py", "README.md", "Dockerfile", "Makefile",
)  # fmt: skip
_VENDORS = ("globex", "initech", "umbrella", "hooli", "stark", "wayne", "wonka", "tyrell", "cyberdyne", "soylent")
_DOC_AREAS = ("architecture", "onboarding", "runbooks", "howto", "decisions", "security", "api", "release-notes")
_TOPICS = ("overview", "setup", "deploy", "rollback", "incident", "monitoring", "backups", "access", "testing",
           "logging", "caching", "migrations", "alerts", "oncall", "glossary", "faq")  # fmt: skip


def _dates(start: date, n: int, step: int) -> list[str]:
    return [(start - timedelta(days=step * i)).isoformat() for i in range(n)]


R6_NEAR_MISSES: list[str] = [
    "finance/invoices/acme/2025-12-15_ACME_INV-1987.pdf",
    "finance/invoices/initech/2026-06-30_INITECH_INV-310.pdf",
    "finance/invoices/hooli/2026-05-31_HOOLI_INV-58.pdf",
]
"""With the GLOBEX invoice, the "four older or unrelated files" among R6's nine hits for {invoice, acme} (§6.6)."""


def generated_paths() -> list[str]:
    """Deterministic workspace paths around the named ones (services, docs, receipts, notes, exports). Apart from
    :data:`R6_NEAR_MISSES`, none mentions invoices, ACME or payments, so the §13 retrievals keep their hits."""
    out = [*R6_NEAR_MISSES]
    out += [f"services/{svc}/{name}" for svc in _SERVICES for name in _SERVICE_FILES]
    out += [f"docs/{area}/{topic}.md" for area in _DOC_AREAS for topic in _TOPICS]
    for v, vendor in enumerate(_VENDORS):
        for i, day in enumerate(_dates(date(2026, 9, 10), 24, 15)):
            out.append(f"finance/receipts/{vendor}/{day}_{vendor.upper()}_R-{100 + 37 * v + i}.pdf")
    out += [f"notes/meetings/{day}_{topic}.md" for day in _dates(date(2026, 9, 23), 60, 3) for topic in _TOPICS[:8]]
    out += [f"data/exports/{day}_{svc}_export.csv" for day in _dates(date(2026, 9, 20), 40, 7)
            for svc in _SERVICES[:20]]  # fmt: skip
    out += [f"archive/{year}/{area}/{topic}.md" for year in range(2015, 2026) for area in _DOC_AREAS
            for topic in _TOPICS]  # fmt: skip
    return out


def workspace_paths(size: int = FILES_SIZE) -> list[str]:
    """``size`` unique paths: the named ones first, then generated ones."""
    paths = list(dict.fromkeys([*NAMED_PATHS, *generated_paths()]))
    if len(paths) < size:
        raise ValueError(f"only {len(paths)} distinct workspace paths can be generated")
    return paths[:size]


def files(paths: Sequence[str] | None = None) -> FileIndex:
    """The ``files`` index (BM25 over path tokens with the §13.1 synonyms; hierarchy ``dirname``)."""
    return FileIndex("files", workspace_paths() if paths is None else list(paths), synonyms=FILE_SYNONYMS)


# --------------------------------------------------------------------------------------------------------------------
# context, catalog, router
# --------------------------------------------------------------------------------------------------------------------


@cache
def default_sources() -> tuple[Registry, Registry, FileIndex]:
    """The full-size ``contacts``, ``accounts`` and ``files`` sources (built once; they are read-only)."""
    return contacts(), accounts(), files()


def scenario_messages(request: str, *, history: bool = False) -> list[dict[str, str]]:
    """The conversation of one request (after the R2 history turns when ``history``)."""
    return [*(R2_HISTORY if history else []), {"role": "user", "content": request}]


def scenario_context(
    request: str | None = None,
    *,
    history: bool = False,
    sources: Sequence[Any] | None = None,
    observations: Sequence[Observation] = (),
    **kw: Any,
) -> Context:
    """The §13.1 context (time, locale, user, sources) with ``request`` as the latest user message."""
    messages = scenario_messages(request, history=history) if request is not None else []
    return Context(
        messages=messages,
        now=SCENARIO_NOW,
        locale=LOCALE,
        user=dict(USER),
        sources=list(default_sources() if sources is None else sources),
        observations=list(observations),
        **kw,
    )


def scenario_tools() -> list[dict[str, Any]]:
    """The six plain OpenAI tools of §13.2 (only ``transfer_funds`` carries ``x-jev``, plus ``get_weather.city``'s
    ``default_from``)."""
    tools: list[dict[str, Any]] = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return tools


def scenario_catalog(sources: Sequence[Any] | None = None) -> Catalog:
    """The §13.2 catalog compiled against the scenario sources."""
    return Catalog.from_openai(scenario_tools(), sources=list(default_sources() if sources is None else sources))


def scenario_router(script: Script, *, context: Context | None = None, **kw: Any) -> tuple[Router, ScriptedBackend]:
    """A Router over the §13 catalog and context whose Jev answers come from ``script``."""
    ctx = context or scenario_context()
    backend = ScriptedBackend(script, model="~typesafe/jev-latest")
    catalog = scenario_catalog(list(ctx.sources.values()))
    return Router(catalog, backend=backend, context=ctx, **kw), backend


__all__ = [
    "ACCOUNT_ROWS",
    "CONTACTS_SIZE",
    "FILES_SIZE",
    "FILE_SYNONYMS",
    "LOCALE",
    "NAMED_PATHS",
    "R2_HISTORY",
    "R6_NEAR_MISSES",
    "SCENARIO_CONTACTS",
    "SCENARIO_NOW",
    "USER",
    "account_rows",
    "accounts",
    "contact_rows",
    "contacts",
    "default_sources",
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

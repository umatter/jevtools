"""Generate the synthetic data files of the app-domain benchmark (deterministic; no randomness).

Run ``python -m jevtools.bench.app._generate`` to rewrite ``domains/*/data/*.json``; ``--check`` exits 1 when a file
is stale (a test runs it). Every row is invented: names, emails, IDs, IBANs and amounts are synthetic.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator
from itertools import product
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent / "domains"

FIRST = ["Nina", "Luca", "Mia", "Noah", "Elena", "David", "Sara", "Jonas", "Lea", "Marco", "Julia", "Felix",
         "Laura", "Simon", "Hannah", "Adrian", "Chiara", "Fabian", "Ines", "Kevin", "Olivia", "Patrick", "Rahel",
         "Timo"]  # fmt: skip
LAST = ["Arnold", "Baumann", "Crameri", "Dubois", "Egli", "Frei", "Gerber", "Huber", "Iten", "Jost", "Kunz",
        "Lüthi", "Moser", "Nussbaum", "Ott", "Pfister", "Roth", "Schmid", "Tanner", "Vogel", "Widmer",
        "Zürcher"]  # fmt: skip


def filler_people(n: int, *, domain: str, skip: set[str]) -> Iterator[tuple[str, str]]:
    """``n`` deterministic ``(name, email)`` pairs that avoid the names in ``skip``."""
    count = 0
    for last, first in product(LAST, FIRST):
        name = f"{first} {last}"
        if name in skip:
            continue
        local = f"{first}.{last}".lower().replace("ü", "ue").replace("ö", "oe").replace("ä", "ae")
        yield name, f"{local}@{domain}"
        count += 1
        if count >= n:
            return


# --------------------------------------------------------------------------------------------------------------------
# inbox: contacts and calendar events
# --------------------------------------------------------------------------------------------------------------------

INBOX_CONTACTS: list[dict[str, Any]] = [
    {"name": "Anna Keller", "email": "anna.keller@acme.com", "aliases": [], "org": "ACME",
     "notes": "Account Manager at ACME"},
    {"name": "Anna Rossi", "email": "anna.rossi@gmail.com", "aliases": [], "org": "",
     "notes": "Friend, personal address"},
    {"name": "Annabel Frey", "email": "annabel.frey@muster.ch", "aliases": [], "org": "Muster",
     "notes": "Finance, Muster AG"},
    {"name": "Bob Meier", "email": "bob.meier@muster.ch", "aliases": [], "org": "Muster", "notes": "Payments team"},
    {"name": "Robert Brown", "email": "rbrown@partner.io", "aliases": ["Bob"], "org": "Partner Inc",
     "notes": "Partner Inc, goes by Bob"},
    {"name": "Carol Liu", "email": "carol.liu@muster.ch", "aliases": [], "org": "Muster", "notes": "Payments team"},
    {"name": "Caroline Weber", "email": "caroline.weber@muster.ch", "aliases": [], "org": "Muster", "notes": "Legal"},
    {"name": "Thomas Becker", "email": "thomas.becker@muster.ch", "aliases": ["Tom"], "org": "Muster",
     "notes": "Engineering lead, goes by Tom"},
    {"name": "Lisa Wong", "email": "lisa.wong@muster.ch", "aliases": [], "org": "Muster", "notes": "Design"},
    {"name": "Lisa Wang", "email": "lisa.wang@globex.com", "aliases": [], "org": "Globex",
     "notes": "Procurement at Globex"},
    {"name": "Maria Garcia", "email": "maria.garcia@muster.ch", "aliases": [], "org": "Muster", "notes": "Marketing"},
    {"name": "Priya Nair", "email": "priya.nair@muster.ch", "aliases": [], "org": "Muster", "notes": "Data science"},
    {"name": "Finance Team", "email": "finance@muster.ch", "aliases": ["finance"], "org": "Muster",
     "notes": "Group address of the finance team"},
]  # fmt: skip


def inbox_contacts() -> list[dict[str, Any]]:
    """The named contacts plus 120 filler contacts at three organisations."""
    rows = [dict(r) for r in INBOX_CONTACTS]
    names = {r["name"] for r in rows}
    orgs = [("muster.ch", "Muster"), ("acme.com", "ACME"), ("globex.com", "Globex")]
    for i, (name, email) in enumerate(filler_people(120, domain="example.org", skip=names)):
        host, org = orgs[i % len(orgs)]
        rows.append({"name": name, "email": email.replace("example.org", host), "aliases": [], "org": org,
                     "notes": f"{org} contact"})  # fmt: skip
    return rows


INBOX_EVENTS: list[dict[str, Any]] = [
    {"id": "evt_101", "title": "1:1 Lisa Wong", "start": "2026-09-25T09:00:00+02:00", "when": "Fri 25 Sep 09:00"},
    {"id": "evt_102", "title": "Budget review", "start": "2026-09-29T14:00:00+02:00", "when": "Tue 29 Sep 14:00"},
    {"id": "evt_103", "title": "ACME quarterly review", "start": "2026-09-24T14:30:00+02:00",
     "when": "Thu 24 Sep 14:30"},
    {"id": "evt_104", "title": "Design sync", "start": "2026-09-28T11:00:00+02:00", "when": "Mon 28 Sep 11:00"},
    {"id": "evt_105", "title": "Payments standup", "start": "2026-09-25T08:45:00+02:00", "when": "Fri 25 Sep 08:45"},
    {"id": "evt_106", "title": "Dentist", "start": "2026-10-01T16:00:00+02:00", "when": "Thu 1 Oct 16:00"},
    {"id": "evt_107", "title": "Offsite planning", "start": "2026-10-06T10:00:00+02:00", "when": "Tue 6 Oct 10:00"},
    {"id": "evt_108", "title": "Globex procurement call", "start": "2026-09-30T15:00:00+02:00",
     "when": "Wed 30 Sep 15:00"},
]  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# crm: companies, contacts, deals, sales reps
# --------------------------------------------------------------------------------------------------------------------

CRM_COMPANIES: list[dict[str, Any]] = [
    {"id": "cmp_101", "name": "ACME AG", "city": "Zurich", "industry": "Manufacturing"},
    {"id": "cmp_102", "name": "Globex", "city": "Basel", "industry": "Chemicals"},
    {"id": "cmp_103", "name": "Müller AG", "city": "Bern", "industry": "Construction"},
    {"id": "cmp_104", "name": "Mueller GmbH", "city": "Munich", "industry": "Tooling"},
    {"id": "cmp_105", "name": "Initech", "city": "Geneva", "industry": "Software"},
    {"id": "cmp_106", "name": "Umbrella Health", "city": "Lausanne", "industry": "Healthcare"},
    {"id": "cmp_107", "name": "Stark Logistics", "city": "Zug", "industry": "Logistics"},
    {"id": "cmp_108", "name": "Wayne Foods", "city": "Lucerne", "industry": "Food"},
]  # fmt: skip
_FILLER_COMPANY = ["Alpine", "Rhine", "Summit", "Lakeside", "Granite", "Helvetia", "Nordwind", "Aare", "Juniper",
                   "Matterhorn", "Silverline", "Vineyard"]  # fmt: skip
_FILLER_SUFFIX = ["Systems", "Partners", "Energy", "Robotics"]


def crm_companies() -> list[dict[str, Any]]:
    rows = [dict(r) for r in CRM_COMPANIES]
    for i, (a, b) in enumerate(product(_FILLER_COMPANY, _FILLER_SUFFIX)):
        rows.append({"id": f"cmp_{200 + i}", "name": f"{a} {b}", "city": "Zurich", "industry": b})
    return rows


CRM_CONTACTS: list[dict[str, Any]] = [
    {"id": "ct_301", "name": "Maya Kunz", "email": "maya.kunz@umbrella-health.ch", "company": "Umbrella Health",
     "role": "Head of IT"},
    {"id": "ct_302", "name": "Leo Tanner", "email": "leo.tanner@initech.ch", "company": "Initech", "role": "CTO"},
    {"id": "ct_303", "name": "Maya Kuhn", "email": "maya.kuhn@globex.ch", "company": "Globex", "role": "Buyer"},
    {"id": "ct_304", "name": "Oliver Stark", "email": "oliver@stark-logistics.ch", "company": "Stark Logistics",
     "role": "CEO"},
]  # fmt: skip


def crm_contacts() -> list[dict[str, Any]]:
    rows = [dict(r) for r in CRM_CONTACTS]
    names = {r["name"] for r in rows}
    companies = [c["name"] for c in CRM_COMPANIES]
    for i, (name, email) in enumerate(filler_people(60, domain="example.com", skip=names)):
        rows.append({"id": f"ct_{400 + i}", "name": name, "email": email, "company": companies[i % len(companies)],
                     "role": "Contact"})  # fmt: skip
    return rows


CRM_DEALS: list[dict[str, Any]] = [
    {"id": "D-1001", "name": "ACME renewal 2027", "company": "ACME AG", "stage": "proposal", "owner": "Jonas Frei"},
    {"id": "D-1002", "name": "ACME expansion", "company": "ACME AG", "stage": "qualified", "owner": "Elena Moser"},
    {"id": "D-1003", "name": "Globex pilot", "company": "Globex", "stage": "qualified", "owner": "Marco Huber"},
    {"id": "D-1004", "name": "Initech data platform", "company": "Initech", "stage": "negotiation",
     "owner": "Priya Nair"},
    {"id": "D-1005", "name": "Umbrella Health CRM rollout", "company": "Umbrella Health", "stage": "proposal",
     "owner": "Jonas Frei"},
    {"id": "D-1006", "name": "Wayne Foods cold chain", "company": "Wayne Foods", "stage": "lead",
     "owner": "Marco Huber"},
    {"id": "D-1017", "name": "Stark Logistics fleet tracking", "company": "Stark Logistics", "stage": "proposal",
     "owner": "Elena Moser"},
]  # fmt: skip


def crm_deals() -> list[dict[str, Any]]:
    rows = [dict(r) for r in CRM_DEALS]
    reps = [r["name"] for r in CRM_REPS]
    for i, company in enumerate(crm_companies()[len(CRM_COMPANIES) :]):
        rows.append({"id": f"D-{1100 + i}", "name": f"{company['name']} onboarding", "company": company["name"],
                     "stage": "lead", "owner": reps[i % len(reps)]})  # fmt: skip
    return rows


CRM_REPS: list[dict[str, Any]] = [
    {"name": "Jonas Frei", "email": "jonas.frei@muster.ch", "team": "Enterprise"},
    {"name": "Elena Moser", "email": "elena.moser@muster.ch", "team": "Enterprise"},
    {"name": "Marco Huber", "email": "marco.huber@muster.ch", "team": "Mid-market"},
    {"name": "Priya Nair", "email": "priya.nair@muster.ch", "team": "Mid-market"},
    {"name": "Jonas Vogel", "email": "jonas.vogel@muster.ch", "team": "SMB"},
]  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# banking: accounts, payees, cards
# --------------------------------------------------------------------------------------------------------------------

BANK_ACCOUNTS: list[dict[str, Any]] = [
    {"id": "acc_7731", "nickname": "Savings", "currency": "CHF", "iban_masked": "CH93…2957", "balance": 12500.0},
    {"id": "acc_2210", "nickname": "Checking", "currency": "CHF", "iban_masked": "CH56…1180", "balance": 3100.0},
    {"id": "acc_4410", "nickname": "Travel savings", "currency": "EUR", "iban_masked": "CH08…4410", "balance": 900.0},
    {"id": "acc_5102", "nickname": "Joint household", "currency": "CHF", "iban_masked": "CH12…5102",
     "balance": 4200.0},
    {"id": "acc_6630", "nickname": "Brokerage cash", "currency": "USD", "iban_masked": "CH77…6630",
     "balance": 2000.0},
]  # fmt: skip

BANK_PAYEES: list[dict[str, Any]] = [
    {"id": "pay_swisscom", "name": "Swisscom", "category": "Telecom", "iban_masked": "CH11…0001"},
    {"id": "pay_sunrise", "name": "Sunrise", "category": "Telecom", "iban_masked": "CH11…0002"},
    {"id": "pay_ewz", "name": "EWZ Electricity", "category": "Utilities", "iban_masked": "CH11…0003"},
    {"id": "pay_primeo", "name": "Primeo Electricity", "category": "Utilities", "iban_masked": "CH11…0004"},
    {"id": "pay_landlord", "name": "Immo Stauffacher (rent)", "category": "Rent", "iban_masked": "CH11…0005"},
    {"id": "pay_helsana", "name": "Helsana health insurance", "category": "Insurance", "iban_masked": "CH11…0006"},
    {"id": "pay_tax", "name": "Canton of Zurich tax office", "category": "Taxes", "iban_masked": "CH11…0007"},
    {"id": "pay_anna", "name": "Anna Rossi", "category": "Friends", "iban_masked": "CH11…0008"},
    {"id": "pay_acme", "name": "ACME AG", "category": "Suppliers", "iban_masked": "CH11…0009"},
    {"id": "pay_gym", "name": "Kraftwerk Gym", "category": "Leisure", "iban_masked": "CH11…0010"},
]  # fmt: skip

BANK_CARDS: list[dict[str, Any]] = [
    {"id": "card_4421", "name": "Visa Platinum", "last4": "4421"},
    {"id": "card_1180", "name": "Mastercard", "last4": "1180"},
    {"id": "card_7702", "name": "Debit card", "last4": "7702"},
]  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# workspace: files, folders, colleagues
# --------------------------------------------------------------------------------------------------------------------

WORKSPACE_NAMED: list[str] = [
    "board/2026-Q3_board_deck.pptx", "board/2026-Q2_board_deck.pptx", "board/2026-Q1_board_deck.pptx",
    "finance/budget/2026_budget_v3.xlsx", "finance/budget/2025_budget_final.xlsx",
    "legal/contracts/drafts/acme_msa_draft_v2.docx", "legal/contracts/signed/globex_msa_2025.pdf",
    "design/mockups/checkout_v4.fig", "design/mockups/onboarding_v2.fig",
    "projects/atlas/roadmap.md", "projects/alpha/notes.txt", "projects/beta/2026-03-02_kickoff_notes.md",
    "reports/weekly/2026-09-18_weekly_report.md", "reports/weekly/2026-09-11_weekly_report.md",
    "reports/weekly/2026-09-04_weekly_report.md", "marketing/pricing/pricing_change_faq.md",
]  # fmt: skip
WORKSPACE_FOLDERS: list[str] = [
    "archive", "archive/board", "board", "design/mockups", "finance/budget", "legal/contracts/drafts",
    "legal/contracts/signed", "marketing/pricing", "projects/alpha", "projects/atlas", "projects/beta",
    "reports/weekly", "shared",
]  # fmt: skip


def workspace_files() -> list[str]:
    paths = list(WORKSPACE_NAMED)
    topics = ["meeting_notes", "draft", "export", "summary", "checklist", "agenda", "retro", "minutes"]
    for folder, topic in product(WORKSPACE_FOLDERS, topics):
        for n in (1, 2, 3):
            paths.append(f"{folder}/{topic}_{n}.md")
    return sorted(dict.fromkeys(paths))


def workspace_people() -> list[dict[str, Any]]:
    keep = {"Maria Garcia", "Priya Nair", "Thomas Becker", "Finance Team", "Caroline Weber", "Lisa Wong"}
    return [r for r in INBOX_CONTACTS if r["name"] in keep]


# --------------------------------------------------------------------------------------------------------------------
# helpdesk: tickets and agents
# --------------------------------------------------------------------------------------------------------------------

HELPDESK_TICKETS: list[dict[str, Any]] = [
    {"id": "INC-1043", "title": "VPN drops every hour", "requester": "Nina Egli", "status": "open", "team": "network"},
    {"id": "INC-1044", "title": "VPN client won't install on macOS", "requester": "Luca Roth", "status": "open",
     "team": "desktop"},
    {"id": "INC-1050", "title": "Printer on the 3rd floor is jammed", "requester": "Mia Ott", "status": "open",
     "team": "desktop"},
    {"id": "INC-1052", "title": "Database backup failed overnight", "requester": "Monitoring", "status": "open",
     "team": "database"},
    {"id": "INC-1057", "title": "Phishing email reported by finance", "requester": "Annabel Frey", "status": "open",
     "team": "security"},
    {"id": "INC-1061", "title": "Laptop screen flickers", "requester": "David Kunz", "status": "open",
     "team": "desktop"},
    {"id": "INC-1063", "title": "Password reset for new hire", "requester": "HR", "status": "open", "team": "desktop"},
    {"id": "INC-1070", "title": "Printer offline in the Basel office", "requester": "Sara Jost", "status": "open",
     "team": "desktop"},
]  # fmt: skip
_TICKET_TOPICS = ["Outlook keeps asking for the password", "Teams audio drops", "Shared drive is slow",
                  "Request for a second monitor", "Badge reader at the door fails", "Excel crashes on startup",
                  "New laptop setup", "Wi-Fi guest access", "Software licence request",
                  "Calendar sync error"]  # fmt: skip


def helpdesk_tickets() -> list[dict[str, Any]]:
    rows = [dict(r) for r in HELPDESK_TICKETS]
    people = [name for name, _ in filler_people(40, domain="muster.ch", skip=set())]
    for i in range(40):
        rows.append({"id": f"INC-{1100 + i}", "title": _TICKET_TOPICS[i % len(_TICKET_TOPICS)],
                     "requester": people[i], "status": "open", "team": "desktop"})  # fmt: skip
    return rows


HELPDESK_AGENTS: list[dict[str, Any]] = [
    {"name": "Priya Shah", "email": "priya.shah@muster.ch", "team": "network"},
    {"name": "Leo Brunner", "email": "leo.brunner@muster.ch", "team": "database"},
    {"name": "Nora Fischer", "email": "nora.fischer@muster.ch", "team": "desktop"},
    {"name": "Sven Keller", "email": "sven.keller@muster.ch", "team": "security"},
    {"name": "Aisha Khan", "email": "aisha.khan@muster.ch", "team": "desktop"},
    {"name": "Ben Ott", "email": "ben.ott@muster.ch", "team": "network"},
]  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# research: datasets, variables, collaborators, analysis files
# --------------------------------------------------------------------------------------------------------------------

RESEARCH_DATASETS: list[dict[str, Any]] = [
    {"id": "ds_101", "name": "survey_2024_wave1", "rows": 4812, "description": "Household survey 2024, wave 1"},
    {"id": "ds_102", "name": "survey_2024_wave2", "rows": 4391, "description": "Household survey 2024, wave 2"},
    {"id": "ds_103", "name": "housing_panel_2015_2025", "rows": 120450,
     "description": "Housing panel, prices and rents 2015-2025"},
    {"id": "ds_104", "name": "customer_churn_q3", "rows": 25003, "description": "Customer churn data, Q3 2026"},
    {"id": "ds_105", "name": "election_polls_2026", "rows": 612, "description": "Election polls 2026"},
    {"id": "ds_106", "name": "clinical_trial_bp", "rows": 842,
     "description": "Clinical trial of a blood pressure drug"},
]  # fmt: skip

RESEARCH_VARIABLES: list[dict[str, Any]] = [
    {"name": name, "label": label, "dataset": dataset}
    for dataset, variables in {
        "survey_2024_wave1": [("age", "Age in years"), ("income", "Household income (CHF)"),
                              ("education", "Highest education level"), ("vote", "Voted in the last election"),
                              ("gender", "Gender"), ("region", "Region")],
        "survey_2024_wave2": [("household_size", "Household size")],
        "housing_panel_2015_2025": [("price", "Sale price (CHF)"), ("rent", "Monthly rent (CHF)"),
                                    ("sqm", "Living area in square metres"), ("rooms", "Number of rooms"),
                                    ("city", "City"), ("year", "Year")],
        "customer_churn_q3": [("churned", "Customer churned (yes/no)"), ("tenure_months", "Tenure in months"),
                              ("monthly_charges", "Monthly charges (CHF)"), ("contract_type", "Contract type"),
                              ("support_calls", "Number of support calls")],
        "election_polls_2026": [("party", "Party"), ("share", "Vote share (%)"), ("pollster", "Pollster")],
        "clinical_trial_bp": [("systolic_bp", "Systolic blood pressure (mmHg)"),
                              ("diastolic_bp", "Diastolic blood pressure (mmHg)"), ("treatment", "Treatment arm"),
                              ("dose_mg", "Dose in mg")],
    }.items()
    for name, label in variables
]  # fmt: skip

RESEARCH_FILES: list[str] = [
    "analysis/clean_data.R", "analysis/fit_models.py", "analysis/churn_model.R", "analysis/polls_trend.R",
    "reports/churn_report.html", "reports/survey_wave2_summary.html", "reports/housing_prices_2025.html",
    "notebooks/eda_housing.ipynb", "notebooks/eda_trial.ipynb",
]  # fmt: skip


def research_collaborators() -> list[dict[str, Any]]:
    keep = {"Priya Nair", "Maria Garcia", "Thomas Becker"}
    rows = [r for r in INBOX_CONTACTS if r["name"] in keep]
    return [*rows, {"name": "Leo Tanner", "email": "leo.tanner@uni-example.ch", "aliases": [], "org": "University",
                    "notes": "Co-author, University"}]  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[], Any]] = {
    "inbox/data/contacts.json": inbox_contacts,
    "inbox/data/events.json": lambda: INBOX_EVENTS,
    "crm/data/companies.json": crm_companies,
    "crm/data/contacts.json": crm_contacts,
    "crm/data/deals.json": crm_deals,
    "crm/data/reps.json": lambda: CRM_REPS,
    "banking/data/accounts.json": lambda: BANK_ACCOUNTS,
    "banking/data/payees.json": lambda: BANK_PAYEES,
    "banking/data/cards.json": lambda: BANK_CARDS,
    "workspace/data/files.json": workspace_files,
    "workspace/data/folders.json": lambda: [{"path": f} for f in WORKSPACE_FOLDERS],
    "workspace/data/people.json": workspace_people,
    "helpdesk/data/tickets.json": helpdesk_tickets,
    "helpdesk/data/agents.json": lambda: HELPDESK_AGENTS,
    "research/data/datasets.json": lambda: RESEARCH_DATASETS,
    "research/data/variables.json": lambda: RESEARCH_VARIABLES,
    "research/data/collaborators.json": research_collaborators,
    "research/data/files.json": lambda: RESEARCH_FILES,
}


def render(value: Any) -> str:
    """One row per line: stable, diffable and compact."""
    if isinstance(value, list):
        return "[\n" + ",\n".join(json.dumps(v, ensure_ascii=False) for v in value) + "\n]\n"
    return json.dumps(value, ensure_ascii=False, indent=1) + "\n"


def main(argv: list[str] | None = None) -> int:
    check = "--check" in (argv if argv is not None else sys.argv[1:])
    stale = []
    for rel, make in GENERATORS.items():
        path = HERE / rel
        text = render(make())
        if path.exists() and path.read_text(encoding="utf-8") == text:
            continue
        stale.append(rel)
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    print(f"{len(stale)} file(s) {'stale' if check else 'written'}" + (f": {', '.join(stale)}" if stale else ""))
    return 1 if check and stale else 0


if __name__ == "__main__":
    sys.exit(main())

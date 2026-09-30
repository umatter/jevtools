"""Generate the held-out app-domain cases (``heldout/<domain>/cases.jsonl``; deterministic, no randomness).

The held-out set runs the same six apps (catalogs, contexts and data of ``domains/``) with cases generated from
templates over the data, so its gold labels follow from the rows rather than from hand labelling: a description that
fits exactly one record expects that record; one that fits several with nothing to tell them apart expects a menu
(``clarify``, every fitting record accepted); a search request expects the search tool. Every description is checked
against the data when the file is generated, so a slip in a template fails generation instead of producing a wrong
gold label. The set exists to decide between engine variants: features are not tuned against it.

    python -m jevtools.bench.app._heldout           # (re)write the files
    python -m jevtools.bench.app._heldout --check   # exit 1 when a file is missing or stale
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
DOMAINS_DIR = HERE / "domains"
HELDOUT_DIR = HERE / "heldout"

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("the a an my our of to for from with in on at and about".split())

Case = dict[str, Any]


_CANON = {"slides": "deck", "presentation": "deck", "pptx": "deck", "fig": "mockup", "xlsx": "budget"}
"""The workspace file index's synonyms (``context.json``), folded the same way on both sides."""


def _fold(word: str) -> str:
    word = _CANON.get(word, word)
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def words(text: str) -> set[str]:
    """Content words of a text (lower case; ``_``, ``-``, ``/`` and ``.`` separate words; plurals and the workspace
    synonyms folded, so "the mockups" fits ``design/mockups/checkout_v4.fig``)."""
    return {_fold(w) for w in _WORD.findall(text.lower().replace("_", " ")) if w not in _STOP}


def fitting(rows: Sequence[Any], description: str, fields: Callable[[Any], str]) -> list[Any]:
    """The rows whose ``fields`` text contains every content word of ``description``."""
    wanted = words(description)
    return [row for row in rows if wanted <= words(fields(row))]


def load(domain: str, name: str) -> Any:
    return json.loads((DOMAINS_DIR / domain / "data" / f"{name}.json").read_text(encoding="utf-8"))


class Builder:
    """Cases of one domain, numbered in order, with their gold checked against the data."""

    def __init__(self, domain: str, prefix: str, taken: set[str]) -> None:
        self.domain, self.prefix, self.taken = domain, prefix, taken
        self.cases: list[Case] = []

    def add(self, message: str, gold: dict[str, Any], *tags: str) -> None:
        if message.casefold() in self.taken:  # never repeat a message of the development bench
            return
        self.taken.add(message.casefold())
        self.cases.append({
            "id": f"{self.prefix}-{len(self.cases) + 1:03d}", "messages": message,
            "context": f"../../domains/{self.domain}/context.json",
            "catalog": f"../../domains/{self.domain}/catalog.json", "gold": gold, "tags": list(tags),
        })  # fmt: skip

    def one(self, message: str, tool: str, args: dict[str, Any], *tags: str, critical: bool = False) -> None:
        """A request that names one record (a shown call is right)."""
        outcomes = ["confirm"] if critical else ["execute", "confirm"]
        self.add(message, {"outcomes_ok": outcomes, "tool": tool, "args": args}, "unique", *tags)

    def several(self, message: str, tool: str, slot: str, values: Sequence[Any], *tags: str,
                args: dict[str, Any] | None = None) -> None:  # fmt: skip
        """A request that fits several records with nothing to tell them apart (a menu is right)."""
        assert len(values) >= 2, (message, values)
        gold: dict[str, Any] = {"outcomes_ok": ["clarify"], "tool": tool, "accepted": {slot: list(values)}}
        if args:
            gold["args"] = args
        self.add(message, gold, "several", *tags)

    def none(self, message: str, *tags: str) -> None:
        self.add(message, {"outcomes_ok": ["abstain"]}, *tags)


def unique(rows: Sequence[Any], description: str, fields: Callable[[Any], str]) -> Any:
    found = fitting(rows, description, fields)
    assert len(found) == 1, f"{description!r} fits {len(found)} rows"
    return found[0]


def ambiguous(rows: Sequence[Any], description: str, fields: Callable[[Any], str]) -> list[Any]:
    found = fitting(rows, description, fields)
    assert len(found) >= 2, f"{description!r} fits {len(found)} rows, expected several"
    return found


MESSAGES = ["the contract is signed", "I'll be ten minutes late", "the offsite moved to Friday",
            "the invoice is paid", "the draft is ready for review", "we need to reschedule",
            "the slides are uploaded", "lunch is on me tomorrow"]  # fmt: skip
EMAIL = ["Email {who} that {msg}", "Send {who} a note that {msg}", "Write to {who}: {msg}",
         "Drop {who} an email saying {msg}"]  # fmt: skip


def inbox(b: Builder) -> None:
    contacts = load("inbox", "contacts")
    events = load("inbox", "events")
    names = [c for c in contacts if not c["aliases"] and len(c["name"].split()) == 2]
    for i, c in enumerate(names[::6][:22]):
        who = c["name"]
        assert len([x for x in contacts if x["name"] == who]) == 1
        b.one(EMAIL[i % len(EMAIL)].format(who=who, msg=MESSAGES[i % len(MESSAGES)]), "send_email",
              {"to": c["email"]}, "full_name")  # fmt: skip
    firsts: dict[str, list[str]] = {}
    for c in contacts:
        firsts.setdefault(c["name"].split()[0], []).append(c["email"])
    shared = [f for f, emails in firsts.items() if len(emails) >= 2]
    for i, first in enumerate(shared[:16]):
        b.several(EMAIL[i % len(EMAIL)].format(who=first, msg=MESSAGES[(i + 3) % len(MESSAGES)]), "send_email",
                  "to", firsts[first], "first_name")  # fmt: skip
    for c in contacts:
        for alias in c["aliases"]:
            # an alias can also be someone's name ("Bob" is Robert Brown's alias and Bob Meier's first name)
            fits = [x["email"] for x in contacts if alias in x["aliases"] or alias.lower() in x["name"].lower().split()]
            message = f"Email {alias} that the numbers are final"
            if len(fits) == 1:
                b.one(message, "send_email", {"to": c["email"]}, "alias")
            else:
                b.several(message, "send_email", "to", fits, "alias")
    title = lambda e: e["title"]  # noqa: E731
    for i, e in enumerate(events):
        desc = e["title"]
        unique(events, desc, title)
        verb = ["Cancel the {d}", "Please cancel {d}", "Call off the {d}"][i % 3]
        b.one(verb.format(d=desc), "cancel_event", {"event_id": e["id"]}, "event")
        move = ["Move the {d} to Monday at 10", "Push the {d} to next Tuesday at 3pm", "Reschedule {d} to Friday 9am"]
        b.one(move[i % 3].format(d=desc), "reschedule_event", {"event_id": e["id"]}, "event")
    for desc in ("review",):
        rows = ambiguous(events, desc, title)
        b.several(f"Cancel the {desc}", "cancel_event", "event_id", [e["id"] for e in rows], "event")
        b.several(f"Move the {desc} to Monday at 10", "reschedule_event", "event_id", [e["id"] for e in rows], "event")
    for msg in ("Thanks, that's all for now", "What can you do?", "Don't email anyone today"):
        b.none(msg, "no_action")


STAGES = ["qualified", "proposal", "negotiation", "won", "lost"]


def crm(b: Builder) -> None:
    deals = load("crm", "deals")
    reps = load("crm", "reps")
    contacts = load("crm", "contacts")
    dname = lambda d: f"{d['name']} {d['company']}"  # noqa: E731
    for i, d in enumerate(deals[::3][:18]):
        if len(fitting(deals, d["name"], dname)) != 1:
            continue
        stage = STAGES[i % len(STAGES)]
        verb = ["Move the {n} deal to {s}", "Set {n} to {s}", "Mark the {n} deal as {s}"][i % 3]
        b.one(verb.format(n=d["name"], s=stage), "update_deal_stage", {"deal_id": d["id"], "stage": stage}, "deal")
    for i, d in enumerate(deals[1::5][:10]):
        b.one(["What's the status of {id}?", "Look up {id}", "Show me deal {id}"][i % 3].format(id=d["id"]),
              "get_deal", {"deal_id": d["id"]}, "typed_id")  # fmt: skip
    groups = ["ACME", "Alpine onboarding", "Rhine onboarding", "Summit onboarding", "Lakeside onboarding"]
    for i, g in enumerate(groups):
        rows = fitting(deals, g, dname)
        if len(rows) < 2:
            continue
        b.several(f"Move the {g} deal to {STAGES[i % 4]}", "update_deal_stage", "deal_id", [d["id"] for d in rows],
                  "deal", args={"stage": STAGES[i % 4]})  # fmt: skip
    rname = lambda r: r["name"]  # noqa: E731
    for i, d in enumerate(deals[2::7][:8]):
        rep = reps[i % len(reps)]
        unique(reps, rep["name"], rname)
        if len(fitting(deals, d["name"], dname)) != 1:
            continue
        b.one(f"Give the {d['name']} deal to {rep['name']}", "assign_deal",
              {"deal_id": d["id"], "owner_email": rep["email"]}, "rep")  # fmt: skip
    jonas = ambiguous(reps, "Jonas", rname)
    for d in deals[4:7]:
        if len(fitting(deals, d["name"], dname)) == 1:
            b.several(f"Assign the {d['name']} deal to Jonas", "assign_deal", "owner_email",
                      [r["email"] for r in jonas], "rep", args={"deal_id": d["id"]})  # fmt: skip
    for i, c in enumerate(contacts[::8][:8]):
        b.one(f"Log a call with {c['name']}: {['discussed pricing', 'follow up next week', 'wants a demo'][i % 3]}",
              "log_call", {"contact_id": c["id"]}, "contact")  # fmt: skip
    for msg in ("How does the pipeline work?", "Never mind", "Don't touch the ACME deals"):
        b.none(msg, "no_action")


def helpdesk(b: Builder) -> None:
    tickets = load("helpdesk", "tickets")
    agents = load("helpdesk", "agents")
    ttitle = lambda t: t["title"]  # noqa: E731
    for i, t in enumerate(tickets[::5][:10]):
        b.one(["Show me {id}", "What's the status of {id}?", "Open ticket {id}"][i % 3].format(id=t["id"]),
              "get_ticket", {"ticket_id": t["id"]}, "typed_id")  # fmt: skip
    for i, t in enumerate(tickets[1::6][:8]):
        a = agents[i % len(agents)]
        b.one(f"Assign {t['id']} to {a['name'].split()[0]}", "assign_ticket",
              {"ticket_id": t["id"], "assignee": a["email"]}, "typed_id", "agent")  # fmt: skip
    titles: dict[str, list[dict[str, Any]]] = {}
    for t in tickets:
        titles.setdefault(t["title"], []).append(t)
    once = [ts[0] for ts in titles.values() if len(ts) == 1]
    many = [ts for ts in titles.values() if len(ts) >= 2]
    verbs = [("Close the ticket about {d}", "close_ticket"), ("Escalate the {d} ticket", "escalate_ticket"),
             ("Show me the ticket about {d}", "get_ticket"), ("Add a comment to the {d} ticket: waiting on the vendor",
             "add_comment")]  # fmt: skip
    for i, t in enumerate(once):
        if len(fitting(tickets, t["title"], ttitle)) != 1:
            continue
        verb, tool = verbs[i % len(verbs)]
        b.one(verb.format(d=t["title"]), tool, {"ticket_id": t["id"]}, "title")
    for i, ts in enumerate(many[:8]):
        verb, tool = verbs[i % len(verbs)]
        b.several(verb.format(d=ts[0]["title"]), tool, "ticket_id", [t["id"] for t in ts], "title")
        t = ts[1]
        b.one(f"Show me {t['requester']}'s {t['title']} ticket", "get_ticket", {"ticket_id": t["id"]}, "requester")
    for msg in ("How do I reset my own password?", "Thanks, that's it", "Don't close anything yet"):
        b.none(msg, "no_action")


def banking(b: Builder) -> None:
    accounts = load("banking", "accounts")
    payees = load("banking", "payees")
    cards = load("banking", "cards")
    for i, a in enumerate(accounts):
        b.one(["What's the balance of my {n} account?", "How much is in {n}?"][i % 2].format(n=a["nickname"]),
              "get_balance", {"account": a["id"]}, "account")  # fmt: skip
        b.one(f"Show the transactions on {a['nickname']} since September 1", "list_transactions",
              {"account": a["id"]}, "account")  # fmt: skip
    for c in cards:
        b.one(f"Freeze my {c['name']}", "freeze_card", {"card_id": c["id"]}, "card")  # write tier
        b.one(f"Freeze the card ending in {c['last4']}", "freeze_card", {"card_id": c["id"]}, "card", "typed_id")
    pname = lambda p: f"{p['name']} {p['category']}"  # noqa: E731
    checking = next(a["id"] for a in accounts if a["nickname"] == "Checking")
    for i, p in enumerate(payees):
        desc = p["name"]
        if len(fitting(payees, desc, pname)) != 1:
            continue
        b.one(f"Pay {desc} {[80, 120, 45, 300][i % 4]} francs from Checking", "pay_bill",
              {"payee_id": p["id"], "from_account": checking}, "payee", critical=True)  # fmt: skip
    for desc in ("electricity", "telecom"):
        rows = ambiguous(payees, desc, pname)
        b.several(f"Pay the {desc} bill, 90 francs from Checking", "pay_bill", "payee_id", [p["id"] for p in rows],
                  "payee", args={"from_account": checking})  # fmt: skip
    for msg in ("What's a good savings rate?", "Don't pay the rent this month", "Thanks!"):
        b.none(msg, "no_action")


def research(b: Builder) -> None:
    datasets = load("research", "datasets")
    variables = load("research", "variables")
    collaborators = load("research", "collaborators")
    dfield = lambda d: f"{d['name']} {d['description']}"  # noqa: E731
    described = {"ds_103": "the housing panel", "ds_104": "the churn data", "ds_105": "the election polls",
                 "ds_106": "the clinical trial"}  # fmt: skip
    for i, d in enumerate(datasets):
        b.one(["Summarize {n}", "Give me an overview of {n}"][i % 2].format(n=d["name"]), "summarize_dataset",
              {"dataset_id": d["id"]}, "typed_id")  # fmt: skip
        if d["id"] in described:
            unique(datasets, described[d["id"]], dfield)
            b.one(f"Summarize {described[d['id']]}", "summarize_dataset", {"dataset_id": d["id"]}, "description")
    survey = ambiguous(datasets, "household survey", dfield)
    b.several("Summarize the household survey", "summarize_dataset", "dataset_id", [d["id"] for d in survey],
              "description")  # fmt: skip
    by_name = {d["name"]: d for d in datasets}
    for v in variables[::2]:
        d = by_name[v["dataset"]]
        b.one(f"Plot {v['label'].split(' (')[0].lower()} from {d['name']}", "plot_variable",
              {"dataset_id": d["id"], "variable": v["name"]}, "variable")  # fmt: skip
    reports = [f for f in load("research", "files") if f.endswith((".html", ".ipynb"))]
    for i, f in enumerate(reports):
        c = collaborators[i % len(collaborators)]
        stem = f.rsplit("/", 1)[1].rsplit(".", 1)[0].replace("_", " ")
        b.one(f"Share the {stem} {'notebook' if f.endswith('ipynb') else 'report'} with {c['name']}", "share_report",
              {"report_path": f, "recipient": c["email"]}, "report")  # fmt: skip
    for msg in ("What is a fixed effect?", "Thanks, looks good", "Don't run anything tonight"):
        b.none(msg, "no_action")


FILES_ONE = {
    "board/2026-Q1_board_deck.pptx": "the Q1 board deck",
    "board/2026-Q2_board_deck.pptx": "the Q2 board deck",
    "design/mockups/checkout_v4.fig": "the checkout mockup",
    "design/mockups/onboarding_v2.fig": "the onboarding mockup",
    "finance/budget/2025_budget_final.xlsx": "the final 2025 budget",
    "finance/budget/2026_budget_v3.xlsx": "the 2026 budget",
    "legal/contracts/drafts/acme_msa_draft_v2.docx": "the ACME MSA draft",
    "legal/contracts/signed/globex_msa_2025.pdf": "the signed Globex MSA",
    "marketing/pricing/pricing_change_faq.md": "the pricing change FAQ",
    "projects/atlas/roadmap.md": "the Atlas roadmap",
    "reports/weekly/2026-09-11_weekly_report.md": "the 2026-09-11 weekly report",
}
FILES_SEVERAL = [
    "the board deck",
    "the weekly report",
    "the budget",
    "the MSA",
    "the mockup",
    "the Alpha project notes",
]
TOPICS = ["pricing", "the budget", "onboarding", "the roadmap", "board meetings", "weekly reports", "contracts",
          "mockups", "the kickoff", "retros"]  # fmt: skip
SEARCH = ["Search my files for anything about {t}", "Find files about {t}", "Which files mention {t}?",
          "Look for anything on {t}", "Is there a doc about {t}?", "Do we have anything on {t}?"]  # fmt: skip


def workspace(b: Builder) -> None:
    files: list[str] = load("workspace", "files")
    people = load("workspace", "people")
    path = lambda f: f  # noqa: E731
    for i, (f, desc) in enumerate(FILES_ONE.items()):
        unique(files, desc, path)
        b.one(["Open {d}", "Show me {d}", "Read {d}"][i % 3].format(d=desc), "read_file", {"path": f}, "file")
        p = people[i % len(people)]
        if i % 2 == 0:
            b.one(f"Share {desc} with {p['name']}", "share_file", {"path": f, "recipient": p["email"]}, "file")
    for i, desc in enumerate(FILES_SEVERAL):
        rows = ambiguous(files, desc, path)
        b.several(["Open {d}", "Show me {d}"][i % 2].format(d=desc), "read_file", "path", rows, "file")
        p = people[i % len(people)]
        b.several(f"Share {desc} with {p['name']}", "share_file", "path", rows, "file",
                  args={"recipient": p["email"]})  # fmt: skip
    for i, topic in enumerate(TOPICS):
        for j in range(3):
            b.add(SEARCH[(i + j) % len(SEARCH)].format(t=topic),
                  {"outcomes_ok": ["execute", "confirm"], "tool": "search_files"}, "search")  # fmt: skip
    for msg in ("How do shared folders work?", "Thanks, that's all", "Don't move anything yet"):
        b.none(msg, "no_action")


BUILDERS: dict[str, tuple[str, Callable[[Builder], None]]] = {
    "inbox": ("h-inbox", inbox), "crm": ("h-crm", crm), "banking": ("h-bank", banking),
    "workspace": ("h-ws", workspace), "helpdesk": ("h-hd", helpdesk), "research": ("h-rs", research),
}  # fmt: skip


def development_messages() -> set[str]:
    out: set[str] = set()
    for path in DOMAINS_DIR.glob("*/cases.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            messages = json.loads(line)["messages"]
            text = messages if isinstance(messages, str) else messages[-1]["content"]
            out.add(str(text).casefold())
    return out


def build() -> Iterator[tuple[Path, str]]:
    taken = development_messages()
    for domain, (prefix, make) in BUILDERS.items():
        builder = Builder(domain, prefix, taken)
        make(builder)
        text = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in builder.cases)
        yield HELDOUT_DIR / domain / "cases.jsonl", text


def main(argv: Iterable[str] | None = None) -> int:
    check = "--check" in (list(argv) if argv is not None else sys.argv[1:])
    stale = []
    for path, text in build():
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if check:
        print(f"{len(stale)} file(s) stale" + (": " + ", ".join(str(p) for p in stale) if stale else ""))
        return 1 if stale else 0
    print(f"wrote {len(BUILDERS)} held-out domain(s) under {HELDOUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

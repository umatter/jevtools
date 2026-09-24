"""08 · Bring your own tools: a small support-desk toolset (not part of the spec's scenario).

How an adopter declares their own tools, sources and hints — everything a Jev ballot needs is built by code:

1. **sources**: two ``jt.Registry`` objects over the app's own rows (tickets, support agents). ``key`` is the value
   that goes into the call, ``label`` is what Jev sees (WYSIWYG), ``match`` drives mention matching, ``provides``
   tags let slots find the registry;
2. **tools**: plain Python functions with ``@jt.tool`` and ``Annotated`` markers (``jt.Ref(source=…)`` for entity
   references, ``Literal`` for enums, ``jt.Span()`` for words copied from the request, ``jt.Noun`` for wording);
3. **hints**: ``jt.hints({...})`` adds ``x-jev`` keys from the outside — here to a tool you do not own (a plain
   OpenAI JSON tool from another team) and to set a tier the verb table cannot infer ("assign");
4. a ``jt.Router`` over the catalog; the host executes the calls it gets back.

The requests: an ambiguous assignee (two Priyas) → clarify menu → click → execute; a priority change → execute; a
knowledge-base search → execute. All data is synthetic; the scripted answers in
``examples/fixtures/helpdesk.answers.json`` are hand-written and illustrative (never evidence about Jev).

Run: ``python examples/08_bring_your_own_tools.py [--backend scripted|sim|live]``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

import _show

import jevtools as jt

# -- 1. your data, as candidate sources ------------------------------------------------------------------------------
TICKETS: list[dict[str, str]] = [
    {"id": "T-1042", "subject": "Printer on the 3rd floor keeps jamming", "customer": "Globex", "opened": "2026-09-22",
     "priority": "normal"},
    {"id": "T-1043", "subject": "VPN drops every hour", "customer": "Initech", "opened": "2026-09-23",
     "priority": "high"},
    {"id": "T-1044", "subject": "Invoice PDF shows the wrong VAT rate", "customer": "Globex", "opened": "2026-09-23",
     "priority": "normal"},
    {"id": "T-1045", "subject": "Password reset link expired", "customer": "Umbrella", "opened": "2026-09-24",
     "priority": "low"},
]  # fmt: skip
AGENTS: list[dict[str, str]] = [
    {"handle": "priya.nair", "name": "Priya Nair", "team": "Hardware"},
    {"handle": "priya.shah", "name": "Priya Shah", "team": "Billing"},
    {"handle": "tom.berger", "name": "Tom Berger", "team": "Network"},
    {"handle": "lena.vogt", "name": "Lena Vogt", "team": "Accounts"},
]

tickets = jt.Registry(
    "tickets", TICKETS, key="id", label="{id} · {subject}", describe="{customer}; opened {opened}; priority {priority}",
    match=["subject", "customer", "id"], provides=["ticket_id"], attrs=["customer", "priority", "opened"],
)  # fmt: skip
agents = jt.Registry(
    "agents", AGENTS, key="handle", label="{name} ({team})", describe="{team} team",
    match=["name", "team"], provides=["agent"], attrs=["team"],
)  # fmt: skip

# -- 2. your tools ---------------------------------------------------------------------------------------------------
TicketId = Annotated[str, jt.Ref(source="tickets")]


@jt.tool  # "assign" is not in the verb table: its tier comes from HINTS below
def assign_ticket(
    ticket_id: TicketId,
    assignee: Annotated[str, jt.Ref(source="agents"), jt.Noun("the support agent who takes the ticket")],
) -> dict[str, str]:
    """Assign a support ticket to an agent."""
    return {"status": "assigned", "ticket": ticket_id, "to": assignee}


@jt.tool  # tier write ("set")
def set_priority(ticket_id: TicketId, priority: Literal["low", "normal", "high", "urgent"]) -> dict[str, str]:
    """Change the priority of a support ticket."""
    return {"status": "updated", "ticket": ticket_id, "priority": priority}


@jt.tool  # tier read ("search"); the query is a span of the user's own words
def search_kb(query: Annotated[str, jt.Span()]) -> list[str]:
    """Search the internal knowledge base for how-to articles."""
    return [f"KB-17 · {query}"]


CLOSE_TICKET: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "close_ticket",
        "description": "Close a support ticket as resolved.",
        "parameters": {"type": "object", "required": ["ticket_id"],
                       "properties": {"ticket_id": {"type": "string", "description": "The ticket to close"}}},
    },
}  # fmt: skip
"""A tool from another team's codebase: plain OpenAI JSON, left untouched."""

# -- 3. hints: x-jev keys added from the outside ---------------------------------------------------------------------
HINTS = jt.hints({
    "assign_ticket": {"risk": "write", "render": "Assign {ticket_id.label} to {assignee.label}"},
    "close_ticket": {"risk": "write"},
    "close_ticket.ticket_id": {"source": "tickets"},
})  # fmt: skip


def close_ticket(ticket_id: str) -> dict[str, str]:
    return {"status": "closed", "ticket": ticket_id}


EXECUTORS: dict[str, Callable[..., Any]] = {
    "assign_ticket": assign_ticket, "set_priority": set_priority, "search_kb": search_kb, "close_ticket": close_ticket,
}  # fmt: skip


def make_router(backend: jt.backends.Backend) -> jt.Router:
    """The catalog (functions + the foreign JSON tool + hints) and a context holding the two sources."""
    sources = [tickets, agents]
    context = jt.Context(now=datetime(2026, 9, 24, 14, 5, tzinfo=ZoneInfo("Europe/Zurich")), locale="en-CH",
                         user={"name": "Sam Muster", "role": "support lead"}, sources=sources)  # fmt: skip
    catalog = jt.Catalog.from_any([assign_ticket, set_priority, search_kb, CLOSE_TICKET], hints=HINTS, sources=sources)
    return jt.Router(catalog, backend=backend, context=context)


# --------------------------------------------------------------------------------------------------------------------

REQUESTS = (
    "Give the Globex printer ticket to Priya",
    "Set the VPN ticket to urgent",
    "Search the knowledge base for how to clear a paper jam",
)


def main(argv: Sequence[str] | None = None) -> dict[str, jt.Decision]:
    args = _show.parse_args(__doc__, argv)
    _show.header("08 · bring your own tools: a support desk", args.backend, ["helpdesk"])
    router = make_router(_show.backend(args.backend, "helpdesk"))
    for spec in router.catalog:
        slots = ", ".join(f"{s.name}: {s.kind}" + (f" ← {s.source}" if isinstance(s.source, str) else "")
                          for s in spec.slots)  # fmt: skip
        _show.kv("tool" if spec is router.catalog.tools[0] else "", f"{spec.name} [{spec.tier.value}]  {slots}")

    results: dict[str, jt.Decision] = {}
    for request in REQUESTS:
        _show.step(f'"{request}"')
        d = router.decide(request)
        _show.decision(d)
        option = _show.first_option(d, ("pick:assignee:0", "ok"))
        if d.pending_id is not None and option is not None:
            _show.step(f"click [{option}]")
            d = router.resume(d.pending_id, selection=option)
            _show.decision(d, slots=False)
        for call in d.tool_calls:  # the host executes what it gets back
            _show.kv("host ran", f"{call.name}(…) → {EXECUTORS[call.name](**call.arguments)}", wrap=False)
        results[request] = d
    return results


if __name__ == "__main__":
    main()

"""Stub resolvers and scenario data for engine tests (plan, decode, router, trace).

The engine only talks to resolvers through the :class:`~jevtools.kinds.base.Resolver` contract, so these small,
configurable resolvers stand in for the real kinds: a Choice resolver over fixed candidates per ``tool.slot`` (with
optional ``present`` Nouls and a ``widen`` hook) and an accept-Noul resolver for text slots.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from jevtools import templates
from jevtools.ballot import BallotQuestion, slot_qid
from jevtools.candidates import (
    Bottom,
    Candidate,
    Channel,
    Pool,
    apply_allow_list,
    assign_labels,
    canonical_order,
    value_key,
)
from jevtools.context import Context
from jevtools.kinds.base import (
    RESOLVERS,
    ResolveContext,
    SlotResult,
    ValueEntry,
    decode_choice,
    elect,
    probe_question,
    register_resolver,
    resolve_default,
    slot_question,
    unasked_result,
)
from jevtools.kinds.enum import CATALOG_DATA, load_catalog, register_catalog
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.wire import Answer, NoulAnswer
from tests.scenario.fixtures import LOCALE, R2_HISTORY, SCENARIO_NOW, USER

Candidates = Mapping[str, Sequence[Candidate]]


def cand(value: Any, channel: str = "user", *, text: str | None = None, label: str = "", **prov: Any) -> Candidate:
    """A candidate with ``prov`` keywords (``anchor=…``, ``whole=True``…); ``attrs=`` sets row attributes."""
    attrs = prov.pop("attrs", {})
    late = prov.pop("late", None)
    return Candidate(value=value, channel=Channel(channel), text=text, label=label, prov=prov, attrs=attrs, late=late)


class StubChoice:
    """A slot-Choice resolver over fixed candidates (``candidates["tool.slot"]``)."""

    def __init__(self, kind: str, candidates: Candidates | None = None, *, present: Sequence[str] = (),
                 widen_with: Candidates | None = None) -> None:  # fmt: skip
        self.kind = kind
        self.candidates: dict[str, list[Candidate]] = {k: list(v) for k, v in (candidates or {}).items()}
        self.present = set(present)
        self.widen_with = {k: list(v) for k, v in (widen_with or {}).items()}
        self.widened: list[str] = []

    def pool(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> Pool:
        raw = self.candidates.get(f"{tool.name}.{slot.name}", [])
        admitted, blocked = apply_allow_list(raw, slot.channels)
        labelled = assign_labels(admitted, slot=slot.name, label_max=rc.limits.label_max)
        return Pool(tool=tool.name, path=slot.path, kind=self.kind, candidates=canonical_order(labelled),
                    evidence_backed=any(c.is_evidence for c in labelled), blocked=blocked)  # fmt: skip

    def questions(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext) -> list[BallotQuestion]:
        default = resolve_default(tool, slot, rc.ctx)
        if not pool.candidates:
            return [probe_question(tool, slot, default)] if default is not None and not default.omit else []
        questions = [slot_question(tool, slot, pool.candidates, default)]
        if f"{tool.name}.{slot.name}" in self.present:
            questions.append(BallotQuestion(
                qid=slot_qid(tool.id, slot.qpath, "present"), family="present", tool=tool.name, path=slot.path,
                kind=slot.kind, stakes=slot.stakes, primitive="noul",
                instructions=templates.present_instructions(tool.intent, slot.noun),
                criteria=dict(templates.PRESENT_CRITERIA),
            ))  # fmt: skip
        return questions

    def decode(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer],
               rc: ResolveContext) -> SlotResult:  # fmt: skip
        questions = [q for q in rc.slot_questions(tool, slot) if q.primitive == "choice"]
        if not questions:
            return unasked_result(slot, resolve_default(tool, slot, rc.ctx))
        question = questions[-1]  # a widen bucket supersedes the round-1 question
        attrs = {c.label: c.attrs for c in pool.candidates}
        return decode_choice(slot, question, answers.get(question.qid), out_of_pool=rc.policy.shapes.out_of_pool,
                             attrs=attrs).with_(normalizer=f"{self.kind}@stub")  # fmt: skip

    def widen(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext,
              stage: str) -> tuple[Pool, list[BallotQuestion]]:  # fmt: skip
        """The ``Widenable`` hook: each stage offers the configured extra candidates in a ``bucket`` Choice."""
        extra = self.widen_with.get(f"{tool.name}.{slot.name}")
        if extra is None:
            return pool, []
        self.widened.append(stage)
        labelled = assign_labels(extra, slot=slot.name, label_max=rc.limits.label_max,
                                 taken=[c.label for c in pool.candidates])  # fmt: skip
        widened = pool.model_copy(update={"candidates": canonical_order([*pool.candidates, *labelled])})
        bucket = slot_question(tool, slot, labelled, resolve_default(tool, slot, rc.ctx), family="bucket",
                               suffix=("bucket", len(self.widened) - 1))  # fmt: skip
        return widened, [bucket]


class StubAccept:
    """An accept-Noul resolver for text slots: one ``T.P.accept.i`` Noul per candidate, elected = argmax n."""

    kind = "text"

    def __init__(self, candidates: Candidates | None = None) -> None:
        self.candidates: dict[str, list[Candidate]] = {k: list(v) for k, v in (candidates or {}).items()}

    def pool(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> Pool:
        raw = self.candidates.get(f"{tool.name}.{slot.name}", [])
        admitted, blocked = apply_allow_list(raw, slot.channels)
        labelled = assign_labels(admitted, slot=slot.name, label_max=rc.limits.label_max)
        return Pool(tool=tool.name, path=slot.path, kind="text", candidates=labelled, blocked=blocked,
                    evidence_backed=any(c.is_evidence for c in labelled))  # fmt: skip

    def questions(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext) -> list[BallotQuestion]:
        content = slot.stakes == "content"
        return [BallotQuestion(
            qid=slot_qid(tool.id, slot.qpath, "accept", i), family="accept", tool=tool.name, path=slot.path,
            kind="text", stakes=slot.stakes, primitive="noul",
            instructions=templates.accept_instructions(tool.intent, slot.noun, str(c.value), content=content),
            criteria=dict(templates.ACCEPT_CONTENT_CRITERIA) if content else None,
            meta={"value": c.value, "label": c.label, "channel": c.channel.value, "late": c.late},
        ) for i, c in enumerate(pool.candidates)]  # fmt: skip

    def decode(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer],
               rc: ResolveContext) -> SlotResult:  # fmt: skip
        questions = [q for q in rc.slot_questions(tool, slot) if q.family == "accept"]
        if not questions:
            return unasked_result(slot, resolve_default(tool, slot, rc.ctx))
        dist: dict[str, float] = {}
        values: dict[str, Any] = {}
        entries: dict[str, ValueEntry] = {}
        for q in questions:
            answer = answers.get(q.qid)
            n = answer.noul if isinstance(answer, NoulAnswer) else 0.0
            value = q.meta["value"]
            key = value_key(value)
            if key not in dist or n > dist[key]:
                dist[key], values[key] = n, value
                entries[key] = ValueEntry(
                    display=str(value),
                    label=q.meta.get("label"),
                    channel=Channel(q.meta["channel"]),
                    late=q.meta.get("late"),
                    p=n,
                )
        result = elect(path=slot.path, kind="text", stakes=slot.stakes, dist=dist, values=values, entries=entries,
                       out_of_pool=1.1, qids=[q.qid for q in questions])  # fmt: skip
        if max(dist.values(), default=0.0) < rc.policy.shapes.accept_min:
            return result.with_(
                value=Bottom.UNCOVERED,
                shape="uncovered_text",
                channel=None,
                factor=None if slot.stakes == "cosmetic" else max(dist.values(), default=0.0),
            )
        return result.with_(normalizer="text@stub")


@contextmanager
def resolvers(*items: Any, currencies: bool = True) -> Iterator[None]:
    """Register stub resolvers (and an ``iso4217`` stub catalog) for the duration of a test."""
    saved = dict(RESOLVERS)
    saved_catalog = CATALOG_DATA.get("iso4217")
    for item in items:
        register_resolver(item.kind, item)
    if currencies:
        register_catalog("iso4217", [{"value": "CHF", "text": "Swiss franc"}, {"value": "EUR", "text": "Euro"},
                                     {"value": "USD", "text": "US dollar"}])  # fmt: skip
    try:
        yield
    finally:
        RESOLVERS.clear()
        RESOLVERS.update(saved)
        if saved_catalog is None:
            CATALOG_DATA.pop("iso4217", None)
        else:
            CATALOG_DATA["iso4217"] = saved_catalog
        load_catalog.cache_clear()


# --------------------------------------------------------------------------------------------------------------------
# Scenario candidates (spec §13.1)
# --------------------------------------------------------------------------------------------------------------------

ANNAS = [
    cand("anna.keller@acme.com", "registry", label="Anna Keller <anna.keller@acme.com>",
         text='Contact matching "Anna": Account Manager at ACME; last emailed 2 days ago.', anchor="Anna",
         attrs={"name": "Anna Keller"}),
    cand("anna.rossi@gmail.com", "registry", label="Anna Rossi <anna.rossi@gmail.com>",
         text='Contact matching "Anna": personal contact; last emailed 3 weeks ago.', anchor="Anna",
         attrs={"name": "Anna Rossi"}),
    cand("annabel.frey@muster.ch", "registry", label="Annabel Frey <annabel.frey@muster.ch>",
         text='Contact similar to "Anna": Finance, the user\'s own company; last emailed 5 months ago.', anchor="Anna",
         attrs={"name": "Annabel Frey"}),
]  # fmt: skip
SUBJECTS = [cand("Running 10 minutes late", "author"), cand("Running late", "author"),
            cand("I'll be 10 minutes late", "user")]  # fmt: skip
BODIES = [cand("Hi ⟨recipient's first name⟩,\n\nI'll be 10 minutes late.\n\nBest,\nSam", "author",
               late={"placeholders": ["to.first_name"]}),
          cand("I'll be 10 minutes late.", "user")]  # fmt: skip
QUERIES = [cand("Anna that I'll be 10 minutes late", "user"), cand("Email Anna that I'll be 10 minutes late", "user")]


def account(key: str, nickname: str, currency: str, iban: str, balance: float, anchor: str | None) -> Candidate:
    """A row of the scenario ``accounts`` registry (balances are attributes, never sent to Jev)."""
    prov: dict[str, Any] = {"source": "accounts", "whole": True}
    if anchor:
        prov["anchor"] = anchor
    return cand(
        key,
        "registry",
        label=f"{nickname} · {currency} · {iban}",
        text=f"Account {nickname}.",
        attrs={"nickname": nickname, "currency": currency, "balance": balance},
        **prov,
    )


def accounts(savings_balance: float = 12000.0) -> list[Candidate]:
    """The four scenario accounts; ``savings``/``checking`` mentions anchor three of them."""
    return [account("acc_7731", "Savings", "CHF", "CH93…2957", savings_balance, "savings"),
            account("acc_2210", "Checking", "CHF", "CH56…1180", 2300.0, "checking"),
            account("acc_4410", "Travel savings", "EUR", "CH08…4410", 800.0, "savings"),
            account("acc_5102", "Joint household", "CHF", "CH12…5102", 5100.0, None)]  # fmt: skip


def scenario_resolvers(*, savings_balance: float = 12000.0, city: Sequence[Candidate] = (),
                       amount: Sequence[Candidate] = (cand("250.00", "user"),)) -> list[Any]:  # fmt: skip
    """Stubs for every non-enum kind of the §13.2 catalog, loaded with the scenario rows."""
    rows = accounts(savings_balance)
    ref = StubChoice("ref", {"send_email.to": ANNAS, "transfer_funds.from_account": rows,
                             "transfer_funds.to_account": rows}, present=["send_email.to"])  # fmt: skip
    return [
        ref,
        StubChoice("span", {"get_weather.city": list(city)}),
        StubChoice("money", {"transfer_funds.amount": list(amount)}),
        StubChoice("temporal"),
        StubChoice("quantity"),
        StubChoice("list"),
        StubAccept({"send_email.subject": SUBJECTS, "send_email.body": BODIES, "search_web.query": QUERIES}),
    ]


# --------------------------------------------------------------------------------------------------------------------
# Scenario context and fixtures
# --------------------------------------------------------------------------------------------------------------------

R2_MESSAGES = [*R2_HISTORY, {"role": "user", "content": "Email Anna that I'll be 10 minutes late"}]
R3_REQUEST = "Move 250 CHF from my savings to checking"


def scenario_context(messages: Any = ()) -> Context:
    """The §13.1 context (time, locale, user) with ``messages`` and no sources (the stubs hold the candidates)."""
    return Context(messages=list(messages) if not isinstance(messages, str) else messages, now=SCENARIO_NOW,
                   locale=LOCALE, user=dict(USER))  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# Scripted answers (spec §13 numbers)
# --------------------------------------------------------------------------------------------------------------------

R2_SCRIPT: dict[str, Any] = {
    "tool": {"send_email": 0.96, "get_weather": 0.01, "search_web": 0.01, "create_event": 0.005, "read_file": 0.005,
             "transfer_funds": 0.005, "NO_TOOL": 0.005, "UNSUPPORTED": 0.0},
    "send_email.authorized": 0.95,
    "send_email.to": {"Anna Keller <anna.keller@acme.com>": 0.86, "Anna Rossi <anna.rossi@gmail.com>": 0.07,
                      "Annabel Frey <annabel.frey@muster.ch>": 0.03, "NOT_STATED": 0.01, "NONE_OF_THESE": 0.03},
    "send_email.to.present": 0.97,
    "send_email.subject.accept.0": 0.93, "send_email.subject.accept.1": 0.4, "send_email.subject.accept.2": 0.6,
    "send_email.body.accept.0": 0.91, "send_email.body.accept.1": 0.88,
}  # fmt: skip
SAVINGS = "Savings · CHF · CH93…2957"
CHECKING = "Checking · CHF · CH56…1180"
TRAVEL = "Travel savings · EUR · CH08…4410"


def r3_script(**overrides: Any) -> dict[str, Any]:
    script: dict[str, Any] = {
        "tool": {"transfer_funds": 0.98, "NO_TOOL": 0.01, "UNSUPPORTED": 0.01},
        "transfer_funds.authorized": 0.98,
        "transfer_funds.from_account": {SAVINGS: 0.95, TRAVEL: 0.04, "NONE_OF_THESE": 0.01},
        "transfer_funds.to_account": {CHECKING: 0.97, SAVINGS: 0.01, "NONE_OF_THESE": 0.02},
        "transfer_funds.amount": {"250.00": 0.99, "NONE_OF_THESE": 0.01},
        "transfer_funds.currency": {"CHF": 0.95, "EUR": 0.01, "NOT_STATED": 0.02, "NONE_OF_THESE": 0.02},
        "transfer_funds.joint": {"250.00 CHF: Savings → Checking": 0.92, "NONE_OF_THESE": 0.08},
    }  # fmt: skip
    script.update(overrides)
    return script

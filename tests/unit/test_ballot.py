"""Ballot → JevRequest is normative: rebuild the verbatim R2 request of spec §13.4 byte for byte."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from jevtools import templates as t
from jevtools.ballot import (
    Ballot,
    BallotOption,
    BallotQuestion,
    SentinelSpec,
    ToolViability,
    flatten_instructions,
    slot_qid,
    tool_qid,
)
from jevtools.candidates import NO_TOOL, NONE_OF_THESE, NOT_STATED, UNSUPPORTED, Channel
from jevtools.canonical import canonical_json
from jevtools.context import Context, build_state
from jevtools.kinds.base import ResolveContext, get_resolver, probe_question, resolve_default
from jevtools.kinds.ref import RefResolver
from jevtools.policy import Policy
from jevtools.spec.catalog import Catalog
from tests.support import load_fixture

MODEL = "~typesafe/jev-latest"
SEND = "send an email from the user to one recipient"
WEB = "search the public web"


def _tool_question(catalog: Catalog) -> BallotQuestion:
    tools = sorted(catalog, key=lambda tool: tool.name)
    return BallotQuestion(
        qid="tool",
        family="tool",
        primitive="choice",
        instructions=t.tool_instructions(),
        options=[
            BallotOption(label=tool.name, value=tool.name, text=t.tool_option_text(tool.description)) for tool in tools
        ],
        sentinels={
            NO_TOOL: SentinelSpec(decodes_to="no_tool", text=t.TOOL_SENTINEL_TEXT[NO_TOOL]),
            UNSUPPORTED: SentinelSpec(decodes_to="unsupported", text=t.TOOL_SENTINEL_TEXT[UNSUPPORTED]),
        },
    )


def _accept(qid: str, intent: str, noun: str, candidate: str, *, content: bool, tool: str, path: str) -> BallotQuestion:
    return BallotQuestion(
        qid=qid,
        family="accept",
        tool=tool,
        path=(path,),
        kind="text",
        stakes="content" if content else "cosmetic",
        primitive="noul",
        instructions=t.accept_instructions(intent, noun, candidate, content=content),
        criteria=dict(t.ACCEPT_CONTENT_CRITERIA) if content else None,
    )


def _noul(
    qid: str, family: str, instructions: str, criteria: dict[str, str], path: tuple[str, ...] = ()
) -> BallotQuestion:
    return BallotQuestion(
        qid=qid,
        family=family,
        tool="send_email",
        path=path,
        primitive="noul",  # type: ignore[arg-type]
        instructions=instructions,
        criteria=criteria,
    )


def r2_ballot(catalog: Catalog, ctx: Context) -> Ballot:
    rc = ResolveContext(ctx=ctx, catalog=catalog)
    weather = catalog["get_weather"]
    city, unit = weather.slot("city"), weather.slot("unit")
    enum = get_resolver("enum")
    unit_questions = enum.questions(weather, unit, enum.pool(weather, unit, rc), rc)
    contacts = [
        (
            "Anna Keller <anna.keller@acme.com>",
            "anna.keller@acme.com",
            'Contact matching "Anna": Account Manager at ACME; last emailed 2 days ago.',
        ),
        (
            "Anna Rossi <anna.rossi@gmail.com>",
            "anna.rossi@gmail.com",
            'Contact matching "Anna": personal contact; last emailed 3 weeks ago.',
        ),
        (
            "Annabel Frey <annabel.frey@muster.ch>",
            "annabel.frey@muster.ch",
            'Contact similar to "Anna": Finance, the user\'s own company; last emailed 5 months ago.',
        ),
    ]
    send = catalog["send_email"]
    to = send.slot("to")
    to_question = BallotQuestion(
        qid=slot_qid(send.id, to.qpath),
        family="slot",
        tool=send.name,
        path=to.path,
        kind=to.kind,
        stakes=to.stakes,
        primitive="choice",
        instructions=t.slot_instructions(send.intent, t.slot_ask(to.noun)),
        options=[
            BallotOption(label=label, value=value, text=text, channel=Channel.REGISTRY, prov={"source": "contacts"})
            for label, value, text in contacts
        ],
        sentinels={
            NOT_STATED: SentinelSpec(decodes_to="missing", text=t.not_stated_text()),
            NONE_OF_THESE: SentinelSpec(decodes_to="uncovered", text=t.NONE_OF_THESE_TEXT),
        },
    )
    questions = [
        _tool_question(catalog),
        probe_question(weather, city, resolve_default(weather, city, ctx)),  # type: ignore[arg-type]
        *unit_questions,
        _accept(
            "search_web.query.accept.0",
            WEB,
            "the search query",
            "Anna that I'll be 10 minutes late",
            content=False,
            tool="search_web",
            path="query",
        ),
        _accept(
            "search_web.query.accept.1",
            WEB,
            "the search query",
            "Email Anna that I'll be 10 minutes late",
            content=False,
            tool="search_web",
            path="query",
        ),
        _noul(tool_qid(send.id, "authorized"), "authorized", t.auth_instructions(SEND), t.AUTH_CRITERIA),
        to_question,
        _noul(
            slot_qid(send.id, "to", "present"),
            "present",
            t.present_instructions(SEND, to.noun),
            t.PRESENT_CRITERIA,
            ("to",),
        ),
        *[RefResolver.verify_question(send, to, option, i) for i, option in enumerate(to_question.options)],
        *[
            _accept(
                f"send_email.subject.accept.{i}",
                SEND,
                "the subject line",
                s,
                content=False,
                tool="send_email",
                path="subject",
            )
            for i, s in enumerate(["Running 10 minutes late", "Running late", "I'll be 10 minutes late"])
        ],
        *[
            _accept(
                f"send_email.body.accept.{i}",
                SEND,
                "the body text of the email",
                b,
                content=True,
                tool="send_email",
                path="body",
            )
            for i, b in enumerate(
                ["Hi ⟨recipient's first name⟩,\n\nI'll be 10 minutes late.\n\nBest,\nSam", "I'll be 10 minutes late."]
            )
        ],
    ]
    viability = [
        ToolViability(
            name=tool.name,
            tier=tool.tier,
            viable="ok",
            speculated=tool.name in ("get_weather", "search_web", "send_email"),
        )
        for tool in catalog
    ]
    return Ballot(
        catalog_sha256=catalog.sha256,
        context_sha256=ctx.sha256,
        policy_version=Policy.default().version,
        state=build_state(ctx),
        tools=viability,
        questions=questions,
    )


def test_r2_request_is_byte_identical(scenario_catalog: Catalog, ctx_default: Context) -> None:
    expected: dict[str, Any] = load_fixture("spec_r2_request.json")
    ballot = r2_ballot(scenario_catalog, ctx_default)
    (request,) = ballot.to_requests(MODEL)
    assert request.to_wire() == expected
    assert ballot.request_bytes(MODEL) == [canonical_json(expected)]


def test_ballot_document_hash_and_round_trip(scenario_catalog: Catalog, ctx_default: Context) -> None:
    ballot = r2_ballot(scenario_catalog, ctx_default)
    doc = ballot.to_doc()
    assert list(doc) == [
        "spec",
        "ballot_sha256",
        "catalog_sha256",
        "context_sha256",
        "policy_version",
        "mode",
        "state",
        "tools",
        "questions",
        "calls",
    ]
    assert list(doc["questions"][6]) == [
        "qid",
        "family",
        "tool",
        "path",
        "kind",
        "stakes",
        "primitive",
        "instructions",
        "criteria_null",
        "options",
        "sentinels",
        "meta",
    ]
    assert doc["questions"][1]["sentinels"]["NOT_STATED"] == {
        "decodes_to": "default",
        "text": "No; the default (Zurich, the user's home city) would be used.",
        "value": "Zurich",
        "channel": "registry",
        "display": "Zurich, the user's home city",
    }
    assert doc["questions"][3]["criteria_null"] is True and doc["calls"] == [[q.qid for q in ballot.questions]]
    loaded = Ballot.from_doc(doc)
    assert loaded == ballot and loaded.sha256 == ballot.sha256 == doc["ballot_sha256"]
    assert ballot.to_json() == canonical_json(doc)
    tampered = {**doc, "policy_version": "other"}
    with pytest.raises(ValueError, match="mismatch"):
        Ballot.from_doc(tampered)


def test_split_calls_and_opaque_ids(scenario_catalog: Catalog, ctx_default: Context) -> None:
    ballot = r2_ballot(scenario_catalog, ctx_default)
    qids = [q.qid for q in ballot.questions]
    split = ballot.model_copy(update={"calls": [qids[:5], qids[5:]]})
    requests = split.to_requests(MODEL)
    assert [list(r.questions) for r in requests] == [qids[:5], qids[5:]]
    assert all(r.state == ballot.state for r in requests)
    (opaque,) = ballot.to_requests(MODEL, id_mode="opaque")
    assert list(opaque.questions)[:3] == ["q0001", "q0002", "q0003"]
    assert ballot.wire_ids("opaque")["send_email.to"] == "q0007"
    assert [q.qid for q in ballot.questions_for("send_email", ["to"])] == [
        "send_email.to", "send_email.to.present", "send_email.to.verify.0", "send_email.to.verify.1",
        "send_email.to.verify.2"]  # fmt: skip
    assert len(ballot.questions_for("send_email")) == 11 and ballot.question("tool").family == "tool"
    with pytest.raises(KeyError):
        ballot.question("nope")


def test_ballot_rejects_bad_plans(scenario_catalog: Catalog, ctx_default: Context) -> None:
    ballot = r2_ballot(scenario_catalog, ctx_default)
    with pytest.raises(ValidationError, match="duplicate"):
        ballot.model_validate({**ballot.model_dump(), "questions": [*ballot.questions, ballot.questions[0]]})
    with pytest.raises(ValidationError, match="exactly once"):
        ballot.model_validate({**ballot.model_dump(), "calls": [["tool"]]})


def test_question_shapes_and_decode_map() -> None:
    option = BallotOption(label="celsius", value="celsius", channel=Channel.AUTHOR)
    question = BallotQuestion(
        qid="get_weather.unit",
        family="slot",
        primitive="choice",
        instructions="?",
        options=[option],
        sentinels={NOT_STATED: SentinelSpec(decodes_to="default", value="celsius", text="d")},
    )
    assert question.labels == ["celsius", "NOT_STATED"]
    assert question.decode_label("celsius") == option and question.decode_label("x") is None
    assert question.decode_label(NOT_STATED).decodes_to == "default"  # type: ignore[union-attr]
    score = BallotQuestion(qid="t.p", family="slot", primitive="score", instructions="?", levels=["low", "high"])
    assert score.to_wire().model_dump()["criteria"] == ["low", "high"] and not score.criteria_null
    for bad in (
        {"primitive": "noul", "options": [option]},
        {"primitive": "choice", "criteria": {"true": "y"}},
        {"primitive": "score"},
        {"primitive": "noul", "levels": ["a"]},
        {"primitive": "choice", "sentinels": {"MAYBE": SentinelSpec(decodes_to="missing")}},
    ):
        with pytest.raises(ValidationError):
            BallotQuestion(qid="t.p", family="slot", instructions="?", **bad)  # type: ignore[arg-type]


def test_flattened_instructions_for_backends_without_object_instructions() -> None:
    accept = BallotQuestion(
        qid="search_web.query.accept.0",
        family="accept",
        primitive="noul",
        instructions={"question": "Would the search query below be a sensible choice?", "candidate": 'Zürich "now"'},
    )
    assert accept.to_wire(object_instructions=False).instructions == (
        'Would the search query below be a sensible choice?\nCandidate: "Zürich \\"now\\""'
    )
    assert flatten_instructions("plain") == "plain"
    ballot = Ballot(catalog_sha256="c", context_sha256="x", policy_version="p", state="s", questions=[accept])
    (request,) = ballot.to_requests("m", object_instructions=False)
    assert isinstance(request.questions["search_web.query.accept.0"].instructions, str)

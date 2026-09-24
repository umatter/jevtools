"""Composite and text resolvers: lists (§4.2.9), records and unions (§4.2.10), free text (§4.2.11)."""

from __future__ import annotations

import pytest

from jevtools.candidates import EXCLUDE, NONE_OF_THESE, NOT_STATED, Bottom, Channel
from jevtools.context import Observation
from jevtools.kinds.text import elect_accept
from tests.kinds_support import choice, custom, decode, noul, resolve, scenario
from tests.scenario_sources import contacts

R5 = "Book a 45 min sync with Bob and Carol next Tuesday at 3pm"
BOB, BROWN = "Bob Meier <bob.meier@muster.ch>", "Robert Brown <rbrown@partner.io>"
CAROL, WEBER = "Carol Liu <carol.liu@muster.ch>", "Caroline Weber <caroline.weber@muster.ch>"

# -- anchored lists ----------------------------------------------------------------------------------------------------


def test_r5_attendees_questions_and_decode() -> None:
    catalog, rc = scenario(R5)
    tool = catalog["create_event"]
    slot = tool.slot("attendees")
    pool, questions = resolve(tool, slot, rc)
    assert [q.qid for q in questions] == [
        "create_event.attendees.m0",
        "create_event.attendees.m1",
        "create_event.attendees.more",
    ]
    m0, m1, more = questions
    assert m0.labels == [BOB, BROWN, EXCLUDE, NONE_OF_THESE] and m1.labels == [CAROL, WEBER, EXCLUDE, NONE_OF_THESE]
    assert m0.sentinels[EXCLUDE].text == '"Bob" is mentioned, but is not one of the invitees.'
    assert m0.sentinels[NONE_OF_THESE].text == '"Bob" is someone not listed.'
    assert more.criteria is None and more.primitive == "noul"
    answers = {
        m0.qid: choice(m0, {BOB: 0.88, BROWN: 0.09, EXCLUDE: 0.01, NONE_OF_THESE: 0.02}),
        m1.qid: choice(m1, {CAROL: 0.93, WEBER: 0.05, EXCLUDE: 0.01, NONE_OF_THESE: 0.01}),
        more.qid: noul(0.04),
    }
    result = decode(tool, slot, pool, answers, rc)
    assert result.value == ["bob.meier@muster.ch", "carol.liu@muster.ch"]
    assert result.factor == pytest.approx(0.88 * 0.93 * 0.96) and round(result.factor, 4) == 0.7857
    assert result.channel is Channel.REGISTRY and result.shape == "ok" and result.probes == {"more": 0.04}
    assert [(a.part, a.value, round(a.p, 2)) for a in result.alternatives] == [
        ("m0", "rbrown@partner.io", 0.09),
        ("m1", "caroline.weber@muster.ch", 0.05),
    ]


def test_anchored_list_exclude_uncovered_and_more() -> None:
    catalog, rc = scenario(R5)
    tool = catalog["create_event"]
    slot = tool.slot("attendees")
    pool, (m0, m1, more) = resolve(tool, slot, rc)
    excluded = {m0.qid: choice(m0, {EXCLUDE: 0.9, BOB: 0.1}), m1.qid: choice(m1, {CAROL: 1.0}), more.qid: noul(0.1)}
    result = decode(tool, slot, pool, excluded, rc)
    assert result.value == ["carol.liu@muster.ch"] and result.factor == pytest.approx(0.9 * 1.0 * 0.9)
    uncovered = {**excluded, m0.qid: choice(m0, {NONE_OF_THESE: 0.7, BOB: 0.3})}
    assert decode(tool, slot, pool, uncovered, rc).shape == "out_of_pool"
    who_else = decode(tool, slot, pool, {**excluded, more.qid: noul(0.7)}, rc)
    assert who_else.shape == "missing" and "more" in who_else.flags


def test_group_mention_expands_to_item_nouls() -> None:
    registry = contacts().__class__(
        "contacts",
        [dict(r) for r in __import__("tests.scenario_sources", fromlist=["CONTACT_ROWS"]).CONTACT_ROWS],
        key="email",
        label="{name} <{email}>",
        match=["name", "aliases", "team"],
        provides=["email", "person"],
        groups="team",
    )
    props = {"attendees": {"type": "array", "items": {"type": "string", "format": "email"}}}
    tool, rc = custom("invite_people", props, "Invite everyone in Payments", sources=[registry], required=["attendees"])
    pool, questions = resolve(tool, tool.slot("attendees"), rc)
    assert [q.family for q in questions] == ["more", "item", "item"]
    more, bob, carol = questions
    assert "Should" in str(bob.instructions) and '"Bob Meier <bob.meier@muster.ch>"' in str(bob.instructions)
    result = decode(
        tool, tool.slot("attendees"), pool, {more.qid: noul(0.05), bob.qid: noul(0.9), carol.qid: noul(0.5)}, rc
    )
    assert result.value == ["bob.meier@muster.ch"] and result.shape == "flag_band"
    assert result.factor == pytest.approx(0.95 * 0.9 * 0.5)


def test_multi_select_and_enumerative_lists() -> None:
    props = {
        "tags": {"type": "array", "items": {"type": "string", "enum": ["bug", "urgent", "ui"]}},
        "names": {"type": "array", "items": {"type": "string"}},
    }
    tool, rc = custom("label_issue", props, 'Label it as an urgent bug and add "Apollo"')
    pool, questions = resolve(tool, tool.slot("tags"), rc)
    assert [q.qid for q in questions] == [
        "label_issue.tags.item.0",
        "label_issue.tags.item.1",
        "label_issue.tags.item.2",
    ]
    assert [c.label for c in pool.candidates] == ["bug", "ui", "urgent"]  # canonical order
    answers = {questions[0].qid: noul(0.95), questions[1].qid: noul(0.1), questions[2].qid: noul(0.85)}
    result = decode(tool, tool.slot("tags"), pool, answers, rc)
    assert result.value == ["bug", "urgent"] and result.factor == pytest.approx(0.95 * 0.9 * 0.85)
    pool, questions = resolve(tool, tool.slot("names"), rc)
    assert "Apollo" in [c.value for c in pool.candidates] and all(q.family == "item" for q in questions)


def test_list_without_anchors_probes_its_default() -> None:
    catalog, rc = scenario("Book a sync next Tuesday at 3pm")
    tool = catalog["create_event"]
    pool, (probe,) = resolve(tool, tool.slot("attendees"), rc)
    assert probe.family == "probe" and probe.sentinels[NOT_STATED].text == "No; the default ([]) would be used."
    result = decode(tool, tool.slot("attendees"), pool, {probe.qid: choice(probe, {NOT_STATED: 0.9})}, rc)
    assert result.value == [] and result.factor == pytest.approx(0.9)


def test_array_of_objects_basic() -> None:
    item = {
        "type": "object",
        "properties": {
            "email": {"type": "string", "format": "email"},
            "role": {"type": "string", "enum": ["required", "optional"]},
        },
    }
    props = {"people": {"type": "array", "items": item}}
    tool, rc = custom("create_meeting", props, "Meet Bob and Carol", sources=[contacts()], required=["people"])
    pool, questions = resolve(tool, tool.slot("people"), rc)
    assert [q.qid for q in questions] == [
        "create_meeting.people.m0.email",
        "create_meeting.people.m0.email.present",
        "create_meeting.people.m0.role",
        "create_meeting.people.m1.email",
        "create_meeting.people.m1.email.present",
        "create_meeting.people.m1.role",
        "create_meeting.people.more",
    ]
    q = {question.qid.removeprefix("create_meeting.people."): question for question in questions}
    answers = {
        q["m0.email"].qid: choice(q["m0.email"], {BOB: 0.9}),
        q["m0.role"].qid: choice(q["m0.role"], {"optional": 0.8, NOT_STATED: 0.2}),
        q["m1.email"].qid: choice(q["m1.email"], {CAROL: 0.95}),
        q["m1.role"].qid: choice(q["m1.role"], {NOT_STATED: 0.9}),
        q["more"].qid: noul(0.1),
    }
    result = decode(tool, tool.slot("people"), pool, answers, rc)
    assert result.value == [{"email": "bob.meier@muster.ch", "role": "optional"}, {"email": "carol.liu@muster.ch"}]
    assert result.factor == pytest.approx(0.9 * 0.8 * 0.95 * 0.9 * 0.9)


# -- records and unions ------------------------------------------------------------------------------------------------


def test_record_flattened_leaves() -> None:
    props = {
        "address": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {"type": "string", "description": "The city"},
                "zip": {"type": "string", "pattern": "^\\d{4}$"},
            },
        }
    }
    tool, rc = custom("ship_parcel", props, "Ship it to Zurich", required=["address"])
    pool, questions = resolve(tool, tool.slot("address"), rc)
    assert [q.qid for q in questions] == ["ship_parcel.address.city"] and pool.meta["viability"] == "ok"
    q = questions[0]
    result = decode(tool, tool.slot("address"), pool, {q.qid: choice(q, {"Zurich": 0.9, NOT_STATED: 0.1})}, rc)
    assert (
        result.value == {"city": "Zurich"}
        and result.factor == pytest.approx(0.9)
        and set(result.parts)
        == {
            "address.city",
            "address.zip",
        }
    )


def test_union_branch_choice() -> None:
    branch_a = {
        "title": "Card",
        "description": "Pay by card",
        "type": "object",
        "required": ["last4"],
        "properties": {"last4": {"type": "string", "pattern": "^\\d{4}$"}},
    }
    branch_b = {
        "title": "Invoice",
        "description": "Pay by invoice",
        "type": "object",
        "properties": {"note": {"type": "string", "enum": ["net30", "net60"]}},
    }
    props = {"method": {"oneOf": [branch_a, branch_b]}}
    tool, rc = custom("pay_order", props, "pay with net60 invoice", required=["method"])
    pool, questions = resolve(tool, tool.slot("method"), rc)
    assert pool.meta["branch_viability"] == {"method.b0": "empty:method.last4", "method.b1": "ok"}
    branch_q, note_q = questions
    assert branch_q.qid == "pay_order.method.branch" and branch_q.labels == ["Card", "Invoice", NONE_OF_THESE]
    answers = {
        branch_q.qid: choice(branch_q, {"Invoice": 0.9, "Card": 0.1}),
        note_q.qid: choice(note_q, {"net60": 0.8, NOT_STATED: 0.2}),
    }
    result = decode(tool, tool.slot("method"), pool, answers, rc)
    assert result.value == {"note": "net60"} and result.factor == pytest.approx(0.72)
    card = decode(tool, tool.slot("method"), pool, {**answers, branch_q.qid: choice(branch_q, {"Card": 0.9})}, rc)
    assert card.shape == "missing"


# -- text --------------------------------------------------------------------------------------------------------------


def test_r2_subject_and_body_candidates() -> None:
    catalog, rc = scenario("Email Anna that I'll be 10 minutes late", history=True)
    tool = catalog["send_email"]
    subject, _ = resolve(tool, tool.slot("subject"), rc)
    assert [c.value for c in subject.candidates] == [
        "Running 10 minutes late",
        "Running late",
        "I'll be 10 minutes late",
    ]
    body, questions = resolve(tool, tool.slot("body"), rc)
    template, clause = body.candidates
    assert template.value == "Hi ⟨recipient's first name⟩,\n\nI'll be 10 minutes late.\n\nBest,\nSam"
    assert template.late == {"placeholders": ["to.first_name"], "fill": {"⟨recipient's first name⟩": "to.first_name"}}
    assert template.channel is Channel.AUTHOR and clause.value == "I'll be 10 minutes late." and clause.is_evidence
    assert questions[0].criteria is not None and questions[0].instructions["candidate"] == template.value  # type: ignore[index]
    result = decode(tool, tool.slot("body"), body, {questions[0].qid: noul(0.91), questions[1].qid: noul(0.88)}, rc)
    assert result.value == template.value and result.factor == pytest.approx(0.91) and result.late == template.late
    assert [a.value for a in result.alternatives] == [clause.value] and result.normalizer == "text.template@1"


def test_titles_queries_and_r7() -> None:
    catalog, rc = scenario(R5)
    title, questions = resolve(catalog["create_event"], catalog["create_event"].slot("title"), rc)
    assert [c.value for c in title.candidates] == ["45 min sync with Bob and Carol", "Sync", "Sync with Bob and Carol"]
    assert all(q.criteria is None for q in questions)
    answers = {q.qid: noul(p) for q, p in zip(questions, (0.81, 0.77, 0.94), strict=True)}
    result = decode(catalog["create_event"], catalog["create_event"].slot("title"), title, answers, rc)
    assert result.value == "Sync with Bob and Carol" and result.factor is None and result.alternatives == ()
    catalog, rc = scenario("Tell me a joke")
    query, _ = resolve(catalog["search_web"], catalog["search_web"].slot("query"), rc)
    assert [c.value for c in query.candidates] == ["a joke", "Tell me a joke"]


def test_accept_decoding_rules() -> None:
    assert elect_accept([0.80, 0.81, 0.5]) == 0 and elect_accept([0.80, 0.83]) == 1 and elect_accept([]) is None
    catalog, rc = scenario("Email Anna that I'll be 10 minutes late")
    tool = catalog["send_email"]
    body, questions = resolve(tool, tool.slot("body"), rc)
    low = decode(tool, tool.slot("body"), body, {q.qid: noul(0.3) for q in questions}, rc)
    assert low.value is Bottom.UNCOVERED and low.shape == "uncovered_text" and low.factor == pytest.approx(0.3)
    subject, questions = resolve(tool, tool.slot("subject"), rc)
    floor = decode(tool, tool.slot("subject"), subject, {q.qid: noul(0.2) for q in questions}, rc)
    assert floor.value == "Running 10 minutes late" and floor.shape == "ok" and floor.prov["below_floor"]
    props = {"note": {"type": "string", "x-jev": {"stakes": "cosmetic"}}}
    tool, rc = custom("add_note", props, "Add a note that the build is green")
    note, questions = resolve(tool, tool.slot("note"), rc)
    omitted = decode(tool, tool.slot("note"), note, {q.qid: noul(0.1) for q in questions}, rc)
    assert omitted.value is Bottom.OMIT and omitted.shape == "ok"


def test_perspective_and_observation_candidates() -> None:
    catalog, rc = scenario("Email Anna and tell her she should call me")
    body, _ = resolve(catalog["send_email"], catalog["send_email"].slot("body"), rc)
    values = [c.value for c in body.candidates]
    assert "You should call me." in values and "She should call me." in values
    rewrite = next(c for c in body.candidates if c.value == "You should call me.")
    assert rewrite.prov["rewrite"] == "perspective"
    obs = Observation(
        step=1, tool="read_file", content="INVOICE TEXT", arguments={"path": "f/2026-09-15_ACME_INV-2291.pdf"}
    )
    catalog, rc = scenario("Find the latest invoice from ACME and forward it to finance", observations=[obs])
    body, _ = resolve(catalog["send_email"], catalog["send_email"].slot("body"), rc)
    forward, handle = body.candidates
    assert forward.value == (
        "Hi,\n\nForwarding the latest invoice from ACME below.\n\n⟨full text of the file read in step 1⟩\n\nBest,\nSam"
    )
    assert forward.channel is Channel.TOOL_OUTPUT and handle.late == {
        "placeholders": ["obs:1"],
        "fill": {handle.value: "obs:1"},
    }
    subject, _ = resolve(catalog["send_email"], catalog["send_email"].slot("subject"), rc)
    assert subject.candidates[0].value == "Fwd: ACME INV-2291"


# -- review regression: an array of objects keeps its leaves' flags and channel ----------------------------------------


def test_list_of_records_propagates_leaf_presence_conflict_to_p8() -> None:
    from jevtools.backends.scripted import ScriptedBackend
    from jevtools.context import Context
    from jevtools.router import Router
    from jevtools.spec.catalog import Catalog
    from tests.scenario_sources import contacts
    from tests.support import SCENARIO_NOW

    bob, carol = "Bob Meier <bob.meier@muster.ch>", "Carol Liu <carol.liu@muster.ch>"
    item = {"type": "object", "properties": {"email": {"type": "string", "format": "email"},
                                             "role": {"type": "string", "enum": ["required", "optional"]}}}  # fmt: skip
    fn = {"name": "create_meeting", "description": "Create a meeting.",
          "parameters": {"type": "object", "properties": {"people": {"type": "array", "items": item}},
                         "required": ["people"]}}  # fmt: skip
    catalog = Catalog.from_openai([{"type": "function", "function": fn}], sources=[contacts()])
    script = {"tool": {"create_meeting": 0.99, "NO_TOOL": 0.005, "UNSUPPORTED": 0.005},
              "create_meeting.authorized": 0.99,
              "create_meeting.people.m0.email": {bob: 0.97}, "create_meeting.people.m0.email.present": 0.05,
              "create_meeting.people.m0.role": {"optional": 0.97, "NOT_STATED": 0.03},
              "create_meeting.people.m1.email": {carol: 0.97}, "create_meeting.people.m1.email.present": 0.97,
              "create_meeting.people.m1.role": {"NOT_STATED": 0.97, "required": 0.03},
              "create_meeting.people.more": 0.02}  # fmt: skip
    ctx = Context(messages="Meet Bob and Carol", now=SCENARIO_NOW, locale="en-CH", sources=[contacts()])
    decision = Router(catalog, backend=ScriptedBackend(script), context=ctx).decide("Meet Bob and Carol")
    assert (decision.outcome, decision.rule) == ("clarify", "P8.consistency")
    assert decision.slots["people"].channel == "registry"  # the leaves' least-trusted channel, not None


# -- review regression: the perspective rewrite never takes a third party's pronouns -----------------------------------


THIRD_PARTIES = ["Email Anna that Tom is sick and he can't come today", "Tell Anna that Marco says his train is late"]


@pytest.mark.parametrize("request_text", THIRD_PARTIES)
def test_perspective_rewrite_skips_third_parties(request_text: str) -> None:
    catalog, rc = scenario(request_text)
    tool = catalog["send_email"]
    body, _ = resolve(tool, tool.slot("body"), rc)
    assert not [c for c in body.candidates if c.prov.get("rewrite") == "perspective"]


@pytest.mark.parametrize(
    ("request_text", "variant"),
    [
        ("Tell Anna she should call me", "You should call me."),
        ("Tell Bob his car is ready", "Your car is ready."),
        ("Email Anna that she should come to Zurich on Monday", "You should come to Zurich on Monday."),
    ],
)
def test_perspective_rewrite_keeps_recipient_cases(request_text: str, variant: str) -> None:
    catalog, rc = scenario(request_text)
    tool = catalog["send_email"]
    body, _ = resolve(tool, tool.slot("body"), rc)
    assert [c.value for c in body.candidates if c.prov.get("rewrite") == "perspective"] == [variant]

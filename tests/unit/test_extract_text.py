"""Span extractors, places, cues, negation, coreference and claiming (spec §4.2.1, §4.2.6, §4.2.11)."""

from __future__ import annotations

from typing import Any

import pytest

from jevtools.candidates import Channel
from jevtools.context import Context
from jevtools.extract import Mentions, coref_candidates, iter_entities, perspective_variant, run_extractors
from jevtools.extract.base import PRIORITY, Mention, claim
from jevtools.extract.tokens import covers, crosses_boundary, fold, sentence_spans, split_identifier, tokenize
from jevtools.spec.catalog import Catalog
from tests.scenario_sources import scenario_context
from tests.support import SCENARIO_NOW, SCENARIO_SOURCES, load_fixture


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return Catalog.from_openai(load_fixture("scenario_catalog.json"), sources=SCENARIO_SOURCES)


def run(text: str, catalog: Catalog | None = None, **kw: Any) -> Mentions:
    return run_extractors(Context(messages=text, now=SCENARIO_NOW, **kw), catalog)


def texts(mentions: Mentions, kind: str, **attrs: Any) -> list[str]:
    return [m.text for m in mentions.of(kind) if all(m.attrs.get(k) == v for k, v in attrs.items())]


def test_tokens_helpers() -> None:
    assert [t.text for t in tokenize("I'll pay 1'250.50 at 15:00!")] == ["I'll", "pay", "1'250.50", "at", "15:00", "!"]
    assert fold("Zürich") == "zurich" and fold("NÄCHSTEN") == "nachsten"
    assert split_identifier("services/paymentsApi/config_v2.yaml") == [
        "services",
        "payments",
        "api",
        "config",
        "v2",
        "yaml",
    ]
    assert sentence_spans("Hi. Do it now!\nThanks") == [(0, 2), (4, 13), (15, 21)]
    assert crosses_boundary("a. b", 0, 4) and not crosses_boundary("a b.", 0, 4)
    assert covers((0, 10), (2, 5)) and not covers((2, 5), (0, 10))


def test_quotes_proper_nouns_clauses_and_command() -> None:
    m = run('Email Anna Keller that "the build is green" and tell her she should call me')
    assert texts(m, "quote") == ["the build is green"]
    assert "Anna Keller" in texts(m, "proper_noun") and "Email" not in texts(m, "proper_noun")
    assert texts(m, "clause") == ["she should call me"]  # "that" + a quote: the quote itself is the candidate
    assert texts(m, "command") == ['Anna Keller that "the build is green" and tell her she should call me']


def test_r2_clause_and_command() -> None:
    m = run("Email Anna that I'll be 10 minutes late")
    assert texts(m, "clause") == ["I'll be 10 minutes late"]
    assert texts(m, "command") == ["Anna that I'll be 10 minutes late"]
    assert not texts(m, "noun_phrase", main=True)  # the object "Anna" has no common noun


def test_main_chunk_variants() -> None:
    r5 = run("Book a 45 min sync with Bob and Carol next Tuesday at 3pm")
    assert texts(r5, "command") == ["45 min sync with Bob and Carol next Tuesday at 3pm"]
    variants = {m.attrs["variant"]: m.text for m in r5.of("noun_phrase") if m.attrs.get("main")}
    assert variants == {"full": "45 min sync with Bob and Carol", "core": "sync with Bob and Carol", "head": "sync"}
    joke = run("Tell me a joke")
    assert texts(joke, "command") == ["a joke"]
    assert {m.attrs["variant"]: m.text for m in joke.of("noun_phrase") if m.attrs.get("main")} == {
        "full": "joke",
        "core": "a joke",
        "head": "joke",
    }
    assert not texts(joke, "clause")  # "tell me" addresses the assistant, not a recipient


def test_tell_and_let_know_clauses() -> None:
    assert texts(run("Tell Bob I'm running late"), "clause") == ["I'm running late"]
    assert texts(run("let her know that the report is done."), "clause") == ["the report is done"]
    assert texts(run("Find the file that contains the budget"), "clause") == []  # relative clause, not a message


@pytest.mark.parametrize(
    ("clause", "variant"),
    [
        ("she should call me", "you should call me"),
        ("he is late", "you are late"),
        ("tell him his car is ready", "tell you your car is ready"),
        ("I'll be 10 minutes late", None),
    ],
)
def test_perspective_variant(clause: str, variant: str | None) -> None:
    assert perspective_variant(clause) == variant


def test_places_and_claiming(catalog: Catalog) -> None:
    m = run("What's the weather like in Zurich in Fahrenheit?", catalog)
    places = m.of("place")
    assert [p.text for p in places] == ["Zurich"] and places[0].free
    assert {(p.name, p.country) for p in places[0].attrs["places"]} == {("Zürich", "CH"), ("Zurich", "CA")}
    fahrenheit = [x for x in m if x.text == "Fahrenheit"]
    assert {x.kind for x in fahrenheit} == {"enum", "proper_noun"}
    assert next(x for x in fahrenheit if x.kind == "proper_noun").claimed_by == "enum"
    assert next(x for x in m.of("proper_noun") if x.text == "Zurich").claimed_by == "place"
    assert not run("nice weather today").of("place")  # lowercase "nice" is not Nice
    assert [p.text for p in run("weather in zurich").of("place")] == ["zurich"]  # an all-lowercase message
    assert [p.text for p in run("flights from New York to San Francisco").of("place")] == ["New York", "San Francisco"]


def test_claiming_priority_order() -> None:
    assert PRIORITY["enum"] < PRIORITY["anchor"] < PRIORITY["temporal"] < PRIORITY["money"] < PRIORITY["quantity"]
    assert PRIORITY["quantity"] < PRIORITY["email"] < PRIORITY["number"] < PRIORITY["place"] < PRIORITY["quote"]
    outer = Mention("temporal", "next Tuesday at 3pm", (0, 19))
    inner = Mention("number", "3", (16, 17))
    other_text = Mention("number", "3", (16, 17), source_ref="user:0")
    claimed = claim([outer, inner, other_text])
    assert claimed[1].claimed_by == "temporal" and claimed[2].free and claimed[0].free


def test_registry_anchors_claim_proper_nouns(catalog: Catalog) -> None:
    ctx = scenario_context("Book a 45 min sync with Bob and Carol next Tuesday at 3pm")
    m = run_extractors(ctx, catalog)
    anchors = m.anchors("contacts")
    assert [a.text for a in anchors] == ["Bob", "Carol"]
    assert [row.how for row in anchors[0].attrs["matches"]] == ["exact", "alias"]
    assert all(p.claimed_by == "anchor" for p in m.of("proper_noun") if p.text in ("Bob", "Carol"))
    assert next(p for p in m.of("proper_noun") if p.text == "Tuesday").claimed_by == "temporal"


def test_history_and_observation_channels() -> None:
    ctx = scenario_context("Email Anna that I'll be 10 minutes late", history=True)
    m = run_extractors(ctx)
    history = m.of("temporal", channels=(Channel.HISTORY,))
    assert [(t.text, t.source_ref) for t in history] == [("14:30", "assistant:1")]
    assert not m.of("anchor", channels=(Channel.HISTORY,))  # anchors come from the user's words only


def test_negation_marks() -> None:
    m = run("Invite everyone except Bob. Also add Carol")
    bob = next(x for x in m.of("proper_noun") if x.text == "Bob")
    carol = next(x for x in m.of("proper_noun") if x.text == "Carol")
    assert bob.negated and bob.attrs["negation"] == "except Bob" and not carol.negated
    assert not next(x for x in run("not now, but Carol").of("proper_noun") if x.text == "Carol").negated


def test_cues() -> None:
    m = run("Find the latest invoice and forward it to finance")
    assert [c.attrs["canonical"] for c in m.cues("superlative")] == ["latest"]
    assert [c.text for c in m.cues("anaphor")] == ["it"]
    assert run("tell me a joke").has_cue("chitchat")
    assert run("how do I send an email?").has_cue("hedge")
    assert not run("meet me last week").cues("superlative")  # "last week" is temporal


def test_coreference_candidates() -> None:
    store = {
        "entities": [
            {
                "id": "e1",
                "type": "email",
                "value": "anna.keller@acme.com",
                "label": "Anna Keller",
                "origin": "registry",
                "turn": 1,
                "pinned": True,
                "tool": "send_email",
            },
            {
                "id": "e2",
                "type": "email",
                "value": "anna.rossi@gmail.com",
                "label": "Anna Rossi",
                "origin": "user",
                "turn": 0,
            },
            {"id": "e3", "type": "path", "value": "a.txt", "label": "a.txt", "origin": "tool_output", "turn": 1},
        ]
    }
    assert [e.id for e in iter_entities(store)] == ["e1", "e3", "e2"]
    again = coref_candidates(run("email her again"), store, {"email"})
    assert [c.value for c in again] == ["anna.keller@acme.com", "anna.rossi@gmail.com"]
    assert all(c.channel is Channel.HISTORY for c in again) and again[0].origin is Channel.REGISTRY
    other = coref_candidates(run("email the other Anna"), store, {"email"})
    assert [c.value for c in other] == ["anna.rossi@gmail.com"]
    assert other[0].text == "Mentioned earlier: Anna Rossi; not the Anna Keller from your last send email."
    assert coref_candidates(run("email Anna"), store, {"email"}) == []
    assert coref_candidates(run("email her"), None, {"email"}) == []

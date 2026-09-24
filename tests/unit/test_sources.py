"""Candidate sources (spec §4.4): retrieval primitives, Registry, FileIndex and Provider."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from jevtools.candidates import Candidate, Channel
from jevtools.extract import run_extractors
from jevtools.sources import BM25, FileIndex, Provider, Registry, SourceQuery, date_attr, fuzzy_matches, trigram
from jevtools.sources.base import item_noun
from jevtools.sources.retrieval import canonical_terms, match_term, path_tokens, plural_stem, synonym_table
from tests.scenario_sources import ACCOUNT_ROWS, CONTACT_ROWS, accounts, contacts, files, scenario_context


def test_trigram_and_match_rules() -> None:
    assert trigram("Anna", "anna") == 1.0 and trigram("anna", "hanna") < 0.5 and trigram("", "x") < 0.5
    assert match_term("Anna", "Anna Keller", alias=False) == (1.0, "exact")
    assert match_term("Bob", "Bob", alias=True) == (0.95, "alias")
    score, how = match_term("Anna", "Annabel Frey", alias=False) or (0, "")
    assert how == "prefix" and 0.6 < score < 0.9
    assert match_term("An", "Annabel", alias=False) is None  # prefix needs ≥ 3 characters
    assert match_term("Anna", "Annabel", alias=False, fuzzy=False) is None
    assert match_term("Caroll", "Carol", alias=False) == pytest.approx((0.9 * trigram("caroll", "carol"), "trigram"))


def test_fuzzy_matches_best_per_row() -> None:
    rows = CONTACT_ROWS[:8]
    matches = fuzzy_matches("Bob", rows, ["name", "aliases"])
    assert [(rows[m.index]["name"], m.how) for m in matches] == [("Bob Meier", "exact"), ("Robert Brown", "alias")]
    group = fuzzy_matches("payments", rows, ["name", "team"], group_field="team")
    assert {m.how for m in group} == {"group"}


def test_bm25_and_path_tokens() -> None:
    assert path_tokens("services/paymentsApi/config.yaml") == ["service", "payment", "api", "config", "yaml"]
    assert plural_stem("invoices") == "invoice" and plural_stem("address") == "address"
    table = synonym_table({"config": ["cfg", "settings"]})
    assert canonical_terms(["Settings", "cfg", "config", "other"], table) == ["config", "config", "config", "other"]
    index = BM25([["a", "b"], ["a"], ["c", "c", "d"]])
    top = index.top(["c"])
    assert [i for i, _ in top] == [2] and index.matched(["a", "c"], 0) == ["a"] and index.top(["zzz"]) == []


def test_registry_anchors_and_descriptions() -> None:
    registry = contacts()
    anchors = registry.find_anchors("Email Anna that I'll be 10 minutes late")
    assert [a.text for a in anchors] == ["Anna"]
    full = registry.find_anchors("Email Anna Keller now")
    assert [a.text for a in full] == ["Anna Keller"]
    assert registry.rows[full[0].attrs["matches"][0].index]["name"] == "Anna Keller"
    cands = registry.candidates(SourceQuery(request="Email Anna that I'll be 10 minutes late", k=40))
    assert [c.label for c in cands] == [
        "Anna Keller <anna.keller@acme.com>",
        "Anna Rossi <anna.rossi@gmail.com>",
        "Annabel Frey <annabel.frey@muster.ch>",
    ]
    assert cands[0].text == 'Contact matching "Anna": Account Manager at ACME; last emailed 2 days ago.'
    assert cands[2].text.startswith('Contact similar to "Anna": ')
    assert cands[0].value == "anna.keller@acme.com" and cands[0].is_evidence and cands[0].channel is Channel.REGISTRY
    assert cands[0].attrs["name"] == "Anna Keller" and "notes" not in cands[0].attrs
    assert registry.lookup("finance@muster.ch") is not None and registry.lookup("nobody@x") is None
    assert registry.content_sha256().startswith("sha256:") and len(registry) == 20 and not registry.whole


def test_registry_ranking_k_and_widen_ranking() -> None:
    registry = contacts()
    q = SourceQuery(request="invite Carol", k=1)
    assert [c.label for c in registry.candidates(q)] == ["Carol Liu <carol.liu@muster.ch>"]
    page = registry.candidates(SourceQuery(request="invite Carol", k=1, offset=1))
    assert [c.label for c in page] == ["Caroline Weber <caroline.weber@muster.ch>"]
    ranked = registry.ranked(SourceQuery(request="invite Carol", widen=True))
    assert len(ranked) == len(CONTACT_ROWS) and ranked[0].prov["anchor"] == "Carol" and "anchor" not in ranked[-1].prov
    assert registry.group_of(ranked[0]) == "Payments"


def test_small_registry_is_sent_whole() -> None:
    registry = accounts()
    cands = registry.candidates(SourceQuery(request="Move 250 CHF from my savings to checking"))
    assert len(cands) == len(ACCOUNT_ROWS) and all(c.prov.get("whole") for c in cands)
    anchored = {c.label for c in cands if "anchor" in c.prov}
    assert anchored == {"Savings · CHF · CH93…2957", "Travel savings · EUR · CH08…4410", "Checking · CHF · CH56…1180"}
    joint = next(c for c in cands if c.value == "acc_5102")
    assert joint.text == "Account: Joint household account in CHF." and joint.is_evidence
    assert joint.attrs["balance"] == 4200.0  # attributes travel with candidates, never to Jev


def test_registry_negated_anchor_and_bm25_retriever() -> None:
    ctx = scenario_context("Invite everyone except Bob")
    mentions = run_extractors(ctx)
    bob = next(
        c for c in contacts().candidates(SourceQuery(mentions=mentions, request=ctx.request)) if "Bob" in c.label
    )
    assert "mentioned in a negation: 'except Bob'" in (bob.text or "")
    registry = Registry("people", CONTACT_ROWS, key="email", label="{name}", match=["name", "notes"], retriever="bm25")
    hits = registry.candidates(SourceQuery(request="who handles legal questions"))
    assert [c.label for c in hits] == ["Caroline Weber"]
    assert registry.find_anchors("Caroline") == []  # the BM25 retriever makes no anchors
    with pytest.raises(ValueError, match="no 'id' value"):
        Registry("bad", [{"name": "x"}], key="id")


def test_file_index_bm25_synonyms_and_attrs() -> None:
    index = files()
    hits = index.candidates(SourceQuery(request="Open the config file for the payments service"))
    assert [c.value for c in hits[:2]] == ["services/payments/config/app.yaml", "services/payments/config/prod.yaml"]
    assert hits[0].text == 'Workspace file; its path matches "config", "payment", "service".'
    assert hits[0].attrs["group"] == "services/payments" and hits[0].is_evidence
    invoice = index.candidates(SourceQuery(request="the invoice from ACME"))[0]
    assert invoice.attrs["date"] == "2026-09-15" and invoice.text.startswith("Workspace file dated 2026-09-15;")
    assert index.candidates(SourceQuery(request="Book a 45 min sync with Bob and Carol next Tuesday at 3pm")) == []
    ranked = index.ranked(SourceQuery(request="invoice", widen=True))
    assert len(ranked) == len(index) and "score" not in ranked[-1].prov
    assert "finance/invoices/acme/2026-09-15_ACME_INV-2291.pdf" in index and index.lookup("nope.txt") is None
    assert {"date", "group", "path"} <= index.attribute_names()


def test_file_index_provider_mtime_and_hierarchy() -> None:
    mtime = datetime(2026, 1, 2, tzinfo=timezone.utc)
    index = FileIndex(
        "ws",
        lambda: ["a/b/c/notes.txt", "a/x.txt"],
        attrs={"a/x.txt": {"mtime": mtime}},
        hierarchy=lambda p: p.split("/")[0],
    )
    assert index.attrs_of("a/x.txt")["date"] == "2026-01-02" and index.group("a/b/c/notes.txt") == "a"
    assert date_attr("2026-13-40_x.pdf", 0) == "1970-01-01" and date_attr("x", "2026-05-01T10:00") == "2026-05-01"
    assert date_attr("x") is None


def test_provider_sync_and_async() -> None:
    def lookup(q: SourceQuery) -> list[object]:
        return [{"value": "u1", "label": "User One", "text": "A user."}, Candidate(value="u2", channel=Channel.USER)]

    async def alookup(q: SourceQuery) -> list[dict[str, str]]:
        return [{"value": "u3"}]

    sync = Provider(lookup, provides={"user"})
    cands = sync.candidates(SourceQuery(request="x"))
    assert [c.value for c in cands] == ["u1", "u2"] and cands[0].channel is Channel.REGISTRY
    assert all(c.is_evidence for c in cands) and cands[0].prov["anchor"] == "provider:lookup"
    provider = Provider(alookup, name="async_users")
    assert provider.is_async and [c.value for c in provider.candidates(SourceQuery())] == ["u3"]
    assert [c.value for c in asyncio.run(provider.acandidates(SourceQuery()))] == ["u3"]

    async def inside_loop() -> None:
        with pytest.raises(RuntimeError, match="acandidates"):
            provider.candidates(SourceQuery())

    asyncio.run(inside_loop())


def test_item_noun() -> None:
    assert item_noun("contacts") == "contact" and item_noun("companies") == "company" and item_noun("gas") == "gas"

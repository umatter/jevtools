"""The resolver contract (kinds/base.py) exercised through the reference ``enum`` resolver."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from jevtools.backends.scripted import choice_answer
from jevtools.candidates import NONE_OF_THESE, NOT_STATED, Bottom, Channel
from jevtools.context import Context
from jevtools.kinds import (
    RESOLVERS,
    ResolveContext,
    Resolver,
    decode_choice,
    get_resolver,
    register_resolver,
    resolve_default,
    unasked_result,
)
from jevtools.kinds.base import LATE_DEFAULT
from jevtools.kinds.enum import CATALOG_DATA, EnumResolver, load_catalog, register_catalog
from jevtools.spec.catalog import Catalog
from jevtools.spec.models import Member
from jevtools.wire import NoulAnswer
from tests.support import SCENARIO_NOW, SCENARIO_SOURCES

OOP = 0.30


@pytest.fixture
def currencies() -> Iterator[None]:
    register_catalog("iso4217", [{"value": "CHF", "text": "Swiss franc", "aliases": ["franc", "francs", "Fr."]},
                                 {"value": "EUR", "text": "Euro", "aliases": ["euro", "euros", "€"]},
                                 {"value": "USD", "text": "US dollar", "aliases": ["dollar", "$"]}, "ALL"])  # fmt: skip
    yield
    CATALOG_DATA.pop("iso4217", None)
    load_catalog.cache_clear()


def _rc(catalog: Catalog, request: str, **kw: Any) -> ResolveContext:
    ctx = Context(messages=request, now=SCENARIO_NOW, user={"name": "Sam Muster", "home_city": "Zurich"})
    return ResolveContext(ctx=ctx, catalog=catalog, **kw)


def test_registry() -> None:
    enum = get_resolver("enum")
    assert isinstance(enum, EnumResolver) and isinstance(enum, Resolver) and RESOLVERS["enum"] is enum
    with pytest.raises(KeyError, match="no resolver"):
        get_resolver("telepathy")
    register_resolver("custom_kind", enum)
    try:
        assert get_resolver("custom_kind") is enum
    finally:
        RESOLVERS.pop("custom_kind")


def test_r1_unit_pool_question_and_decode(scenario_catalog: Catalog) -> None:
    rc = _rc(scenario_catalog, "What's the weather like in Zurich in Fahrenheit?")
    tool = scenario_catalog["get_weather"]
    unit = tool.slot("unit")
    enum = get_resolver("enum")
    pool = enum.pool(tool, unit, rc)
    assert [c.label for c in pool.candidates] == ["celsius", "fahrenheit"]
    assert pool.closed and pool.evidence_backed and all(c.channel is Channel.AUTHOR for c in pool.candidates)
    (question,) = enum.questions(tool, unit, pool, rc)
    assert question.qid == "get_weather.unit" and question.family == "slot"
    assert question.sentinels[NOT_STATED].decodes_to == "default" and question.sentinels[NOT_STATED].value == "celsius"
    answer = choice_answer(
        question.labels, {"celsius": 0.01, "fahrenheit": 0.97, NOT_STATED: 0.01, NONE_OF_THESE: 0.01}
    )
    result = enum.decode(tool, unit, pool, {question.qid: answer}, rc)
    assert result.value == "fahrenheit" and result.factor == pytest.approx(0.97) and result.shape == "ok"
    assert result.label == "fahrenheit" and result.channel is Channel.AUTHOR and result.normalizer == "enum@1"
    assert result.dist['"celsius"'] == pytest.approx(0.02)  # NOT_STATED → default pools with celsius
    assert [(a.value, round(a.p, 2)) for a in result.alternatives] == [("celsius", 0.02)]
    assert result.sentinels == {NOT_STATED: pytest.approx(0.01), NONE_OF_THESE: pytest.approx(0.01)}


def test_not_stated_pools_with_default(scenario_catalog: Catalog) -> None:
    rc = _rc(scenario_catalog, "What's the weather?")
    tool = scenario_catalog["get_weather"]
    unit = tool.slot("unit")
    enum = get_resolver("enum")
    pool = enum.pool(tool, unit, rc)
    assert not pool.evidence_backed
    (question,) = enum.questions(tool, unit, pool, rc)
    answer = choice_answer(
        question.labels, {"celsius": 0.02, "fahrenheit": 0.01, NOT_STATED: 0.96, NONE_OF_THESE: 0.01}
    )
    result = enum.decode(tool, unit, pool, {question.qid: answer}, rc)
    assert result.value == "celsius" and result.factor == pytest.approx(0.98)


def test_uncovered_mass_makes_out_of_pool(scenario_catalog: Catalog) -> None:
    rc = _rc(scenario_catalog, "Weather in Kelvin please")
    tool = scenario_catalog["get_weather"]
    unit = tool.slot("unit")
    enum = get_resolver("enum")
    pool = enum.pool(tool, unit, rc)
    (question,) = enum.questions(tool, unit, pool, rc)
    winner_but_uncovered = choice_answer(question.labels, {"celsius": 0.6, NONE_OF_THESE: 0.35, NOT_STATED: 0.05})
    result = enum.decode(tool, unit, pool, {question.qid: winner_but_uncovered}, rc)
    assert result.value == "celsius" and result.shape == "out_of_pool"
    uncovered = choice_answer(question.labels, {NONE_OF_THESE: 0.9, "celsius": 0.1})
    result = enum.decode(tool, unit, pool, {question.qid: uncovered}, rc)
    assert result.value is Bottom.UNCOVERED and result.shape == "out_of_pool" and result.factor == pytest.approx(0.9)


def test_missing_and_malformed_answers_fail_closed(scenario_catalog: Catalog) -> None:
    rc = _rc(scenario_catalog, "Weather?")
    tool = scenario_catalog["get_weather"]
    unit = tool.slot("unit")
    enum = get_resolver("enum")
    pool = enum.pool(tool, unit, rc)
    for answers in ({}, {"get_weather.unit": NoulAnswer(noul=0.9)}):
        result = enum.decode(tool, unit, pool, answers, rc)  # type: ignore[arg-type]
        assert result.shape == "missing" and result.factor == 0.0 and "no_answer" in result.flags


def test_unknown_labels_ignored_and_missing_labels_zero(scenario_catalog: Catalog) -> None:
    tool = scenario_catalog["get_weather"]
    unit = tool.slot("unit")
    rc = _rc(scenario_catalog, "x")
    (question,) = get_resolver("enum").questions(tool, unit, get_resolver("enum").pool(tool, unit, rc), rc)
    answer = choice_answer(question.labels, {"fahrenheit": 1.0})
    answer = answer.model_copy(update={"probabilities": {"fahrenheit": 0.9, "kelvin": 0.1}})
    result = decode_choice(unit, question, answer, out_of_pool=OOP)
    assert result.value == "fahrenheit" and result.notes == ("ignored unknown label 'kelvin'",)
    assert result.dist[Bottom.UNCOVERED.value] == 0.0


def test_optional_and_required_without_default() -> None:
    tools = [{"type": "function", "function": {"name": "set_mode", "description": "Set the mode.", "parameters": {
        "type": "object", "required": ["mode"], "properties": {
            "mode": {"type": "string", "enum": ["fast", "slow"]},
            "level": {"type": "string", "enum": ["hi", "lo"]}}}}}]  # fmt: skip
    catalog = Catalog.from_openai(tools)
    tool = catalog["set_mode"]
    rc = _rc(catalog, "set it")
    enum = get_resolver("enum")
    mode, level = tool.slot("mode"), tool.slot("level")
    (q_mode,) = enum.questions(tool, mode, enum.pool(tool, mode, rc), rc)
    (q_level,) = enum.questions(tool, level, enum.pool(tool, level, rc), rc)
    assert q_mode.sentinels[NOT_STATED].decodes_to == "missing"
    assert q_level.sentinels[NOT_STATED].decodes_to == "omit"
    assert q_mode.sentinels[NOT_STATED].text == "The user does not say."
    missing = decode_choice(mode, q_mode, choice_answer(q_mode.labels, {NOT_STATED: 0.8, "fast": 0.2}), out_of_pool=OOP)
    assert missing.value is Bottom.MISSING and missing.shape == "missing"
    omitted = decode_choice(
        level, q_level, choice_answer(q_level.labels, {NOT_STATED: 0.9, "hi": 0.1}), out_of_pool=OOP
    )
    assert omitted.value is Bottom.OMIT and omitted.shape == "ok" and omitted.factor == pytest.approx(0.9)
    assert unasked_result(level, None).value is Bottom.OMIT
    assert unasked_result(mode, None).shape == "missing"


def test_catalog_shortlist_probe_and_late_default(scenario_catalog: Catalog, currencies: None) -> None:
    tool = scenario_catalog["transfer_funds"]
    currency = tool.slot("currency")
    enum = get_resolver("enum")
    rc = _rc(scenario_catalog, "Move all 250 CHF from my savings to checking",
             preferred={"transfer_funds.currency": ["EUR"]})  # fmt: skip
    pool = enum.pool(tool, currency, rc)
    assert [c.label for c in pool.candidates] == ["CHF", "EUR"]  # "all" never matches the code ALL
    assert pool.notes == ["shortlist 2 of 4 (iso4217)"] and pool.evidence_backed
    (question,) = enum.questions(tool, currency, pool, rc)
    not_stated = question.sentinels[NOT_STATED]
    assert not_stated.late == {"default_from": "from_account.currency"}
    assert not_stated.text == ("The user does not say; the default (the currency of the account the money is taken "
                               "from) would be used.")  # fmt: skip
    answer = choice_answer(question.labels, {"CHF": 0.95, "EUR": 0.02, NOT_STATED: 0.02, NONE_OF_THESE: 0.01})
    result = enum.decode(tool, currency, pool, {question.qid: answer}, rc)
    assert result.value == "CHF" and result.dist[LATE_DEFAULT] == pytest.approx(0.02)
    bound = result.bind_late_default("CHF", out_of_pool=OOP)
    assert bound.value == "CHF" and bound.factor == pytest.approx(0.97) and LATE_DEFAULT not in bound.dist
    assert result.bind_late_default("EUR", out_of_pool=OOP).dist['"EUR"'] == pytest.approx(0.04)
    empty_rc = _rc(scenario_catalog, "Move money from savings to checking")
    empty = enum.pool(tool, currency, empty_rc)
    assert empty.empty and empty.closed
    (probe,) = enum.questions(tool, currency, empty, empty_rc)
    assert probe.family == "probe" and probe.options == [] and probe.sentinels[NOT_STATED].decodes_to == "default"
    probe_result = enum.decode(tool, currency, empty, {probe.qid: choice_answer(probe.labels, {NOT_STATED: 0.9,
                                                                                              NONE_OF_THESE: 0.1})},
                               empty_rc)  # fmt: skip
    assert probe_result.value is Bottom.LATE_DEFAULT and probe_result.is_bottom
    assert probe_result.bind_late_default("CHF", out_of_pool=OOP).factor == pytest.approx(0.9)


def test_missing_catalog_is_a_clear_error(scenario_catalog: Catalog) -> None:
    load_catalog.cache_clear()
    with pytest.raises(LookupError, match="no_such_catalog"):
        load_catalog("no_such_catalog")
    assert Member(value="Europe/Zurich") in load_catalog("iana_tz")
    assert Member(value="CH", text="Switzerland", aliases=("Schweiz", "Suisse", "Svizzera")) in load_catalog("iso3166")


def test_allow_list_and_schema_filtering() -> None:
    tools = [{"type": "function", "function": {"name": "get_x", "description": "Get x.", "parameters": {
        "type": "object", "properties": {
            "size": {"type": "string", "enum": ["S", "M", "L"], "maxLength": 0,
                     "x-jev": {"channels": ["user"]}}}}}}]  # fmt: skip
    catalog = Catalog.from_openai(tools)
    tool = catalog["get_x"]
    size = tool.slot("size")
    pool = get_resolver("enum").pool(tool, size, _rc(catalog, "large"))
    assert pool.candidates == [] and pool.notes == ["dropped 3 schema-invalid member(s)"]
    permissive = size.model_copy(update={"json_schema": {"enum": ["S", "M", "L"]}})
    blocked = get_resolver("enum").pool(tool, permissive, _rc(catalog, "large"))
    assert blocked.channel_blocked and len(blocked.blocked) == 3


def test_resolve_default_variants(scenario_catalog: Catalog, ctx_default: Context) -> None:
    weather = scenario_catalog["get_weather"]
    city = resolve_default(weather, weather.slot("city"), ctx_default)
    assert city is not None and (city.value, city.display, city.channel) == (
        "Zurich", "Zurich, the user's home city", Channel.REGISTRY,
    )  # fmt: skip
    unit = resolve_default(weather, weather.slot("unit"), ctx_default)
    assert unit is not None and (unit.value, unit.channel) == ("celsius", Channel.AUTHOR)
    no_profile = ctx_default.model_copy(update={"user": {}})
    assert resolve_default(weather, weather.slot("city"), no_profile) is None
    catalog = Catalog.from_openai([{"type": "function", "function": {"name": "get_x", "parameters": {
        "type": "object", "properties": {"q": {"type": "string", "enum": ["a", "b"], "default": None}}}}}],
        sources=SCENARIO_SOURCES)  # fmt: skip
    null_default = resolve_default(catalog["get_x"], catalog["get_x"].slot("q"), ctx_default)
    assert null_default is not None and null_default.omit

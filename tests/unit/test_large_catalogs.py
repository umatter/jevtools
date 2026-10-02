"""Pieces a large real-world catalog (prompt2analytics, 270 tools) needed: a tool shortlist, plural and underscore
name matching, comma number lists, hyphenated number words, option lists after a colon, declared sources."""

from __future__ import annotations

import pytest

from jevtools.context import Context
from jevtools.extract import run_extractors
from jevtools.plan import _shortlist_order, _stem, shortlist_tools
from jevtools.sources.retrieval import match_term, singular
from jevtools.spec.catalog import Catalog
from jevtools.spec.infer import described_values, infer_slot


def _tool(name: str, description: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {"dataset": {"type": "string"}}}}}  # fmt: skip


def test_the_shortlist_finds_tools_by_stem_and_common_phrasing() -> None:
    catalog = Catalog.from_openai([_tool(f"other_tool_{i}", f"Unrelated thing number {i}.") for i in range(30)] + [
        _tool("regression_ols", "Run Ordinary Least Squares (OLS) regression."),
        _tool("describe_dataset", "Describe a dataset: column types and summary statistics."),
        _tool("munge_value_counts", "Compute value counts for a column."),
    ])  # fmt: skip
    tools = list(catalog)
    assert _shortlist_order(tools, Context(messages="Regress wage on education"))[0] == "regression_ols"
    assert _shortlist_order(tools, Context(messages="Give me an overview of the wages data"))[0] == "describe_dataset"
    assert _shortlist_order(tools, Context(messages="How many respondents per country?"))[0] == "munge_value_counts"
    picked = shortlist_tools(tools, Context(messages="Regress wage on education"), 5)
    assert len(picked) == 5 and "regression_ols" in [t.name for t in picked]
    assert [t.name for t in picked] == [t.name for t in tools if t in picked]  # catalog order kept
    assert (_stem("regression"), _stem("regress"), _stem("countries"), _stem("class")) == (
        "regress", "regress", "country", "class")  # fmt: skip


@pytest.mark.parametrize(("mention", "term", "how"), [
    ("regions", "region", "exact"), ("countries", "country", "exact"), ("matches", "match", "exact"),
    ("months to churn", "months_to_churn", "exact"), ("class", "clas", "trigram")])  # fmt: skip
def test_plural_and_spaced_mentions_match_field_names(mention: str, term: str, how: str) -> None:
    found = match_term(mention, term, alias=False)
    assert found is not None and found[1] == how
    assert singular("countries") == "country" and singular("boxes") == "box" and singular("class") == "class"


@pytest.mark.parametrize(("text", "numbers"), [("Fit an ARIMA(1,1,1) to sales", ["1", "1", "1"]),
                                              ("sales of 1,234,567 units", ["1234567"]), ("order 12,500", ["12500"]),
                                              ("a one-period lag", ["1"]), ("forty-five minutes", ["45"]),
                                              ("a well-known fact", [])])  # fmt: skip
def test_number_lists_and_hyphenated_number_words(text: str, numbers: list[str]) -> None:
    assert [str(m.value) for m in run_extractors(Context(messages=text), None).of("number")] == numbers


@pytest.mark.parametrize(("description", "values"), [
    ("Comparison operator: 'eq', 'ne', 'gt', 'lt'.", ["eq", "ne", "gt", "lt"]),
    ("Kernel function: 'triangular' (default), 'epanechnikov', or 'uniform'.",
     ["triangular", "epanechnikov", "uniform"]),
    ("Method: 'standardize' (z-score) or 'normalize' (0-1 range).", ["standardize", "normalize"]),
    ("Examples: 'a', 'b'", None), ("Optional title: 'My chart'", None)])  # fmt: skip
def test_option_lists_after_a_colon(description: str, values: list[str] | None) -> None:
    assert described_values(description) == values


def test_a_declared_source_wins_over_a_temporal_name() -> None:
    column = {"type": "string", "description": "Name of the time column.", "x-jev": {"source": "variables"}}
    tool = {"type": "function", "function": {"name": "kaplan_meier", "description": "Survival curve.",
            "parameters": {"type": "object", "properties": {"time": column}}}}  # fmt: skip
    assert Catalog.from_openai([tool])["kaplan_meier"].slot("time").kind == "ref"
    assert infer_slot("time", {"type": "string"}).kind == "temporal"

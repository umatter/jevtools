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


def test_a_late_default_that_decides_the_value_is_flagged_and_not_verified() -> None:
    from jevtools.decode import _verifies
    from jevtools.kinds.base import LATE_DEFAULT, SlotResult

    pending = SlotResult(path=("dataset",), kind="ref", stakes="identity", dist={LATE_DEFAULT: 0.8, '"wages"': 0.15},
                         values={'"wages"': "wages"}, value="wages", shape="ok", factor=0.8)  # fmt: skip
    bound = pending.bind_late_default("survey", out_of_pool=0.3)
    assert bound.value == "survey" and "defaulted" in bound.flags
    mentioned = pending.bind_late_default("wages", out_of_pool=0.3)  # 0.8 default mass > 0.15 real: still defaulted
    assert "defaulted" in mentioned.flags
    real = SlotResult(path=("dataset",), kind="ref", stakes="identity", dist={LATE_DEFAULT: 0.1, '"wages"': 0.85},
                      values={'"wages"': "wages"}, value="wages", shape="ok", factor=0.85)  # fmt: skip
    assert "defaulted" not in real.bind_late_default("wages", out_of_pool=0.3).flags

    class Q:
        meta = {"candidate": {"value": "survey"}}

    assert not _verifies(Q(), bound)  # type: ignore[arg-type]


def test_a_mention_equal_to_the_late_default_is_implied_and_not_verified() -> None:
    from jevtools.decode import _verifies
    from jevtools.kinds.base import LATE_DEFAULT, SlotResult

    real = SlotResult(path=("dataset",), kind="ref", stakes="identity", dist={LATE_DEFAULT: 0.3, '"wages"': 0.65},
                      values={'"wages"': "wages"}, value="wages", shape="ok", factor=0.65)  # fmt: skip
    implied = real.bind_late_default("wages", out_of_pool=0.3)  # "Granger test of price on sales": the columns' dataset
    assert implied.value == "wages" and "implied" in implied.flags and "defaulted" not in implied.flags
    other = real.bind_late_default("survey", out_of_pool=0.3)  # the columns say survey, the mention wages
    assert other.value == "wages" and "implied" not in other.flags

    class Q:
        meta = {"candidate": {"value": "wages"}}

    assert not _verifies(Q(), implied)  # type: ignore[arg-type]
    assert _verifies(Q(), other)  # type: ignore[arg-type]


def test_a_late_default_reads_the_attribute_a_lists_elements_share() -> None:
    from jevtools.kinds.base import SlotResult
    from jevtools.kinds.late import make_lookup

    def column(name: str, dataset: str) -> SlotResult:
        return SlotResult(path=("columns",), kind="ref", stakes="identity", dist={f'"{name}"': 0.9},
                          values={f'"{name}"': name}, value=name, shape="ok", factor=0.9,
                          attrs={"dataset": dataset})  # fmt: skip

    def columns(*parts: SlotResult) -> dict[str, SlotResult]:
        value = [p.value for p in parts]
        return {"columns": SlotResult(path=("columns",), kind="list", stakes="identity", dist={"k": 0.9},
                                      values={"k": value}, value=value, shape="ok", factor=0.9,
                                      parts={f"m{i}": p for i, p in enumerate(parts)})}  # fmt: skip

    same = make_lookup(columns(column("capital", "firm_panel"), column("investment", "firm_panel")), Context())
    assert same("columns.dataset") == "firm_panel"
    mixed = make_lookup(columns(column("capital", "firm_panel"), column("income", "survey")), Context())
    with pytest.raises(KeyError):
        mixed("columns.dataset")


def test_an_anchor_naming_a_siblings_value_leaves_the_list() -> None:
    from jevtools.kinds.base import SlotResult
    from jevtools.kinds.listing import _names_sibling

    def anchor(dist: dict[str, float], value: object) -> SlotResult:
        return SlotResult(path=("x",), kind="ref", stakes="identity", dist=dist,
                          values={k: k.strip('"') for k in dist if k.startswith('"')}, value=value, shape="ok",
                          factor=max(dist.values()))  # fmt: skip

    taken = {'"firm_id"', '"investment"'}
    firm = anchor({'"firm_id"': 0.47, "⊥excluded": 0.52}, "⊥excluded")  # "Firm fixed effects": elected EXCLUDE
    assert _names_sibling(firm, taken)
    assert _names_sibling(anchor({'"investment"': 0.9}, "investment"), taken)
    assert not _names_sibling(anchor({'"capital"': 0.98}, "capital"), taken)
    assert not _names_sibling(firm, set())

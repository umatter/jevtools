"""The BFCL data layer and the port of BFCL's AST checker (Python semantics)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jevtools.bench.bfcl import (
    CATEGORIES,
    BfclCase,
    check,
    check_call,
    load_category,
    standardize_string,
    to_json_schema,
    to_openai_tool,
    value_matches,
)

DATA = Path(__file__).parent / "data"

TRIANGLE = {
    "name": "calculate_triangle_area",
    "description": "Calculate the area of a triangle given its base and height.",
    "parameters": {"type": "dict", "required": ["base", "height"], "properties": {
        "base": {"type": "integer", "description": "The base of the triangle."},
        "height": {"type": "float", "description": "The height of the triangle."},
        "unit": {"type": "string", "description": "The unit of measure"},
        "sides": {"type": "tuple", "items": {"type": "integer"}, "description": "Side lengths"},
        "meta": {"type": "dict", "properties": {"k": {"type": "any"}}, "description": "Extra"},
    }},
}  # fmt: skip
ANSWER = {"calculate_triangle_area": {"base": [10], "height": [5.0], "unit": ["units", ""], "sides": [[3, 4], ""],
                                      "meta": [""]}}  # fmt: skip


def case(category: str = "simple_python", truth: list[dict] | None = None) -> BfclCase:
    return BfclCase(id="t", category=category, messages=[], functions=[TRIANGLE],
                    ground_truth=[ANSWER] if truth is None else truth)  # fmt: skip


def test_schema_dialect_becomes_json_schema() -> None:
    schema = to_json_schema(TRIANGLE["parameters"])
    assert schema["type"] == "object"
    assert schema["properties"]["height"]["type"] == "number"
    assert schema["properties"]["sides"]["type"] == "array"
    assert schema["properties"]["meta"]["type"] == "object"
    assert "type" not in schema["properties"]["meta"]["properties"]["k"]  # "any": no constraint
    tool = to_openai_tool(TRIANGLE)
    assert tool["function"]["name"] == "calculate_triangle_area" and tool["type"] == "function"


def test_standardize_string_matches_bfcl() -> None:
    assert standardize_string("April 1, 2024") == standardize_string("april 1 2024") == "april12024"
    assert standardize_string("New-York_City/NY.") == "newyorkcityny"
    assert standardize_string("it's") == 'it"s'


@pytest.mark.parametrize(
    ("args", "valid", "error_type"),
    [
        ({"base": 10, "height": 5}, True, None),  # int accepted for a float parameter
        ({"base": 10, "height": 5.0, "unit": "Units"}, True, None),  # strings compared case-insensitively
        ({"base": 10, "height": 5.0, "sides": [3, 4]}, True, None),
        ({"base": 10}, False, "missing_required"),
        ({"base": 11, "height": 5}, False, "value_error:others"),
        ({"base": 10, "height": 5, "unit": "cm"}, False, "value_error:string"),
        ({"base": 10, "height": 5, "colour": "red"}, False, "unexpected_param"),
        ({"base": "10", "height": 5}, False, "type_error:simple"),
        ({"base": 10, "height": 5, "sides": [4, 3]}, False, "value_error:list/tuple"),  # list order matters
    ],
)
def test_check_call(args: dict, valid: bool, error_type: str | None) -> None:
    result = check_call(TRIANGLE, {"calculate_triangle_area": args}, ANSWER)
    assert (result.valid, result.error_type) == (valid, error_type)


def test_wrong_function_and_count() -> None:
    assert check(case(), [{"other": {}}]).error_type == "wrong_func_name"
    assert check(case(), []).error_type == "wrong_count"


def test_missing_optional_without_empty_marker() -> None:
    truth = [{"calculate_triangle_area": {"base": [10], "height": [5.0], "unit": ["cm"]}}]
    assert check(case(truth=truth), [{"calculate_triangle_area": {"base": 10, "height": 5}}]).error_type == (
        "missing_optional"
    )


def test_irrelevance_and_relevance() -> None:
    assert check(case("irrelevance", []), []).valid
    assert not check(case("irrelevance", []), [{"calculate_triangle_area": {}}]).valid
    assert check(case("live_relevance", []), [{"calculate_triangle_area": {}}]).valid
    assert not check(case("live_relevance", []), []).valid


def test_parallel_matches_in_any_order() -> None:
    truth = [{"calculate_triangle_area": {"base": [1], "height": [1.0]}},
             {"calculate_triangle_area": {"base": [2], "height": [2.0]}}]  # fmt: skip
    calls = [
        {"calculate_triangle_area": {"base": 2, "height": 2}},
        {"calculate_triangle_area": {"base": 1, "height": 1}},
    ]
    assert check(case("parallel", truth), calls).valid
    assert check(case("parallel", truth), calls[:1]).error_type == "wrong_count"


def test_dict_values() -> None:
    fn = {"name": "f", "parameters": {"type": "dict", "required": ["d"], "properties": {"d": {"type": "dict"}}}}
    answer = {"f": {"d": [{"size": ["large"], "note": ["", "hot"]}]}}
    assert check_call(fn, {"f": {"d": {"size": "Large"}}}, answer).valid
    assert check_call(fn, {"f": {"d": {"size": "small"}}}, answer).error_type == "value_error:dict_value"
    assert check_call(fn, {"f": {"d": {"size": "large", "x": 1}}}, answer).error_type == "value_error:dict_value"


def test_value_matches() -> None:
    assert value_matches(10, [10.0]) and value_matches("10", [10], "integer")
    assert value_matches(True, [True]) and not value_matches(True, [1])
    assert value_matches("new york", ["New York"]) and not value_matches("", ["", "x"])
    assert value_matches([3, 4], [[3, 4]]) and not value_matches([4, 3], [[3, 4]])


def test_load_the_vendored_sample() -> None:
    cases = load_category(DATA, "simple_python")
    assert [c.id for c in cases][:2] == ["simple_python_0", "simple_python_1"]
    first = cases[0]
    assert first.messages[0]["role"] == "user" and first.functions[0]["name"] == "calculate_triangle_area"
    assert first.ground_truth == [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
    assert load_category(DATA, "irrelevance", limit=1)[0].ground_truth == []
    with pytest.raises(ValueError, match="unsupported"):
        load_category(DATA, "multi_turn_base")
    assert "simple_java" not in CATEGORIES

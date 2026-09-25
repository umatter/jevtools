"""BFCL (Berkeley Function Calling Leaderboard) data and its AST checker, for single-turn categories.

BFCL (https://github.com/ShishirPatil/gorilla, Apache-2.0) states each test case as user messages plus a list of
function documents in a JSON-Schema dialect (``dict``, ``float``, ``tuple``, ``any``), and each answer as the
acceptable values of every parameter: ``{"calculate_triangle_area": {"base": [10], "unit": ["units", ""]}}``, where
``""`` means the parameter may be omitted. This module loads those files, converts the functions into OpenAI tools,
and checks a call the way BFCL's ``ast_checker`` does for Python (string values compared case-insensitively without
spaces and ``,./-_*^``, an ``int`` accepted for a ``float``, one level of nested checks, dict and list-of-dict values).

Only the single-turn categories are supported. The ``parallel`` categories need several calls per turn, which
jevtools does not decide (``ext.parallel``), so they are scored as BFCL scores them and will fail on the call count;
Java and JavaScript categories, multi-turn, memory and web-search categories are out of scope.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BFCL_REPO = "ShishirPatil/gorilla"
BFCL_VERSION = "v4"
"""The data file prefix (``BFCL_v4_<category>.json``)."""
BFCL_DATA_URL = "https://raw.githubusercontent.com/{repo}/{ref}/berkeley-function-call-leaderboard/bfcl_eval/data"

SINGLE_CALL = ("simple_python", "multiple", "live_simple", "live_multiple")
"""Categories whose answer is exactly one call."""
PARALLEL = ("parallel", "parallel_multiple", "live_parallel", "live_parallel_multiple")
"""Categories whose answer is several calls in one turn (not decided by jevtools; scored as BFCL scores them)."""
IRRELEVANCE = ("irrelevance", "live_irrelevance")
"""Categories where the right answer is to call no function."""
RELEVANCE = ("live_relevance",)
"""Categories where the right answer is to call some function (any arguments)."""
CATEGORIES = (*SINGLE_CALL, *IRRELEVANCE, *RELEVANCE, *PARALLEL)
"""Supported categories (Python only)."""
DEFAULT_CATEGORIES = (*SINGLE_CALL, *IRRELEVANCE, *RELEVANCE)
"""What ``jevtools bench bfcl`` runs unless told otherwise (the parallel categories are opt-in)."""

_TYPE_MAP: dict[str, str | None] = {"dict": "object", "float": "number", "tuple": "array", "any": None}
_PY_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str, "integer": int, "float": float, "boolean": bool, "array": list, "tuple": list, "dict": dict,
    "any": str,
}  # fmt: skip
"""BFCL's ``PYTHON_TYPE_MAPPING``."""
_NESTED = ("array", "tuple")
_STANDARDIZE = re.compile(r"[ ,./\-_*^]")


# --------------------------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BfclCase:
    """One BFCL test case: the messages of its (single) turn, its functions and its acceptable answers."""

    id: str
    category: str
    messages: list[dict[str, Any]]
    functions: list[dict[str, Any]]
    ground_truth: list[dict[str, dict[str, list[Any]]]] = field(default_factory=list)
    """One ``{function: {param: [acceptable values]}}`` per expected call; empty for (ir)relevance categories."""

    @property
    def tools(self) -> list[dict[str, Any]]:
        """The functions as OpenAI tools (JSON Schema)."""
        return [to_openai_tool(f) for f in self.functions]

    def function(self, name: str) -> dict[str, Any] | None:
        """The BFCL function document named ``name``."""
        return next((f for f in self.functions if f.get("name") == name), None)


def data_file(directory: Path, category: str, *, answers: bool = False) -> Path:
    """``<dir>/BFCL_v4_<category>.json`` or ``<dir>/possible_answer/BFCL_v4_<category>.json``."""
    name = f"BFCL_{BFCL_VERSION}_{category}.json"
    return directory / "possible_answer" / name if answers else directory / name


def has_answers(category: str) -> bool:
    """Whether BFCL ships a ``possible_answer`` file for ``category``."""
    return category not in (*IRRELEVANCE, *RELEVANCE)


def load_category(directory: str | Path, category: str, *, limit: int | None = None) -> list[BfclCase]:
    """Load one category's cases (and answers) from a BFCL data directory (see :func:`download`)."""
    if category not in CATEGORIES:
        raise ValueError(f"unsupported BFCL category {category!r}; supported: {', '.join(CATEGORIES)}")
    directory = Path(directory)
    rows = _read_jsonl(data_file(directory, category))
    answers: dict[str, Any] = {}
    if has_answers(category):
        answers = {a["id"]: a["ground_truth"] for a in _read_jsonl(data_file(directory, category, answers=True))}
    cases: list[BfclCase] = []
    for row in rows[:limit] if limit is not None else rows:
        turns = row.get("question") or [[]]
        cases.append(BfclCase(id=str(row["id"]), category=category, messages=list(turns[0]),
                              functions=list(row.get("function") or []),
                              ground_truth=list(answers.get(str(row["id"]), []))))  # fmt: skip
    return cases


def download(
    directory: str | Path, categories: Iterable[str] = DEFAULT_CATEGORIES, *, ref: str = "main", repo: str = BFCL_REPO
) -> list[Path]:
    """Fetch the data (and answer) files of ``categories`` from GitHub into ``directory``; returns the files written.

    ``ref`` pins a branch, tag or commit of the BFCL repository so that results are reproducible.
    """
    directory = Path(directory)
    base = BFCL_DATA_URL.format(repo=repo, ref=ref)
    written: list[Path] = []
    for category in categories:
        targets = [(data_file(directory, category), f"{base}/{data_file(Path(), category).name}")]
        if has_answers(category):
            name = data_file(Path(), category).name
            targets.append((data_file(directory, category, answers=True), f"{base}/possible_answer/{name}"))
        for path, url in targets:
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https GitHub URL
                path.write_bytes(response.read())
            written.append(path)
    return written


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------------------------------------------------
# Function documents → OpenAI tools
# --------------------------------------------------------------------------------------------------------------------


def to_json_schema(schema: Any) -> Any:
    """BFCL's schema dialect → JSON Schema: ``dict`` → ``object``, ``float`` → ``number``, ``tuple`` → ``array``,
    ``any`` → no type constraint. Everything else is kept."""
    if isinstance(schema, list):
        return [to_json_schema(s) for s in schema]
    if not isinstance(schema, Mapping):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str) and value in _TYPE_MAP:
            mapped = _TYPE_MAP[value]
            if mapped is not None:
                out[key] = mapped
        elif key == "properties" and isinstance(value, Mapping):
            out[key] = {name: to_json_schema(sub) for name, sub in value.items()}
        else:
            out[key] = to_json_schema(value)
    return out


def to_openai_tool(function: Mapping[str, Any]) -> dict[str, Any]:
    """A BFCL function document as an OpenAI function tool (name and description unchanged)."""
    parameters = to_json_schema(function.get("parameters") or {"type": "object", "properties": {}})
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})
    return {"type": "function", "function": {"name": function["name"],
                                             "description": function.get("description", ""),
                                             "parameters": parameters}}  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# The AST checker (Python semantics of bfcl_eval/eval_checker/ast_eval/ast_checker.py)
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """Whether a model output matches the answer, and BFCL's error type when it does not."""

    valid: bool
    error_type: str | None = None
    error: str | None = None


def standardize_string(text: str) -> str:
    """BFCL's ``standardize_string``: drop spaces and ``,./-_*^``, lowercase, single → double quotes."""
    return _STANDARDIZE.sub("", text).lower().replace("'", '"')


def check(case: BfclCase, calls: Sequence[Mapping[str, Mapping[str, Any]]]) -> CheckResult:
    """Score ``calls`` (``[{function_name: {param: value}}]``) against ``case`` as BFCL does for its category."""
    if case.category in IRRELEVANCE:
        return CheckResult(True) if not calls else CheckResult(False, "irrelevance:called", "a function was called")
    if case.category in RELEVANCE:
        return CheckResult(True) if calls else CheckResult(False, "relevance:no_call", "no function was called")
    if case.category in PARALLEL:
        return _check_parallel(case, calls)
    if len(calls) != len(case.ground_truth):
        return CheckResult(False, "wrong_count", f"expected {len(case.ground_truth)} call(s), got {len(calls)}")
    expected = next(iter(case.ground_truth[0]))
    description = case.function(expected)  # simple: the only function; multiple: the one the answer names
    if description is None:
        return CheckResult(False, "missing_description", f"no function document for {expected!r}")
    return check_call(description, calls[0], case.ground_truth[0])


def _check_parallel(case: BfclCase, calls: Sequence[Mapping[str, Mapping[str, Any]]]) -> CheckResult:
    if len(calls) != len(case.ground_truth):
        return CheckResult(False, "wrong_count", f"expected {len(case.ground_truth)} call(s), got {len(calls)}")
    matched: set[int] = set()
    for answer in case.ground_truth:
        description = case.function(next(iter(answer)))
        if description is None:
            return CheckResult(False, "missing_description", "no function document")
        for index, call in enumerate(calls):
            if index not in matched and check_call(description, call, answer).valid:
                matched.add(index)
                break
        else:
            return CheckResult(False, "parallel:cannot_find_match", "an expected call has no matching output")
    return CheckResult(True)


def check_call(
    description: Mapping[str, Any], call: Mapping[str, Mapping[str, Any]], answer: Mapping[str, Mapping[str, list[Any]]]
) -> CheckResult:
    """BFCL's ``simple_function_checker`` (Python): name, required parameters, types and values, optional ones."""
    possible = next(iter(answer.values()))
    name = str(description["name"])
    if name not in call:
        return CheckResult(False, "wrong_func_name", f"function {name!r} not in the output")
    params = dict(call[name])
    details: Mapping[str, Any] = description.get("parameters", {}).get("properties", {})
    for required in description.get("parameters", {}).get("required", []):
        if required not in params:
            return CheckResult(False, "missing_required", f"missing required parameter {required!r}")
    for param, value in params.items():
        if param not in details or param not in possible:
            return CheckResult(False, "unexpected_param", f"unexpected parameter {param!r}")
        result = _check_param(param, value, possible[param], details[param])
        if not result.valid:
            return result
    for param, acceptable in possible.items():
        if param not in params and "" not in acceptable:
            return CheckResult(False, "missing_optional", f"optional parameter {param!r} not provided")
    return CheckResult(True)


def _check_param(param: str, value: Any, acceptable: list[Any], detail: Mapping[str, Any]) -> CheckResult:
    expected = str(detail.get("type", "any"))
    py_type = _PY_TYPES.get(expected, str)
    nested = None
    if expected in _NESTED:
        nested = _PY_TYPES.get(str((detail.get("items") or {}).get("type", "any")), str)
    if expected == "tuple" and isinstance(value, tuple):
        value = list(value)
    if expected == "float" and type(value) is int:
        value = float(value)
    typed = _type_check(value, acceptable, py_type, nested)
    if not typed.valid:
        return CheckResult(False, typed.error_type, f"parameter {param!r}: {typed.error}")
    is_variable = typed.error == "variable"
    if not is_variable:
        if py_type is dict:
            return _dict_check(param, value, acceptable)
        if py_type is list and nested is dict:
            return _list_dict_check(param, value, acceptable)
        if py_type is str:
            options = [standardize_string(a) for a in acceptable if isinstance(a, str)]
            if standardize_string(value) not in options:
                return CheckResult(False, "value_error:string", f"parameter {param!r}: {value!r} not in {acceptable}")
            return CheckResult(True)
        if py_type is list:
            return _list_check(param, value, acceptable)
    if value not in acceptable:
        return CheckResult(False, "value_error:others", f"parameter {param!r}: {value!r} not in {acceptable}")
    return CheckResult(True)


def _answer_type(acceptable: Sequence[Any]) -> type | None:
    return next((type(a) for a in acceptable if a != ""), None)


def _type_check(value: Any, acceptable: Sequence[Any], py_type: Any, nested: Any) -> CheckResult:
    """BFCL's ``type_checker``; ``error="variable"`` marks a value typed like the answer rather than the schema."""
    answer_type = _answer_type(acceptable)
    variable = answer_type is not None and answer_type is not py_type
    if type(value) is py_type:
        if nested is None:
            return CheckResult(True, None, "variable" if variable else None)
        for item in acceptable:
            if isinstance(item, list) and all(_type_check(v, item, nested, None).valid for v in value):
                return CheckResult(True, None, "variable" if variable else None)
        return CheckResult(False, "type_error:nested", f"nested type check failed for {value!r}")
    if answer_type is not None and type(value) is answer_type:
        return CheckResult(True, None, "variable")
    return CheckResult(False, "type_error:simple", f"expected {getattr(py_type, '__name__', py_type)}, got "
                                                    f"{type(value).__name__} ({value!r})")  # fmt: skip


def _standard(value: Any) -> Any:
    return standardize_string(value) if isinstance(value, str) else value


def _list_check(param: str, value: list[Any], acceptable: Sequence[Any]) -> CheckResult:
    output = [_standard(v) for v in value]
    options = [[_standard(v) for v in a] for a in acceptable if isinstance(a, list)]
    if output not in options:
        return CheckResult(False, "value_error:list/tuple", f"parameter {param!r}: {value!r} not in {acceptable}")
    return CheckResult(True)


def _dict_check(param: str, value: Mapping[str, Any], acceptable: Sequence[Any]) -> CheckResult:
    result = CheckResult(False, "dict_checker:unclear", f"parameter {param!r}: no acceptable dict")
    for option in acceptable:
        if option == "" or not isinstance(option, Mapping):
            continue
        ok = True
        for key, item in value.items():
            if key not in option or _standard(item) not in [_standard(a) for a in option[key]]:
                result, ok = (
                    CheckResult(False, "value_error:dict_value", f"parameter {param!r}: bad key {key!r}"),
                    False,
                )
                break
        if ok:
            for key, choices in option.items():
                if key not in value and "" not in choices:
                    result, ok = (
                        CheckResult(False, "value_error:dict_key", f"parameter {param!r}: missing {key!r}"),
                        False,
                    )
                    break
        if ok:
            return CheckResult(True)
    return result


def _list_dict_check(param: str, value: Sequence[Any], acceptable: Sequence[Any]) -> CheckResult:
    result = CheckResult(False, "list_dict_checker:unclear", f"parameter {param!r}: no acceptable list of dicts")
    for option in acceptable:
        if not isinstance(option, list) or len(option) != len(value):
            result = CheckResult(False, "value_error:list_dict_count", f"parameter {param!r}: wrong length")
            continue
        if all(_dict_check(param, v, [o]).valid for v, o in zip(value, option, strict=True)):
            return CheckResult(True)
    return result


# --------------------------------------------------------------------------------------------------------------------
# Value matching used by the oracle (one parameter value against its acceptable values)
# --------------------------------------------------------------------------------------------------------------------


def value_matches(value: Any, acceptable: Sequence[Any], bfcl_type: str = "any") -> bool:
    """Whether ``value`` would pass BFCL's check for a parameter whose acceptable values are ``acceptable``."""
    if isinstance(value, bool):
        return any(isinstance(a, bool) and a == value for a in acceptable)
    if isinstance(value, (int, float)):
        return any(isinstance(a, (int, float)) and not isinstance(a, bool) and float(a) == float(value)
                   for a in acceptable)  # fmt: skip
    if isinstance(value, str):
        wanted = standardize_string(value)
        if any(isinstance(a, str) and a != "" and standardize_string(a) == wanted for a in acceptable):
            return True
        if bfcl_type in ("integer", "float"):  # a numeric string elected for a numeric parameter
            try:
                number = float(value)
            except ValueError:
                return False
            return value_matches(number, acceptable, bfcl_type)
        return False
    if isinstance(value, list):
        return _list_check("", value, acceptable).valid
    if isinstance(value, Mapping):
        return _dict_check("", value, acceptable).valid
    return value in acceptable


__all__ = [
    "BFCL_DATA_URL",
    "BFCL_REPO",
    "BFCL_VERSION",
    "CATEGORIES",
    "DEFAULT_CATEGORIES",
    "IRRELEVANCE",
    "PARALLEL",
    "RELEVANCE",
    "SINGLE_CALL",
    "BfclCase",
    "CheckResult",
    "check",
    "check_call",
    "data_file",
    "download",
    "has_answers",
    "load_category",
    "standardize_string",
    "to_json_schema",
    "to_openai_tool",
    "value_matches",
]

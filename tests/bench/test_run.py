"""The oracle, the runner and the ``jevtools bench bfcl`` command over the vendored BFCL sample.

Oracle numbers are the ceiling of what jevtools can emit, and simulator numbers test plumbing; neither is evidence
about Jev.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from jevtools.backends.scripted import ScriptedBackend
from jevtools.backends.simulator import LexicalSimulator
from jevtools.bench import load_category, run_bfcl, run_case
from jevtools.cli import main

DATA = Path(__file__).parent / "data"
SAMPLE = ("simple_python", "multiple", "irrelevance", "live_simple", "live_multiple", "live_irrelevance",
          "live_relevance", "parallel")  # fmt: skip


def cases(category: str) -> dict:
    return {c.id: c for c in load_category(DATA, category)}


def test_oracle_reaches_the_answer_when_every_value_is_on_the_ballot() -> None:
    record = run_case(cases("simple_python")["simple_python_0"], "oracle")
    assert (record.outcome, record.proposal, record.strict, record.ceiling) == ("execute", True, True, True)
    assert record.call == {"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}
    assert [(c.param, c.status) for c in record.coverage] == [("base", "covered"), ("height", "covered"),
                                                               ("unit", "omitted")]  # fmt: skip


def test_oracle_attributes_misses() -> None:
    simple = cases("simple_python")
    miss = run_case(simple["simple_python_7"], "oracle")  # the unit "inches" is never nominated
    assert not miss.proposal and miss.attribution == "value_not_nominated"
    assert {c.param: c.status for c in miss.coverage}["unit"] == "uncovered"
    typed = run_case(simple["simple_python_13"], "oracle")  # "y=x^2" is nominated; [1, 3] fails BFCL's float check
    assert typed.attribution == "wrong_call" and typed.error_type == "type_error:nested"
    skipped = run_case(simple["simple_python_34"], "oracle")
    assert skipped.attribution == "tool_not_speculated" and skipped.rule.startswith("P6")


def test_oracle_abstains_on_irrelevance() -> None:
    for case in cases("irrelevance").values():
        record = run_case(case, "oracle")
        assert (record.outcome, record.proposal, record.strict) == ("abstain", True, True)


def test_list_values_keep_mention_order() -> None:
    record = run_case(cases("multiple")["multiple_15"], "oracle")
    assert record.proposal and record.call is not None
    assert record.call["arguments"]["coordinates"] == [43.653225, -79.383186]


def test_non_oracle_runs_record_the_ceiling() -> None:
    case = cases("simple_python")["simple_python_0"]
    uniform = ScriptedBackend({})  # every answer uniform: nothing is elected with confidence
    record = run_case(case, uniform)
    assert record.ceiling is True and not record.proposal
    assert record.attribution in ("no_call", "wrong_call")
    assert record.jev_calls == 1


def test_report_summary_and_render() -> None:
    report = run_bfcl([c for name in SAMPLE for c in load_category(DATA, name)], "oracle")
    summary = report.summary()
    assert set(summary) == {*SAMPLE, "ALL"}
    assert summary["irrelevance"]["proposal"] == 1.0 and summary["parallel"]["proposal"] == 0.0
    assert summary["ALL"]["ceiling"] == summary["ALL"]["proposal"]  # an oracle run is its own ceiling
    table = report.render()
    assert table.splitlines()[0].startswith("| category | n | ceiling | proposal | strict |")
    doc = json.loads(report.to_json())
    assert doc["mode"] == "oracle" and len(doc["records"]) == len(report.records)


def test_simulator_run_is_deterministic() -> None:
    sample = list(cases("simple_python").values())[:2]
    first = run_bfcl(sample, LexicalSimulator()).to_json()
    second = run_bfcl(sample, LexicalSimulator()).to_json()
    assert json.loads(first)["summary"] == json.loads(second)["summary"]


@pytest.mark.parametrize("backend", ["oracle", "sim"])
def test_cli_bench(backend: str, tmp_path: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    report = tmp_path / "report.json"
    code = main(["bench", "bfcl", "--data", str(DATA), "--categories", "simple_python,irrelevance",
                 "--backend", backend, "--out", str(report)], out=out, err=err)  # fmt: skip
    assert code == 0, err.getvalue()
    text = out.getvalue()
    assert "| simple_python | 5 |" in text and "| irrelevance | 3 |" in text
    assert ("ceiling of what jevtools can emit" in text) if backend == "oracle" else ("never evidence" in text)
    assert json.loads(report.read_text())["summary"]["simple_python"]["n"] == 5


def test_cli_bench_missing_data(tmp_path: Path) -> None:
    err = io.StringIO()
    assert main(["bench", "bfcl", "--data", str(tmp_path)], out=io.StringIO(), err=err) == 1
    assert "--download" in err.getvalue()

"""The app-domain benchmark: bundled domains, the oracle over §11.1 gold labels, the runner and ``jevtools bench app``.

Oracle numbers are the ceiling of what jevtools can emit, and simulator numbers test plumbing; neither is evidence
about Jev.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from jevtools.backends.simulator import LexicalSimulator
from jevtools.bench.app import DOMAINS, domains_dir, load_domain, run_app, run_domains
from jevtools.bench.app._generate import main as generate
from jevtools.bench.oracle import ANY, EvalGold
from jevtools.cli import main
from jevtools.eval.dataset import EvalCase, parse_case

CEILING_MISSES = {"inbox-17"}
"""Cases the oracle cannot get right: "the head of legal" names a contact by an attribute no match field holds."""


@pytest.fixture(scope="module")
def oracle_report():  # type: ignore[no-untyped-def]
    return run_domains()


def case(domain: str, case_id: str) -> EvalCase:
    return next(c for c in load_domain(domain) if c.id == case_id)


def test_every_bundled_domain_loads_and_its_gold_names_real_tools() -> None:
    assert set(DOMAINS) == {p.name for p in domains_dir().iterdir() if (p / "cases.jsonl").is_file()}
    for name in DOMAINS:
        cases = load_domain(name)
        assert len(cases) >= 15
        assert len({c.id for c in cases}) == len(cases)
        for c in cases:
            tools = {t["function"]["name"]: t["function"] for t in c.catalog_tools() or []}
            ctx = c.build_context()
            assert ctx.sources, c.id
            if c.gold.tool is not None:
                assert c.gold.tool in tools, c.id
                assert set(c.gold.slots) <= set(tools[c.gold.tool]["parameters"]["properties"]), c.id


def test_domain_data_matches_its_generator() -> None:
    assert generate(["--check"]) == 0


def test_unknown_domain_is_a_clear_error() -> None:
    with pytest.raises(FileNotFoundError, match="no app-bench domain 'nope'"):
        load_domain("nope")


def test_eval_gold_marks_unchecked_arguments_and_clarify_splits() -> None:
    doc = {"id": "x", "messages": "Email Anna", "context": "context.json", "catalog": "catalog.json",
           "gold": {"outcomes_ok": ["clarify"], "tool": "send_email",
                    "accepted": {"to": ["anna.keller@acme.com", "anna.rossi@gmail.com"]}}}  # fmt: skip
    c = parse_case(doc, base_dir=domains_dir() / "inbox")
    gold = EvalGold(c, c.catalog_tools() or [])
    assert gold.tool == "send_email" and not gold.no_tool and gold.split
    assert gold.params["to"] == ["anna.keller@acme.com", "anna.rossi@gmail.com"]
    assert gold.params["subject"] == [ANY, ""]  # not checked: any value, or none
    assert gold.matches("Hi", gold.params["subject"]) and not gold.matches("", [ANY])
    assert gold.matches(45.0, [45]) and not gold.matches("anna@x.ch", gold.params["to"])


def test_oracle_ceiling_over_the_bundled_domains(oracle_report) -> None:  # type: ignore[no-untyped-def]
    missed = {r.record.case_id for r in oracle_report.records if not r.record.correct}
    assert missed == CEILING_MISSES
    summary = oracle_report.summary()["ALL"]
    assert summary["n"] == sum(len(load_domain(d)) for d in DOMAINS)
    assert summary["wrong_executions"] == 0
    assert summary["injection_hits"] == 0 and summary["injection_cases"] == 6
    assert summary["clarify_useful"] == 1.0


def test_oracle_decisions_worth_pinning(oracle_report) -> None:  # type: ignore[no-untyped-def]
    by_id = {r.record.case_id: r.record for r in oracle_report.records}
    # critical tier: a perfect judge reaches the confirm band (joint question answered), never execute
    assert (by_id["bank-01"].outcome, by_id["bank-01"].rule) == ("confirm", "P9.critical.confirm_band")
    # an infeasible transfer (50,000 from a 12,500 account) is a consistency problem, not a tool menu
    assert by_id["bank-04"].rule == "P8.consistency"
    # a 50/50 between two reps named Jonas is a menu, not a confirm card that guesses
    assert (by_id["crm-08"].outcome, by_id["crm-08"].rule) == ("clarify", "P9.write.ambiguous")
    # typed identifiers anchor their rows: "D-1017", "INC-1052", and "ticket 1100" for INC-1100
    assert by_id["crm-01"].arguments == {"deal_id": "D-1017", "stage": "negotiation"}
    assert by_id["hd-15"].arguments["ticket_id"] == "INC-1100"
    # "since September 1" said on 24 September 2026 is the past one
    assert by_id["bank-10"].arguments == {"account": "acc_2210", "since": "2026-09-01"}
    # "last week's weekly report" is the latest report (a superlative over the matching files)
    assert by_id["ws-07"].arguments == {"path": "reports/weekly/2026-09-18_weekly_report.md"}
    # file names and names given with "called" are nominated verbatim
    assert by_id["ws-13"].arguments["new_name"] == "notes_old.txt"
    assert by_id["crm-05"].arguments["title"] == "Data platform phase 2"  # a title is sentence-cased


def test_report_renders_and_serializes(oracle_report, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    table = oracle_report.render(tags=True)
    assert all(f"| {d} |" in table for d in DOMAINS) and "| ALL |" in table and "| injection |" in table
    path = tmp_path / "report.json"
    oracle_report.save(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["mode"] == "oracle" and len(doc["records"]) == len(oracle_report.records)
    assert {"ceiling", "correct", "within_ceiling", "calls_right", "stages"} <= set(doc["summary"]["ALL"])


def test_a_live_style_run_records_the_ceiling_next_to_its_own_accuracy() -> None:
    cases = [("inbox", case("inbox", "inbox-01")), ("inbox", case("inbox", "inbox-17"))]
    report = run_app(cases, LexicalSimulator())
    assert report.mode == "simulator"
    assert [r.ceiling for r in report.records] == [True, False]
    summary = report.summary()["inbox"]
    assert summary["ceiling"] == 0.5 and summary["within_ceiling"] in (0.0, 1.0)


def test_cli_bench_app(tmp_path: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    path = tmp_path / "app.json"
    assert main(["bench", "app", "--domains", "helpdesk", "--tags", "--out", str(path)], out=out, err=err) == 0
    text = out.getvalue()
    assert "16 case(s) in 1 domain(s) on oracle" in text and "| helpdesk | 16 |" in text and "| identifier |" in text
    assert "not a measurement of Jev" in text and path.is_file()
    assert main(["bench", "app", "--domains", "nope"], out=io.StringIO(), err=err) == 1
    assert "no app-bench domain 'nope'" in err.getvalue()


def test_cli_bench_app_on_a_custom_directory(tmp_path: Path) -> None:
    domain = tmp_path / "mine"
    domain.mkdir()
    source = domains_dir() / "helpdesk"
    for name in ("catalog.json", "context.json"):
        (domain / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    (domain / "data").mkdir()
    for file in (source / "data").iterdir():
        (domain / "data" / file.name).write_text(file.read_text(encoding="utf-8"), encoding="utf-8")
    first = (source / "cases.jsonl").read_text(encoding="utf-8").splitlines()[0]
    (domain / "cases.jsonl").write_text(first + "\n", encoding="utf-8")
    out = io.StringIO()
    assert main(["bench", "app", "--dir", str(tmp_path), "--backend", "sim"], out=out, err=io.StringIO()) == 0
    assert "1 case(s) in 1 domain(s) on simulator" in out.getvalue() and "| mine | 1 |" in out.getvalue()

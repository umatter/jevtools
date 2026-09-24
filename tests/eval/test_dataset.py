"""The §11.1 dataset: gold match modes, JSONL loading and case contexts/catalogs from files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevtools.backends.scripted import ScriptedBackend
from jevtools.eval.dataset import EvalCase, Gold, dump, load, loads, same_value, select, template_matches
from jevtools.eval.harness import router_factory_for
from jevtools.policy import Outcome
from jevtools.sources import Registry
from tests.eval.support import R2_BODY
from tests.scenario import scripts
from tests.scenario.fixtures import R2_HISTORY, SCENARIO_CONTACTS, scenario_tools

SPEC_LINE = {
    "id": "r2-anna-hist",
    "messages": [*R2_HISTORY, {"role": "user", "content": "Email Anna that I'll be 10 minutes late"}],
    "context": "fixtures/ctx_default.json",
    "catalog": "fixtures/catalog.json",
    "gold": {"outcomes_ok": ["execute", "confirm"], "tool": "send_email", "args": {"to": "anna.keller@acme.com"},
             "match": {"to": "exact", "body": "accepted_set", "subject": "ignore"},
             "accepted": {"body": [R2_BODY, "I'll be 10 minutes late."]}},
    "tags": ["name_collision", "history"],
}  # fmt: skip


def write_dataset(tmp_path: Path) -> Path:
    """The spec's example case with its context and catalog files next to it."""
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    ctx = {"now": "2026-09-24T14:05:00+02:00", "tz": "Europe/Zurich", "locale": "en-CH",
           "user": {"name": "Sam Muster", "home_city": "Zurich"},
           "sources": [{"name": "contacts", "rows": SCENARIO_CONTACTS, "key": "email", "label": "{name} <{email}>",
                        "describe": "{notes}", "match": ["name", "aliases", "team"],
                        "provides": ["email", "person"]}]}  # fmt: skip
    (fixtures / "ctx_default.json").write_text(json.dumps(ctx), encoding="utf-8")
    (fixtures / "catalog.json").write_text(json.dumps(scenario_tools()), encoding="utf-8")
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(SPEC_LINE) + "\n\n", encoding="utf-8")
    return path


def test_gold_match_modes() -> None:
    gold = Gold.model_validate(SPEC_LINE["gold"])
    assert gold.mode("to") == "exact" and gold.mode("body") == "accepted_set" and gold.mode("subject") == "ignore"
    assert gold.mode("unlisted") == "ignore" and gold.slots == ["to", "body"]
    good = {"to": "anna.keller@acme.com", "subject": "anything", "body": R2_BODY}
    assert gold.call_matches("send_email", good)
    assert gold.call_matches("send_email", {**good, "body": "I'll be 10 minutes late."})
    assert not gold.call_matches("send_email", {**good, "body": "Hello"})
    assert not gold.call_matches("send_email", {**good, "to": "anna.rossi@gmail.com"})
    assert not gold.call_matches("send_email", {"to": "anna.keller@acme.com"})  # a checked argument is missing
    assert not gold.call_matches("read_file", good) and not gold.call_matches(None, None)
    assert gold.allows("confirm") and not gold.allows(Outcome.CLARIFY)


def test_gold_without_tool_and_defaults() -> None:
    none = Gold(outcomes_ok=[Outcome.ABSTAIN])
    assert none.call_matches(None, None) and not none.call_matches("search_web", {})
    implicit = Gold.model_validate({"outcomes_ok": ["execute"], "tool": "t", "args": {"n": 45},
                                    "accepted": {"s": ["a", "b"]}})  # fmt: skip
    assert implicit.mode("n") == "exact" and implicit.mode("s") == "accepted_set"
    assert implicit.accepts("n", 45.0) and implicit.values("s") == ["a", "b"]


@pytest.mark.parametrize(
    "gold",
    [
        {"outcomes_ok": []},
        {"outcomes_ok": ["execute"], "tool": "t", "match": {"x": "exact"}},
        {"outcomes_ok": ["execute"], "tool": "t", "match": {"x": "accepted_set"}},
        {"outcomes_ok": ["execute"], "args": {"x": 1}},
        {"outcomes_ok": ["maybe"]},
    ],
)
def test_invalid_gold_is_rejected(gold: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Gold.model_validate(gold)


def test_same_value_and_templates() -> None:
    assert same_value(45, 45.0) and same_value({"a": [1, 2]}, {"a": [1, 2]}) and not same_value("1", 1)
    template = "Hi ⟨recipient's first name⟩,\n\nI'll be 10 minutes late.\n\nBest,\nSam"
    assert template_matches(template, R2_BODY)
    assert not template_matches(template, "Hi Anna, see you")
    assert template_matches("x", "x") and not template_matches(1, "1")


def test_load_resolves_files_and_builds_the_router(tmp_path: Path) -> None:
    path = write_dataset(tmp_path)
    [c] = load(path)
    assert c.id == "r2-anna-hist" and c.base_dir == tmp_path and c.request.startswith("Email Anna")
    tools = c.catalog_tools()
    assert tools is not None and [t["function"]["name"] for t in tools][:2] == ["get_weather", "send_email"]
    ctx = c.build_context()
    assert ctx.locale == "en-CH" and ctx.timezone_name == "Europe/Zurich" and ctx.user["home_city"] == "Zurich"
    assert isinstance(ctx.sources["contacts"], Registry) and len(ctx.sources["contacts"]) == len(SCENARIO_CONTACTS)
    assert ctx.request == c.request and len(ctx.messages) == 3

    backend = ScriptedBackend(scripts.R2, model="~typesafe/jev-latest")
    router = router_factory_for(backend)(c)
    decision = router.decide(c.messages)
    assert decision.outcome is Outcome.CONFIRM and decision.call is not None
    assert c.gold.call_matches(decision.call.name, decision.call.arguments)


def test_load_errors_name_the_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(SPEC_LINE) + "\n" + json.dumps(SPEC_LINE) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:2: duplicate case id"):
        load(path)
    path.write_text('{"id": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:1: invalid case"):
        load(path)


def test_dump_round_trip_and_select(tmp_path: Path) -> None:
    cases = loads(json.dumps(SPEC_LINE) + "\n" + json.dumps({**SPEC_LINE, "id": "b", "tags": ["x"]}))
    dump(cases, tmp_path / "out.jsonl")
    again = load(tmp_path / "out.jsonl")
    assert [c.model_dump() for c in again] == [c.model_dump() for c in cases]
    assert [c.id for c in select(again, tags=["x"])] == ["b"]
    assert [c.id for c in select(again, ids=["r2-anna-hist"])] == ["r2-anna-hist"]
    assert EvalCase.model_validate({"id": "s", "messages": "hi", "gold": {"outcomes_ok": ["abstain"]}}).request == "hi"


def test_variants_keep_the_base_directory(tmp_path: Path) -> None:
    [c] = load(write_dataset(tmp_path))
    v = c.variant("masked", messages=[{"role": "user", "content": "Email [someone]"}], meta={"k": 1})
    assert v.id == "r2-anna-hist~masked" and v.base_dir == tmp_path
    assert v.meta == {"k": 1, "variant_of": "r2-anna-hist"} and v.catalog_tools() == c.catalog_tools()


def test_run_dataset_end_to_end(tmp_path: Path) -> None:
    from jevtools.eval.harness import run_dataset

    path = write_dataset(tmp_path)
    report = run_dataset(path, ScriptedBackend(scripts.R2, model="~typesafe/jev-latest"), keep_traces=False)
    [rec] = report.records
    assert rec.correct and rec.outcome == "confirm" and rec.locale == "en-CH" and rec.trace is None
    assert report.meta["dataset"] == str(path) and report.meta["backend"] == "scripted"


def test_source_paths_in_a_context_file_resolve_against_that_file(tmp_path: Path) -> None:
    """A context document's relative source paths (``rows_file``, ``paths_file``, ``path``) are relative to the
    context file, as for ``jevtools verify --context``: a golden context loads from a dataset elsewhere (#14)."""
    golden = Path(__file__).resolve().parents[1] / "golden" / "R1" / "context.json"
    line = {"id": "r1", "messages": [{"role": "user", "content": "What's the weather like in Zurich?"}],
            "context": str(golden), "gold": {"outcomes_ok": ["execute"]}}  # fmt: skip
    data = tmp_path / "deep" / "cases.jsonl"
    data.parent.mkdir()
    data.write_text(json.dumps(line) + "\n", encoding="utf-8")
    [case] = load(data)
    ctx = case.build_context()
    assert {"contacts", "files"} <= set(ctx.sources) and len(ctx.sources["contacts"]) > 0
    # an inline context document keeps resolving against the dataset directory
    (tmp_path / "deep" / "rows.json").write_text(json.dumps(SCENARIO_CONTACTS), encoding="utf-8")
    inline = {**line, "context": {"sources": [{"name": "contacts", "type": "registry", "rows_file": "rows.json",
                                               "key": "email", "match": ["name"]}]}}  # fmt: skip
    [case2] = loads(json.dumps(inline), base_dir=data.parent)
    assert len(case2.build_context().sources["contacts"]) == len(SCENARIO_CONTACTS)

"""Golden conformance fixtures (spec §10.2), replayed from the files alone (``case.json`` drives everything):

1. ``compile(catalog, context)`` → ``ballot.json`` byte-equal;
2. ``ballot.to_requests(model)`` → ``request*.json`` byte-equal (and every later request of the case too);
3. ``(ballot, responses)`` → ``decision.json`` byte-equal, ignoring ``created_at``;
4. ``verify(trace.json)`` passes (and the regenerated trace equals ``trace.json`` once normalized).

The stored responses are the scripted §13.3 numbers ([I]): these tests pin bytes and policy branches, never Jev's
accuracy. Regenerate with ``jevtools fixtures --update`` (with a spec version bump).
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from jevtools.canonical import canonical_json
from jevtools.cli import main
from jevtools.trace import Trace, verify
from tests.golden.cases import (
    EVIDENCE,
    GOLDEN_DIR,
    SHARED,
    ReplayBackend,
    case_names,
    check_fixtures,
    loop_doc,
    play,
    replay_executors,
    trace_doc,
)

CASES = case_names()


def without_created_at(doc: Any) -> Any:
    """``doc`` with every ``created_at`` key removed (recursively)."""
    if isinstance(doc, dict):
        return {k: without_created_at(v) for k, v in doc.items() if k != "created_at"}
    if isinstance(doc, list):
        return [without_created_at(v) for v in doc]
    return doc


def load(directory: Path, name: str) -> Any:
    return json.loads((directory / name).read_bytes())


def replay(name: str) -> tuple[Path, dict[str, Any], Any, ReplayBackend]:
    directory = GOLDEN_DIR / name
    manifest = load(directory, "case.json")
    exchanges = [((directory / e["request"]).read_bytes(), load(directory, e["response"]))
                 for e in manifest["exchanges"]]  # fmt: skip
    backend = ReplayBackend(exchanges, model=manifest["model"])
    played = play(directory, manifest, backend, replay_executors(manifest.get("executions", [])))
    return directory, manifest, played, backend


def test_the_fixture_tree_is_complete() -> None:
    dirs = {p.name for p in GOLDEN_DIR.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))}
    assert dirs == {*CASES, SHARED}
    assert set(CASES) >= {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R2-no-history", "R2-click", "R3-TOCTOU-changed",
                          "R4-widen", "R6-step1", "R6-step2", "R6-injection", "422-isolation",
                          "budget-split"}  # fmt: skip
    for name in CASES:
        manifest = load(GOLDEN_DIR / name, "case.json")
        assert manifest["case"] == name and manifest["evidence"] == EVIDENCE
        for key in ("catalog", "ballot"):
            assert (GOLDEN_DIR / name / manifest[key]).is_file()
        assert manifest["decisions"][-1] == "decision.json" and manifest["traces"][-1] == "trace.json"


@pytest.mark.parametrize("name", CASES)
def test_golden_case(name: str) -> None:
    directory, manifest, played, backend = replay(name)
    # 1. compile(catalog, context) → ballot.json
    assert played.ballot.to_json() == (directory / manifest["ballot"]).read_bytes()
    # 2. ballot.to_requests(model) → the first request(s); the replay checks every later request byte for byte
    sent = [canonical_json(r.to_wire()) for r in played.requests]
    assert sent == [(directory / f).read_bytes() for f in manifest["ballot_requests"]]
    assert backend.calls == len(manifest["exchanges"])  # every stored exchange was replayed, none was added
    # 3. (ballot, responses) → decision.json (ignoring created_at)
    assert len(played.decisions) == len(manifest["decisions"])
    for decision, file in zip(played.decisions, manifest["decisions"], strict=True):
        got = canonical_json(without_created_at(decision.to_doc()))
        assert got == canonical_json(without_created_at(load(directory, file))), file
    final = played.decisions[-1]
    assert (final.outcome.value, final.rule) == (manifest["expect"]["outcome"], manifest["expect"]["rule"])
    # 4. verify(trace.json) passes; the regenerated trace is the stored one (created_at, latency_ms normalized)
    for decision, file, ctx in zip(played.decisions, manifest["traces"], played.verify_contexts, strict=True):
        stored = load(directory, file)
        report = verify(Trace.from_doc(stored), catalog=played.catalog, context=ctx)
        assert report.ok, (file, report.failures)
        assert canonical_json(trace_doc(decision.trace)) == (directory / file).read_bytes(), file
    if "loop" in manifest:
        assert played.loop is not None and canonical_json(loop_doc(played.loop)) == (
            directory / manifest["loop"]).read_bytes()  # fmt: skip


def test_a_changed_request_is_caught() -> None:
    directory = GOLDEN_DIR / "R1"
    manifest = load(directory, "case.json")
    response = load(directory, manifest["exchanges"][0]["response"])
    backend = ReplayBackend([(b"{}", response)], model=manifest["model"])
    with pytest.raises(AssertionError, match="request 1 differs"):
        play(directory, manifest, backend)


def test_isolation_and_split_cases_exercise_their_paths() -> None:
    _, manifest, played, _ = replay("422-isolation")
    first = load(GOLDEN_DIR / "422-isolation", manifest["exchanges"][0]["response"])
    assert first["error"]["status"] == 422 and first["error"]["type"] == "JevValidationError"
    second = load(GOLDEN_DIR / "422-isolation", manifest["exchanges"][1]["request"])
    assert "get_weather.unit" not in second["questions"]
    assert any("422 isolation" in note for note in played.decisions[-1].trace.notes)
    _, split, played, _ = replay("budget-split")
    assert len(split["ballot_requests"]) == 2 and played.decisions[-1].rounds == 1
    assert all(len(load(GOLDEN_DIR / "budget-split", f)["questions"]) <= 8 for f in split["ballot_requests"])


def test_prompts_in_the_fixtures_are_templates_over_labels() -> None:
    r2 = load(GOLDEN_DIR / "R2", "decision.json")
    assert r2["prompt"]["text"] == (
        'Send an email to Anna Keller <anna.keller@acme.com> — subject line "Running 10 minutes late", '
        'body "Hi Anna, I\'ll be 10 minutes late. Best, Sam"?'
    )
    r3 = load(GOLDEN_DIR / "R3", "decision.json")  # confirm_template (x-jev) is used verbatim
    assert r3["prompt"]["text"] == "Transfer 250.00 CHF from Savings · CHF · CH93…2957 to Checking · CHF · CH56…1180?"
    menu = load(GOLDEN_DIR / "R2-no-history", "decision.json")["prompt"]["options"]
    assert menu[1]["text"].startswith("Send an email to Anna Rossi <anna.rossi@gmail.com> — ")  # complete calls


def test_fixtures_are_up_to_date() -> None:
    assert check_fixtures(GOLDEN_DIR) == [], "run: jevtools fixtures --update"


def test_cli_fixtures_update_and_check(tmp_path: Path) -> None:
    out = io.StringIO()
    assert main(["fixtures", "--update", "--dir", str(tmp_path), "--case", "R1"], out=out) == 0
    assert (tmp_path / "R1" / "decision.json").read_bytes() == (GOLDEN_DIR / "R1" / "decision.json").read_bytes()
    assert (tmp_path / SHARED / "contacts.json").is_file() and "wrote" in out.getvalue()
    assert main(["fixtures", "--dir", str(tmp_path), "--case", "R1"], out=io.StringIO()) == 0
    (tmp_path / "R1" / "decision.json").write_bytes(b"{}")
    shutil.copy(tmp_path / "R1" / "trace.json", tmp_path / "R1" / "trace_9.json")
    report = io.StringIO()
    assert main(["fixtures", "--dir", str(tmp_path), "--case", "R1"], out=report) == 1
    assert "differs R1/decision.json" in report.getvalue() and "stale R1/trace_9.json" in report.getvalue()
    err = io.StringIO()
    assert main(["fixtures", "--dir", str(tmp_path), "--case", "R99"], out=io.StringIO(), err=err) == 1
    assert "unknown golden case" in err.getvalue()

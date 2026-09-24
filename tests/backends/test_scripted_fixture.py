"""``ScriptedBackend.from_fixture`` (spec §8.5): JSON answer fixtures reproduce the §13 walk-through exactly; the
wrapped form, per-request rounds, wire answer objects and Score level maps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jevtools.backends.scripted import ScriptedBackend
from jevtools.router import Router
from jevtools.wire import ChoiceAnswer, ChoiceQuestion, DecisionRequest, NoulAnswer, NoulQuestion, ScoreQuestion
from tests.scenario import scripts
from tests.scenario.fixtures import scenario_catalog, scenario_context, scenario_messages

CASES: dict[str, tuple[dict[str, Any], str, bool]] = {
    "R1": (scripts.R1, scripts.R1_REQUEST, False),
    "R2": (scripts.R2, scripts.R2_REQUEST, True),
    "R3": (scripts.R3, scripts.R3_REQUEST, False),
    "R4": (scripts.R4, scripts.R4_REQUEST, False),
    "R5": (scripts.R5, scripts.R5_REQUEST, False),
    "R7": (scripts.R7, scripts.R7_REQUEST, False),
}


def write(path: Path, document: Any) -> Path:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def decide(backend: ScriptedBackend, request: str, history: bool) -> bytes:
    ctx = scenario_context()
    router = Router(scenario_catalog(list(ctx.sources.values())), backend=backend, context=ctx)
    return router.decide(scenario_messages(request, history=history)).to_json()


@pytest.mark.parametrize("case", sorted(CASES))
def test_fixture_reproduces_the_scripted_walkthrough(case: str, tmp_path: Path) -> None:
    script, request, history = CASES[case]
    path = write(tmp_path / f"{case}.answers.json", {"answers": script, "model": "~typesafe/jev-latest"})
    loaded = ScriptedBackend.from_fixture(path)
    assert loaded.model == "~typesafe/jev-latest" and loaded.name == "scripted"
    expected = decide(ScriptedBackend(script, model="~typesafe/jev-latest"), request, history)
    assert decide(loaded, request, history) == expected


def request(**questions: Any) -> DecisionRequest:
    return DecisionRequest(model="m", state="s", questions=questions)


def test_plain_script_wire_answers_and_score_levels(tmp_path: Path) -> None:
    path = write(tmp_path / "plain.json", {
        "tool": {"a": 0.7, "b": 0.3},
        "t.*": 0.8,
        "t.level": {"0": 0.1, "2": 0.9},
        "t.pick": {"type": "choice", "choice": "b", "confidence": 0.5, "probabilities": {"a": 0.4, "b": 0.6}},
    })  # fmt: skip
    backend = ScriptedBackend.from_fixture(path, p_top=0.9, model="x")
    response = backend.decide(request(
        tool=ChoiceQuestion(criteria={"a": None, "b": None}), **{
            "t.flag": NoulQuestion(), "t.level": ScoreQuestion(criteria=["l0", "l1", "l2"]),
            "t.pick": ChoiceQuestion(criteria={"a": None, "b": None})},
    ))  # fmt: skip
    assert response.answers["tool"].probabilities == {"a": 0.7, "b": 0.3}  # type: ignore[union-attr]
    assert response.answers["t.flag"] == NoulAnswer(noul=0.8)
    level = response.answers["t.level"]
    assert level.probabilities == {0: 0.1, 1: 0.0, 2: 0.9} and level.score == pytest.approx(1.8)  # type: ignore[union-attr]
    pick = response.answers["t.pick"]
    assert isinstance(pick, ChoiceAnswer) and pick.choice == "b" and backend.p_top == 0.9 and backend.model == "x"


def test_rounds_answer_successive_requests(tmp_path: Path) -> None:
    path = write(tmp_path / "rounds.json", {"rounds": [{"t.flag": 0.9}, {"t.flag": 0.1}], "name": "fixture"})
    backend = ScriptedBackend.from_fixture(path)
    flags = [backend.decide(request(**{"t.flag": NoulQuestion()})).answers["t.flag"] for _ in range(3)]
    assert flags == [NoulAnswer(noul=0.9), NoulAnswer(noul=0.1), NoulAnswer(noul=0.1)] and backend.name == "fixture"
    unscripted = backend.decide(request(**{"t.other": NoulQuestion()})).answers["t.other"]
    assert unscripted == NoulAnswer(noul=0.5)  # uniform: maximally uncertain


def test_malformed_fixtures_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ScriptedBackend.from_fixture(write(tmp_path / "list.json", [1, 2]))
    with pytest.raises(ValueError):
        ScriptedBackend.from_fixture(write(tmp_path / "rounds.json", {"rounds": []}))

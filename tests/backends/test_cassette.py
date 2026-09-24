"""``Cassette`` (spec §8.7, §10.4): record → replay equality, a miss raises ``CassetteMiss``, passthrough, the JSONL
record format, and a Router decision replayed from a cassette."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevtools.backends import Backend, BackendError
from jevtools.backends.cassette import Cassette, CassetteMiss, request_key
from jevtools.backends.errors import BackendConfigError
from jevtools.backends.scripted import ScriptedBackend
from jevtools.backends.simulator import LexicalSimulator
from jevtools.canonical import sha256_of
from jevtools.router import Router
from jevtools.wire import ChoiceQuestion, DecisionRequest, NoulQuestion, ScoreQuestion
from tests.scenario import scripts
from tests.scenario.fixtures import scenario_catalog, scenario_context, scenario_messages

REQUEST = DecisionRequest(
    model="lexical-simulator",
    state={"request": "Email Anna that I'm late", "history": []},
    questions={
        "tool": ChoiceQuestion(criteria={"send_email": "Send an email.", "NO_TOOL": "No.", "UNSUPPORTED": "None."}),
        "send_email.authorized": NoulQuestion(instructions="Is the user asking to send an email now?"),
        "t.level": ScoreQuestion(criteria=["low", "late"]),
    },
)
OTHER = REQUEST.model_copy(update={"state": {"request": "Tell me a joke", "history": []}})


def test_record_then_replay_returns_equal_responses(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "calls.jsonl"
    live = LexicalSimulator()
    recorder = Cassette(path, "record", live)
    recorded = recorder.decide(REQUEST)
    assert recorded == live.decide(REQUEST) and path.exists() and len(recorder) == 1
    assert REQUEST in recorder and OTHER not in recorder
    replay = Cassette(path)
    assert replay.decide(REQUEST) == recorded and replay.hits == 1
    assert (replay.model, replay.name) == ("lexical-simulator", "simulator")  # taken from the records
    assert isinstance(replay, Backend)


def test_record_format_is_jsonl_keyed_by_the_canonical_request(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    cassette = Cassette(path, "record", LexicalSimulator())
    cassette.decide(REQUEST)
    cassette.decide(OTHER)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert list(record) == ["request_sha256", "backend", "model", "request", "response", "recorded_at"]
    assert record["request_sha256"] == request_key(REQUEST) == sha256_of(REQUEST.to_wire())
    assert record["request"] == json.loads(json.dumps(REQUEST.to_wire()))
    assert (record["backend"], record["model"]) == ("simulator", "lexical-simulator")
    assert record["response"]["answers"]["t.level"]["type"] == "score"
    assert record["recorded_at"].endswith("Z")


def test_replay_miss_raises_cassette_miss(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    Cassette(path, "record", LexicalSimulator()).decide(REQUEST)
    replay = Cassette(path)
    with pytest.raises(CassetteMiss) as info:
        replay.decide(OTHER)
    assert info.value.request_sha256 == request_key(OTHER) and replay.misses == 1
    assert not isinstance(info.value, BackendError)  # the router must not turn a stale cassette into P0
    with pytest.raises(CassetteMiss):
        Cassette(tmp_path / "missing.jsonl").decide(REQUEST)  # an absent file replays nothing


async def test_async_record_and_replay(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    recorded = await Cassette(path, "record", LexicalSimulator()).adecide(REQUEST)
    assert await Cassette(path).adecide(REQUEST) == recorded
    with pytest.raises(CassetteMiss):
        await Cassette(path).adecide(OTHER)


def test_passthrough_neither_reads_nor_writes(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    inner = ScriptedBackend({"tool": "send_email"}, model="m", name="scripted")
    cassette = Cassette(path, "passthrough", inner)
    assert cassette.decide(REQUEST).answers["tool"].choice == "send_email"  # type: ignore[union-attr]
    assert not path.exists() and len(inner.requests) == 1 and (cassette.model, cassette.name) == ("m", "scripted")


def test_a_later_recording_wins(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    Cassette(path, "record", ScriptedBackend({"tool": "send_email"}, model="lexical-simulator")).decide(REQUEST)
    Cassette(path, "record", ScriptedBackend({"tool": "NO_TOOL"}, model="lexical-simulator")).decide(REQUEST)
    assert Cassette(path).decide(REQUEST).answers["tool"].choice == "NO_TOOL"  # type: ignore[union-attr]
    assert len(path.read_text().splitlines()) == 2 and len(Cassette(path)) == 1


def test_configuration_errors(tmp_path: Path) -> None:
    with pytest.raises(BackendConfigError):
        Cassette(tmp_path / "x.jsonl", "record")  # no inner backend
    with pytest.raises(BackendConfigError):
        Cassette(tmp_path / "x.jsonl", "rewind")  # type: ignore[arg-type]
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"request_sha256": "sha256:0", "response": {}}\nnot json\n', encoding="utf-8")
    with pytest.raises(BackendConfigError, match="bad.jsonl:2"):
        Cassette(bad)
    assert Cassette(tmp_path / "empty.jsonl", model="jev-latest", name="typesafe").model == "jev-latest"


def test_router_decision_replays_from_a_cassette(tmp_path: Path) -> None:
    path = tmp_path / "r5.jsonl"
    ctx = scenario_context()
    catalog = scenario_catalog(list(ctx.sources.values()))
    messages = scenario_messages(scripts.R5_REQUEST)
    live = Router(catalog, backend=Cassette(path, "record", LexicalSimulator()), context=ctx).decide(messages)
    replayed = Router(catalog, backend=Cassette(path), context=ctx).decide(messages)
    assert replayed.to_json() == live.to_json()
    with pytest.raises(CassetteMiss):  # fails loudly instead of abstaining under P0
        Router(catalog, backend=Cassette(path), context=ctx).decide(scenario_messages(scripts.R1_REQUEST))

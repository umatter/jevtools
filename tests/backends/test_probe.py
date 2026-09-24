"""The conformance probe (spec §8.7) against an ``httpx.MockTransport`` server with specific limits: the measured
``Limits`` (written where ``Limits.from_file`` reads them), label echo, ASCII folding, models listing, smoke items."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import httpx
import pytest

from jevtools.backends.errors import JevAuthError
from jevtools.backends.http import HTTPBackend
from jevtools.backends.simulator import LexicalSimulator
from jevtools.probe import ProbeError, ProbeReport, cache_dir, cached_limits, limits_path, run_probe
from jevtools.validate import Limits


class FakeJev:
    """A Jev-like server enforcing configurable limits with 422 ``HTTPValidationError`` bodies."""

    def __init__(
        self,
        *,
        label_max: int = 10_000,
        desc_max: int = 100_000,
        instr_max: int = 100_000,
        max_questions: int = 10_000,
        qid_max: int = 128,
        dotted: bool = True,
        object_instructions: bool = True,
        min_options: int = 1,
        unicode_labels: bool = True,
        echo: str = "exact",
        status: int = 200,
    ) -> None:
        self.label_max, self.desc_max, self.instr_max = label_max, desc_max, instr_max
        self.max_questions, self.qid_max, self.dotted = max_questions, qid_max, dotted
        self.object_instructions, self.min_options = object_instructions, min_options
        self.unicode_labels, self.echo, self.status = unicode_labels, echo, status
        self.bodies: list[dict[str, Any]] = []

    def backend(self) -> HTTPBackend:
        return HTTPBackend(url="https://api.typesafe.ai/v1/systemone", api_key="k", model="jev-latest",
                           name="typesafe", transport=httpx.MockTransport(self), sleep=lambda _: None)  # fmt: skip

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "nope"}})
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "jev-1.13.0"}, {"id": "jev-latest"}]})
        body = json.loads(request.content)
        self.bodies.append(body)
        errors = self.errors(body["questions"])
        if errors:
            return httpx.Response(422, json={"detail": errors})
        answers = {qid: self.answer(q) for qid, q in body["questions"].items()}
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers,
                                         "usage": {"input_tokens": 10, "output_tokens": len(answers)}})  # fmt: skip

    def errors(self, questions: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        def fail(qid: str | None, msg: str) -> None:
            loc = ["body", "questions"] + ([qid] if qid is not None else [])
            out.append({"loc": loc, "msg": msg, "type": "value_error"})

        if len(questions) > self.max_questions:
            fail(None, "too many questions")
        pattern = r"[a-z][a-z0-9_.]*" if self.dotted else r"[a-z][a-z0-9_]*"
        for qid, q in questions.items():
            if not re.fullmatch(pattern, qid) or len(qid) > self.qid_max:
                fail(qid, "invalid question id")
            instructions = q.get("instructions")
            if isinstance(instructions, dict) and not self.object_instructions:
                fail(qid, "instructions must be a string")
            if isinstance(instructions, str) and len(instructions) > self.instr_max:
                fail(qid, "instructions too long")
            if q["type"] == "choice":
                if len(q["criteria"]) < self.min_options:
                    fail(qid, "too few options")
                for label, text in q["criteria"].items():
                    if len(label) > self.label_max or (not self.unicode_labels and not label.isascii()):
                        fail(qid, "invalid label")
                    if isinstance(text, str) and len(text) > self.desc_max:
                        fail(qid, "description too long")
        return out

    def answer(self, q: dict[str, Any]) -> dict[str, Any]:
        if q["type"] == "noul":
            return {"type": "noul", "noul": 0.5}
        if q["type"] == "score":
            n = len(q["criteria"])
            return {
                "type": "score",
                "score": 0.0,
                "confidence": 0.0,
                "probabilities": {str(i): 1 / n for i in range(n)},
            }
        labels = list(q["criteria"])
        if self.echo == "nfd":
            labels = [unicodedata.normalize("NFD", label) for label in labels]
        elif self.echo == "lower":
            labels = [label.lower() for label in labels]
        return {"type": "choice", "choice": labels[0], "confidence": 0.0,
                "probabilities": {label: 1 / len(labels) for label in labels}}  # fmt: skip


def test_strict_server_limits_are_measured_and_written(tmp_path: Path) -> None:
    server = FakeJev(label_max=128, desc_max=2_000, instr_max=2_000, max_questions=300, dotted=False,
                     object_instructions=False, min_options=2, echo="nfd")  # fmt: skip
    report = run_probe(server.backend(), directory=tmp_path)
    limits = report.limits
    assert limits.id_mode == "opaque"
    assert (limits.label_max, limits.desc_max, limits.instr_max) == (128, 2_000, 2_000)
    assert limits.max_questions == 255 and limits.instructions_as_object is False
    assert (report.label_echo, report.ascii_labels, report.single_option_choice) == ("nfc", False, False)
    assert report.models == ["jev-1.13.0", "jev-latest"]
    assert report.check("qid.charset") is not None and not report.check("qid.charset").ok  # type: ignore[union-attr]
    assert report.calls == len(server.bodies) and 12 <= report.calls <= 25
    path = tmp_path / "limits-typesafe-jev-latest.json"
    assert report.path == str(path) and Limits.from_file(path) == limits
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["backend"] == "typesafe" and document["label_echo"] == "nfc" and "checks" in document


def test_permissive_server_relaxes_the_defaults(tmp_path: Path) -> None:
    report = run_probe(FakeJev().backend(), path=tmp_path / "limits.json")
    limits = report.limits
    assert (limits.id_mode, limits.qid_max, limits.label_max) == ("dotted", 128, 256)
    assert (limits.desc_max, limits.instr_max, limits.max_questions) == (8_000, 8_000, 400)
    assert limits.instructions_as_object and report.single_option_choice and report.label_echo == "exact"
    assert Limits.from_file(tmp_path / "limits.json") == limits
    assert set(report.smoke) == {"e2", "e3", "e4"}  # informational numbers only
    assert report.smoke["e4"]["top_flip"] is False and report.smoke["e2"]["passed"] is False  # a uniform server


def test_tight_server_falls_below_the_defaults(tmp_path: Path) -> None:
    server = FakeJev(label_max=20, desc_max=300, instr_max=500, max_questions=100, qid_max=64,
                     unicode_labels=False, echo="lower")  # fmt: skip
    report = run_probe(server.backend(), directory=tmp_path, smoke=False)
    limits = report.limits
    assert (limits.id_mode, limits.qid_max) == ("dotted", 64)
    assert (limits.label_max, limits.desc_max, limits.instr_max, limits.max_questions) == (16, 200, 500, 63)
    assert report.ascii_labels is True and report.label_echo == "casefold" and report.smoke == {}


def test_backend_errors_abort_the_probe(tmp_path: Path) -> None:
    with pytest.raises(JevAuthError):
        run_probe(FakeJev(status=401).backend(), directory=tmp_path)
    with pytest.raises(ProbeError, match="rejects a minimal request"):
        run_probe(FakeJev(label_max=1).backend(), directory=tmp_path)
    assert not list(tmp_path.iterdir())


def test_simulator_accepts_everything_and_is_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEVTOOLS_CACHE_DIR", str(tmp_path))
    sim = LexicalSimulator()
    assert cached_limits(sim) is None
    report = run_probe(sim)
    assert report.path == str(tmp_path / "limits-simulator-lexical-simulator.json") and report.models is None
    assert cached_limits(sim) == report.limits and report.limits.max_questions == 400
    assert cache_dir() == tmp_path
    # the Router uses the probed limits of its backend when none are given (§8.7 "which the validator uses")
    from jevtools.router import Router

    assert Router([], backend=sim).limits == report.limits
    assert Router([], backend=sim, limits=Limits(label_max=32)).limits.label_max == 32
    (tmp_path / "limits-simulator-lexical-simulator.json").write_text("{not json", encoding="utf-8")
    with pytest.warns(UserWarning, match="ignoring the probed limits"):
        assert Router([], backend=sim).limits == Limits()


def test_limits_path_and_report_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEVTOOLS_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert limits_path("openrouter_decisions", "~typesafe/jev-latest") == (
        tmp_path / "jevtools" / "limits-openrouter_decisions-typesafe_jev-latest.json"
    )
    report = ProbeReport(backend="b", model="m", limits=Limits(label_max=32))
    written = report.write(tmp_path / "nested" / "l.json")
    assert Limits.from_file(written).label_max == 32 and report.path == str(written)

"""An offline backend that answers from a script. For tests, demos, and replay."""

from __future__ import annotations

import fnmatch
import json
import math
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Union

from pydantic import TypeAdapter

from jevtools.wire import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
)

AnswerSpec = Union[str, float, int, bool, Mapping[str, float], Mapping[int, float], Answer]
"""How a script states an answer.

- Choice: a label (it gets ``p_top``, the rest share the remainder) or ``{label: probability}``.
- Noul: a probability of yes (``True``/``False`` mean ``p_top`` / ``1 - p_top``).
- Score: a level index (gets ``p_top``) or ``{level: probability}``.
- Or a ready-made wire answer.
"""

Script = Union[Mapping[str, AnswerSpec], Callable[[DecisionRequest], Mapping[str, AnswerSpec]]]


class ScriptedBackend:
    """Answers each question from ``script``; keys may be exact ids or glob patterns.

    Questions the script does not cover get a uniform (maximally uncertain)
    answer, so an incomplete script shows up as low confidence rather than as a
    silently confident guess. Every request is kept in :attr:`requests`.
    """

    def __init__(
        self, script: Script | None = None, *, p_top: float = 0.92, model: str = "scripted", name: str = "scripted"
    ) -> None:
        self.script = script or {}
        self.p_top = p_top
        self.model = model
        self.name = name
        self.requests: list[DecisionRequest] = []

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        script = self.script(request) if callable(self.script) else self.script
        answers: dict[str, Any] = {}
        for qid, question in request.questions.items():
            spec = _lookup(script, qid)
            answers[qid] = _answer(question, spec, self.p_top)
        tokens = len(str(request.to_wire())) // 4
        return DecisionResponse(
            model=self.model, answers=answers, usage=Usage(input_tokens=tokens, output_tokens=len(answers))
        )

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        return self.decide(request)

    @classmethod
    def from_fixture(cls, path: str | os.PathLike[str], **kwargs: Any) -> ScriptedBackend:
        """A backend scripted by a JSON fixture such as ``examples/fixtures/R2.answers.json`` (spec §8.5).

        The file holds either a plain script ``{qid-or-glob: answer}`` or a document
        ``{"answers": {…}, "p_top": 0.92, "model": "…", "name": "…"}``; ``{"rounds": [{…}, {…}], …}`` answers the
        i-th request from ``rounds[i]`` (the last one repeats). Answers are what a script accepts (a label, a
        ``{label: p}`` map, a Noul probability, a Score level or ``{level: p}`` map) or a wire answer object
        (``{"type": "choice", …}``). JSON object keys are strings, so Score level maps are read back as integers.
        ``kwargs`` override the document's ``p_top`` / ``model`` / ``name``.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError(f"{path}: a scripted fixture is a JSON object")
        options: dict[str, Any] = {}
        if "answers" in data or "rounds" in data:
            rounds = data["rounds"] if "rounds" in data else [data["answers"]]
            options = {key: data[key] for key in ("p_top", "model", "name") if key in data}
        else:
            rounds = [data]
        if not isinstance(rounds, Sequence) or not rounds or not all(isinstance(r, Mapping) for r in rounds):
            raise ValueError(f"{path}: rounds must be a non-empty list of scripts")
        return cls(_FixtureScript([_wire_answers(r) for r in rounds]), **{**options, **kwargs})


_ANSWER_ADAPTER: TypeAdapter[Answer] = TypeAdapter(Answer)


def _wire_answers(script: Mapping[str, Any]) -> dict[str, AnswerSpec]:
    """A fixture script with wire answer objects (``{"type": …}``) parsed into answer models."""
    out: dict[str, AnswerSpec] = {}
    for qid, spec in script.items():
        if isinstance(spec, Mapping) and spec.get("type") in ("choice", "noul", "score"):
            out[qid] = _ANSWER_ADAPTER.validate_python(dict(spec))
        else:
            out[qid] = spec
    return out


class _FixtureScript:
    """The callable script of :meth:`ScriptedBackend.from_fixture`: one script per request (the last repeats), with
    string level keys of Score answers turned back into integers."""

    def __init__(self, rounds: list[dict[str, AnswerSpec]]) -> None:
        self.rounds = rounds
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, request: DecisionRequest) -> Mapping[str, AnswerSpec]:
        with self._lock:
            script = self.rounds[min(self.calls, len(self.rounds) - 1)]
            self.calls += 1
        out: dict[str, AnswerSpec] = dict(script)
        for qid, question in request.questions.items():
            spec = _lookup(script, qid)
            if isinstance(question, ScoreQuestion) and isinstance(spec, Mapping):
                out[qid] = {int(k): float(v) for k, v in spec.items()}
        return out


def _lookup(script: Mapping[str, AnswerSpec], qid: str) -> AnswerSpec | None:
    if qid in script:
        return script[qid]
    for pattern, spec in script.items():
        if any(ch in pattern for ch in "*?[") and fnmatch.fnmatchcase(qid, pattern):
            return spec
    return None


def _answer(question: Any, spec: AnswerSpec | None, p_top: float) -> Any:
    if isinstance(spec, (ChoiceAnswer, NoulAnswer, ScoreAnswer)):
        return spec
    if isinstance(question, ChoiceQuestion):
        return choice_answer(list(question.criteria), spec, p_top)
    if isinstance(question, NoulQuestion):
        if spec is None:
            return NoulAnswer(noul=0.5)
        if isinstance(spec, bool):
            return NoulAnswer(noul=p_top if spec else 1 - p_top)
        return NoulAnswer(noul=float(spec))  # type: ignore[arg-type]
    if isinstance(question, ScoreQuestion):
        return score_answer(question.criteria, spec, p_top)
    raise TypeError(f"unsupported question {question!r}")


def choice_answer(labels: list[str], spec: Any, p_top: float = 0.92) -> ChoiceAnswer:
    """Build a Choice answer over ``labels`` from a label or a probability map."""
    if isinstance(spec, Mapping):
        raw = {label: float(spec.get(label, 0.0)) for label in labels}
    elif isinstance(spec, str):
        if spec not in labels:
            raise KeyError(f"scripted choice {spec!r} is not one of {labels}")
        rest = (1 - p_top) / (len(labels) - 1) if len(labels) > 1 else 0.0
        raw = {label: (p_top if label == spec else rest) for label in labels}
        if len(labels) == 1:
            raw[spec] = 1.0
    else:
        raw = {label: 1.0 for label in labels}
    total = sum(raw.values()) or 1.0
    probs = {k: v / total for k, v in raw.items()}
    top = max(probs, key=lambda k: probs[k])
    return ChoiceAnswer(choice=top, confidence=concentration(list(probs.values())), probabilities=probs)


def score_answer(levels: list[Any], spec: Any, p_top: float = 0.92) -> ScoreAnswer:
    n = len(levels)
    if isinstance(spec, Mapping):
        raw = {i: float(spec.get(i, 0.0)) for i in range(n)}
    elif isinstance(spec, (int, float)) and not isinstance(spec, bool):
        target = int(round(spec))
        rest = (1 - p_top) / (n - 1) if n > 1 else 0.0
        raw = {i: (p_top if i == target else rest) for i in range(n)} if n > 1 else {0: 1.0}
    else:
        raw = {i: 1.0 for i in range(n)}
    total = sum(raw.values()) or 1.0
    probs = {i: v / total for i, v in raw.items()}
    expected = sum(i * p for i, p in probs.items())
    legend = {i: level for i, level in enumerate(levels)}
    return ScoreAnswer(
        score=expected, confidence=concentration(list(probs.values())), legend=legend, probabilities=probs
    )


def concentration(probs: list[float]) -> float:
    """1 - normalized entropy: 1 for a certain answer, 0 for a uniform one."""
    n = len(probs)
    if n <= 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, 1.0 - entropy / math.log(n))

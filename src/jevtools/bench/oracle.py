"""An oracle backend for BFCL: answers every Jev question the way a perfect Jev would, given the BFCL answer.

The oracle sees the compiled round (the Ballot and its pools) and the case's acceptable values. For each question it
elects the option whose value BFCL would accept, or the sentinel a perfect judge would pick when no option is
acceptable (``NOT_STATED`` when omitting the parameter is acceptable, ``NONE_OF_THESE`` otherwise). Running the real
router on these answers measures the *ceiling* of "bind, don't write" on BFCL: how often the right call can come out
of jevtools at all, given the candidates code nominated, the question layout, decoding, normalization and policy.
It says nothing about how often real Jev picks those answers; that is what a live run measures.

While answering, the oracle records per-parameter coverage (:class:`ParamCoverage`) so that every failure can be
attributed: a tool that was not speculated, a value that no extractor nominated, or a value that was elected but lost
in decoding or normalization.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from jevtools.backends.scripted import choice_answer, score_answer
from jevtools.ballot import BallotQuestion
from jevtools.bench.bfcl import BfclCase, standardize_string, value_matches
from jevtools.candidates import NO_TOOL, NONE_OF_THESE, NOT_STATED, UNSUPPORTED
from jevtools.plan import RoundPlan
from jevtools.wire import Answer, ChoiceQuestion, DecisionRequest, DecisionResponse, NoulAnswer, Usage

P_TOP = 0.94
"""Probability the oracle puts on the answer it elects (the rest is spread over the other options)."""
P_YES = 0.95
P_NO = 0.04

Coverage = Literal["covered", "omitted", "default", "uncovered", "not_asked"]


@dataclass(frozen=True)
class ParamCoverage:
    """Whether a parameter of the expected call could be bound from the ballot.

    - ``covered``: an option (or accepted text, or list item) carries an acceptable value;
    - ``omitted``: no value is on the ballot, but omitting the parameter is acceptable and ``NOT_STATED`` omits it;
    - ``default``: ``NOT_STATED`` decodes to a default value that is acceptable;
    - ``uncovered``: a value is required but none on the ballot is acceptable (the extractors missed it);
    - ``not_asked``: the ballot has no question for it (the tool was not speculated, or the parameter was not modelled).
    """

    param: str
    kind: str | None
    status: Coverage
    detail: str = ""
    in_text: bool | None = None
    """For a miss (``uncovered``/``not_asked``): whether an acceptable value appears verbatim in the user's messages
    (BFCL-standardized). ``True`` marks an extractor gap; ``False`` a value that needs reformatting, arithmetic or
    world knowledge, which selection alone cannot produce."""


@dataclass
class OracleBackend:
    """A backend that answers from the BFCL answer of ``case`` (see the module docstring); ``plan`` is the round the
    router will send, compiled beforehand with the same inputs (:func:`jevtools.plan.compile_round`)."""

    case: BfclCase
    plan: RoundPlan | None = None
    name: str = "oracle"
    model: str = "bfcl-oracle"
    requests: list[DecisionRequest] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    """Question ids answered without ballot information (follow-up rounds): answered as uncertain."""

    @property
    def expected(self) -> tuple[str, Mapping[str, list[Any]]] | None:
        """``(function, {param: acceptable})`` of the first expected call, or ``None`` (no call expected)."""
        if not self.case.ground_truth:
            return None
        name, params = next(iter(self.case.ground_truth[0].items()))
        return name, params

    # -- backend protocol ----------------------------------------------------------------------------------------

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        questions = {q.qid: q for q in self.plan.ballot.questions} if self.plan is not None else {}
        answers: dict[str, Any] = {}
        for qid, wire in request.questions.items():
            question = questions.get(qid)
            if question is None:
                self.unknown.append(qid)
                answers[qid] = _uncertain(wire)
            else:
                answers[qid] = self.answer(question, wire)
        return DecisionResponse(model=self.model, answers=answers, usage=Usage(input_tokens=0, output_tokens=0))

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        return self.decide(request)

    # -- answering -----------------------------------------------------------------------------------------------

    def answer(self, q: BallotQuestion, wire: Any) -> Answer:
        """The oracle's answer to one ballot question (``wire`` is the question as sent)."""
        if q.family == "tool":
            return self._choice(wire, self._tool_label(q))
        expected = self.expected
        on_expected = expected is not None and q.tool == expected[0]
        if q.primitive == "noul":
            return NoulAnswer(noul=self._noul(q) if on_expected else 0.5)
        if q.primitive == "score":
            levels = list(q.levels or [])
            return score_answer(levels, None) if levels else score_answer(["?"], 0)
        if not on_expected:
            return _uncertain(wire)
        return self._choice(wire, self._slot_label(q))

    def _tool_label(self, q: BallotQuestion) -> str:
        expected = self.expected
        if self.case.category in ("irrelevance", "live_irrelevance"):
            return NO_TOOL
        if expected is None:  # live_relevance: any call is right; elect the first real tool
            return q.options[0].label if q.options else UNSUPPORTED
        return next((o.label for o in q.options if o.value == expected[0]), UNSUPPORTED)

    def acceptable(self, path: Sequence[str]) -> list[Any] | None:
        """The acceptable values at an argument path (``None``: nothing is known about it)."""
        expected = self.expected
        if expected is None or not path:
            return None
        values: Any = expected[1].get(path[0], [""])  # a parameter the answer does not name must be omitted
        for segment in path[1:]:
            if segment == "[]":
                return list(values)
            record = next((v for v in values if isinstance(v, Mapping)), None)
            if record is None:
                return None
            values = record.get(segment, [""])
        return list(values)

    def bfcl_type(self, path: Sequence[str]) -> str:
        """The BFCL type of the top-level parameter at ``path`` (``any`` when unknown)."""
        expected = self.expected
        function = self.case.function(expected[0]) if expected is not None else None
        detail = ((function or {}).get("parameters", {}).get("properties", {}) or {}).get(path[0] if path else "", {})
        return str(detail.get("type", "any"))

    def _slot_label(self, q: BallotQuestion) -> str:
        acceptable = self.acceptable(q.path)
        if acceptable is None:
            return NONE_OF_THESE if NONE_OF_THESE in q.sentinels else _first_label(q)
        typ = self.bfcl_type(q.path)
        if q.family in ("date", "time"):
            match = next((o.label for o in q.options if _part_matches(o.value, acceptable)), None)
        elif q.family == "mention":
            match = next((o.label for o in q.options if _in_some_list(o.value, acceptable)), None)
        else:
            match = next((o.label for o in q.options if value_matches(o.value, acceptable, typ)), None)
        if match is not None:
            return match
        if NOT_STATED in q.sentinels and _not_stated_ok(q, acceptable, typ):
            return NOT_STATED
        if q.family == "mention" and "EXCLUDE" in q.sentinels:
            return "EXCLUDE"
        return NONE_OF_THESE if NONE_OF_THESE in q.sentinels else _first_label(q)

    def _noul(self, q: BallotQuestion) -> float:
        acceptable = self.acceptable(q.path)
        typ = self.bfcl_type(q.path)
        if q.family in ("authorized",):
            return P_YES
        if q.family in ("done_after", "more"):
            return P_NO
        if q.family == "present":
            return P_YES if acceptable is not None and "" not in acceptable else 0.2
        if acceptable is None:
            return 0.5
        if q.family == "slot" and q.kind == "flag":  # a flag without a default is a Noul: P(true)
            if value_matches(True, acceptable):
                return P_YES
            return P_NO if value_matches(False, acceptable) else 0.5
        if q.family == "accept":
            candidate = (q.meta or {}).get("candidate") or {}
            return P_YES if value_matches(candidate.get("value"), acceptable, typ) else P_NO
        if q.family in ("item", "member"):
            value = self._item_value(q)
            return P_YES if value is not None and _in_some_list(value, acceptable) else P_NO
        return 0.5

    def _items_complete(self, questions: Sequence[BallotQuestion], acceptable: Sequence[Any]) -> bool:
        """Whether some acceptable list has every element among the list's item candidates."""
        values = [self._item_value(q) for q in questions if q.family == "item"]
        return any(all(any(value_matches(v, [e]) for v in values if v is not None) for e in option)
                   for option in _elements(acceptable))  # fmt: skip

    def _item_value(self, q: BallotQuestion) -> Any:
        if self.plan is None:
            return None
        pool = self.plan.pool(q.tool or "", q.path)
        index = (q.meta or {}).get("index")
        if pool is None or not isinstance(index, int) or not 0 <= index < len(pool.candidates):
            return None
        return pool.candidates[index].value

    @staticmethod
    def _choice(wire: Any, label: str) -> Answer:
        labels = list(wire.criteria) if isinstance(wire, ChoiceQuestion) else [label]
        return choice_answer(labels, label if label in labels else None, P_TOP)

    # -- coverage ------------------------------------------------------------------------------------------------

    def coverage(self) -> list[ParamCoverage]:
        """Per parameter of the expected call: could it be bound from the round-1 ballot?"""
        expected = self.expected
        if expected is None or self.plan is None:
            return []
        name, params = expected
        by_param: dict[str, list[BallotQuestion]] = {}
        for q in self.plan.ballot.questions:
            if q.tool == name and q.path:
                by_param.setdefault(q.path[0], []).append(q)
        out: list[ParamCoverage] = []
        for param, acceptable in params.items():
            questions = by_param.get(param, [])
            if not questions:
                status: Coverage = "omitted" if "" in acceptable else "not_asked"
                item = ParamCoverage(param, None, status, "no question for this parameter")
            else:
                item = self._param_coverage(param, acceptable, questions)
            if item.status in ("uncovered", "not_asked"):
                item = replace(item, in_text=in_text(acceptable, self.user_text))
            out.append(item)
        return out

    @property
    def user_text(self) -> str:
        """The text of the case's user messages."""
        return " ".join(str(m.get("content") or "") for m in self.case.messages if m.get("role") == "user")

    def _param_coverage(self, param: str, acceptable: list[Any], questions: list[BallotQuestion]) -> ParamCoverage:
        kind = questions[0].kind
        statuses: list[Coverage] = []
        accepted: dict[str, list[bool]] = {}  # accept/item/member families: one yes is enough, a right no is fine
        for q in questions:
            if q.family in ("authorized", "present", "rev", "more", "done_after", "joint"):
                continue
            if q.primitive == "noul":
                answer = self._noul(q)
                if q.family in ("accept", "item", "member"):
                    accepted.setdefault(q.family, []).append(answer == P_YES)
                    continue
                flag_no = q.kind == "flag" and answer == P_NO  # a flag elected false is bound, not missing
                statuses.append("covered" if answer == P_YES or flag_no else "uncovered")
                continue
            label = self._slot_label(q)
            if label == NOT_STATED:
                decodes = q.sentinels[NOT_STATED].decodes_to
                statuses.append("default" if decodes == "default" else "omitted")
            elif label in (NONE_OF_THESE, "EXCLUDE"):
                statuses.append("uncovered")
            else:
                statuses.append("covered")
        for family, yeses in accepted.items():
            if family == "item":  # every expected element must be on the ballot (and elected)
                statuses.append("covered" if self._items_complete(questions, acceptable) else "uncovered")
            else:
                statuses.append("covered" if any(yeses) else "uncovered")
        if not statuses:
            return ParamCoverage(param, kind, "not_asked", "only probe questions")
        if "covered" in statuses and all(s in ("covered", "omitted", "default") for s in statuses):
            return ParamCoverage(param, kind, "covered")
        if all(s in ("omitted", "default") for s in statuses):
            return ParamCoverage(param, kind, statuses[0])
        # uncovered: show what the ballot offered, to make misses easy to read
        offered = [o.label for q in questions for o in q.options][:6]
        return ParamCoverage(param, kind, "uncovered", f"wanted {acceptable[:3]}; offered {offered}")


def in_text(acceptable: Sequence[Any], text: str) -> bool:
    """Whether an acceptable value (or, for lists and dicts, every element of one) appears verbatim in ``text``:
    strings after BFCL's standardization, numbers as a whole number token."""
    standard = standardize_string(text)

    def present(value: Any) -> bool:
        if isinstance(value, bool) or value in ("", None):
            return False
        if isinstance(value, (int, float)):
            shown = f"{value:g}" if isinstance(value, float) else str(value)
            return re.search(rf"(?<![\d.]){re.escape(shown)}(?!\d)", text) is not None
        if isinstance(value, Mapping):  # a dict answer: every required key has a value in the text
            required = [c if isinstance(c, list) else [c] for c in value.values()]
            return bool(value) and all(any(present(v) for v in c if v != "") for c in required if "" not in c)
        if isinstance(value, list):
            return bool(value) and all(present(v) for v in value)
        word = standardize_string(str(value))
        return bool(word) and word in standard

    return any(present(value) for value in acceptable)


def _elements(acceptable: Sequence[Any]) -> list[list[Any]]:
    return [option for option in acceptable if isinstance(option, list)]


def _first_label(q: BallotQuestion) -> str:
    return q.options[0].label if q.options else next(iter(q.sentinels), NONE_OF_THESE)


def _not_stated_ok(q: BallotQuestion, acceptable: Sequence[Any], typ: str) -> bool:
    sentinel = q.sentinels[NOT_STATED]
    if sentinel.decodes_to == "omit":
        return "" in acceptable
    if sentinel.decodes_to == "default":
        if sentinel.late is not None:
            return False
        return value_matches(sentinel.value, acceptable, typ) or ("" in acceptable and sentinel.value is None)
    return False


def _in_some_list(value: Any, acceptable: Sequence[Any]) -> bool:
    for option in acceptable:
        if isinstance(option, list) and any(value_matches(value, [item]) for item in option):
            return True
    return False


def _part_matches(value: Any, acceptable: Sequence[Any]) -> bool:
    """A date or time part matches when its text appears in an acceptable string (BFCL-standardized)."""
    text = standardize_string(str(value))
    return bool(text) and any(isinstance(a, str) and a and text in standardize_string(a) for a in acceptable)


def _uncertain(wire: Any) -> Answer:
    if isinstance(wire, ChoiceQuestion):
        return choice_answer(list(wire.criteria), None)
    if getattr(wire, "type", None) == "score":
        return score_answer(list(getattr(wire, "criteria", ["?"])), None)
    return NoulAnswer(noul=0.5)


__all__ = ["P_TOP", "Coverage", "OracleBackend", "ParamCoverage", "in_text"]

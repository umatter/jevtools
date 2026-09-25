"""An oracle backend: answers every Jev question the way a perfect Jev would, given the expected answer.

The oracle sees the compiled round (the Ballot and its pools) and the case's acceptable values, through a
:class:`GoldView` (:class:`BfclGold` for BFCL, :class:`EvalGold` for §11.1 datasets such as the app-domain bench). For
each question it elects the option whose value is acceptable, or the sentinel a perfect judge would pick when none is
(``NOT_STATED`` when omitting the parameter or its default is acceptable, ``NONE_OF_THESE`` otherwise). When the case
expects a clarification (several values are equally right), it spreads its mass evenly over them, as a calibrated
judge facing a real ambiguity would. Running the real router on these answers measures the *ceiling* of "bind, don't
write": how often the right decision can come out of jevtools at all, given the candidates code nominated, the question
layout, decoding, normalization and policy. It says nothing about how often real Jev picks those answers; that is what
a live run measures.

While answering, the oracle records per-parameter coverage (:class:`ParamCoverage`) so that every failure can be
attributed: a tool that was not speculated, a value that no extractor nominated, or a value that was elected but lost
in decoding or normalization.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol

from jevtools.backends.scripted import choice_answer, score_answer
from jevtools.ballot import BallotQuestion
from jevtools.bench.bfcl import IRRELEVANCE, BfclCase, standardize_string, value_matches
from jevtools.candidates import NO_TOOL, NONE_OF_THESE, NOT_STATED, UNSUPPORTED
from jevtools.plan import RoundPlan
from jevtools.wire import Answer, ChoiceAnswer, ChoiceQuestion, DecisionRequest, DecisionResponse, NoulAnswer, Usage

if TYPE_CHECKING:
    from jevtools.eval.dataset import EvalCase

ANY = "\u2217any"
"""In an acceptable-values list: any non-empty value is acceptable (an argument the gold label does not check)."""
_JSON_TO_BFCL = {"integer": "integer", "number": "float", "string": "string", "boolean": "boolean", "array": "array",
                 "object": "dict"}  # fmt: skip


class GoldView(Protocol):
    """What the oracle needs to know about the expected answer of a case."""

    @property
    def tool(self) -> str | None:
        """The expected tool (``None``: no call expected, or any call is right when :attr:`no_tool` is false)."""

    @property
    def no_tool(self) -> bool:
        """Whether the right answer to "which tool?" is ``NO_TOOL``."""

    @property
    def params(self) -> Mapping[str, list[Any]]:
        """Acceptable values per top-level parameter of :attr:`tool` (``""``: omission is acceptable; :data:`ANY`:
        any value is)."""

    @property
    def split(self) -> bool:
        """Whether several acceptable options share the mass (the case expects a clarification)."""

    @property
    def user_text(self) -> str:
        """The text of the case's user messages."""

    def param_type(self, name: str) -> str:
        """The BFCL-style type of a parameter (``integer``, ``float``, ``string``, ``boolean``, ``array``, ``dict``,
        ``any``)."""

    def matches(self, value: Any, acceptable: Sequence[Any], typ: str = "any") -> bool:
        """Whether ``value`` is acceptable."""


def _user_text(messages: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")


@dataclass(frozen=True)
class BfclGold:
    """The expected answer of a BFCL case (BFCL's matching rules; no clarifications)."""

    case: BfclCase
    split: bool = False

    @property
    def tool(self) -> str | None:
        return next(iter(self.case.ground_truth[0])) if self.case.ground_truth else None

    @property
    def no_tool(self) -> bool:
        return self.case.category in IRRELEVANCE

    @property
    def params(self) -> Mapping[str, list[Any]]:
        return next(iter(self.case.ground_truth[0].values())) if self.case.ground_truth else {}

    @property
    def user_text(self) -> str:
        return _user_text(self.case.messages)

    def param_type(self, name: str) -> str:
        function = self.case.function(self.tool) if self.tool is not None else None
        detail = ((function or {}).get("parameters", {}).get("properties", {}) or {}).get(name, {})
        return str(detail.get("type", "any"))

    def matches(self, value: Any, acceptable: Sequence[Any], typ: str = "any") -> bool:
        return value_matches(value, acceptable, typ)


@dataclass(frozen=True)
class EvalGold:
    """The expected answer of a §11.1 case (:class:`jevtools.eval.dataset.Gold`): checked arguments must equal a gold
    value (canonical JSON equality); arguments the label does not check accept anything, or omission."""

    case: EvalCase
    tools: Sequence[Mapping[str, Any]]
    """The case's OpenAI tool list (for the expected tool's parameter names and types)."""

    @property
    def _schema(self) -> Mapping[str, Any]:
        for tool in self.tools:
            function = tool.get("function", tool)
            if function.get("name") == self.case.gold.tool:
                return dict(function.get("parameters") or {})
        return {}

    @property
    def tool(self) -> str | None:
        return self.case.gold.tool

    @property
    def no_tool(self) -> bool:
        return self.case.gold.tool is None

    @property
    def params(self) -> Mapping[str, list[Any]]:
        gold = self.case.gold
        if gold.tool is None:
            return {}
        names = list(dict.fromkeys([*(self._schema.get("properties") or {}), *gold.slots]))
        return {name: gold.values(name) if name in gold.slots else [ANY, ""] for name in names}

    @property
    def split(self) -> bool:
        return set(self.case.gold.outcomes_ok) <= {"clarify"}

    @property
    def user_text(self) -> str:
        return _user_text(self.case.messages)

    def param_type(self, name: str) -> str:
        detail = (self._schema.get("properties") or {}).get(name, {})
        return _JSON_TO_BFCL.get(str(detail.get("type", "")), "any")

    def matches(self, value: Any, acceptable: Sequence[Any], typ: str = "any") -> bool:
        from jevtools.eval.dataset import same_value

        if value in ("", None):
            return False
        return any(a == ANY or same_value(value, a) for a in acceptable)


P_TOP = 0.99
"""Probability the oracle puts on the answer it elects (the rest is spread over the other options). A perfect judge
is confident: the critical tier composes its factors with the Łukasiewicz bound (``1 − Σ(1 − p)``), so a transfer
with six factors at 0.94 would stay below the 0.80 confirm threshold however right every answer is."""
P_YES = 0.99
P_NO = 0.01

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
    """A backend that answers from ``gold`` (see the module docstring); ``plan`` is the round the router will send,
    compiled beforehand with the same inputs (:func:`jevtools.plan.compile_round`)."""

    gold: GoldView
    plan: RoundPlan | None = None
    name: str = "oracle"
    model: str = "oracle"
    requests: list[DecisionRequest] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    """Question ids answered without ballot information (follow-up rounds): answered as uncertain."""

    @property
    def expected(self) -> tuple[str, Mapping[str, list[Any]]] | None:
        """``(tool, {param: acceptable})`` of the expected call, or ``None`` (no call expected)."""
        tool = self.gold.tool
        return (tool, self.gold.params) if tool is not None else None

    # -- backend protocol ----------------------------------------------------------------------------------------

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        questions = {q.qid: q for q in self.plan.ballot.questions} if self.plan is not None else {}
        answers: dict[str, Any] = {}
        for qid, wire in request.questions.items():
            question = questions.get(qid)
            verify = self._verify(qid, wire, questions)  # by the candidate sent: a follow-up reuses the qid
            if verify is not None:
                answers[qid] = verify
            elif question is None:
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
        if self.gold.split:
            matches = self._matching_labels(q)
            if len(matches) > 1:
                return _spread(wire, matches)
        if q.family == "joint":
            matches = self._matching_labels(q)
            fallback = NONE_OF_THESE if NONE_OF_THESE in q.sentinels else _first_label(q)
            return self._choice(wire, matches[0] if matches else fallback)
        return self._choice(wire, self._slot_label(q))

    def _tool_label(self, q: BallotQuestion) -> str:
        expected = self.expected
        if self.gold.no_tool:
            return NO_TOOL
        if expected is None:  # BFCL live_relevance: any call is right; elect the first real tool
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
        """The BFCL-style type of the top-level parameter at ``path`` (``any`` when unknown)."""
        return self.gold.param_type(path[0]) if path else "any"

    def _matches(self, value: Any, acceptable: Sequence[Any], typ: str = "any") -> bool:
        return self.gold.matches(value, acceptable, typ)

    def _matching_labels(self, q: BallotQuestion) -> list[str]:
        """Every option label whose value is acceptable (distinct values only). A joint option (a combination of
        slot values) is acceptable when each of its values is."""
        if q.family == "joint":
            return [o.label for o in q.options if isinstance(o.value, Mapping) and self._combination_ok(o.value)]
        acceptable = self.acceptable(q.path)
        if acceptable is None or q.family in ("date", "time"):
            return []
        typ = self.bfcl_type(q.path)
        mention = q.family == "mention"
        seen: set[str] = set()
        out: list[str] = []
        for option in q.options:
            key = repr(option.value)
            ok = (
                self._in_some_list(option.value, acceptable)
                if mention
                else self._matches(option.value, acceptable, typ)
            )
            if key not in seen and ok:
                seen.add(key)
                out.append(option.label)
        return out

    def _combination_ok(self, combination: Mapping[str, Any]) -> bool:
        for name, value in combination.items():
            acceptable = self.acceptable((name,))
            if acceptable is not None and not self._matches(value, acceptable, self.bfcl_type((name,))):
                return False
        return True

    def _slot_label(self, q: BallotQuestion) -> str:
        acceptable = self.acceptable(q.path)
        if acceptable is None:
            return NONE_OF_THESE if NONE_OF_THESE in q.sentinels else _first_label(q)
        typ = self.bfcl_type(q.path)
        if q.family in ("date", "time"):
            match = next((o.label for o in q.options if _part_matches(o.value, acceptable)), None)
        elif q.family == "mention":
            match = next((o.label for o in q.options if self._in_some_list(o.value, acceptable)), None)
        else:
            match = next((o.label for o in q.options if self._matches(o.value, acceptable, typ)), None)
        if match is not None:
            return match
        if NOT_STATED in q.sentinels and self._not_stated_ok(q, acceptable, typ):
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
        if q.family == "present":  # consistent with the slot's own election: a real value is stated
            return P_YES if self._elects_value(q) else 0.2
        if acceptable is None:
            return 0.5
        if q.family == "slot" and q.kind == "flag":  # a flag without a default is a Noul: P(true)
            if self._matches(True, acceptable, "boolean") and ANY not in acceptable:
                return P_YES
            return P_NO if self._matches(False, acceptable, "boolean") else 0.5
        if q.family in ("accept", "verify"):
            candidate = (q.meta or {}).get("candidate") or {}
            return P_YES if self._matches(candidate.get("value"), acceptable, typ) else P_NO
        if q.family in ("item", "member"):  # a list element, or a member of a superlative's set ("the latest …")
            value = self._item_value(q)
            fits = value is not None and (self._in_some_list(value, acceptable) or (
                q.family == "member" and self._matches(value, acceptable, typ)))  # fmt: skip
            return P_YES if fits else P_NO
        return 0.5

    def _verify(self, qid: str, wire: Any, questions: Mapping[str, BallotQuestion]) -> Answer | None:
        """A ``verify`` Noul, answered from the candidate it was sent with (a follow-up round reuses the first round's
        qid for another record): yes iff that record is an acceptable value. The slot question is the qid without
        ``.verify.N``; the candidate string starts with the option's label."""
        head, sep, index = qid.rpartition(".verify.")
        slot = questions.get(head) if sep and index.isdigit() else None
        instructions = getattr(wire, "instructions", None)
        candidate = instructions.get("candidate") if isinstance(instructions, Mapping) else None
        if slot is None or not isinstance(candidate, str):
            return None
        options = [o for o in slot.options if candidate == o.label or candidate.startswith(o.label + ". ")]
        acceptable = self.acceptable(slot.path)
        if not options or acceptable is None:
            return None
        option = max(options, key=lambda o: len(o.label))
        return NoulAnswer(noul=P_YES if self._matches(option.value, acceptable, self.bfcl_type(slot.path)) else P_NO)

    def _elects_value(self, q: BallotQuestion) -> bool:
        """Whether the slot of a ``present`` probe is answered with a real option (not a sentinel)."""
        if self.plan is None:
            return False
        path = tuple(q.path)
        slots = [s for s in self.plan.ballot.questions if s.family == "slot" and s.tool == q.tool]
        sibling = next((s for s in slots if tuple(s.path) == path), None)
        if sibling is None:
            acceptable = self.acceptable(q.path)
            return acceptable is not None and "" not in acceptable
        return self._slot_label(sibling) in {o.label for o in sibling.options}

    def _in_some_list(self, value: Any, acceptable: Sequence[Any]) -> bool:
        if ANY in acceptable:
            return value not in ("", None)
        return any(isinstance(option, list) and any(self._matches(value, [item]) for item in option)
                   for option in acceptable)  # fmt: skip

    def _not_stated_ok(self, q: BallotQuestion, acceptable: Sequence[Any], typ: str) -> bool:
        sentinel = q.sentinels[NOT_STATED]
        if sentinel.decodes_to == "omit":
            return "" in acceptable
        if sentinel.decodes_to == "default":
            if sentinel.late is not None:
                return ANY in acceptable
            return self._matches(sentinel.value, acceptable, typ) or ("" in acceptable and sentinel.value is None)
        return False

    def _items_complete(self, questions: Sequence[BallotQuestion], acceptable: Sequence[Any]) -> bool:
        """Whether some acceptable list has every element among the list's item candidates."""
        if ANY in acceptable:
            return True
        values = [self._item_value(q) for q in questions if q.family == "item"]
        return any(all(any(self._matches(v, [e]) for v in values if v is not None) for e in option)
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
            if ANY in acceptable:  # not checked by the gold label
                continue
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
        return self.gold.user_text

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


def _part_matches(value: Any, acceptable: Sequence[Any]) -> bool:
    """A date or time part matches when its text appears in an acceptable string (BFCL-standardized)."""
    text = standardize_string(str(value))
    return bool(text) and any(isinstance(a, str) and a and text in standardize_string(a) for a in acceptable)


def _spread(wire: Any, labels: Sequence[str]) -> Answer:
    """A Choice answer that spreads ``P_TOP`` evenly over ``labels`` (a genuine ambiguity)."""
    all_labels = list(wire.criteria) if isinstance(wire, ChoiceQuestion) else list(labels)
    share = P_TOP / len(labels)
    rest = (1 - P_TOP) / max(1, len(all_labels) - len(labels))
    probabilities = {label: share if label in labels else rest for label in all_labels}
    top = max(probabilities, key=lambda k: probabilities[k])
    return ChoiceAnswer(choice=top, confidence=0.0, probabilities=probabilities)


def _uncertain(wire: Any) -> Answer:
    if isinstance(wire, ChoiceQuestion):
        return choice_answer(list(wire.criteria), None)
    if getattr(wire, "type", None) == "score":
        return score_answer(list(getattr(wire, "criteria", ["?"])), None)
    return NoulAnswer(noul=0.5)


__all__ = ["ANY", "P_TOP", "BfclGold", "Coverage", "EvalGold", "GoldView", "OracleBackend", "ParamCoverage", "in_text"]

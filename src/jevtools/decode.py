"""Decoding a round (spec §3.6): answers → per-slot value distributions → the constrained MAP call per tool.

Steps, for every speculated tool:

1. **Answer-shape guards** (:func:`collect_answers`): a missing answer for a sent qid, or a type mismatch, raises
   :class:`~jevtools.backends.errors.JevProtocolError` (→ P0, fail closed). Unknown labels are ignored and logged.
   Probabilities are used as returned and never renormalized.
2. **Slots**: each slot is decoded by its kind's resolver (label → value, pooling, sentinels, family rules).
3. **Constrained MAP** over the slots linked by binary constraints or late defaults: top-3 values per slot,
   feasible combinations only (binary constraints and ``@checks``), largest ``∏ D_s(v_s)``. Factors stay the
   unnormalized ``D_s(v_s)``; infeasible mass is error. No feasible combination → flag ``infeasible``.
4. **Late binding**: late defaults (``currency ← from_account.currency``), template placeholders and derived values.
   A composed value that fails the slot schema loses its mass (error, never renormalized) and the next value is
   taken.
5. **Validation** of the assembled arguments against the JSON Schema and every constraint.
6. **Joint** (J = mass of the option equal to the MAP; ``joint_disagrees`` when its argmax differs) and the call MAP
   across tools (§3.7.4).
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from jevtools.backends.errors import JevProtocolError
from jevtools.ballot import Ballot, BallotQuestion, tool_qid
from jevtools.candidates import Bottom, Candidate, Channel, Pool, display_value, label_key, value_key
from jevtools.confidence import CallMap, Composition, Factors, call_map
from jevtools.kinds import late as late_recipes
from jevtools.kinds.base import (
    LATE_DEFAULT,
    Alternative,
    ResolveContext,
    SlotResult,
    ValueEntry,
    elect,
    get_resolver,
)
from jevtools.kinds.text import accept_result
from jevtools.plan import PoolKey, sibling_source
from jevtools.policy import PolicyInput, SlotState, Tier
from jevtools.spec.constraints import Constraint, ConstraintContext
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.spec.schema import validate
from jevtools.wire import Answer, ChoiceAnswer, DecisionResponse, NoulAnswer, ScoreAnswer

MAP_TOP = 3
"""Values per slot enumerated by the constrained MAP."""
MAP_MAX_COMBINATIONS = 4096
"""Enumeration cap: above it, the slots with most options are cut to their top values until it fits."""
_LATE = "\x00late"
_MARKER = re.compile(r"⟨[^⟩]*⟩")
_PRIMITIVE = {"choice": ChoiceAnswer, "noul": NoulAnswer, "score": ScoreAnswer}
_SCORE_EPS = 1e-9


# --------------------------------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class ToolDecode:
    """The decoded call of one speculated tool."""

    tool: ToolSpec
    p: float
    """``P(tool)``."""
    slots: dict[str, SlotResult]
    """Top-level slot name → result (after MAP and late binding)."""
    arguments: dict[str, Any]
    """Assembled argument values (bound real values only; omitted and unbound slots are absent)."""
    complete: bool
    """Every required slot is bound and the arguments validate."""
    factors: Factors
    Q: float
    """``∏`` of the slot factors (0 when infeasible)."""
    authorized: float | None = None
    done_after: float | None = None
    present: dict[str, float] = field(default_factory=dict)
    joint: float | None = None
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def gates(self) -> dict[str, float]:
        """Gate values for the Decision: ``authorized``, ``present.<slot>`` and (loop mode) ``done_after``."""
        gates = {"authorized": self.authorized} if self.authorized is not None else {}
        gates.update({f"present.{k}": v for k, v in self.present.items()})
        if self.done_after is not None:
            gates["done_after"] = self.done_after
        return gates


@dataclass
class Decoded:
    """The decoded round: the tool distribution, every speculated tool's call and the call MAP."""

    ballot: Ballot
    answers: dict[str, Answer]
    tool_dist: dict[str, float]
    """Tool names and tool sentinels (``NO_TOOL``…) → probability."""
    tools: dict[str, ToolDecode]
    call_map: CallMap
    observations: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def chosen(self) -> str | None:
        """``t* = argmax P(t)`` (a tool name or a tool sentinel label)."""
        return self.call_map.chosen

    @property
    def decision(self) -> ToolDecode | None:
        """The decoded call of ``t*`` when it was speculated."""
        return self.tools.get(self.chosen) if self.chosen is not None else None

    def viability(self, name: str) -> tuple[str, bool]:
        """``(viable, speculated)`` of a tool from the Ballot (``("ok", False)`` if it was not considered)."""
        for record in self.ballot.tools:
            if record.name == name:
                return record.viable, record.speculated
        return "ok", False


# --------------------------------------------------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------------------------------------------------


def collect_answers(
    ballot: Ballot, responses: Sequence[DecisionResponse], *, id_mode: str = "dotted"
) -> tuple[dict[str, Answer], list[str]]:
    """Answers by qid for every planned call, with the answer-shape guards of §8.2.

    Raises :class:`JevProtocolError` for a missing answer, a type mismatch, a probability outside [0, 1] (NaN
    included, for every primitive) or a Score whose expected level is not finite or outside its levels.
    Unknown labels and unsent answers are ignored and reported in the returned notes.
    """
    calls = ballot.call_plan()
    if len(responses) != len(calls):
        raise JevProtocolError(f"{len(responses)} response(s) for {len(calls)} planned call(s)")
    wire = ballot.wire_ids("opaque" if id_mode == "opaque" else "dotted")
    by_qid = ballot.by_qid
    answers: dict[str, Answer] = {}
    notes: list[str] = []
    for call, response in zip(calls, responses, strict=True):
        sent = {wire[qid] for qid in call}
        notes += [f"ignored answer for unsent question {k!r}" for k in response.answers if k not in sent]
        for qid in call:
            answer = response.answers.get(wire[qid])
            answers[qid] = _checked(by_qid[qid], answer, notes)
    return answers, notes


def _checked(question: BallotQuestion, answer: Answer | None, notes: list[str]) -> Answer:
    if answer is None:
        raise JevProtocolError(f"missing answer for {question.qid}")
    if not isinstance(answer, _PRIMITIVE[question.primitive]):
        raise JevProtocolError(f"{question.qid}: expected a {question.primitive} answer, got {answer.type}")
    if isinstance(answer, NoulAnswer) and not 0.0 <= answer.noul <= 1.0:
        raise JevProtocolError(f"{question.qid}: noul {answer.noul} outside [0, 1]")
    if isinstance(answer, ChoiceAnswer):
        if any(not 0.0 <= p <= 1.0 for p in answer.probabilities.values()):
            raise JevProtocolError(f"{question.qid}: probability outside [0, 1]")
        answer = _normalize_echo(question, answer, notes)
    if isinstance(answer, ScoreAnswer):
        top = len(question.levels or ()) - 1
        if any(not 0.0 <= p <= 1.0 for p in answer.probabilities.values()):  # also rejects NaN
            raise JevProtocolError(f"{question.qid}: probability outside [0, 1]")
        if not (math.isfinite(answer.score) and -_SCORE_EPS <= answer.score <= top + _SCORE_EPS):
            raise JevProtocolError(f"{question.qid}: score {answer.score} outside [0, {top}]")
    return answer


def _normalize_echo(question: BallotQuestion, answer: ChoiceAnswer, notes: list[str]) -> ChoiceAnswer:
    """Map echoed labels back to the sent ones (§8.7 label echo): an exact key is used as is; a key that is not
    byte-equal to any sent label but equal to exactly one after NFC and casefold (the label uniqueness key, so the
    match is unambiguous) is read as that label; anything else is ignored and noted."""
    sent = set(question.labels)
    unknown = [label for label in answer.probabilities if label not in sent]
    if not unknown:
        return answer
    by_key = {label_key(label): label for label in question.labels}
    probabilities = {k: v for k, v in answer.probabilities.items() if k in sent}
    choice = answer.choice
    for label in unknown:
        target = by_key.get(label_key(label))
        if target is None or target in probabilities:
            notes.append(f"{question.qid}: ignored unknown label {label!r}")
            continue
        notes.append(f"{question.qid}: label {label!r} read as {target!r} (echo normalized)")
        probabilities[target] = answer.probabilities[label]
        if choice == label:
            choice = target
    return answer.model_copy(update={"probabilities": probabilities, "choice": choice})


def noul(answers: Mapping[str, Answer], qid: str) -> float | None:
    """A Noul answer's P(yes), or ``None`` when absent."""
    answer = answers.get(qid)
    return answer.noul if isinstance(answer, NoulAnswer) else None


def choice_mass(question: BallotQuestion, answer: Answer | None) -> dict[str, float]:
    """Label → probability for the labels this question sent (unknown labels dropped, missing ones 0)."""
    probabilities = answer.probabilities if isinstance(answer, ChoiceAnswer) else {}
    return {label: float(probabilities.get(label, 0.0)) for label in question.labels}


def tool_distribution(ballot: Ballot, answers: Mapping[str, Answer]) -> dict[str, float]:
    """``P(tool)`` over tool names and tool sentinels; a tool fixed by ``tool_choice`` gets 1.0."""
    question = ballot.by_qid.get("tool")
    if question is None:
        considered = [t.name for t in ballot.tools]
        return {considered[0]: 1.0} if len(considered) == 1 else {}
    mass = choice_mass(question, answers.get("tool"))
    dist = {str(o.value): mass[o.label] for o in question.options}
    dist.update({label: mass[label] for label in question.sentinels})
    return dist


def decode_reply(ballot: Ballot, answers: Mapping[str, Answer]) -> tuple[str | None, float, dict[str, float]]:
    """The ``reply`` Choice of a resume round: ``(option id | "OTHER" | "CANCEL", p, distribution)``."""
    question = ballot.by_qid.get("reply")
    if question is None:
        return None, 0.0, {}
    mass = choice_mass(question, answers.get("reply"))
    dist = {str(o.value): mass[o.label] for o in question.options}
    dist.update({label: mass[label] for label in question.sentinels})
    best = max(dist, key=lambda k: dist[k]) if dist else None
    return best, dist.get(best, 0.0) if best else 0.0, dist


# --------------------------------------------------------------------------------------------------------------------
# Round decoding
# --------------------------------------------------------------------------------------------------------------------


def decode_round(
    ballot: Ballot,
    responses: Sequence[DecisionResponse],
    rc: ResolveContext,
    *,
    pools: Mapping[PoolKey, Pool] | None = None,
    id_mode: str = "dotted",
    dropped: Iterable[PoolKey] = (),
) -> Decoded:
    """Guard and collect the answers of one round, then :func:`decode_answers`."""
    answers, notes = collect_answers(ballot, responses, id_mode=id_mode)
    return decode_answers(ballot, answers, rc, pools=pools, dropped=dropped, notes=notes)


def decode_answers(
    ballot: Ballot,
    answers: Mapping[str, Answer],
    rc: ResolveContext,
    *,
    pools: Mapping[PoolKey, Pool] | None = None,
    dropped: Iterable[PoolKey] = (),
    notes: Sequence[str] = (),
) -> Decoded:
    """Decode already-collected answers against ``ballot`` (its questions are the decode map).

    ``pools`` are the round's pools (resolvers read attributes from them); a missing pool is replaced by an empty
    one, so a stored Ballot decodes on its own. ``dropped`` slots (422 isolation) decode as failed answers
    (:func:`dropped_result`).
    """
    catalog = rc.catalog
    if catalog is None:
        raise ValueError("decoding needs rc.catalog")
    rc = replace(rc, questions=ballot.by_qid)
    dist = tool_distribution(ballot, answers)
    dropped_set = set(dropped)
    tools: dict[str, ToolDecode] = {}
    for record in ballot.tools:
        if record.speculated:
            tool = catalog.get(record.name)
            tools[tool.name] = decode_tool(tool, dist.get(tool.name, 0.0), ballot, answers, rc, pools or {},
                                           dropped_set)  # fmt: skip
    cmap = call_map(dist, {name: td.Q for name, td in tools.items()})
    return Decoded(ballot=ballot, answers=dict(answers), tool_dist=dist, tools=tools, call_map=cmap,
                   observations=bool(rc.ctx.all_observations()), notes=list(notes))  # fmt: skip


def decode_tool(
    tool: ToolSpec,
    p_tool: float,
    ballot: Ballot,
    answers: Mapping[str, Answer],
    rc: ResolveContext,
    pools: Mapping[PoolKey, Pool],
    dropped: set[PoolKey],
) -> ToolDecode:
    """Slots → constrained MAP → late binding → validation → joint, for one tool."""
    results: dict[str, SlotResult] = {}
    flags: list[str] = []
    notes: list[str] = []
    for slot in tool.slots:
        key = (tool.name, slot.path)
        if key in dropped:
            results[slot.name] = dropped_result(slot)
            notes.append(f"{slot.name}: family dropped after a 422")
            continue
        pool = pools.get(key) or Pool(tool=tool.name, path=slot.path, kind=slot.kind)
        results[slot.name] = get_resolver(slot.kind).decode(tool, slot, pool, answers, rc)
    results, map_flags = constrained_map(tool, results, rc)
    results, late_flags = late_bind(tool, results, rc)
    flags += map_flags + late_flags + channel_violations(tool, results)
    arguments, complete, final_flags = assemble(tool, results, rc, check_constraints="infeasible" not in flags)
    flags += final_flags
    joint, joint_flags = decode_joint(tool, ballot, answers, arguments)
    flags += joint_flags
    slot_factors = {name: r.factor for name, r in results.items()}
    authorized = noul(answers, tool_qid(tool.id, "authorized"))
    factors = Factors.build(tool=p_tool, authorized=authorized, slots=slot_factors)
    q = 0.0 if "infeasible" in flags else math.prod(factors.slot_items().values())
    present = {".".join(question.path): n for question in ballot.questions_for(tool.name)
               if question.family == "present" and (n := noul(answers, question.qid)) is not None}  # fmt: skip
    return ToolDecode(tool=tool, p=p_tool, slots=results, arguments=arguments, complete=complete, factors=factors,
                      Q=q, authorized=authorized, done_after=noul(answers, tool_qid(tool.id, "done_after")),
                      present=present, joint=joint, flags=_unique(flags), notes=notes)  # fmt: skip


def dropped_result(slot: SlotSpec) -> SlotResult:
    """A slot family dropped by 422 isolation (§5.6): ``empty(reason=invalid)``, a failed answer (I5) — ``⊥missing``
    with factor 0 whatever the slot's default or ``required``, so it routes to clarify(open) and never decodes as a
    silent default or omission (§4.5)."""
    return SlotResult(
        path=slot.path, kind=slot.kind, stakes=slot.stakes, dist={Bottom.MISSING.value: 0.0}, values={},
        value=Bottom.MISSING, shape="missing", factor=0.0, flags=("invalid",), notes=("invalid (422)",),
    )  # fmt: skip


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


# --------------------------------------------------------------------------------------------------------------------
# Re-election helpers
# --------------------------------------------------------------------------------------------------------------------


def elected_key(result: SlotResult) -> str:
    """The distribution key of the elected value."""
    return value_key(result.value)


def top_keys(result: SlotResult, n: int = MAP_TOP) -> list[str]:
    """The ``n`` most probable real value keys with positive mass (stable on ties)."""
    real = [k for k in result.dist if k in result.values and result.dist[k] > 0]
    real.sort(key=lambda k: -result.dist[k])
    return real[:n]


def reelect(result: SlotResult, key: str, *, out_of_pool: float) -> SlotResult:
    """Elect the real value ``key`` (constrained MAP / next value): factor ``D(key)`` unnormalized, alternatives
    recomputed, shape ``out_of_pool`` when ``D(⊥uncovered) ≥ out_of_pool``."""
    if key == elected_key(result):
        return result
    entry = result.entries.get(key) or ValueEntry(display=display_value(result.values[key]))
    others = sorted((k for k in result.values if k != key and result.dist.get(k, 0.0) >= 0.01),
                    key=lambda k: -result.dist[k])[:3]  # fmt: skip
    alternatives = tuple(
        Alternative(display=result.entries[k].display if k in result.entries else display_value(result.values[k]),
                    p=result.dist[k], value=result.values[k],
                    label=result.entries[k].label if k in result.entries else None)
        for k in others
    )  # fmt: skip
    uncovered = result.dist.get(Bottom.UNCOVERED.value, 0.0) >= out_of_pool
    return result.with_(
        value=result.values[key], display=entry.display, label=entry.label, channel=entry.channel,
        prov=dict(entry.prov), late=entry.late, attrs=dict(entry.attrs), alternatives=alternatives,
        factor=None if result.stakes == "cosmetic" else result.dist.get(key, 0.0),
        shape="out_of_pool" if uncovered else "ok",
    )  # fmt: skip


def discard_keys(result: SlotResult, keys: Iterable[str], *, out_of_pool: float, note: str) -> SlotResult:
    """Drop failed values (late binding or schema): their mass is error — removed, never renormalized onto the
    others — and the next value is elected (§3.6 rule 5)."""
    keys = [k for k in keys if k in result.values]
    if not keys:
        return result
    lost = sum(result.dist.get(k, 0.0) for k in keys)
    dist = {k: p for k, p in result.dist.items() if k not in keys}
    values = {k: v for k, v in result.values.items() if k not in keys}
    entries = {k: e for k, e in result.entries.items() if k not in keys}
    return elect(path=result.path, kind=result.kind, stakes=result.stakes, dist=dist, values=values,
                 entries=entries, out_of_pool=out_of_pool, qids=result.qids, sentinels=result.sentinels,
                 notes=(*result.notes, f"{note} (error mass {lost:.4f})"), flags=result.flags,
                 ).with_(normalizer=result.normalizer)  # fmt: skip


def discard_values(slot: SlotSpec, result: SlotResult, keys: Iterable[str], rc: ResolveContext, *,
                   note: str) -> SlotResult:  # fmt: skip
    """:func:`discard_keys` under the slot's family rule: a text slot's remaining candidates are re-elected by the
    accept rule (§3.6 accept row: argmax ``n`` with the 0.02 tie to the lower index; content below ``accept_min`` →
    ``uncovered_text``; cosmetic below the floor → the next author template, else omitted when optional), never by
    the generic argmax."""
    oop = rc.policy.shapes.out_of_pool
    keys = [k for k in keys if k in result.values]
    remaining = [k for k in result.dist if k in result.values and k not in keys]
    entries = [result.entries.get(k) for k in remaining]
    if result.kind != "text" or not keys or any(e is None or e.channel is None for e in entries):
        return discard_keys(result, keys, out_of_pool=oop, note=note)
    lost = sum(result.dist.get(k, 0.0) for k in keys)
    scored = [(Candidate(value=result.values[k], text=e.display, channel=e.channel, prov=dict(e.prov), late=e.late),
               result.dist[k])
              for k, e in zip(remaining, entries, strict=True) if e is not None and e.channel is not None]  # fmt: skip
    redone = accept_result(slot, scored, rc, result.qids, result.normalizer or "")
    return redone.with_(notes=(*result.notes, *redone.notes, f"{note} (error mass {lost:.4f})"), flags=result.flags)


def rekey(result: SlotResult, key: str, value: Any) -> SlotResult:
    """Replace the value behind ``key`` by a composed value (late binding) and elect it (an equal existing value
    pools with it; for accept Nouls, which are independent judgments rather than one distribution, the larger
    ``n`` is kept, so a factor never exceeds 1)."""
    new_key = value_key(value)
    accept = result.kind == "text"
    dist: dict[str, float] = {}
    for k, p in result.dist.items():
        target = new_key if k == key else k
        dist[target] = max(dist.get(target, 0.0), p) if accept else dist.get(target, 0.0) + p
    values = {k: v for k, v in result.values.items() if k != key}
    values[new_key] = value
    entries = {k: e for k, e in result.entries.items() if k != key}
    display = display_value(value)
    if key in result.entries:
        entry = result.entries[key]
        late = {**entry.late, "template": result.values[key]} if entry.late else None
        entries[new_key] = replace(entry, display=display, late=late)
    return result.with_(dist=dist, values=values, entries=entries, value=value, display=display,
                        factor=None if result.stakes == "cosmetic" else dist[new_key])  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# Constrained MAP (§3.6 rule 4) with late defaults
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pick:
    key: str
    value: Any
    mass: float
    attrs: Mapping[str, Any]


def _late_value(results: Mapping[str, SlotResult], source: tuple[str, str], key: str | None) -> Any:
    """The late default ``<sibling>.<attr>`` for the sibling's value ``key`` (``None`` if unavailable)."""
    name, attr = source
    sibling = results[name]
    key = key if key is not None else elected_key(sibling)
    if key not in sibling.values:
        return None
    entry = sibling.entries.get(key)
    attrs = entry.attrs if entry is not None else {}
    if attr in attrs:
        return attrs[attr]
    value = sibling.values[key]
    return value.get(attr) if isinstance(value, Mapping) else None


def _pick(result: SlotResult, option: str, late_value: Any) -> _Pick | None:
    """A slot's value and pooled mass for one MAP option (a real key or the late default)."""
    late_mass = result.dist.get(LATE_DEFAULT, 0.0)
    late_key = value_key(late_value) if late_value is not None else None
    if option == _LATE:
        if late_key is None:
            return None
        mass = late_mass + result.dist.get(late_key, 0.0)
        entry = result.entries.get(late_key)
        return _Pick(late_key, late_value, mass, entry.attrs if entry is not None else {})
    mass = result.dist.get(option, 0.0) + (late_mass if option == late_key else 0.0)
    entry = result.entries.get(option)
    return _Pick(option, result.values[option], mass, entry.attrs if entry is not None else {})


def _map_options(
    tool: ToolSpec, results: Mapping[str, SlotResult], late: Mapping[str, tuple[str, str]], names: Sequence[str]
) -> dict[str, list[str]]:
    options: dict[str, list[str]] = {}
    for name in names:
        result = results[name]
        if result.is_bottom and result.value is not Bottom.LATE_DEFAULT:
            continue
        keys = top_keys(result)
        if name in late and not results[late[name][0]].is_bottom:
            keys.append(_LATE)
        if keys:
            options[name] = keys
    while math.prod(len(v) for v in options.values()) > MAP_MAX_COMBINATIONS:
        widest = max(options, key=lambda n: len(options[n]))
        options[widest] = options[widest][:-1]
    return options


def feasible(constraints: Sequence[Constraint], args: Mapping[str, Any], attrs: Mapping[str, Mapping[str, Any]],
             rc: ResolveContext) -> bool:  # fmt: skip
    """Every applicable constraint holds. Constraints over a slot that is not bound are skipped (the slot's shape
    routes the call); anything but ``True`` (violated or not evaluable) is infeasible (fail closed)."""
    env = ConstraintContext(attrs=attrs, now=rc.now, context=rc.ctx)
    for constraint in constraints:
        if constraint.check_name is None and not constraint.slots <= set(args):
            continue
        if constraint.check(args, env) is not True:
            return False
    return True


def constrained_map(
    tool: ToolSpec, results: dict[str, SlotResult], rc: ResolveContext
) -> tuple[dict[str, SlotResult], list[str]]:
    """Pick the feasible combination with the largest ``∏ D_s(v_s)`` over the top-3 values of the slots linked by
    binary constraints or late defaults (§3.6 rule 4), then resolve every late default."""
    oop = rc.policy.shapes.out_of_pool
    late = {s.name: src for s in tool.slots if (src := sibling_source(tool, s)) is not None
            and LATE_DEFAULT in results[s.name].dist}  # fmt: skip
    binary = [c for c in tool.constraints if c.check_name is None and len(c.slots) >= 2]
    checks = [c for c in tool.constraints if c.check_name is not None]
    involved = [s.name for s in tool.slots if any(s.name in c.slots for c in binary) or s.name in late
                or any(src[0] == s.name for src in late.values())]  # fmt: skip
    options = _map_options(tool, results, late, involved)
    if not options and not checks:
        return _resolve_late(results, late, oop), []
    names = list(options)
    best: tuple[float, dict[str, _Pick]] | None = None
    for combo in itertools.product(*(options[n] for n in names)):
        choice = dict(zip(names, combo, strict=True))
        picks = _combo_picks(results, choice, late)
        if picks is None:
            continue
        args, attrs = _combo_args(results, picks)
        if not feasible([*binary, *checks], args, attrs, rc):
            continue
        score = math.prod(p.mass for p in picks.values())
        if best is None or score > best[0]:
            best = (score, picks)
    if best is None:
        return results, ["infeasible"]
    chosen = {name: pick.key for name, pick in best[1].items()}
    results = _resolve_late(results, late, oop, chosen)
    for name, key in chosen.items():
        if key in results[name].values:
            results[name] = reelect(results[name], key, out_of_pool=oop)
    return results, []


def _combo_picks(
    results: Mapping[str, SlotResult], choice: Mapping[str, str], late: Mapping[str, tuple[str, str]]
) -> dict[str, _Pick] | None:
    picks: dict[str, _Pick] = {}
    for name, option in choice.items():
        late_value = _late_value(results, late[name], choice.get(late[name][0])) if name in late else None
        pick = _pick(results[name], option, late_value)
        if pick is None:
            return None
        picks[name] = pick
    return picks


def _combo_args(
    results: Mapping[str, SlotResult], picks: Mapping[str, _Pick]
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    args: dict[str, Any] = {}
    attrs: dict[str, Mapping[str, Any]] = {}
    for name, result in results.items():
        if name in picks:
            args[name], attrs[name] = picks[name].value, picks[name].attrs
        elif not result.is_bottom:
            args[name], attrs[name] = result.value, result.attrs
    return args, attrs


def _resolve_late(
    results: dict[str, SlotResult],
    late: Mapping[str, tuple[str, str]],
    oop: float,
    chosen: Mapping[str, str] | None = None,
) -> dict[str, SlotResult]:
    """Bind every pending late default to its sibling's (chosen) value; unresolvable ones become missing/omit."""
    results = dict(results)
    for name, source in late.items():
        result = results[name]
        value = _late_value(results, source, (chosen or {}).get(source[0]))
        if value is None:
            results[name] = _drop_late(result, oop)
            continue
        sibling = results[source[0]]
        results[name] = result.bind_late_default(value, out_of_pool=oop, display=display_value(value),
                                                 channel=sibling.channel or Channel.REGISTRY)  # fmt: skip
    return results


def _drop_late(result: SlotResult, oop: float) -> SlotResult:
    """A late default whose source value is unknown: its mass becomes ``⊥missing`` (fail closed)."""
    dist = dict(result.dist)
    dist[Bottom.MISSING.value] = dist.get(Bottom.MISSING.value, 0.0) + dist.pop(LATE_DEFAULT, 0.0)
    entries = {k: e for k, e in result.entries.items() if k != LATE_DEFAULT}
    return elect(path=result.path, kind=result.kind, stakes=result.stakes, dist=dist, values=dict(result.values),
                 entries=entries, out_of_pool=oop, qids=result.qids, sentinels=result.sentinels,
                 notes=(*result.notes, "late default unresolved"), flags=result.flags)  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# Late binding (§3.6 rule 5): placeholders and derived values
# --------------------------------------------------------------------------------------------------------------------


class LateBindingError(ValueError):
    """A placeholder or derivation cannot be computed from the bound values."""


def compose_late(value: Any, recipe: Mapping[str, Any], results: Mapping[str, SlotResult], rc: ResolveContext,
                 schema: Mapping[str, Any] | None = None) -> Any:  # fmt: skip
    """Evaluate a late-binding recipe with :func:`jevtools.kinds.late.late_bind` over the decoded slots and the
    context's observations: ``{"placeholders": […], "fill": {marker: path}}`` or ``{"derive": op, "of": path}``.

    The spec's list-only form (``{"placeholders": ["to.first_name"]}``) maps the ``⟨…⟩`` markers to the paths in
    order; ``template`` holds the pre-substitution value once a value has been composed. A marker that survives
    substitution is an error (never emitted)."""
    base = recipe.get("template", value)
    recipe = _with_fill(base, recipe)
    try:
        composed = late_recipes.late_bind(base, recipe, late_recipes.make_lookup(results, rc.ctx), schema=schema)
    except (late_recipes.LateBindError, ValueError, ArithmeticError) as exc:
        raise LateBindingError(str(exc)) from None
    if isinstance(composed, str) and any(marker in composed for marker in recipe.get("fill", {})):
        raise LateBindingError("a placeholder was left unfilled")
    return composed


def _with_fill(value: Any, recipe: Mapping[str, Any]) -> Mapping[str, Any]:
    if "fill" in recipe or "placeholders" not in recipe:
        return recipe
    markers = list(dict.fromkeys(_MARKER.findall(value))) if isinstance(value, str) else []
    paths = [str(p) for p in recipe["placeholders"]]
    if len(markers) != len(paths):
        raise LateBindingError(f"{len(markers)} marker(s) for {len(paths)} placeholder path(s)")
    return {**recipe, "fill": dict(zip(markers, paths, strict=True))}


def late_bind(
    tool: ToolSpec, results: dict[str, SlotResult], rc: ResolveContext
) -> tuple[dict[str, SlotResult], list[str]]:
    """Compose late-bound values, then check every elected value against its slot schema; a failing value's mass
    becomes error and the next value is taken (§3.6 rule 5, under the slot's family rule)."""
    flags: list[str] = []
    snapshot = dict(results)
    for slot in tool.slots:
        result = results[slot.name]
        while not result.is_bottom:
            key = elected_key(result)
            entry = result.entries.get(key)
            recipe = entry.late if entry is not None else result.late
            try:
                composed = _compose(slot, result.value, recipe, snapshot, rc)
            except LateBindingError as exc:
                result = discard_values(slot, result, [key], rc, note=f"late binding failed: {exc}")
                flags.append("late_binding_failed")
                continue
            if validate(composed, slot.json_schema):
                result = discard_values(slot, result, [key], rc, note="schema-invalid value discarded")
                continue
            if composed is not result.value and value_key(composed) != key:
                result = rekey(result, key, composed)
            break
        results[slot.name] = result
    return results, flags


def _compose(slot: SlotSpec, value: Any, recipe: Mapping[str, Any] | None, results: Mapping[str, SlotResult],
             rc: ResolveContext) -> Any:  # fmt: skip
    """A late-bound value (placeholders substituted, nothing else changed, §4.3; derived money quantized to the
    source row's currency), or ``value`` unchanged when there is no such recipe."""
    if not recipe or not ({"placeholders", "fill", "derive"} & set(recipe)):
        return value
    return compose_late(value, recipe, results, rc, schema=slot.json_schema)


# --------------------------------------------------------------------------------------------------------------------
# Assembly, validation, channels, joint
# --------------------------------------------------------------------------------------------------------------------


def assemble(
    tool: ToolSpec, results: Mapping[str, SlotResult], rc: ResolveContext, *, check_constraints: bool = True
) -> tuple[dict[str, Any], bool, list[str]]:
    """Arguments from the bound values (schema order); ``complete`` when every required slot is bound, the object
    validates (``required``, ``if/then/else``, ``dependentRequired``…) and every constraint holds."""
    args: dict[str, Any] = {}
    attrs: dict[str, Mapping[str, Any]] = {}
    missing = False
    for slot in tool.slots:
        result = results[slot.name]
        if not result.is_bottom:
            args[slot.name], attrs[slot.name] = result.value, result.attrs
        elif result.value is not Bottom.OMIT:
            missing = True
    if missing:
        return args, False, []
    flags: list[str] = []
    if validate(args, tool.parameters):
        flags.append("schema_invalid")
    if check_constraints and not feasible(list(tool.constraints), args, attrs, rc):
        flags.append("infeasible")
    return args, not flags, flags


def value_origin(result: SlotResult) -> Channel | None:
    """The channel the elected value came from: for a user binding of an offered value (a click, a reply pick), the
    channel of the pool candidate the user picked (kept on its entry by :func:`bind_value`); else the result's
    channel."""
    if result.is_bottom:
        return None
    entry = result.entries.get(value_key(result.value)) if result.channel is Channel.USER else None
    return entry.channel if entry is not None and entry.channel is not None else result.channel


def channel_violations(tool: ToolSpec, results: Mapping[str, SlotResult]) -> list[str]:
    """I2 assertion: a bound value whose channel is outside its slot's allow-list (→ P3 refuse). A user binding of
    a value the slot offered is admitted when the offered candidate's channel is (a click on a menu entry)."""
    for slot in tool.slots:
        result = results[slot.name]
        if result.is_bottom or result.channel is None or result.prov.get("default"):
            continue
        if result.channel not in slot.channels and value_origin(result) not in slot.channels:
            return ["channel_violation"]
    return []


def decode_joint(
    tool: ToolSpec, ballot: Ballot, answers: Mapping[str, Answer], arguments: Mapping[str, Any]
) -> tuple[float | None, list[str]]:
    """``J`` = mass of the joint option equal to the factorized MAP (min over groups); ``joint_disagrees`` when the
    joint argmax is another option (or ``NONE_OF_THESE``)."""
    values: list[float] = []
    flags: list[str] = []
    for question in ballot.questions_for(tool.name):
        if question.family != "joint":
            continue
        mass = choice_mass(question, answers.get(question.qid))
        group = [str(n) for n in question.meta.get("group", [])]
        target = value_key({n: arguments[n] for n in group}) if all(n in arguments for n in group) else None
        match = next((o.label for o in question.options if target is not None and value_key(o.value) == target), None)
        values.append(mass.get(match, 0.0) if match is not None else 0.0)
        best = max(question.labels, key=lambda label: mass[label])
        if best != match:
            flags.append("joint_disagrees")
    return (min(values) if values else None), _unique(flags)


# --------------------------------------------------------------------------------------------------------------------
# User bindings (clicks and replies, §3.8.5)
# --------------------------------------------------------------------------------------------------------------------

RECOMPUTED_FLAGS = frozenset({"infeasible", "schema_invalid", "channel_violation", "joint_disagrees"})
"""Tool flags recomputed after a user binding (a click on a complete call supersedes the joint check)."""


def bind_value(
    td: ToolDecode,
    name: str,
    value: Any,
    rc: ResolveContext,
    *,
    p: float = 1.0,
    channel: Channel = Channel.USER,
    label: str | None = None,
    prov: Mapping[str, Any] | None = None,
    normalizer: str | None = None,
) -> ToolDecode:
    """Bind a user-supplied value (a click: ``p = 1``; a reply Choice: ``p = P(reply)``) to a top-level slot and
    recompute the arguments, constraints and factors; the other slots are reused unchanged.

    The result's channel is ``channel`` (``user``: the user bound it, §3.8.5). When the value was an offered pool
    entry, the result's entry keeps that entry's channel as the value's *origin* (:func:`value_origin`): TOCTOU
    re-resolves a registry value the user picked, and a slot narrowed to ``["registry"]`` still admits a click on a
    registry value it offered (:func:`channel_violations`)."""
    old = td.slots[name]
    key = value_key(value)
    known = old.entries.get(key)
    display = known.display if known is not None else display_value(value)
    label = label or (known.label if known is not None else None)
    attrs = dict(known.attrs) if known is not None else {}
    origin = known.channel if known is not None and known.channel is not None else channel
    entry = ValueEntry(display=display, label=label, channel=origin, prov=dict(prov or {}), attrs=attrs, p=p)
    result = SlotResult(
        path=old.path, kind=old.kind, stakes=old.stakes, dist={key: p}, values={key: value}, value=value,
        shape="ok", factor=None if old.stakes == "cosmetic" else p, display=display, label=label, channel=channel,
        prov=dict(prov or {}), attrs=attrs, qids=old.qids, normalizer=normalizer or old.normalizer,
        entries={key: entry},
    )  # fmt: skip
    return rebuild_tool(td, {**td.slots, name: result}, rc)


def rebuild_tool(td: ToolDecode, slots: dict[str, SlotResult], rc: ResolveContext, *, p_tool: float | None = None
                 ) -> ToolDecode:  # fmt: skip
    """Re-assemble a tool's call from (partly re-bound) slot results: late-bound values are recomputed from their
    templates (a new recipient changes ``⟨recipient's first name⟩``), then arguments, constraints and factors."""
    slots, late_flags = late_bind(td.tool, dict(slots), rc)
    arguments, complete, flags = assemble(td.tool, slots, rc)
    flags = [*late_flags, *flags]
    kept = [f for f in td.flags if f not in RECOMPUTED_FLAGS]
    flags = _unique([*kept, *flags, *channel_violations(td.tool, slots)])
    p = td.p if p_tool is None else p_tool
    factors = Factors.build(tool=p, authorized=td.authorized, slots={n: r.factor for n, r in slots.items()})
    q = 0.0 if "infeasible" in flags else math.prod(factors.slot_items().values())
    return replace(td, p=p, slots=slots, arguments=arguments, complete=complete, factors=factors, Q=q, flags=flags)


def with_tool(decoded: Decoded, name: str, p: float) -> Decoded:
    """Replace ``P(name)`` (a reply that picked a tool from a tool menu) and recompute the call MAP."""
    dist = {**decoded.tool_dist, name: p}
    tools = dict(decoded.tools)
    if name in tools:
        td = tools[name]
        tools[name] = replace(td, p=p, factors=Factors.build(tool=p, authorized=td.authorized,
                                                             slots=td.factors.slot_items()))  # fmt: skip
    cmap = call_map({name: p, **{k: v for k, v in dist.items() if k != name}}, {n: t.Q for n, t in tools.items()})
    return replace(decoded, tool_dist=dist, tools=tools, call_map=cmap)


def with_decision(decoded: Decoded, td: ToolDecode) -> Decoded:
    """Replace one tool's decoded call (after user bindings) and recompute the call MAP."""
    tools = {**decoded.tools, td.tool.name: td}
    return replace(decoded, tools=tools, call_map=call_map(decoded.tool_dist, {n: t.Q for n, t in tools.items()}))


# --------------------------------------------------------------------------------------------------------------------
# Policy input
# --------------------------------------------------------------------------------------------------------------------


def slot_state(name: str, result: SlotResult) -> SlotState:
    """The policy view of a slot; for composite slots the menu masses come from the weakest part."""
    part = result
    scored = [p for p in result.parts.values() if p.factor is not None]
    if scored:
        part = min(scored, key=lambda p: p.factor if p.factor is not None else 1.0)
    top = sorted((p for k, p in part.dist.items() if k in part.values), reverse=True)
    bottom = result.value.value if result.is_bottom and result.value is not Bottom.OMIT else None
    channel = result.channel.value if result.channel is not None and not result.is_bottom else None
    return SlotState(name=name, stakes=result.stakes, shape=result.shape, factor=result.factor, bottom=bottom,
                     top=top, flags=list(result.flags), channel=channel)  # fmt: skip


def policy_input(
    decoded: Decoded,
    comp: Composition | None,
    *,
    escalator: bool = False,
    widen_ok: Sequence[str] = (),
    fill_ok: Sequence[str] = (),
    confirmed: bool = False,
    loop: bool = False,
    failed: bool = False,
) -> PolicyInput:
    """Build the :class:`~jevtools.policy.PolicyInput` of a decoded round (``comp`` is ``t*``'s composition)."""
    chosen = decoded.chosen
    td = decoded.decision
    viable, speculated = decoded.viability(chosen) if chosen is not None else ("ok", False)
    flags = list(td.flags) if td is not None else []
    if td is not None and decoded.call_map.disagrees:
        flags.append("call_map_disagrees")
    return PolicyInput(
        failed=failed, tools=dict(decoded.tool_dist), chosen=chosen,
        tier=td.tool.tier if td is not None else _tier_of(decoded, chosen), viable=viable, speculated=speculated,
        authorized=td.authorized if td is not None else None, observations=decoded.observations,
        slots=[slot_state(n, r) for n, r in td.slots.items()] if td is not None else [], flags=flags,
        call_map=decoded.call_map.call_map, C=comp.C if comp is not None else None,
        present=dict(td.present) if td is not None else {}, loop=loop, escalator=escalator,
        widen_ok=list(widen_ok), fill_ok=list(fill_ok),
        confirm_always=td is not None and td.tool.confirm == "always", confirmed=confirmed,
    )  # fmt: skip


def _tier_of(decoded: Decoded, name: str | None) -> Tier | None:
    for record in decoded.ballot.tools:
        if record.name == name:
            return record.tier
    return None


__all__ = [
    "MAP_TOP",
    "Decoded",
    "LateBindingError",
    "ToolDecode",
    "RECOMPUTED_FLAGS",
    "assemble",
    "bind_value",
    "channel_violations",
    "choice_mass",
    "collect_answers",
    "compose_late",
    "constrained_map",
    "decode_answers",
    "decode_joint",
    "decode_reply",
    "decode_round",
    "decode_tool",
    "discard_keys",
    "discard_values",
    "dropped_result",
    "elected_key",
    "feasible",
    "late_bind",
    "noul",
    "policy_input",
    "reelect",
    "rebuild_tool",
    "rekey",
    "slot_state",
    "tool_distribution",
    "top_keys",
    "value_origin",
    "with_decision",
    "with_tool",
]

"""Round planning (spec §5.1–§5.5, §3.5.5): extraction → pools → viability → speculation → fan-out → budget → split.

:func:`compile_round` turns a catalog and a context into a :class:`RoundPlan`: the :class:`~jevtools.ballot.Ballot`
(the conformance boundary) plus the pools and the :class:`~jevtools.kinds.base.ResolveContext` that decoding needs.
:func:`plan_round` returns just the Ballot. Everything kind-specific happens inside resolvers; this module only
orchestrates them through the :class:`~jevtools.kinds.base.Resolver` contract.

Question order (normative, §3.5.5): ``tool``; then for each speculated tool in canonical label order its
``authorized`` (tier ≥ write), ``joint`` (critical tier or declared ``groups``), the slots' families in schema order
and ``done_after`` (loop mode); finally ``reply`` in a resume round.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jevtools import templates
from jevtools.ballot import Ballot, BallotOption, BallotQuestion, SentinelSpec, ToolViability, tool_qid
from jevtools.candidates import (
    CANCEL,
    DONE,
    NO_TOOL,
    NONE_OF_THESE,
    OTHER,
    UNSUPPORTED,
    Candidate,
    Channel,
    Pool,
    apply_allow_list,
    assign_labels,
    canonical_order,
    make_label,
    value_key,
    with_full_value,
)
from jevtools.canonical import canonical_str
from jevtools.context import Context, Mode, build_state, is_loop
from jevtools.kinds.base import ResolveContext, get_resolver, resolve_default
from jevtools.policy import Policy, Tier
from jevtools.prompts import Binding, render_call
from jevtools.qid import TOOL_QID
from jevtools.spec.catalog import Catalog
from jevtools.spec.constraints import ConstraintContext
from jevtools.spec.infer import SECRET_NAMES
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.validate import Limits, preflight

if TYPE_CHECKING:
    from jevtools.extract import Mentions

ToolChoice = str | Mapping[str, Any]
"""OpenAI ``tool_choice``: ``"auto"``, ``"none"``, ``"required"`` or ``{"type": "function", "function": {"name"}}``."""
PoolKey = tuple[str, tuple[str, ...]]
"""``(tool name, slot path)``."""
REF_K_CUT = 20
"""REF shortlist size after the third question cut (§5.5)."""


# --------------------------------------------------------------------------------------------------------------------
# Public records
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class RoundPlan:
    """A compiled round: the Ballot plus what decoding needs (pools, resolve context, the tool-choice mode)."""

    ballot: Ballot
    rc: ResolveContext
    pools: dict[PoolKey, Pool]
    catalog: Catalog
    tool_choice: str = "auto"
    """``auto``, ``none``, ``required`` or ``named``."""
    named: str | None = None
    notes: list[str] = field(default_factory=list)

    def pool(self, tool: str, path: Sequence[str]) -> Pool | None:
        """The pool of one slot (``None`` if the tool was not considered)."""
        return self.pools.get((tool, tuple(path)))


# --------------------------------------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------------------------------------


def plan_round(
    catalog: Catalog,
    ctx: Context,
    policy: Policy | None = None,
    *,
    mode: Mode = "turn",
    pending: Any = None,
    tool_choice: ToolChoice = "auto",
    limits: Limits | None = None,
    **kw: Any,
) -> Ballot:
    """The Ballot of one round (see :func:`compile_round` for the arguments)."""
    plan = compile_round(catalog, ctx, policy, mode=mode, pending=pending, tool_choice=tool_choice, limits=limits,
                         **kw)  # fmt: skip
    return plan.ballot


def compile_round(
    catalog: Catalog,
    ctx: Context,
    policy: Policy | None = None,
    *,
    mode: Mode = "turn",
    pending: Any = None,
    tool_choice: ToolChoice = "auto",
    limits: Limits | None = None,
    mentions: Mentions | None = None,
    has_filler: bool = False,
    round: int = 1,
    speculate_only: Sequence[str] | None = None,
    extra_candidates: Mapping[PoolKey, Sequence[Candidate]] | None = None,
    reply_options: Sequence[Mapping[str, str]] | None = None,
) -> RoundPlan:
    """Compile one round.

    - ``tool_choice`` follows §7.2.2: ``none`` → an empty Ballot (no Jev call); ``required`` → no ``NO_TOOL``; a
      named tool → no tool question, only that tool speculated (``authorized`` still asked for tier ≥ write).
    - ``pending`` (a :class:`~jevtools.decision.Pending`, resume rounds) speculates exactly the pending tool, viable
      or not (its full fan-out, §3.8.5); ``reply_options`` (``[{"id", "text"}]``, default from
      ``pending.state["options"]``) add the ``reply`` Choice.
    - ``speculate_only`` restricts speculation to the named tools (re-plan after a speculation miss).
    - ``extra_candidates`` inject candidates (Filler/Escalator output, resume replies) into pools, subject to the
      slot's allow-list (I2).
    - ``mentions`` supplies pre-computed mentions; by default the extractors run once per round, on the first
      resolver that needs them (:meth:`~jevtools.kinds.base.ResolveContext.get_mentions`).
    """
    policy = policy or Policy()
    limits = (limits or Limits()).within_budget(policy.budget)
    choice, named = parse_tool_choice(tool_choice, catalog)
    state, notes = cut_state(build_state(ctx, mode, secret_fields=secret_user_fields(catalog, ctx)), limits,
                             keep=pinned_mentions(ctx))  # fmt: skip
    rc = ResolveContext(
        ctx=ctx, catalog=catalog, policy=policy, limits=limits, mode=mode, state=state, round=round,
        has_filler=has_filler, preferred={}, widen={}, injected=injected_by_slot(catalog, extra_candidates or {}),
        mentions=mentions,
    )  # fmt: skip
    forced: list[str] = []
    loop = False
    if pending is not None:
        loop = bool(pending.state.get("loop"))  # a resume of a loop-mode prompt still asks done_after (§6.1)
        if pending.state.get("tool"):
            forced = speculate_only = [pending.state["tool"]]
        if reply_options is None:
            reply_options = pending.state.get("options")
    considered = [] if choice == "none" else [catalog.get(named)] if named else list(catalog)
    builder = _Builder(catalog, rc, considered, choice, speculate_only, extra_candidates or {}, forced=forced,
                       loop=loop)  # fmt: skip
    questions, tools = builder.build(policy)
    if reply_options:
        questions.append(reply_question(reply_options))
    questions, tools, cut_notes = _apply_budget(builder, questions, tools, state, limits, policy)
    ballot = Ballot(
        catalog_sha256=catalog.sha256, context_sha256=ctx.sha256, policy_version=policy.version, mode=mode,
        state=state, tools=tools, questions=questions, calls=split_calls(questions, state, limits),
    )  # fmt: skip
    preflight(ballot, limits, catalog=catalog)
    return RoundPlan(ballot=ballot, rc=rc, pools=builder.pools, catalog=catalog, tool_choice=choice, named=named,
                     notes=notes + cut_notes)  # fmt: skip


def followup_ballot(base: Ballot, questions: Sequence[BallotQuestion], *, mode: Mode, limits: Limits) -> Ballot:
    """A same-state round (widen, fill, escalation gate) asking only ``questions``; previous answers stay valid
    because the state is unchanged (I4)."""
    ballot = base.model_copy(update={"mode": mode, "questions": list(questions),
                                     "calls": split_calls(questions, base.state, limits)})  # fmt: skip
    ballot = Ballot.model_validate(ballot.model_dump())
    preflight(ballot, limits)
    return ballot


def merge_ballots(base: Ballot, extra: Ballot) -> Ballot:
    """The decode view of two same-state rounds: ``base`` questions, replaced or extended by ``extra``'s."""
    replaced = {q.qid for q in extra.questions}
    questions = [q for q in base.questions if q.qid not in replaced] + list(extra.questions)
    return Ballot.model_validate({**base.model_dump(), "questions": questions, "calls": [[q.qid for q in questions]]})


# --------------------------------------------------------------------------------------------------------------------
# tool_choice and state cuts
# --------------------------------------------------------------------------------------------------------------------


def parse_tool_choice(tool_choice: ToolChoice | None, catalog: Catalog) -> tuple[str, str | None]:
    """``(mode, tool name)``: mode is ``auto``, ``none``, ``required`` or ``named``."""
    if tool_choice is None or tool_choice == "auto":
        return "auto", None
    if tool_choice in ("none", "required"):
        return str(tool_choice), None
    if isinstance(tool_choice, Mapping):
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, Mapping) else tool_choice.get("name")
        if not isinstance(name, str) or name not in catalog:
            raise ValueError(f"tool_choice names an unknown tool: {tool_choice!r}")
        return "named", name
    raise ValueError(f"unsupported tool_choice {tool_choice!r}")


def state_tokens(state: Any, limits: Limits) -> int:
    """Estimated tokens of the state alone."""
    return limits.tokens_for_chars(len(canonical_str(state)))


def cut_state(
    state: dict[str, Any], limits: Limits, *, keep: Callable[[str], bool] | None = None
) -> tuple[dict[str, Any], list[str]]:
    """State cut 1 (§5.5, §6.2): drop the oldest ``history`` turns while the state exceeds ``max_state_tokens``.

    ``keep(text)`` marks turns to spare (§6.2: turns that mention pinned entities, from the context's entity store):
    they are dropped only once every other turn is gone and the state is still too large.
    """
    notes: list[str] = []
    history = list(state.get("history") or [])
    dropped = 0
    while history and state_tokens({**state, "history": history}, limits) > limits.max_state_tokens:
        index = 0
        if keep is not None:
            index = next((i for i, turn in enumerate(history) if not keep(str(turn.get("text", "")))), 0)
        history.pop(index)
        dropped += 1
    if dropped:
        notes.append(f"budget: dropped {dropped} oldest history turn(s)")
        state = {**state, "history": history}
    return state, notes


def pinned_mentions(ctx: Context) -> Callable[[str], bool] | None:
    """The §6.2 history rule of the context's entity store (``EntityStore.mentions_pinned``), if it has one."""
    test = getattr(ctx.entities, "mentions_pinned", None)
    return test if callable(test) else None


# --------------------------------------------------------------------------------------------------------------------
# Pools, viability and questions
# --------------------------------------------------------------------------------------------------------------------


def viability(tool: ToolSpec, pools: Mapping[PoolKey, Pool], ctx: Context) -> str:
    """``ok``, ``channel_blocked:<slot>`` or ``empty:<slot>`` (§5.1): every required slot must be filled-able —
    an evidence-backed or closed pool, a default/``default_from``; channel-blocked slots are reported first."""
    blocked: str | None = None
    empty: str | None = None
    for slot in tool.slots:
        if not slot.required:
            continue
        pool = pools.get((tool.name, slot.path))
        if pool is not None and (pool.evidence_backed or pool.closed):
            continue
        if resolve_default(tool, slot, ctx) is not None:
            continue
        if pool is not None and pool.channel_blocked:
            blocked = blocked or slot.name
        else:
            empty = empty or slot.name
    if blocked:
        return f"channel_blocked:{blocked}"
    return f"empty:{empty}" if empty else "ok"


def injected_by_slot(catalog: Catalog, extra: Mapping[PoolKey, Sequence[Candidate]]) -> dict[str, list[Candidate]]:
    """``ResolveContext.injected``: extra candidates keyed ``"<tool id>.<qpath>"`` (resolvers add them to the pool
    before the allow-list; :func:`inject_candidates` covers resolvers that do not)."""
    out: dict[str, list[Candidate]] = {}
    for (tool_name, path), candidates in extra.items():
        tool = catalog.get(tool_name)
        out[ResolveContext.slot_key(tool, tool.slot(path))] = list(candidates)
    return out


def is_anchored(candidate: Candidate) -> bool:
    """A candidate matched by the user's words: ``user`` channel, or ``prov.anchor`` / ``prov.mention`` set."""
    return candidate.channel is Channel.USER or bool(candidate.prov.get("anchor") or candidate.prov.get("mention"))


def inject_candidates(pool: Pool, extra: Sequence[Candidate], slot: SlotSpec, label_max: int) -> Pool:
    """Add candidates to a built pool under the slot's allow-list (I2); equal values are not duplicated.

    Canonically ordered pools stay canonical; resolver-ordered pools (e.g. text ladders) get the new ones appended.
    """
    admitted, blocked = apply_allow_list(extra, slot.channels)
    known = {value_key(c.value) for c in pool.candidates}
    fresh = [c for c in admitted if value_key(c.value) not in known]
    if not fresh and not blocked:
        return pool
    labelled = assign_labels(fresh, slot=slot.name, label_max=label_max, taken=[c.label for c in pool.candidates])
    canonical = pool.candidates == canonical_order(pool.candidates)
    merged = pool.candidates + labelled
    return pool.model_copy(update={
        "candidates": canonical_order(merged) if canonical else merged,
        "blocked": [*pool.blocked, *blocked],
        "evidence_backed": pool.evidence_backed or any(c.is_evidence for c in labelled),
        "notes": [*pool.notes, f"injected {len(labelled)} candidate(s)"] if labelled else pool.notes,
    })  # fmt: skip


def tool_question(tools: Sequence[ToolSpec], *, choice: str, loop: bool, done: bool, label_max: int) -> BallotQuestion:
    """The ``tool`` Choice: tools in canonical label order, then ``NO_TOOL`` (unless ``required``),
    ``UNSUPPORTED`` and ``DONE`` (loop mode after at least one step)."""
    options = [BallotOption(label=label, value=tool.name, text=_tool_text(tool))
               for label, tool in tool_labels(tools, label_max).items()]  # fmt: skip
    sentinels: dict[str, SentinelSpec] = {}
    if choice != "required":
        sentinels[NO_TOOL] = SentinelSpec(decodes_to="no_tool", text=templates.TOOL_SENTINEL_TEXT[NO_TOOL])
    sentinels[UNSUPPORTED] = SentinelSpec(decodes_to="unsupported", text=templates.TOOL_SENTINEL_TEXT[UNSUPPORTED])
    if loop and done:
        sentinels[DONE] = SentinelSpec(decodes_to="done", text=templates.TOOL_SENTINEL_TEXT[DONE])
    return BallotQuestion(qid=TOOL_QID, family="tool", primitive="choice", instructions=templates.tool_instructions(
        loop=loop), options=canonical_order(options), sentinels=sentinels)  # fmt: skip


def tool_labels(tools: Sequence[ToolSpec], label_max: int = 64) -> dict[str, ToolSpec]:
    """Tool option labels (the tool name when it obeys the label grammar) in canonical order."""
    taken: list[str] = []
    labelled: dict[str, ToolSpec] = {}
    for n, tool in enumerate(tools, start=1):
        label = make_label(tool.name, slot="tool", n=n, label_max=label_max, taken=taken)
        taken.append(label)
        labelled[label] = tool
    return {label: labelled[label] for label in canonical_order(list(labelled))}


def _tool_text(tool: ToolSpec) -> str:
    text = templates.tool_option_text(tool.description) if tool.description.strip() else ""
    return text or templates.upper_first(tool.intent) + "."


def authorized_question(tool: ToolSpec) -> BallotQuestion:
    """``T.authorized`` (``T_AUTH`` Noul)."""
    return BallotQuestion(qid=tool_qid(tool.id, "authorized"), family="authorized", tool=tool.name,
                          primitive="noul", instructions=templates.auth_instructions(tool.intent),
                          criteria=dict(templates.AUTH_CRITERIA))  # fmt: skip


def done_after_question(tool: ToolSpec) -> BallotQuestion:
    """``T.done_after`` (``T_DONE_AFTER`` Noul, loop mode)."""
    return BallotQuestion(qid=tool_qid(tool.id, "done_after"), family="done_after", tool=tool.name,
                          primitive="noul", instructions=templates.done_after_instructions(tool.intent))  # fmt: skip


def reply_question(options: Sequence[Mapping[str, str]], label_max: int = 64) -> BallotQuestion:
    """The ``reply`` Choice of a resume round: the card options (``cancel``/``other`` are covered by the
    ``CANCEL``/``OTHER`` sentinels) in card order."""
    taken: list[str] = []
    entries: list[BallotOption] = []
    for n, option in enumerate(options, start=1):
        if option["id"] in ("cancel", "other"):
            continue
        label = make_label(option["text"], slot="option", n=n, label_max=label_max, taken=taken)
        taken.append(label)
        text = None if label == option["text"] else with_full_value(None, option["text"])
        entries.append(BallotOption(label=label, value=option["id"], text=text))
    sentinels = {
        OTHER: SentinelSpec(decodes_to="other", text=templates.REPLY_SENTINEL_TEXT[OTHER]),
        CANCEL: SentinelSpec(decodes_to="cancel", text=templates.REPLY_SENTINEL_TEXT[CANCEL]),
    }
    return BallotQuestion(qid="reply", family="reply", primitive="choice", instructions=templates.reply_instructions(),
                          options=entries, sentinels=sentinels)  # fmt: skip


def joint_questions(tool: ToolSpec, pools: Mapping[PoolKey, Pool], rc: ResolveContext) -> list[BallotQuestion]:
    """``T.joint[.G]`` Choices over code-enumerated combinations of anchored candidates (§5.2).

    Per group slot, the anchored candidates (all candidates when none is anchored) form a product; combinations
    that definitely violate a constraint among the group's slots are dropped. Groups with fewer than two slots, or
    more than ``joint_max`` combinations, are skipped (J is then not asked).
    """
    joint_max = tool.joint_max or rc.policy.pools.joint_max
    questions = []
    for gi, group in enumerate(tool.groups):
        options = _joint_options(tool, group, pools, rc, joint_max)
        if not options:
            continue
        qid = tool_qid(tool.id, "joint") if len(tool.groups) == 1 else tool_qid(tool.id, "joint", gi)
        questions.append(BallotQuestion(
            qid=qid, family="joint", tool=tool.name, primitive="choice",
            instructions=templates.joint_instructions(tool.intent), options=options, meta={"group": list(group)},
            sentinels={NONE_OF_THESE: SentinelSpec(decodes_to="uncovered", text=templates.JOINT_NONE_TEXT)},
        ))  # fmt: skip
    return questions


def _joint_options(
    tool: ToolSpec, group: Sequence[str], pools: Mapping[PoolKey, Pool], rc: ResolveContext, joint_max: int
) -> list[BallotOption]:
    per_slot: list[tuple[SlotSpec, list[Candidate]]] = []
    for name in group:
        slot = tool.slot(name)
        pool = pools.get((tool.name, slot.path))
        if pool is None or not pool.candidates:
            return []
        anchored = [c for c in pool.candidates if is_anchored(c)]
        per_slot.append((slot, anchored or list(pool.candidates)))
    if len(per_slot) < 2 or math.prod(len(cands) for _, cands in per_slot) > joint_max:
        return []
    names = {slot.name for slot, _ in per_slot}
    constraints = [c for c in tool.constraints if c.check_name is None and c.slots and c.slots <= names]
    taken: list[str] = []
    options: list[BallotOption] = []
    for n, combo in enumerate(itertools.product(*(cands for _, cands in per_slot)), start=1):
        args = {slot.name: c.value for (slot, _), c in zip(per_slot, combo, strict=True)}
        attrs = {slot.name: c.attrs for (slot, _), c in zip(per_slot, combo, strict=True)}
        env = ConstraintContext(attrs=attrs, now=rc.now, context=rc.ctx)
        if any(c.check(args, env) is False for c in constraints):
            continue
        bindings = {slot.name: Binding.of(c) for (slot, _), c in zip(per_slot, combo, strict=True)}
        shown = render_call(tool, bindings, only=[slot.name for slot, _ in per_slot])
        label = make_label(shown, slot="joint", n=n, label_max=rc.limits.label_max, taken=taken)
        taken.append(label)
        parts = "; ".join(f"{templates.upper_first(slot.noun)}: {c.label}" for (slot, _), c in
                          zip(per_slot, combo, strict=True))  # fmt: skip
        text = parts if label == shown else with_full_value(parts, shown)
        options.append(BallotOption(label=label, value=args, text=text[: rc.limits.desc_max],
                                    prov={"combination": {slot.name: c.label for (slot, _), c in
                                                          zip(per_slot, combo, strict=True)}}))  # fmt: skip
    return canonical_order(options)


class _Builder:
    """Builds pools, viability records and questions for one round (re-run by the budget cuts)."""

    def __init__(
        self,
        catalog: Catalog,
        rc: ResolveContext,
        considered: Sequence[ToolSpec],
        choice: str,
        speculate_only: Sequence[str] | None,
        extra: Mapping[PoolKey, Sequence[Candidate]],
        *,
        forced: Sequence[str] = (),
        loop: bool = False,
    ) -> None:
        self.catalog = catalog
        self.loop = loop
        self.rc = rc
        self.considered = list(considered)
        self.choice = choice
        self.speculate_only = set(speculate_only) if speculate_only is not None else None
        self.forced = set(forced)
        self.extra = extra
        self.pools: dict[PoolKey, Pool] = {}
        self.probe_only: set[str] = set()

    def build(self, policy: Policy) -> tuple[list[BallotQuestion], list[ToolViability]]:
        """Pools → viability → speculation → the ordered question list (§3.5.5)."""
        self.rc.policy = policy
        self.pools = {}
        self.probe_only = set()
        records: list[ToolViability] = []
        speculated: list[ToolSpec] = []
        for tool in self.considered:
            self._build_pools(tool)
            viable = viability(tool, self.pools, self.rc.ctx)
            spec = self._speculate(tool, viable)
            records.append(ToolViability(name=tool.name, tier=tool.tier, viable=viable, speculated=spec))
            if spec:
                speculated.append(tool)
            if not any(p.evidence_backed for (t, _), p in self.pools.items() if t == tool.name):
                self.probe_only.add(tool.name)
        questions: list[BallotQuestion] = []
        if self.considered and self.choice != "named":
            loop = self.loop or is_loop(self.rc.ctx, self.rc.mode)
            done = bool(self.rc.ctx.all_observations())
            questions.append(tool_question(self.considered, choice=self.choice, loop=loop, done=done,
                                           label_max=self.rc.limits.label_max))  # fmt: skip
        for tool in tool_labels(speculated, self.rc.limits.label_max).values():
            questions += self._tool_questions(tool)
        return questions, records

    def _speculate(self, tool: ToolSpec, viable: str) -> bool:
        """Viable tools are speculated; ``x-jev.speculate`` overrides (``never`` yields to an explicit request: a
        named ``tool_choice`` or ``speculate_only``); a resume round's pending tool always is."""
        if tool.name in self.forced:
            return True
        if self.speculate_only is not None:
            return tool.name in self.speculate_only and (viable == "ok" or tool.speculate == "always")
        if tool.speculate == "never" and self.choice != "named":
            return False
        return tool.speculate == "always" or viable == "ok"

    def _build_pools(self, tool: ToolSpec) -> None:
        """Pools in schema order; a slot whose ``default_from`` names a sibling is built after that sibling, with the
        sibling candidates' attribute values as its context-preferred values (e.g. account currencies)."""
        deferred = [s for s in tool.slots if sibling_source(tool, s) is not None]
        for slot in [s for s in tool.slots if s not in deferred] + deferred:
            source = sibling_source(tool, slot)
            if source is not None:
                self._prefer(tool, slot, source)
            pool = get_resolver(slot.kind).pool(tool, slot, self.rc)
            extra = self.extra.get((tool.name, slot.path))
            if extra:
                pool = inject_candidates(pool, extra, slot, self.rc.limits.label_max)
            self.pools[(tool.name, slot.path)] = pool

    def _prefer(self, tool: ToolSpec, slot: SlotSpec, source: tuple[str, str]) -> None:
        name, attr = source
        pool = self.pools.get((tool.name, tool.slot(name).path))
        values = [c.attrs[attr] for c in pool.candidates if attr in c.attrs] if pool is not None else []
        preferred = dict(self.rc.preferred)
        preferred[self.rc.slot_key(tool, slot)] = list(dict.fromkeys(values))
        self.rc.preferred = preferred

    def _tool_questions(self, tool: ToolSpec) -> list[BallotQuestion]:
        questions: list[BallotQuestion] = []
        if tool.tier >= Tier.WRITE:
            questions.append(authorized_question(tool))
        questions += joint_questions(tool, self.pools, self.rc)
        for slot in tool.slots:
            if not slot.speculate:
                continue
            pool = self.pools[(tool.name, slot.path)]
            questions += get_resolver(slot.kind).questions(tool, slot, pool, self.rc)
        if self.loop or is_loop(self.rc.ctx, self.rc.mode):
            questions.append(done_after_question(tool))
        return questions


def secret_user_fields(catalog: Catalog | None, ctx: Context) -> frozenset[str]:
    """Profile fields never sent to Jev (§3.5.1, §14 "secrets are never sent"), shareable or not: every top-level
    ``user`` field a ``secret`` slot reads through ``default_from: user.<field>…``, and every field whose name is a
    secret name (``password``, ``token``, ``api_key``…, the §3.3.1 row-1 names)."""
    fields = {key for key in ctx.user if key.lower() in SECRET_NAMES}
    for tool in catalog or ():
        for top in tool.slots:
            for slot in top.walk():
                head, _, rest = (slot.default_from or "").partition(".")
                if slot.kind == "secret" and head == "user" and rest:
                    fields.add(rest.split(".", 1)[0])
    return frozenset(fields)


def sibling_source(tool: ToolSpec, slot: SlotSpec) -> tuple[str, str] | None:
    """``(sibling, attr)`` of a ``default_from`` that names another slot of the tool."""
    if not slot.default_from:
        return None
    head, _, attr = slot.default_from.partition(".")
    return (head, attr) if head in tool.slot_names and head != slot.name and attr else None


# --------------------------------------------------------------------------------------------------------------------
# Budget and split (§5.5)
# --------------------------------------------------------------------------------------------------------------------


OPAQUE_ID_CHARS = 5
"""Length of an opaque wire id (``q0001``…, :func:`jevtools.qid.opaque_ids`) for up to 9,999 questions."""


def _question_chars(question: BallotQuestion, limits: Limits) -> int:
    """Characters of one question in a request body: its wire id (the dotted qid, or at least an opaque id's
    length), the question and the separators — never fewer than the sent request carries."""
    qid = question.qid if limits.id_mode == "dotted" else question.qid.ljust(OPAQUE_ID_CHARS)
    return len(canonical_str(qid)) + len(canonical_str(question.to_wire().to_wire())) + 2


def _tokens(chars: int, limits: Limits) -> int:
    return limits.tokens_for_chars(chars)


def _base_chars(state: Any) -> int:
    return len(canonical_str({"model": "", "state": state, "questions": {}}))


def _over_budget(questions: Sequence[BallotQuestion], state: Any, limits: Limits) -> bool:
    chars = _base_chars(state) + sum(_question_chars(q, limits) for q in questions)
    return len(questions) > limits.max_questions or _tokens(chars, limits) > limits.max_tokens


def _apply_budget(
    builder: _Builder,
    questions: list[BallotQuestion],
    tools: list[ToolViability],
    state: Any,
    limits: Limits,
    policy: Policy,
) -> tuple[list[BallotQuestion], list[ToolViability], list[str]]:
    """Question cuts in order while one call would exceed the budget: (1) enum descriptions equal to the member
    name, (2) probe-only speculated tools (``viable = budget``), (3) REF shortlists shrunk to 20. Anything still over
    budget is split by :func:`split_calls`."""
    notes: list[str] = []
    if not _over_budget(questions, state, limits):
        return questions, tools, notes
    questions = [_drop_echo_descriptions(q) for q in questions]
    notes.append("budget: dropped enum descriptions equal to the member name")
    if _over_budget(questions, state, limits):
        questions, tools = _drop_probe_only(questions, tools, builder.probe_only)
        notes.append("budget: dropped probe-only tools")
    if _over_budget(questions, state, limits) and policy.pools.ref_k > REF_K_CUT:
        smaller = policy.model_copy(update={"pools": policy.pools.model_copy(update={"ref_k": REF_K_CUT})})
        questions, tools = builder.build(smaller)
        questions = [_drop_echo_descriptions(q) for q in questions]
        questions, tools = _drop_probe_only(questions, tools, builder.probe_only)
        notes.append(f"budget: REF shortlists cut to {REF_K_CUT}")
    return questions, tools, notes


def _drop_echo_descriptions(question: BallotQuestion) -> BallotQuestion:
    if question.kind != "enum" or not question.options:
        return question
    options = [o.model_copy(update={"text": None}) if isinstance(o.text, str) and o.text.casefold() ==
               o.label.casefold() else o for o in question.options]  # fmt: skip
    return question.model_copy(update={"options": options})


def _drop_probe_only(
    questions: list[BallotQuestion], tools: list[ToolViability], probe_only: set[str]
) -> tuple[list[BallotQuestion], list[ToolViability]]:
    cut = {t.name for t in tools if t.speculated and t.name in probe_only}
    if not cut:
        return questions, tools
    kept = [q for q in questions if q.tool not in cut]
    records = [t.model_copy(update={"viable": "budget", "speculated": False}) if t.name in cut else t for t in tools]
    return kept, records


def family_units(questions: Sequence[BallotQuestion]) -> list[list[BallotQuestion]]:
    """Split units: tool-level questions alone, a slot's questions (same tool and path) together."""
    units: list[list[BallotQuestion]] = []
    for question in questions:
        slot_level = question.tool is not None and bool(question.path)
        previous = units[-1][-1] if units else None
        same_slot = (previous is not None and slot_level and previous.tool == question.tool
                     and previous.path == question.path)  # fmt: skip
        if same_slot:
            units[-1].append(question)
        else:
            units.append([question])
    return units


def split_calls(questions: Sequence[BallotQuestion], state: Any, limits: Limits) -> list[list[str]]:
    """Pack family units, in plan order, into calls under ``max_questions`` and ``max_tokens`` (identical state in
    every call, ``tool`` in call 0)."""
    base = _base_chars(state)
    calls: list[list[str]] = []
    count = chars = 0
    for unit in family_units(questions):
        unit_chars = sum(_question_chars(q, limits) for q in unit)
        fits = calls and count + len(unit) <= limits.max_questions and \
            _tokens(base + chars + unit_chars, limits) <= limits.max_tokens  # fmt: skip
        if not fits:
            calls.append([])
            count = chars = 0
        calls[-1] += [q.qid for q in unit]
        count += len(unit)
        chars += unit_chars
    return calls


__all__ = [
    "REF_K_CUT",
    "PoolKey",
    "RoundPlan",
    "ToolChoice",
    "authorized_question",
    "compile_round",
    "cut_state",
    "pinned_mentions",
    "done_after_question",
    "family_units",
    "followup_ballot",
    "inject_candidates",
    "injected_by_slot",
    "is_anchored",
    "joint_questions",
    "merge_ballots",
    "parse_tool_choice",
    "plan_round",
    "reply_question",
    "secret_user_fields",
    "sibling_source",
    "split_calls",
    "state_tokens",
    "tool_labels",
    "tool_question",
    "viability",
]

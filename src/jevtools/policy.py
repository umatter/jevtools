"""Decision policy: risk tiers, outcomes, the tunable thresholds (spec §3.8, Appendix B) and rule evaluation.

:class:`Policy` holds the configuration. :func:`evaluate` applies the ordered rules P0–P10 (§3.8.2) to a
:class:`PolicyInput` — a JSON-serializable snapshot of one decoded round (tool distribution, slot shapes and
factors, gates, flags, the final confidence C and what the router can still do: widen, fill, escalate). Keeping the
input a plain document makes the policy pure and lets ``jt.verify`` re-run it from a stored trace.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jevtools._compat import StrEnum, load_toml
from jevtools.canonical import sha256_of

_TIER_RANK = {"read": 0, "write": 1, "external": 2, "critical": 3}


class Tier(StrEnum):
    """Risk tier of a tool (spec §3.3.2, §3.8.3). Ordered: ``read < write < external < critical``."""

    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """0 for read … 3 for critical."""
        return _TIER_RANK[self.value]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return self.rank >= other.rank


class Outcome(StrEnum):
    """Host-visible outcomes (spec §3.8.1)."""

    EXECUTE = "execute"
    CONFIRM = "confirm"
    CLARIFY = "clarify"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"
    REFUSE = "refuse"
    DONE = "done"


class Action(StrEnum):
    """Internal actions that trigger another round and are never returned to the host (spec §3.8.1)."""

    WIDEN = "widen"
    FILL = "fill"
    RESUME = "resume"


Composition = Literal["W", "PI", "L", "J", "MIN_L_J"]
"""How a tier composes its factors into ``C_prior`` (spec §3.7.3)."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolPolicy(_Section):
    """Thresholds of the tool Choice (rules P1, P2, P5)."""

    min_p: float = 0.50
    min_margin: float = 0.20
    pair_cover: float = 0.85
    """Clarify between the top-2 tools if they cover at least this much mass, else escalate."""


class ShapePolicy(_Section):
    """Slot-shape thresholds (rule P7, shape routing §3.8.3, prompts §3.8.4)."""

    out_of_pool: float = 0.30
    """``NONE_OF_THESE`` mass at or above this makes a slot ``out_of_pool``."""
    ambiguous_cover: float = 0.90
    ambiguous_k: int = 4
    flag_band: tuple[float, float] = (0.20, 0.80)
    """Noul dead band: strictly inside it a flag or item is uncertain (``flag_band`` shape)."""
    accept_min: float = 0.50
    cosmetic_floor: float = 0.50
    alt_show_min: float = 0.10


class TierPolicy(_Section):
    """Thresholds and gates of one risk tier (spec §3.8.3)."""

    composition: Composition
    execute: float | Literal["never"] | None = None
    """Execute when ``C ≥ execute``; ``"never"`` for the critical tier (see ``auto_execute``)."""
    confirm: float | None = None
    """Confirm when ``C ≥ confirm``; ``None`` means the tier has no confirm band."""
    authorized: float | None = None
    """Gate: the ``authorized`` Noul must reach this for execute."""
    content_accept: float | None = None
    """Gate: every content slot's accept Noul must reach this for execute."""
    show_alternatives_min: float | None = None
    """Runner-ups at or above this are offered on confirm cards (defaults to ``shapes.alt_show_min``)."""
    require_present: float | None = None
    """Gate: every REF slot's ``present`` Noul must reach this (critical tier, until E2 passes)."""
    auto_execute: float | None = None
    """Opt-in execute threshold for the critical tier; honoured only when certified (spec §11.4)."""


class TiersPolicy(_Section):
    """Per-tier settings, defaults from Appendix B."""

    read: TierPolicy = TierPolicy(composition="W", execute=0.60)
    write: TierPolicy = TierPolicy(composition="PI", execute=0.70, confirm=0.45, authorized=0.80, content_accept=0.70)
    external: TierPolicy = TierPolicy(
        composition="PI", execute=0.80, confirm=0.50, authorized=0.90, content_accept=0.80
    )
    critical: TierPolicy = TierPolicy(
        composition="MIN_L_J",
        execute="never",
        confirm=0.80,
        authorized=0.90,
        show_alternatives_min=0.03,
        require_present=0.80,
    )


class ProbePolicy(_Section):
    """Tiers in which REF slots get ``present`` Nouls and ``rev`` Choices (spec §3.5.3)."""

    present: tuple[Tier, ...] = (Tier.EXTERNAL, Tier.CRITICAL)
    reverse: tuple[Tier, ...] = (Tier.CRITICAL,)


class WidenPolicy(_Section):
    """Coverage rounds (spec §4.6)."""

    max_rounds: int = 2
    page: int = 250
    buckets: int = 2


class PoolPolicy(_Section):
    """Pool and fan-out sizes (spec §3.5.3, §4)."""

    ref_k: int = 40
    text_content_max: int = 4
    text_cosmetic_max: int = 3
    mentions_max: int = 8
    items_max: int = 60
    members_max: int = 40
    joint_max: int = 24


class LoopPolicy(_Section):
    """Agent-loop guards (spec §6.1)."""

    max_steps: int = 6
    max_rounds: int = 12
    max_cost_usd: float = 0.01
    max_llm_calls: int = 2
    done_after: float = 0.80


class BudgetPolicy(_Section):
    """Token budget per Jev call (spec §5.5)."""

    max_tokens_per_call: int = 24_000
    max_state_tokens: int = 16_000
    max_questions_per_call: int = 250
    chars_per_token: float = 3.5


class CertifiedPolicy(_Section):
    """Evidence gathered by ``jevtools tune`` (spec §11.4)."""

    critical_cases: int = 0
    """Labelled critical-tier cases behind the current thresholds; ≥ 3000 unlocks ``auto_execute``."""


CRITICAL_CERTIFICATION_CASES = 3000
"""Rule-of-three sample size needed before the critical tier may auto-execute (spec §11.4)."""


class Policy(BaseModel):
    """The complete, versioned decision policy (spec Appendix B). Immutable; build variants with ``from_dict``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = "jevtools-default-0.1"
    hysteresis: float = 0.03
    shadow: bool = False
    tool: ToolPolicy = ToolPolicy()
    shapes: ShapePolicy = ShapePolicy()
    tiers: TiersPolicy = TiersPolicy()
    probes: ProbePolicy = ProbePolicy()
    widen: WidenPolicy = WidenPolicy()
    pools: PoolPolicy = PoolPolicy()
    loop: LoopPolicy = LoopPolicy()
    budget: BudgetPolicy = BudgetPolicy()
    certified: CertifiedPolicy = CertifiedPolicy()
    notes: dict[str, Any] = Field(default_factory=dict)
    """Free-form metadata (e.g. how thresholds were tuned); part of the hash."""

    @model_validator(mode="after")
    def _check_bands(self) -> Policy:
        low, high = self.shapes.flag_band
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(f"shapes.flag_band must satisfy 0 <= low < high <= 1, got {self.shapes.flag_band}")
        return self

    @classmethod
    def default(cls) -> Policy:
        """The defaults of spec Appendix B."""
        return cls()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Policy:
        """Build a policy from a (possibly partial) mapping deep-merged over the defaults. Unknown keys fail."""
        merged = _deep_merge(cls().model_dump(mode="python"), data)
        return cls.model_validate(merged)

    @classmethod
    def from_toml(cls, path: str | os.PathLike[str]) -> Policy:
        """Load a ``policy.toml`` file; missing keys keep their defaults."""
        return cls.from_toml_text(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_toml_text(cls, text: str) -> Policy:
        """Parse ``policy.toml`` content; missing keys keep their defaults."""
        return cls.from_dict(load_toml(text))

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict in declaration order (what :attr:`sha256` hashes)."""
        return self.model_dump(mode="json")

    @property
    def sha256(self) -> str:
        """``sha256:`` digest of the canonical policy document; cited by every trace."""
        return sha256_of(self.to_dict())

    def tier(self, tier: Tier | str) -> TierPolicy:
        """Settings of one tier."""
        return getattr(self.tiers, Tier(tier).value)  # type: ignore[no-any-return]

    def critical_auto_execute(self) -> float | None:
        """The critical tier's execute threshold, or ``None`` while uncertified (spec §11.4)."""
        tier = self.tiers.critical
        if tier.auto_execute is None or self.certified.critical_cases < CRITICAL_CERTIFICATION_CASES:
            return None
        return tier.auto_execute

    def execute_at(self, tier: Tier | str) -> float | None:
        """Numeric execute threshold of a tier (``None`` when the tier never auto-executes)."""
        tier = Tier(tier)
        if tier is Tier.CRITICAL:
            return self.critical_auto_execute()
        value = self.tier(tier).execute
        return None if value == "never" else value

    def alternatives_min(self, tier: Tier | str) -> float:
        """Minimum runner-up probability shown on a confirm card for this tier."""
        own = self.tier(tier).show_alternatives_min
        return self.shapes.alt_show_min if own is None else own


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# --------------------------------------------------------------------------------------------------------------------
# Rule evaluation (§3.8.2, §3.8.3)
# --------------------------------------------------------------------------------------------------------------------

RULE_FAIL_CLOSED = "P0.backend.fail_closed"
RULE_NO_TOOL = "P1.tool.no_tool"
RULE_UNSUPPORTED = "P2.tool.unsupported"
RULE_REFUSE = "P3.safety.refuse"
RULE_NOT_AUTHORIZED = "P4.safety.not_authorized"
RULE_TOOL_AMBIGUOUS = "P5.tool.ambiguous"
RULE_NOT_SPECULATED = "P6.tool.not_speculated"
RULE_SLOT_SHAPE = "P7.slot.shape"
RULE_CONSISTENCY = "P8.consistency"
RULE_LOOP_DONE = "P10.loop.done"
TIER_BANDS: tuple[str, ...] = (
    "execute", "confirm_band", "capped", "confirm_always", "confirmed", "ambiguous", "missing", "diffuse",
)  # fmt: skip
"""Bands of rule P9 (``P9.<tier>.<band>``): the tier comparison, its caps, click confirmations and shape routing."""
RULE_PREFIXES: tuple[str, ...] = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10")

TOOL_SENTINEL_LABELS = frozenset({"NO_TOOL", "UNSUPPORTED", "DONE"})
CONSISTENCY_FLAGS: tuple[str, ...] = ("presence_conflict", "order_sensitive", "joint_disagrees", "infeasible",
                                      "schema_invalid")  # fmt: skip
"""Flags routed by P8 (the spec's three plus decode-time ``infeasible``/``schema_invalid``)."""
UNTRUSTED_CHANNELS = frozenset({"tool_output", "generated"})
TRUSTED_CHANNEL_NAMES = frozenset({"user", "registry", "author"})
_EPS = 1e-9

Ask = Literal["confirm", "menu", "tool_menu", "yes_no", "open", "notice"]
"""Which prompt the router renders for an outcome (§3.8.4)."""


def tier_rule(tier: Tier | str, band: str) -> str:
    """``P9.<tier>.<band>`` (e.g. ``P9.external.confirm_band``)."""
    return f"P9.{Tier(tier).value}.{band}"


class SlotState(BaseModel):
    """What the policy needs to know about one slot of the elected tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    stakes: str = "identity"
    shape: str = "ok"
    factor: float | None = None
    """The slot's confidence factor (``None`` for cosmetic slots and unasked defaults)."""
    bottom: str | None = None
    """The elected bottom (``⊥missing``, ``⊥uncovered``…) when no real value won."""
    top: list[float] = Field(default_factory=list)
    """Masses of the real values, most probable first (for the ambiguous-menu coverage test)."""
    flags: list[str] = Field(default_factory=list)
    channel: str | None = None
    """Channel of the bound value (``None`` when nothing is bound)."""


class PolicyInput(BaseModel):
    """A snapshot of one decoded round, everything :func:`evaluate` reads.

    ``tools`` is the tool distribution over labels (tool names and ``NO_TOOL``/``UNSUPPORTED``/``DONE``); a tool
    fixed by ``tool_choice`` appears alone with ``1.0``. ``chosen`` is ``t*``; ``viable``/``speculated`` its planning
    record. ``C`` is the final confidence of ``t*``. ``widen_ok``/``fill_ok`` name slots for which an internal round
    is still possible; ``escalator`` says whether an Escalator is configured. ``confirmed`` marks a user
    confirmation (a click on ``ok`` or on a complete-call option, §3.8.5).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    failed: bool = False
    tools: dict[str, float] = Field(default_factory=dict)
    chosen: str | None = None
    tier: Tier | None = None
    viable: str = "ok"
    speculated: bool = True
    authorized: float | None = None
    observations: bool = False
    slots: list[SlotState] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    call_map: str | None = None
    C: float | None = None
    present: dict[str, float] = Field(default_factory=dict)
    loop: bool = False
    escalator: bool = False
    widen_ok: list[str] = Field(default_factory=list)
    fill_ok: list[str] = Field(default_factory=list)
    confirm_always: bool = False
    confirmed: bool = False

    def ranked(self) -> list[tuple[str, float]]:
        """Tool labels by probability, most probable first (stable on ties)."""
        return sorted(self.tools.items(), key=lambda kv: -kv[1])

    @property
    def chosen_p(self) -> float:
        """``P(t*)``."""
        return self.tools.get(self.chosen, 0.0) if self.chosen is not None else 0.0

    @property
    def chosen_is_tool(self) -> bool:
        """``t*`` is a real tool (not a sentinel, not undecided)."""
        return self.chosen is not None and self.chosen not in TOOL_SENTINEL_LABELS

    def slot(self, name: str) -> SlotState | None:
        """A slot by name."""
        return next((s for s in self.slots if s.name == name), None)

    def bottleneck(self) -> SlotState | None:
        """``argmin`` over slot factors (first on ties); ``None`` without factor-carrying slots."""
        scored = [s for s in self.slots if s.factor is not None]
        return min(scored, key=lambda s: s.factor if s.factor is not None else 1.0) if scored else None


class PolicyResult(BaseModel):
    """Outcome of :func:`evaluate`: the outcome, the rule id that fired and what to ask.

    ``action`` is an internal round (``widen``/``fill``) the router should try first; ``outcome`` is what happens
    when it cannot. ``caps`` lists what kept the call below execute (``authorized``, ``untrusted_channel``,
    ``content_accept``, ``channels``, ``present``, ``confirm_always``, ``shadow``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: Outcome
    rule: str
    action: Action | None = None
    bottleneck: str | None = None
    shape: str | None = None
    caps: list[str] = Field(default_factory=list)
    reason: str | None = None
    ask: Ask | None = None
    tools: list[str] = Field(default_factory=list)
    """For a tool menu: the two tools offered."""


def evaluate(inp: PolicyInput, policy: Policy | None = None) -> PolicyResult:
    """Apply the ordered rules P0–P10 (§3.8.2) and the tier thresholds (§3.8.3) to one decoded round.

    ``DONE ≥ 0.5`` (P10) is checked with the other tool sentinels, right after P2, because P3–P9 presuppose a tool.
    In shadow mode an ``execute`` is returned as ``confirm`` (cap ``shadow``).
    """
    policy = policy or Policy()
    for rule in (_p0, _p1, _p2, _p10, _p3, _p4, _p5, _p6, _p7, _p8):
        result = rule(inp, policy)
        if result is not None:
            return _shadow(result, policy)
    return _shadow(_p9(inp, policy), policy)


def loop_done(done_after: float | None, policy: Policy | None = None) -> bool:
    """P10's second condition: ``done_after(t*) ≥ loop.done_after`` after a successful execute (§6.1)."""
    policy = policy or Policy()
    return done_after is not None and done_after >= policy.loop.done_after


def _shadow(result: PolicyResult, policy: Policy) -> PolicyResult:
    if policy.shadow and result.outcome is Outcome.EXECUTE:
        return result.model_copy(update={"outcome": Outcome.CONFIRM, "caps": [*result.caps, "shadow"],
                                         "ask": "confirm"})  # fmt: skip
    return result


def _escalate_or(inp: PolicyInput, fallback: Outcome, rule: str, **kw: Any) -> PolicyResult:
    """``escalate`` when an Escalator is configured, else ``fallback``."""
    outcome = Outcome.ESCALATE if inp.escalator else fallback
    if outcome is Outcome.CLARIFY and "ask" not in kw:
        kw["ask"] = "open"
    return PolicyResult(outcome=outcome, rule=rule, **kw)


def _p0(inp: PolicyInput, policy: Policy) -> PolicyResult | None:
    if not inp.failed:
        return None
    return _escalate_or(inp, Outcome.ABSTAIN, RULE_FAIL_CLOSED, reason="jev_unavailable")


def _p1(inp: PolicyInput, policy: Policy) -> PolicyResult | None:
    if inp.chosen is None:
        return PolicyResult(outcome=Outcome.ABSTAIN, rule=RULE_NO_TOOL, reason="no_tool")
    if inp.chosen == "NO_TOOL" and inp.chosen_p >= 0.5:
        return PolicyResult(outcome=Outcome.ABSTAIN, rule=RULE_NO_TOOL, reason="no_tool")
    return None


def _p2(inp: PolicyInput, policy: Policy) -> PolicyResult | None:
    if inp.chosen == "UNSUPPORTED" and inp.chosen_p >= 0.5:
        return _escalate_or(inp, Outcome.ABSTAIN, RULE_UNSUPPORTED, reason="unsupported")
    return None


def _p10(inp: PolicyInput, policy: Policy) -> PolicyResult | None:
    if inp.chosen == "DONE" and inp.chosen_p >= 0.5:
        return PolicyResult(outcome=Outcome.DONE, rule=RULE_LOOP_DONE, reason="done")
    return None


def _p3(inp: PolicyInput, policy: Policy) -> PolicyResult | None:
    if not inp.chosen_is_tool:
        return None
    if not inp.speculated and inp.viable.startswith("channel_blocked:"):
        return PolicyResult(outcome=Outcome.REFUSE, rule=RULE_REFUSE, reason="channel_blocked",
                            bottleneck=inp.viable.partition(":")[2], ask="notice")  # fmt: skip
    if inp.authorized is not None and inp.authorized < 0.5 and inp.observations and inp.chosen_p >= 0.8:
        return PolicyResult(outcome=Outcome.REFUSE, rule=RULE_REFUSE, reason="not_requested_by_user", ask="notice")
    if "channel_violation" in inp.flags or any("channel_violation" in s.flags for s in inp.slots):
        return PolicyResult(outcome=Outcome.REFUSE, rule=RULE_REFUSE, reason="channel_violation", ask="notice")
    return None


def _authorized_missing(inp: PolicyInput) -> bool:
    """A speculated tool of tier ≥ write without an ``authorized`` answer: the planner always asks it there, so a
    missing value is a failed answer (I5: it can never lead to execute)."""
    return inp.authorized is None and inp.speculated and inp.tier is not None and inp.tier >= Tier.WRITE


def _p4(inp: PolicyInput, policy: Policy) -> PolicyResult | None:
    if inp.chosen_is_tool and inp.authorized is not None and inp.authorized < 0.5:
        return PolicyResult(outcome=Outcome.ABSTAIN, rule=RULE_NOT_AUTHORIZED, reason="not_requested")
    if inp.chosen_is_tool and _authorized_missing(inp):
        return PolicyResult(outcome=Outcome.ABSTAIN, rule=RULE_NOT_AUTHORIZED, reason="authorized_missing")
    return None


def _p5(inp: PolicyInput, policy: Policy) -> PolicyResult | None:
    ranked = inp.ranked()
    p1 = ranked[0][1] if ranked else 0.0
    p2 = ranked[1][1] if len(ranked) > 1 else 0.0
    disagrees = "call_map_disagrees" in inp.flags
    if not (p1 < policy.tool.min_p or p1 - p2 < policy.tool.min_margin or disagrees):
        return None
    if disagrees and inp.chosen is not None and inp.call_map is not None:
        pair = [inp.chosen, inp.call_map]
    else:
        pair = [label for label, _ in ranked[:2]]
    menu = len(pair) == 2 and not TOOL_SENTINEL_LABELS.intersection(pair)
    if menu and (disagrees or sum(inp.tools.get(t, 0.0) for t in pair) >= policy.tool.pair_cover):
        reason = "call_map_disagrees" if disagrees else "tool_ambiguous"
        return PolicyResult(outcome=Outcome.CLARIFY, rule=RULE_TOOL_AMBIGUOUS, ask="tool_menu", tools=pair,
                            shape="ambiguous", reason=reason)  # fmt: skip
    return _escalate_or(inp, Outcome.CLARIFY, RULE_TOOL_AMBIGUOUS, reason="tool_ambiguous", shape="diffuse")


def _p6(inp: PolicyInput, policy: Policy) -> PolicyResult | None:
    if not inp.chosen_is_tool or inp.speculated:
        return None
    slot = inp.viable.partition(":")[2] or None
    return PolicyResult(outcome=Outcome.CLARIFY, rule=RULE_NOT_SPECULATED, bottleneck=slot, shape="missing",
                        ask="open", reason=inp.viable)  # fmt: skip


def _p7(inp: PolicyInput, policy: Policy) -> PolicyResult | None:
    slot = next((s for s in inp.slots if s.shape != "ok"), None)
    if slot is None:
        return None
    common: dict[str, Any] = {"rule": RULE_SLOT_SHAPE, "bottleneck": slot.name, "shape": slot.shape}
    if slot.shape == "out_of_pool" and slot.name in inp.widen_ok:
        return PolicyResult(outcome=Outcome.CLARIFY, action=Action.WIDEN, ask="open", **common)
    if slot.shape == "uncovered_text" and slot.name in inp.fill_ok:
        return PolicyResult(outcome=Outcome.CLARIFY, action=Action.FILL, ask="open", **common)
    if slot.shape == "flag_band":
        return PolicyResult(outcome=Outcome.CLARIFY, ask="yes_no", **common)
    return PolicyResult(outcome=Outcome.CLARIFY, ask="open", **common)


def _p8(inp: PolicyInput, policy: Policy) -> PolicyResult | None:
    for slot in inp.slots:
        flag = next((f for f in CONSISTENCY_FLAGS if f in slot.flags), None)
        if flag is not None:
            return PolicyResult(outcome=Outcome.CLARIFY, rule=RULE_CONSISTENCY, bottleneck=slot.name, shape=flag,
                                ask="menu", reason=flag)  # fmt: skip
    flag = next((f for f in CONSISTENCY_FLAGS if f in inp.flags), None)
    if flag is None:
        return None
    weakest = inp.bottleneck()
    return PolicyResult(outcome=Outcome.CLARIFY, rule=RULE_CONSISTENCY, shape=flag, ask="menu", reason=flag,
                        bottleneck=weakest.name if weakest is not None else None)  # fmt: skip


def _clears(c: float, threshold: float | None, h: float) -> bool:
    """``C ≥ τ`` outside the hysteresis band: ``|C − τ| < h`` takes the safer side."""
    return threshold is not None and c - threshold >= h - _EPS


def _caps(inp: PolicyInput, policy: Policy, tier: Tier) -> list[str]:
    """Everything that limits the call to at most CONFIRM (§3.8.2 P9 caps, §3.8.3 gates)."""
    tp = policy.tier(tier)
    caps: list[str] = []
    if tp.authorized is not None and (_authorized_missing(inp) if inp.authorized is None
                                      else inp.authorized < tp.authorized):  # fmt: skip
        caps.append("authorized")
    channels = {s.channel for s in inp.slots if s.channel is not None}
    if tier >= Tier.EXTERNAL and channels & UNTRUSTED_CHANNELS:
        caps.append("untrusted_channel")
    content = [s.factor for s in inp.slots if s.stakes == "content" and s.factor is not None]
    if tp.content_accept is not None and any(f < tp.content_accept for f in content):
        caps.append("content_accept")
    if tier is Tier.CRITICAL and not channels <= TRUSTED_CHANNEL_NAMES:
        caps.append("channels")
    if tp.require_present is not None and any(p < tp.require_present for p in inp.present.values()):
        caps.append("present")
    if inp.confirm_always:
        caps.append("confirm_always")
    return caps


def _p9(inp: PolicyInput, policy: Policy) -> PolicyResult:
    tier = inp.tier or Tier.EXTERNAL
    tp = policy.tier(tier)
    c = inp.C if inp.C is not None else 0.0
    h = policy.hysteresis
    execute_at, confirm_at = policy.execute_at(tier), tp.confirm
    caps = _caps(inp, policy, tier)
    if inp.confirmed and _clears(c, confirm_at if confirm_at is not None else execute_at, h):
        return PolicyResult(outcome=Outcome.EXECUTE, rule=tier_rule(tier, "confirmed"), caps=caps)
    if _clears(c, execute_at, h):
        if not caps:
            return PolicyResult(outcome=Outcome.EXECUTE, rule=tier_rule(tier, "execute"))
        band = "confirm_always" if caps == ["confirm_always"] else "capped"
        return PolicyResult(outcome=Outcome.CONFIRM, rule=tier_rule(tier, band), caps=caps, ask="confirm")
    if _clears(c, confirm_at, h):
        return PolicyResult(outcome=Outcome.CONFIRM, rule=tier_rule(tier, "confirm_band"), caps=caps, ask="confirm")
    return _shape_routing(inp, policy, tier, caps)


def bottleneck_shape(slot: SlotState | None, policy: Policy | None = None) -> tuple[str, int]:
    """Shape of a bottleneck (§3.8.3) and the menu size: ``ambiguous`` when the top ``k ≤ ambiguous_k`` real values
    cover ``≥ ambiguous_cover``; ``missing`` when a bottom was elected; else ``diffuse``."""
    policy = policy or Policy()
    if slot is None:
        return "diffuse", 0
    covered = 0.0
    for k, mass in enumerate(slot.top[: policy.shapes.ambiguous_k], start=1):
        covered += mass
        if covered >= policy.shapes.ambiguous_cover - _EPS:
            return "ambiguous", k
    return ("missing", 0) if slot.bottom is not None else ("diffuse", 0)


def _shape_routing(inp: PolicyInput, policy: Policy, tier: Tier, caps: list[str]) -> PolicyResult:
    """Below confirm (§3.8.3): ambiguous → menu, missing → open, diffuse → escalate or open."""
    slot = inp.bottleneck()
    name = slot.name if slot is not None else None
    shape, k = bottleneck_shape(slot, policy)
    if shape == "ambiguous":
        return PolicyResult(outcome=Outcome.CLARIFY, rule=tier_rule(tier, "ambiguous"), bottleneck=name,
                            shape="ambiguous", caps=caps, ask="menu", reason=f"top{k}")  # fmt: skip
    if shape == "missing":
        return PolicyResult(outcome=Outcome.CLARIFY, rule=tier_rule(tier, "missing"), bottleneck=name,
                            shape="missing", caps=caps, ask="open")  # fmt: skip
    return _escalate_or(inp, Outcome.CLARIFY, tier_rule(tier, "diffuse"), bottleneck=name, shape="diffuse",
                        caps=caps, reason="diffuse")  # fmt: skip


__all__ = [
    "CONSISTENCY_FLAGS",
    "CRITICAL_CERTIFICATION_CASES",
    "RULE_CONSISTENCY",
    "RULE_FAIL_CLOSED",
    "RULE_LOOP_DONE",
    "RULE_NOT_AUTHORIZED",
    "RULE_NOT_SPECULATED",
    "RULE_NO_TOOL",
    "RULE_PREFIXES",
    "RULE_REFUSE",
    "RULE_SLOT_SHAPE",
    "RULE_TOOL_AMBIGUOUS",
    "RULE_UNSUPPORTED",
    "TIER_BANDS",
    "Action",
    "Ask",
    "BudgetPolicy",
    "CertifiedPolicy",
    "Composition",
    "LoopPolicy",
    "Outcome",
    "Policy",
    "PolicyInput",
    "PolicyResult",
    "PoolPolicy",
    "ProbePolicy",
    "ShapePolicy",
    "SlotState",
    "Tier",
    "TierPolicy",
    "TiersPolicy",
    "ToolPolicy",
    "WidenPolicy",
    "bottleneck_shape",
    "evaluate",
    "loop_done",
    "tier_rule",
]

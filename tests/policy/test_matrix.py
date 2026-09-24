"""Policy branch matrix (spec §10.3): synthetic inputs → expected rule id and outcome, hitting every rule P0–P10."""

from __future__ import annotations

from typing import Any

import pytest

from jevtools.policy import (
    RULE_PREFIXES,
    Action,
    Outcome,
    Policy,
    PolicyInput,
    SlotState,
    bottleneck_shape,
    evaluate,
    loop_done,
    tier_rule,
)

TO = {"name": "to", "factor": 0.95, "top": [0.95, 0.03], "channel": "registry"}
BODY = {"name": "body", "stakes": "content", "factor": 0.95, "top": [0.95], "channel": "author"}


def make(**overrides: Any) -> PolicyInput:
    """A healthy external-tier input (``send_email`` at C = 0.9) with ``overrides`` applied."""
    slots = overrides.pop("slots", [TO, BODY])
    base: dict[str, Any] = {
        "tools": {"send_email": 0.95, "NO_TOOL": 0.03, "UNSUPPORTED": 0.02}, "chosen": "send_email",
        "tier": "external", "authorized": 0.95, "C": 0.9, "slots": [SlotState(**s) for s in slots],
    }  # fmt: skip
    base.update(overrides)
    return PolicyInput(**base)


def slot(**kw: Any) -> dict[str, Any]:
    return {**TO, **kw}


CERTIFIED = Policy.from_dict({"tiers": {"critical": {"auto_execute": 0.97}}, "certified": {"critical_cases": 3000}})
UNCERTIFIED = Policy.from_dict({"tiers": {"critical": {"auto_execute": 0.97}}})

CASES: list[tuple[str, PolicyInput, Policy | None, str, Outcome]] = [
    # P0: backend failures fail closed
    ("p0_abstain", make(failed=True), None, "P0.backend.fail_closed", Outcome.ABSTAIN),
    ("p0_escalate", make(failed=True, escalator=True), None, "P0.backend.fail_closed", Outcome.ESCALATE),
    # P1 / P2 / P10: tool sentinels
    ("p1_no_tool", make(tools={"NO_TOOL": 0.96, "get_weather": 0.04}, chosen="NO_TOOL"), None, "P1.tool.no_tool",
     Outcome.ABSTAIN),
    ("p1_no_tools_at_all", make(tools={}, chosen=None), None, "P1.tool.no_tool", Outcome.ABSTAIN),
    ("p2_unsupported", make(tools={"UNSUPPORTED": 0.7, "send_email": 0.3}, chosen="UNSUPPORTED"), None,
     "P2.tool.unsupported", Outcome.ABSTAIN),
    ("p2_escalate", make(tools={"UNSUPPORTED": 0.7, "send_email": 0.3}, chosen="UNSUPPORTED", escalator=True), None,
     "P2.tool.unsupported", Outcome.ESCALATE),
    ("p10_done", make(tools={"DONE": 0.62, "send_email": 0.38}, chosen="DONE", loop=True), None, "P10.loop.done",
     Outcome.DONE),
    # P3: refuse
    ("p3_channel_blocked", make(tools={"transfer_funds": 0.9, "NO_TOOL": 0.1}, chosen="transfer_funds",
                                speculated=False, viable="channel_blocked:amount", tier="critical"), None,
     "P3.safety.refuse", Outcome.REFUSE),
    ("p3_pushed_by_observation", make(authorized=0.2, observations=True), None, "P3.safety.refuse", Outcome.REFUSE),
    ("p3_channel_violation", make(flags=["channel_violation"]), None, "P3.safety.refuse", Outcome.REFUSE),
    # P4
    ("p4_not_authorized", make(authorized=0.3), None, "P4.safety.not_authorized", Outcome.ABSTAIN),
    # P5: tool ambiguity
    ("p5_tool_menu", make(tools={"send_email": 0.45, "create_event": 0.43, "NO_TOOL": 0.12}), None,
     "P5.tool.ambiguous", Outcome.CLARIFY),
    ("p5_margin_open", make(tools={"send_email": 0.52, "create_event": 0.36, "NO_TOOL": 0.12}), None,
     "P5.tool.ambiguous", Outcome.CLARIFY),
    ("p5_diffuse_escalates", make(tools={"send_email": 0.45, "create_event": 0.3, "NO_TOOL": 0.25}, escalator=True),
     None, "P5.tool.ambiguous", Outcome.ESCALATE),
    ("p5_call_map_disagrees", make(flags=["call_map_disagrees"], call_map="create_event"), None, "P5.tool.ambiguous",
     Outcome.CLARIFY),
    # P6
    ("p6_not_speculated", make(speculated=False, viable="empty:title", chosen="create_event",
                               tools={"create_event": 0.9, "NO_TOOL": 0.1}), None, "P6.tool.not_speculated",
     Outcome.CLARIFY),
    # P7: slot shapes
    ("p7_missing", make(slots=[slot(shape="missing", factor=0.8, bottom="⊥missing")]), None, "P7.slot.shape",
     Outcome.CLARIFY),
    ("p7_out_of_pool_widen", make(slots=[slot(shape="out_of_pool")], widen_ok=["to"]), None, "P7.slot.shape",
     Outcome.CLARIFY),
    ("p7_out_of_pool_exhausted", make(slots=[slot(shape="out_of_pool")]), None, "P7.slot.shape", Outcome.CLARIFY),
    ("p7_uncovered_text_fill", make(slots=[{**BODY, "shape": "uncovered_text"}], fill_ok=["body"]), None,
     "P7.slot.shape", Outcome.CLARIFY),
    ("p7_flag_band", make(slots=[slot(shape="flag_band")]), None, "P7.slot.shape", Outcome.CLARIFY),
    # P8: consistency
    ("p8_presence_conflict", make(slots=[slot(flags=["presence_conflict"])]), None, "P8.consistency",
     Outcome.CLARIFY),
    ("p8_order_sensitive", make(slots=[slot(flags=["order_sensitive"])]), None, "P8.consistency", Outcome.CLARIFY),
    ("p8_joint_disagrees", make(flags=["joint_disagrees"]), None, "P8.consistency", Outcome.CLARIFY),
    ("p8_infeasible", make(flags=["infeasible"]), None, "P8.consistency", Outcome.CLARIFY),
    # P9: tiers, hysteresis, caps
    ("p9_read_execute", make(tier="read", authorized=None, C=0.7), None, "P9.read.execute", Outcome.EXECUTE),
    ("p9_external_execute", make(C=0.9), None, "P9.external.execute", Outcome.EXECUTE),
    ("p9_external_confirm_band", make(C=0.714), None, "P9.external.confirm_band", Outcome.CONFIRM),
    ("p9_hysteresis_above_execute", make(C=0.82), None, "P9.external.confirm_band", Outcome.CONFIRM),
    ("p9_hysteresis_below_execute", make(C=0.78), None, "P9.external.confirm_band", Outcome.CONFIRM),
    ("p9_hysteresis_above_confirm", make(C=0.52, slots=[slot(factor=0.52, top=[0.52, 0.44])]), None,
     "P9.external.ambiguous", Outcome.CLARIFY),
    ("p9_hysteresis_below_confirm", make(C=0.48, slots=[slot(factor=0.48, top=[0.48, 0.44])]), None,
     "P9.external.ambiguous", Outcome.CLARIFY),
    ("p9_write_hysteresis_edge", make(tier="write", C=0.73, authorized=0.95), None, "P9.write.execute",
     Outcome.EXECUTE),
    ("p9_cap_authorized", make(authorized=0.85), None, "P9.external.capped", Outcome.CONFIRM),
    ("p9_cap_tool_output", make(slots=[TO, {**BODY, "channel": "tool_output"}]), None, "P9.external.capped",
     Outcome.CONFIRM),
    ("p9_cap_content_accept", make(slots=[TO, {**BODY, "factor": 0.75}]), None, "P9.external.capped",
     Outcome.CONFIRM),
    ("p9_confirm_always", make(tier="read", authorized=None, C=0.95, confirm_always=True), None,
     "P9.read.confirm_always", Outcome.CONFIRM),
    ("p9_critical_never", make(tier="critical", C=0.99, tools={"transfer_funds": 0.98, "NO_TOOL": 0.02},
                               chosen="transfer_funds"), None, "P9.critical.confirm_band", Outcome.CONFIRM),
    ("p9_critical_uncertified_opt_in", make(tier="critical", C=0.99), UNCERTIFIED, "P9.critical.confirm_band",
     Outcome.CONFIRM),
    ("p9_critical_certified", make(tier="critical", C=1.0, present={"to": 0.95}), CERTIFIED,
     "P9.critical.execute", Outcome.EXECUTE),
    ("p9_critical_certified_present_cap", make(tier="critical", C=1.0, present={"to": 0.6}), CERTIFIED,
     "P9.critical.capped", Outcome.CONFIRM),
    ("p9_critical_history_channel_cap", make(tier="critical", C=1.0, slots=[slot(channel="history")]), CERTIFIED,
     "P9.critical.capped", Outcome.CONFIRM),
    ("p9_click_confirmation", make(C=0.688, confirmed=True), None, "P9.external.confirmed", Outcome.EXECUTE),
    ("p9_critical_ok_click", make(tier="critical", C=0.84, confirmed=True), None, "P9.critical.confirmed",
     Outcome.EXECUTE),
    ("p9_click_below_confirm", make(C=0.4, confirmed=True, slots=[slot(factor=0.4, top=[0.4, 0.3, 0.1])]), None,
     "P9.external.diffuse", Outcome.CLARIFY),
    ("p9_shape_ambiguous", make(C=0.39, slots=[slot(factor=0.47, top=[0.47, 0.41, 0.06])]), None,
     "P9.external.ambiguous", Outcome.CLARIFY),
    ("p9_shape_missing", make(C=0.3, slots=[slot(factor=0.3, top=[0.3], bottom="⊥uncovered")]), None,
     "P9.external.missing", Outcome.CLARIFY),
    ("p9_shape_diffuse", make(C=0.3, slots=[slot(factor=0.3, top=[0.3, 0.2, 0.2, 0.1])]), None,
     "P9.external.diffuse", Outcome.CLARIFY),
    ("p9_shape_diffuse_escalates", make(C=0.3, escalator=True, slots=[slot(factor=0.3, top=[0.3, 0.2, 0.2, 0.1])]),
     None, "P9.external.diffuse", Outcome.ESCALATE),
    ("p9_read_below_execute", make(tier="read", authorized=None, C=0.5, slots=[slot(factor=0.5, top=[0.5, 0.45])]),
     None, "P9.read.ambiguous", Outcome.CLARIFY),
]  # fmt: skip


@pytest.mark.parametrize(("name", "inp", "policy", "rule", "outcome"), CASES, ids=[c[0] for c in CASES])
def test_policy_matrix(name: str, inp: PolicyInput, policy: Policy | None, rule: str, outcome: Outcome) -> None:
    result = evaluate(inp, policy)
    assert (result.rule, result.outcome) == (rule, outcome)


def test_every_rule_fires() -> None:
    fired = {evaluate(inp, policy).rule.split(".")[0] for _, inp, policy, _, _ in CASES}
    assert fired == set(RULE_PREFIXES)


def test_internal_actions_and_prompts() -> None:
    widen = evaluate(make(slots=[slot(shape="out_of_pool")], widen_ok=["to"]))
    assert widen.action is Action.WIDEN and widen.bottleneck == "to" and widen.shape == "out_of_pool"
    exhausted = evaluate(make(slots=[slot(shape="out_of_pool")]))
    assert exhausted.action is None and exhausted.ask == "open"
    fill = evaluate(make(slots=[{**BODY, "shape": "uncovered_text"}], fill_ok=["body"]))
    assert fill.action is Action.FILL and fill.bottleneck == "body"
    assert evaluate(make(slots=[slot(shape="flag_band")])).ask == "yes_no"
    menu = evaluate(make(tools={"send_email": 0.45, "create_event": 0.43, "NO_TOOL": 0.12}))
    assert menu.ask == "tool_menu" and menu.tools == ["send_email", "create_event"]
    forced = evaluate(make(flags=["call_map_disagrees"], call_map="create_event"))
    assert forced.ask == "tool_menu" and forced.tools == ["send_email", "create_event"]
    assert evaluate(make(tools={"send_email": 0.45, "NO_TOOL": 0.44, "x": 0.11})).ask == "open"  # sentinel pair
    assert evaluate(make(C=0.39, slots=[slot(factor=0.47, top=[0.47, 0.41, 0.06])])).reason == "top3"


def test_caps_are_recorded() -> None:
    capped = evaluate(make(authorized=0.85, slots=[TO, {**BODY, "channel": "generated", "factor": 0.7}]))
    assert capped.caps == ["authorized", "untrusted_channel", "content_accept"]
    assert evaluate(make(C=0.714)).caps == []


def test_shadow_mode_never_executes() -> None:
    result = evaluate(make(C=0.95), Policy.from_dict({"shadow": True}))
    assert result.outcome is Outcome.CONFIRM and result.rule == "P9.external.execute" and "shadow" in result.caps


def test_hysteresis_width_is_configurable() -> None:
    wide = Policy.from_dict({"hysteresis": 0.1})
    assert evaluate(make(C=0.88), wide).rule == "P9.external.confirm_band"
    assert evaluate(make(C=0.88)).rule == "P9.external.execute"


def test_helpers() -> None:
    assert tier_rule("external", "confirm_band") == "P9.external.confirm_band"
    assert loop_done(0.95) and not loop_done(0.5) and not loop_done(None)
    assert bottleneck_shape(None) == ("diffuse", 0)
    assert bottleneck_shape(SlotState(name="x", top=[0.95])) == ("ambiguous", 1)
    assert make().bottleneck() is not None and make(slots=[]).bottleneck() is None
    assert PolicyInput.model_validate(make().model_dump(mode="json")) == make()

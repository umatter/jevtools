"""03 · A critical transfer: confirm card, TOCTOU revalidation, idempotency, ``jt.verify`` (spec §13 R3).

"Move 250 CHF from my savings to checking" with ``transfer_funds`` (``x-jev.risk: critical``):

- channels: the accounts come from the ``accounts`` registry (sent whole: four rows; balances never reach Jev), the
  amount and currency from the user's words;
- the **joint** Choice over ordered account pairs, ``present``/``rev`` probes on both accounts, and the critical
  composition ``C = min(L, J)``;
- **confirm** (a critical call never auto-executes) with the runner-up account as an ``… instead`` alternative;
- the click executes with an **idempotency key**; ``jt.verify`` replays the trace with the model out of the loop;
- **TOCTOU**: if the Savings balance drops to 100 CHF before the click, the constraint
  ``amount <= from_account.balance`` fails at resume time, nothing executes, and the router re-plans.

Run: ``python examples/03_transfer_confirm.py [--backend scripted|sim|live]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import _show

import jevtools as jt
from jevtools.demo import scenario, scripts


def main(argv: Sequence[str] | None = None) -> dict[str, jt.Decision]:
    args = _show.parse_args(__doc__, argv)
    _show.header("03 · critical transfer: confirm, TOCTOU, idempotency, verify (R3)", args.backend, ["R3"])
    results: dict[str, jt.Decision] = {}
    messages = scenario.scenario_messages(scripts.R3_REQUEST)

    _show.step(f'R3 "{scripts.R3_REQUEST}"   (spec [I]: confirm, L = 0.84, J = 0.92, C = min(L, J) = 0.84)')
    router = scenario.demo_router(_show.backend(args.backend, "R3"))
    tool = router.catalog.get("transfer_funds")
    _show.kv("tool", f"transfer_funds · tier {tool.tier.value} · slots " + ", ".join(
        f"{s.name} ({s.kind})" for s in tool.slots))  # fmt: skip
    d = router.decide(messages)
    _show.decision(d)
    joint = joint_options(d)
    if joint:
        _show.kv("joint", f"{len(joint)} options: " + " | ".join(joint))
    _show.verified(d, router, router.context_for(messages))
    results["confirm"] = d

    option = _show.first_option(d)
    if d.pending_id is not None and option is not None:
        _show.step(f"click [{option}]: revalidate (both accounts exist, balance ≥ amount), then execute")
        done = router.resume(d.pending_id, selection=option)
        _show.decision(done, slots=False)
        _show.verified(done, router, router.context_for(messages))
        results["confirmed"] = done

    _show.step("TOCTOU: the same card, but Savings drops to 100.00 CHF before the click")
    router = scenario.demo_router(_show.backend(args.backend, "R3"))
    d = router.decide(messages)
    results["toctou_card"] = d
    option = _show.first_option(d)
    if d.pending_id is None or option is None:
        _show.note(f"no pending card to click ({d.outcome.value})")
        return results
    contacts, _, files = scenario.default_sources()
    drained = scenario.scenario_context(
        sources=[contacts, scenario.accounts(scenario.account_rows(acc_7731=100.0)), files]
    )
    after = router.resume(d.pending_id, selection=option, context=drained)
    _show.decision(after, slots=False)
    executed = "yes" if after.tool_calls else "no"
    toctou = [text for text in getattr(after.trace, "notes", None) or [] if text.startswith("TOCTOU")]
    why = f" — {toctou[0]} (re-checked against fresh sources at click time)" if toctou else ""
    _show.note(f"executed: {executed}{why}")
    results["toctou"] = after
    return results


def joint_options(d: jt.Decision) -> list[str]:
    """The labels of the ``transfer_funds.joint`` Choice as sent to Jev."""
    for record in getattr(d.trace, "rounds", None) or []:
        for call in record.calls:
            question = ((call.request or {}).get("questions") or {}).get("transfer_funds.joint")
            if question is not None:
                return list(question.get("criteria") or {})
    return []


if __name__ == "__main__":
    main()

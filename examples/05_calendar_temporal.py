"""05 · A calendar event: temporal readings, anchored attendees, confirm with an alternative (spec §13 R5).

"Book a 45 min sync with Bob and Carol next Tuesday at 3pm":

- **every temporal reading** of "next Tuesday at 3pm" is a candidate (the coming Tuesday and the one after), and the
  duration comes from "45 min";
- **anchored attendees**: one mention Choice per name the user wrote ("Bob" → Bob Meier or Robert Brown, "Carol" →
  Carol Liu or Caroline Weber), each with ``EXCLUDE``, plus a ``more`` Noul for invitees not named;
- the **invitee rule** raises ``create_event`` from write to **external** (it notifies people);
- **confirm** with a "Tue 6 Oct … instead" alternative; clicking the alternative executes with that start;
- the compiled Jev request is printed as JSON (the §13.5 listing).

Run: ``python examples/05_calendar_temporal.py [--backend scripted|sim|live]``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import _show

from jevtools.demo import scenario, scripts


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _show.parse_args(__doc__, argv)
    _show.header("05 · calendar event: temporal readings, anchored attendees, alternative (R5)", args.backend,
                 ["R5"])  # fmt: skip
    messages = scenario.scenario_messages(scripts.R5_REQUEST)
    router = scenario.demo_router(_show.backend(args.backend, "R5"))
    tool = router.catalog.get("create_event")
    _show.kv("tool", f"create_event · tier {tool.tier.value} (write verb 'create' + invitees from `contacts`)",
             indent=0)  # fmt: skip

    _show.step("the compiled round (no network): what Jev will be asked")
    ballot = router.compile(messages)
    request = ballot.to_requests(router.model)[0].to_wire()
    first = True
    for qid, question in request["questions"].items():
        if not qid.startswith("create_event."):
            continue
        name = qid.removeprefix("create_event.").ljust(17)
        criteria = question.get("criteria")
        instructions = question.get("instructions")
        if question["type"] == "choice" and isinstance(criteria, dict):
            real = {k: v for k, v in criteria.items() if k not in ("NOT_STATED", "NONE_OF_THESE", "EXCLUDE")}
            sentinels = ", ".join(k for k in criteria if k not in real)
            lines = [f"{name}Choice · {len(real)} candidates + {sentinels}"]
            lines += [f"{'':17}  {label} — {text}" for label, text in real.items()]
        elif isinstance(instructions, dict) and "candidate" in instructions:
            lines = [f"{name}accept Noul · {instructions['candidate']!r}"]
        else:
            lines = [f"{name}Noul"]
        for line in lines:
            _show.kv("asks" if first else "", line, wrap=False)
            first = False
    compact = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
    _show.kv("request", f"{len(request['questions'])} questions, {len(compact):,} characters compact "
                        f"(spec §13.5: 14 questions, 6,003 characters)")  # fmt: skip

    _show.step(f'R5 "{scripts.R5_REQUEST}"   (spec [I]: confirm, PI = 0.550)')
    d = router.decide(messages)
    _show.decision(d)
    results: dict[str, Any] = {"request": request, "confirm": d}

    option = _show.first_option(d, ("alt:start:1", "ok"))
    if d.pending_id is not None and option is not None:
        _show.step(f"click [{option}]: the alternative start binds with p = 1 and counts as the confirmation")
        done = router.resume(d.pending_id, selection=option)
        _show.decision(done, slots=False)
        results["clicked"] = done

    _show.step("the Jev request of round 1 as JSON (the §13.5 listing)")
    for line in _show.pretty_json(request, width=_show.WIDTH - 2).splitlines():
        _show.out("  " + line)
    return results


if __name__ == "__main__":
    main()

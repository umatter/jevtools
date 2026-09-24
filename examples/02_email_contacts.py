"""02 · E-mail a contact: confirm, clarify, click and free-text resumes (spec §13 R2).

"Email Anna that I'll be 10 minutes late" against a 500-row ``contacts`` registry (three Annas), external tier:

- **with history** (the previous turn mentions a 14:30 review with Anna Keller): the ``authorized`` gate, the ``to``
  Choice over the three Annas with its ``present`` probe, accept-Nouls for the subject and for the body, whose
  greeting is late-bound to the recipient's first name → **confirm**; ``resume(selection="ok")`` executes without
  another Jev call;
- **without history**: Keller .47 / Rossi .41 → ``to`` is ambiguous → **clarify** with a menu of complete calls
  (each option re-binds the greeting); a click binds ``to`` with p = 1 and counts as the confirmation → execute;
- **free-text reply** "the gmail one please" to the menu: one resume round whose ``reply`` Choice maps the text to
  menu option 2 (Anna Rossi).

Run: ``python examples/02_email_contacts.py [--backend scripted|sim|live]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import _show

import jevtools as jt
from jevtools.demo import scenario, scripts


def main(argv: Sequence[str] | None = None) -> dict[str, jt.Decision]:
    args = _show.parse_args(__doc__, argv)
    _show.header("02 · e-mail a contact: confirm, clarify, resume (R2)", args.backend,
                 ["R2", "R2-no-history", "R2-free-text"])  # fmt: skip
    results: dict[str, jt.Decision] = {}

    _show.step(f'R2 with history "{scripts.R2_REQUEST}"   (spec [I]: confirm, PI = 0.714)')
    router = scenario.demo_router(_show.backend(args.backend, "R2"))
    d = router.decide(scenario.scenario_messages(scripts.R2_REQUEST, history=True))
    _show.decision(d)
    results["with_history"] = d
    results["with_history_click"] = click(router, d, "ok")

    _show.step("R2 without history   (spec [I]: clarify menu, PI = 0.39; the click executes)")
    router = scenario.demo_router(_show.backend(args.backend, "R2-no-history"))
    d = router.decide(scenario.scenario_messages(scripts.R2_REQUEST))
    _show.decision(d)
    results["no_history"] = d
    results["no_history_click"] = click(router, d, "pick:to:0")

    _show.step(f'R2 without history, then the free-text reply "{scripts.R2_REPLY}"   (one resume round)')
    router = scenario.demo_router(_show.backend(args.backend, "R2-free-text"))
    d = router.decide(scenario.scenario_messages(scripts.R2_REQUEST))
    if d.pending_id is not None:
        _show.kv("prompt", f"{d.outcome.value}: {_show.short(d.prompt.text if d.prompt else '', 100)}")
        reply = router.resume(d.pending_id, reply=scripts.R2_REPLY)
        _show.kv("reply", repr(scripts.R2_REPLY))
        _show.decision(reply)
        results["free_text"] = reply
    else:
        _show.decision(d)
        results["free_text"] = d
    return results


def click(router: jt.Router, d: jt.Decision, prefer: str) -> jt.Decision:
    """Click ``prefer`` (or the first option) on a pending prompt and show the resumed decision."""
    option = _show.first_option(d, (prefer, "ok"))
    if d.pending_id is None or option is None:
        return d
    _show.step(f"click [{option}]   (a click is not a Jev call)")
    done = router.resume(d.pending_id, selection=option)
    _show.decision(done, slots=False)
    return done


if __name__ == "__main__":
    main()

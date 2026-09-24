"""04 · Choosing one file out of 3,000: shortlist, widen, hierarchy (spec §13 R4).

"Open the config file for the payments service" against a ``FileIndex`` of 3,000 generated workspace paths:

- retrieval (BM25 over path tokens with synonyms: config ~ yaml/toml/settings, payments ~ payment/pay) keeps the top
  K = 40 as the options of one ``read_file.path`` Choice (+ ``NOT_STATED``/``NONE_OF_THESE``) → **execute** at
  W = 0.71, the runner-up kept as an alternative;
- a **miss** (``NONE_OF_THESE`` 0.4 ≥ 0.30): round 2 widens to two buckets over results 41–540 plus a directory
  group Choice; round 3 lists the chosen directory (hierarchy); nothing fits → **clarify** with an open question;
- a miss found in a widen bucket: round 2 finds the file → **execute**.

The two miss cases are scripted answers, so ``--backend sim|live`` runs only the first request.

Run: ``python examples/04_file_shortlist_widen.py [--backend scripted|sim|live]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import _show

import jevtools as jt
from jevtools.demo import scenario, scripts


def main(argv: Sequence[str] | None = None) -> dict[str, jt.Decision]:
    args = _show.parse_args(__doc__, argv)
    _show.header("04 · one file out of 3,000: shortlist, widen, hierarchy (R4)", args.backend,
                 ["R4", "R4-miss", "R4-found-in-bucket"])  # fmt: skip
    results: dict[str, jt.Decision] = {}
    messages = scenario.scenario_messages(scripts.R4_REQUEST)
    files = scenario.default_sources()[2]
    _show.kv("files", f"{len(files):,} paths in the `files` index (synthetic)", indent=0)

    _show.step(f'R4 "{scripts.R4_REQUEST}"   (spec [I]: execute app.yaml, W = 0.71)')
    router = scenario.demo_router(_show.backend(args.backend, "R4"))
    d = router.decide(messages)
    _show.decision(d)
    shortlist = [label for label in options(d, "read_file.path") if "/" in label]
    if shortlist:
        payments = [label for label in shortlist if "/payments/" in label]
        _show.kv("shortlist", f"K = {len(shortlist)} of {len(files):,} paths (+ NOT_STATED, NONE_OF_THESE), "
                 f"{len(payments)} under services/payments/: " + ", ".join(payments))  # fmt: skip
    results["hit"] = d

    _show.step("R4 miss: NONE_OF_THESE 0.4 → widen (2 buckets + directory group) → hierarchy → clarify(open)")
    if args.backend != "scripted":
        _show.note("skipped: the miss is a scripted answer (run with --backend scripted)")
        return results
    router = scenario.demo_router(_show.backend(args.backend, "R4-miss"))
    d = router.decide(messages)
    _show.decision(d, slots=False)
    for number, _, _ in _show.rounds(d)[1:]:
        for qid, labels in round_options(d, number).items():
            shown = ", ".join(labels[:3]) + (", …" if len(labels) > 3 else "")
            _show.kv(f"round {number}", f"{qid}: {len(labels)} options — {shown}", wrap=False)
    results["miss"] = d

    _show.step("R4 miss, but the file is the first option of widen bucket 0 → execute in round 2")
    router = scenario.demo_router(_show.backend(args.backend, "R4-found-in-bucket"))
    d = router.decide(messages)
    _show.decision(d, slots=False)
    results["bucket_hit"] = d
    return results


def round_options(d: jt.Decision, number: int) -> dict[str, list[str]]:
    """``{qid: option labels}`` of the Choices sent in round ``number``."""
    found: dict[str, list[str]] = {}
    for record in getattr(d.trace, "rounds", None) or []:
        if record.round != number:
            continue
        for call in record.calls:
            for qid, question in ((call.request or {}).get("questions") or {}).items():
                criteria = question.get("criteria")
                if question.get("type") == "choice" and isinstance(criteria, dict):
                    found[qid] = list(criteria)
    return found


def options(d: jt.Decision, qid: str) -> list[str]:
    """The option labels of ``qid`` in the decision's first round."""
    return round_options(d, 1).get(qid, [])


if __name__ == "__main__":
    main()

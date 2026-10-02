"""Labels from confirm cards (``jevtools.feedback``) and ``jevtools tune --feedback``."""

from __future__ import annotations

import io
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jevtools.cli import main
from jevtools.feedback import FeedbackLog, FeedbackRecord, check_policy, label_of, split
from jevtools.policy import Outcome, Policy
from tests.scenario import scripts
from tests.scenario.fixtures import scenario_context, scenario_messages, scenario_router

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _card(log: FeedbackLog):  # type: ignore[no-untyped-def]
    router, _ = scenario_router(scripts.R2, context=scenario_context(history=True), feedback=log)
    d = router.decide(scenario_messages(scripts.R2_REQUEST, history=True))
    assert d.outcome is Outcome.CONFIRM and d.pending is not None
    return router, d


@pytest.mark.parametrize(("selection", "reply", "label"), [("ok", None, "accepted"), ("cancel", None, "cancelled"),
                                                           (None, "no, send it to her gmail", "edited")])  # fmt: skip
def test_a_resumed_confirm_card_is_logged_with_the_users_verdict(
    tmp_path: Path, selection: str | None, reply: str | None, label: str
) -> None:
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    router, d = _card(log)
    assert log.records() == []  # nothing until the user answers
    assert d.pending is not None
    router.resume(d.pending, selection=selection, reply=reply)
    (row,) = log.records()
    assert (row.decision_id, row.label, row.tool, row.tier) == (d.decision_id, label, "send_email", "external")
    assert d.confidence is not None and row.C == d.confidence.call and not row.shadow
    router.resume(d.pending, selection="ok") if selection != "ok" else None  # a second resume logs nothing new
    assert len(log.records()) == 1


def test_without_a_log_nothing_is_kept() -> None:
    router, _ = scenario_router(scripts.R2, context=scenario_context(history=True))
    router.decide(scenario_messages(scripts.R2_REQUEST, history=True))
    assert router._cards == {}


def test_shadow_mode_labels_would_be_executions(tmp_path: Path) -> None:
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    router, _ = scenario_router(scripts.R1, feedback=log, policy=Policy(shadow=True))
    d = router.decide(scripts.R1_REQUEST)
    assert d.outcome is Outcome.CONFIRM and d.pending is not None  # an execute shown as a card
    router.resume(d.pending, selection="ok")
    (row,) = log.records()
    assert row.shadow and row.label == "accepted"


def _row(i: int, c: float, right: bool, tier: str = "read") -> FeedbackRecord:
    return FeedbackRecord(decision_id=f"d{i}", at=T0 + timedelta(minutes=i), label="accepted" if right else "edited",
                          tool="t", arguments={}, outcome="confirm", rule="P9", tier=tier, composition="W", C=c, W=c,
                          PI=c, L=c)  # fmt: skip


def test_rows_become_tuning_records_and_split_by_time() -> None:
    rows = [_row(i, 0.5 + i / 20, i % 2 == 0) for i in range(10)]
    record = rows[0].to_eval_record()
    assert record.call_match and not record.wrong_if_executed and record.W == 0.5 and record.tier == "read"
    assert not rows[1].to_eval_record().call_match and rows[1].to_eval_record().wrong_if_executed
    fit, held = split(list(reversed(rows)), 0.2)
    assert [r.decision_id for r in held] == ["d8", "d9"] and len(fit) == 8  # the newest rows are checked
    assert (label_of("confirm"), label_of("cancel"), label_of("bind"), label_of(None)) == (
        "accepted", "cancelled", "edited", "edited")  # fmt: skip


def test_check_policy_counts_what_a_policy_would_execute() -> None:
    rows = [_row(0, 0.95, True), _row(1, 0.9, False), _row(2, 0.5, False), _row(3, 0.3, True, tier="write")]
    checks = check_policy(rows, Policy())  # read executes at W ≥ 0.60
    assert (checks["read"].n, checks["read"].executed, checks["read"].wrong) == (3, 2, 1)
    assert checks["read"].upper is not None and checks["read"].status == "over"  # 1 wrong in 2 is far over 5%
    assert checks["write"].executed == 0 and checks["write"].ok


def _log(path: Path, n: int, drift_from: int | None = None) -> FeedbackLog:
    """Read-tier rows: C below 0.6 (wrong) or above 0.8 (right); from row ``drift_from`` on, a third of the
    high-confidence calls are wrong (the app's traffic changed)."""
    rng = random.Random(7)
    log = FeedbackLog(path)
    with log.path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            high = rng.random() < 0.5
            c = round(0.8 + 0.2 * rng.random() if high else 0.6 * rng.random(), 3)
            drifted = drift_from is not None and i >= drift_from and rng.random() < 1 / 3
            fh.write(_row(i, c, high and not drifted).model_dump_json() + "\n")
    return log


def _tune(log: FeedbackLog, out_dir: Path) -> tuple[int, str]:
    out = io.StringIO()
    code = main(["tune", "--feedback", str(log.path), "--out", str(out_dir)], out=out, err=io.StringIO())
    return code, out.getvalue()


def test_tune_feedback_switches_when_the_newest_rows_agree(tmp_path: Path) -> None:
    code, text = _tune(_log(tmp_path / "feedback.jsonl", 2500), tmp_path)
    assert "tuned on 2000 row(s), checked on the newest 500" in text and (tmp_path / "policy.toml").is_file()
    assert code == 0 and "switch to it" in text, text
    tuned = Policy.from_toml(tmp_path / "policy.toml")
    assert isinstance(tuned.tiers.read.execute, float) and 0.55 < tuned.tiers.read.execute < 0.8


def test_tune_feedback_keeps_the_policy_when_the_newest_rows_drift(tmp_path: Path) -> None:
    code, text = _tune(_log(tmp_path / "feedback.jsonl", 2500, drift_from=2000), tmp_path)
    read = next(line for line in text.splitlines() if line.strip().startswith("read") and "%" in line)  # the table
    assert code == 2 and read.endswith("over") and "keep the current one" in text, text


def test_too_few_executions_are_unproven_not_over() -> None:
    rows = [_row(i, 0.95, True) for i in range(20)]  # 20 right executions: 0% observed, bound ~14%
    check = check_policy(rows, Policy())["read"]
    assert check.executed == 20 and check.wrong == 0 and check.status == "unproven"


def test_the_proxy_config_builds_a_feedback_log(tmp_path: Path) -> None:
    from jevtools.serve.config import ServeConfig

    config = ServeConfig.model_validate({"feedback_log": "labels/feedback.jsonl"})
    config._base_dir = tmp_path
    log = config.build_feedback()
    assert log is not None and log.path == tmp_path / "labels" / "feedback.jsonl"
    assert ServeConfig().build_feedback() is None


def test_a_shadow_share_keeps_labels_coming_after_the_switch(tmp_path: Path) -> None:
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    shown = executed = 0
    for i in range(40):
        router, _ = scenario_router(scripts.R1, feedback=log, policy=Policy(shadow_share=0.25))
        d = router.decide(f"{scripts.R1_REQUEST} ({i})")
        if d.outcome is Outcome.CONFIRM:
            shown += 1
            assert d.pending is not None and d.trace is not None and "shadow" in d.trace.outcome.get("caps", [])
            done = router.resume(d.pending, selection="ok")
            assert done.outcome is Outcome.EXECUTE  # the confirmed call runs: no second sample, no loop
        else:
            executed += d.outcome is Outcome.EXECUTE
    assert 3 <= shown <= 18 and shown + executed == 40  # about a quarter, deterministic per decision id
    assert len(log.records()) == shown and all(r.shadow and r.label == "accepted" for r in log.records())


def test_arguments_are_kept_only_on_request(tmp_path: Path) -> None:
    router, d = _card(FeedbackLog(tmp_path / "a.jsonl"))
    assert d.call is not None and d.call.arguments
    plain, kept = FeedbackLog(tmp_path / "plain.jsonl"), FeedbackLog(tmp_path / "kept.jsonl", keep_arguments=True)
    assert plain.add(d, "accepted").arguments == {}  # type: ignore[union-attr]
    assert kept.add(d, "accepted").arguments == dict(d.call.arguments)  # type: ignore[union-attr]

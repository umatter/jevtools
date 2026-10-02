# Piloting jevtools in an app

Thresholds and calibration do not transfer between apps (BENCH "Does one calibration fit every bench?"): the same
confidence was right 95% of the time on one benchmark and 69% on another. A pilot collects the app's own labels,
tunes on them, and switches only when the newest labels confirm the tuned policy. It needs no hand labelling: users
label calls by answering confirm cards.

## 1. Wire it up

```python
import jevtools as jt
from jevtools.feedback import FeedbackLog

log = FeedbackLog("feedback.jsonl")                      # arguments are not stored (keep_arguments=True to keep them)
router = jt.Router(tools, backend=jt.backends.auto(), policy=jt.Policy(shadow=True), feedback=log)

decision = router.decide(messages)
# show decision.prompt (a confirm card); when the user answers:
router.resume(decision.pending, selection="ok")          # or "cancel", another option, or reply="free text"
```

The proxy does the same with `feedback_log = "feedback.jsonl"` and a policy file with `shadow = true` in
`jevtools.toml`. Optionally, give the router an Escalator (`escalator=OpenAICompatibleEscalator("openai/gpt-4o-mini")`)
so values no extractor finds can be drafted (README "Benchmarks": BFCL and When2Call gains, a few more calls where
none fits).

## 2. What is logged

One JSONL row per answered confirm card: decision id, time, tool, tier, the confidence parts (W, Π, L, J, C), the
outcome and rule, whether the card was shown only because of shadow mode, and the verdict:

| Verdict | Meaning | Counts as |
|---|---|---|
| `accepted` | the user confirmed the call as proposed | right |
| `edited` | the user changed a value, picked another option or answered in free text | wrong |
| `cancelled` | the user cancelled | wrong |
| `undone` | logged by the app (`log.add(decision, "undone")`) when a user reverts an executed call | wrong |

Arguments are dropped unless `keep_arguments=True`: they usually hold personal data, and tuning does not use them.
Cards resumed in another process than the one that showed them are not labelled (the card lives in the router).

## 3. How many labels

A wrong-execution budget is shown by a 95% upper bound, so even with no wrong calls a tier needs, among the newest
checked rows:

| Budget | Executions needed (none wrong) |
|---|---:|
| 5% (read default) | ~60 |
| 2% (write default) | ~150 |
| 1% (external default) | ~300 |
| 0.1% (critical) | ~3,000 (and `jevtools tune` certifies the critical tier only then) |

The check uses the newest 20% of the log, so plan for about five times these numbers in total per tier. On the
held-out app bench's live decisions treated as cards (629 of them), the tuner could not yet prove the write and
external budgets.

## 4. Tune and switch

```bash
jevtools tune --feedback feedback.jsonl --out tuned/
```

It tunes on the older 80% and applies the current and the tuned policy to the newest 20%:

- exit 0, **switch**: every tier's wrong-execution bound is within budget on the newest rows;
- exit 2, **keep the current policy**: a tier is over budget (the traffic may have changed);
- exit 3, **collect more labels**: the observed rates are within budget but too few executions prove it.

After switching, leave full shadow mode but keep a sample of executions as cards, so the log keeps receiving labels
and a drift shows up at the next check:

```toml
shadow = false
shadow_share = 0.05     # 5% of would-be executions are shown as cards (cap "shadow")
```

Run the check on a schedule (weekly, or every few thousand cards) and after every Jev model or jevtools upgrade:
both move the confidence scale.

# Benchmarks

jevtools ships two benchmarks, and both run offline:

- **App domains** (`jevtools bench app`): assistants over an app's own data, the setting jevtools is built for.
  Arguments come from the user's words, the app's records and closed sets.
- **BFCL** (`jevtools bench bfcl`): the Berkeley Function Calling Leaderboard. It is the stress test outside that
  setting, where many values have to be written or reformatted rather than chosen.

Both report a **ceiling** first. The ceiling is the accuracy of an **oracle**, a backend that answers every Jev
question perfectly from the gold label. It shows how often jevtools *can* produce the right decision, given the
candidates code nominated, the question layout, decoding and policy. **It is not a measurement of Jev.** With an API
key, the same commands run live Jev and report its accuracy next to the ceiling. Both have been run live (below).

## App domains

```bash
jevtools bench app                                   # the oracle ceiling over all six domains
jevtools bench app --backend sim --tags              # plumbing check on the offline simulator, per-tag table
OPENROUTER_API_KEY=… jevtools bench app --backend auto --out live.json   # live Jev next to the ceiling
jevtools bench app --dir my_domains/                 # your own domains (same layout, see below)
```

| Option | Meaning |
|---|---|
| `--domains a,b` | only these domains (default: all bundled ones) |
| `--dir DIR` | a directory of domains, `DIR/<name>/cases.jsonl`, instead of the bundled ones |
| `--backend` | `oracle` (default), `sim`, or a Jev backend: `auto`, `typesafe`, `openrouter_decisions`, `openrouter_systemone`, `cassette:<path>` |
| `--limit N` | at most N cases per domain |
| `--tags` | also print the table per case tag |
| `--out report.json` | the full report: per-case records (outcome, rule, call, menu, failure stage, usage) and the summary |

From Python: `from jevtools.bench.app import run_domains; print(run_domains(backend=jt.backends.auto()).render())`.

### The domains

Six synthetic apps with 104 labelled cases (99 until 2026-09-26, when five search cases were added to
workspace; the live sections below say which count they ran on). The data is generated deterministically
(`python -m jevtools.bench.app._generate --check`) and ships with the package. Every domain runs at a fixed clock
(Thursday 2026-09-24 14:05, Europe/Zurich) for the user Sam Muster.

| Domain | Tools | App data | Cases |
|---|---|---|---:|
| inbox | `send_email`, `create_event`, `reschedule_event`, `cancel_event`, `list_events` | 133 contacts (two Annas, two Bobs, two Lisas, aliases such as Tom), 8 calendar events | 20 |
| crm | `create_deal`, `update_deal_stage`, `assign_deal`, `log_call`, `get_deal` | 56 companies (Müller AG and Mueller GmbH), 64 contacts, 55 deals (`D-1001`…), 5 reps (two named Jonas) | 17 |
| banking | `transfer_funds` and `pay_bill` (both critical, with balance constraints), `freeze_card`, `get_balance`, `list_transactions` | 5 accounts (Savings and Travel savings), 10 payees (two electricity providers), 3 cards | 16 |
| workspace | `read_file`, `share_file`, `move_file`, `rename_file`, `search_files` | 328 file paths (three board decks, dated weekly reports), 13 folders, 6 colleagues | 15 |
| helpdesk | `get_ticket`, `assign_ticket`, `set_priority` (P1–P4 described), `add_comment`, `close_ticket`, `escalate_ticket` | 48 tickets (`INC-1043`…, two VPN and two printer tickets), 6 agents | 16 |
| research | `summarize_dataset`, `run_regression` (outcome, predictor list, model family), `plot_variable`, `share_report`, `schedule_job` | 6 datasets (two survey waves), 25 variables, 9 R/Python scripts, reports and notebooks, 4 collaborators | 15 |

Each case is a §11.1 line: messages, the domain's `context.json` and `catalog.json`, and a gold label with the
allowed outcomes, the tool and the arguments that must match (`accepted` lists alternatives). Arguments the label
does not name are not checked. The same files are `jevtools eval` datasets, so a live run can go straight into
`jevtools tune`.

The tags say what a case tests:

| Tag | Example |
|---|---|
| `registry`, `alias`, `qualifier`, `near_name` | "Email Tom…" (alias of Thomas Becker); "Bob from Partner Inc"; Maya Kunz next to Maya Kuhn |
| `collision` | "Email Anna…" with two Annas: the right answer is a menu, not a guess |
| `identifier`, `id_literal`, `partial_id`, `id_by_title` | "Close INC-1063", "Assign ticket 1100 to Aisha", "Cancel the budget review" |
| `temporal`, `ambiguous_time` | "tomorrow at 10am"; "next Tuesday" (both readings accepted); "since September 1" |
| `enum`, `enum_description` | "bump it to high priority" → P2 via the member descriptions; "logistic regression" → `logit` |
| `list`, `money`, `critical`, `constraint`, `default_from`, `derived` | predictors lists; "89.90 from checking"; a transfer above the balance; "half of my savings" |
| `file`, `fuzzy_path`, `relative_file`, `folder`, `span`, `named`, `text` | "the Q3 board deck"; "last week's weekly report"; "rename … to notes_old.txt"; "a deal called …" |
| `history`, `coref`, `context_clue` | "Move it to proposal" after the assistant named the deal |
| `injection` | an observation plants an address, amount or path; `meta.planted` lists the values that must never reach a call (6 cases) |
| `missing`, `hedge`, `chitchat`, `negation`, `unsupported` | "Set up a meeting with Maria"; "How do I…?"; "Don't email Anna…"; "Order a taxi" |

### What is measured

| Column | Meaning |
|---|---|
| **ceiling** | the oracle's accuracy (see above) |
| **correct** | the outcome is one the gold allows and, if a call is shown (execute or confirm), it is the gold call |
| **within ceiling** | accuracy on the cases the oracle gets right: in a live run, Jev's judgment separated from coverage |
| **calls right** | accuracy on the cases that want a call shown |
| **clarify useful** | among clarify menus over tools or gold-named arguments, the share that offers the right one |
| **wrong executions** | calls that executed although executing them is wrong: the safety number |
| **injections** | cases where a planted value reached a shown call, out of the injection cases |
| **failure stages** | where a wrong decision went wrong (SPEC §11.2): `backend`, `plan` (tool not asked), `extractor` (value not nominated), `model` (the answers), `policy` (thresholds) |

### Baseline: the ceiling (oracle)

jevtools 0.1.0 on 2026-09-25. These are the oracle's numbers, **not Jev's**.

| Domain | n | Ceiling | Wrong executions | Injections | Failure stages |
|---|---:|---:|---:|---:|---|
| inbox | 20 | 95% | 0 | 0/1 | plan 1 |
| crm | 17 | 100% | 0 | 0/1 | |
| banking | 16 | 100% | 0 | 0/1 | |
| workspace | 20 | 100% | 0 | 0/1 | |
| helpdesk | 16 | 100% | 0 | 0/1 | |
| research | 15 | 100% | 0 | 0/1 | |
| **all** | 104 | 99% | 0 | 0/6 | plan 1 |

The one miss is kept on purpose. In inbox-17, "Email the head of legal…", the role is only in the contact's
`notes`, which is not a `match` field, so `send_email` is never asked about. An app would add a `role` field to
`match`. The case shows that the ceiling depends on how the app describes its data.

On the offline `LexicalSimulator` (a word-overlap test double, **not a model**), 31% of the cases are correct, with
0 wrong executions and 0 of 6 injections reaching a call. That shows the policy fails safe with a weak backend. It
says nothing about Jev.

The ceiling is high because these are the cases jevtools is built for, and because the bench was used to fix the
gaps it found (below). Treat it as a regression gate for that setting. A live run is the real test: Jev must still
pick Anna Keller over Anna Rossi from history, read "high priority" as P2, prefer the past "September 1" after
"since", and ignore instructions inside observations.

### First live run

2026-09-25, OpenRouter Decisions (`~typesafe/jev-latest`), one run per case, about $0.012 for the 99 cases. The
first run exposed a question-wording bug (ref slots asked for "the recipient's email address", which Jev read as
"did the user type an address?"; DECISIONS "First live run"), so there are two columns. A single domain moves by
about ±10 points between runs.

| Domain | n | Ceiling | Live, before the fix | Live, after | Calls right (after) | Wrong executions | Injections |
|---|---:|---:|---:|---:|---:|---:|---:|
| inbox | 20 | 95% | 55% | 60% | 38% | 0 | 0/1 |
| crm | 17 | 100% | 88% | 94% | 91% | 0 | 0/1 |
| banking | 16 | 100% | 81% | 88% | 89% | 0 | 0/1 |
| workspace | 15 | 100% | 53% | 73% | 60% | 3 | 0/1 |
| helpdesk | 16 | 100% | 69% | 69% | 55% | 0 | 0/1 |
| research | 15 | 100% | 87% | 100% | 100% | 0 | 0/1 |
| **all** | 99 | 99% | 72% | 80% | 71% | 3 | 0/6 |

After the fix, 11 misses are at the policy stage and 8 at the model stage. Most policy misses bind the right call but
clarify (`P9.<tier>.diffuse`), because the composed confidence is below the tier's prior thresholds, which are not
tuned. The 3 wrong executions (ws-01, ws-06, ws-07) run the read-only `search_files` where the gold opens the file.

### Replays and negative controls

`--replays N` decides every case N times and pools the records; the report adds each replay's accuracy and the cases
whose correctness flips. `--controls` adds the **negative controls**: the E2 variants of the cases (§11.2), each with
its gold rows removed from the registries, kept only when even the oracle can no longer show the gold call (a date,
an enum member or a path in a file index cannot be removed that way). The right call is impossible there, so the
safe decisions are clarify, abstain or escalate, and a shown call is a **false binding**. The bundled domains give 66
controls; the oracle is safe on all of them.

Live, 2026-09-25, `--replays 3 --controls` (about $0.06, 3.5 minutes):

| Domain | Correct (pooled) | Calls right | Wrong executions | Controls | Safe | False bindings |
|---|---:|---:|---:|---:|---:|---:|
| inbox | 68% | 51% | 0 | 30 | 90% | 3 |
| crm | 94% | 91% | 0 | 42 | 93% | 3 |
| banking | 94% | 93% | 0 | 39 | 100% | 0 |
| workspace | 69% | 60% | 8 | 15 | 100% | 0 |
| helpdesk | 69% | 55% | 0 | 39 | 100% | 0 |
| research | 100% | 100% | 0 | 33 | 100% | 0 |
| **all** | **82%** | 74% | 8 | 198 | 97% | 6 |

Correct per replay was 83%, 81% and 82%, and 6 of 99 cases flipped, so the whole-bench number is stable to about ±1
point while a single domain is not. No planted value reached a call (0 of 18).

The 6 false bindings are two controls, each wrong in all three replays, and both are **look-alike substitutions**:
with "the budget review" removed, "Cancel the budget review" gave a confirm card for *ACME quarterly review*; with
"ACME renewal" removed, "Move the ACME renewal to negotiation" gave one for *ACME expansion*. Both at C ≈ 0.6, in the
confirm band; none executed. The `rev` probe (reverse option order) does not address this. The 8 wrong executions
are the read-only `search_files` instead of `read_file` (ws-01, ws-06, ws-07).

### The look-alike check

The controls' false bindings led to the `verify` Noul (SPEC §3.8.3, DECISIONS "look-alike check"): before a call
is shown in the write tier or above, Jev is asked whether the elected record, shown with its match note, is the one
the request refers to; a no turns the card into a menu. Live, same settings (`--replays 3 --controls`):

| | Before | With `verify` |
|---|---:|---:|
| Correct (pooled) | 82% (83 / 81 / 82) | 81% (81 / 79 / 83) |
| False bindings in 198 controls | 6 | **0** |
| `unverified` menus on the 297 normal decisions | – | 0 |
| Decisions needing a second Jev round | 0 | 1 of 297 |
| Input tokens per decision | 2,817 | 3,315 |
| Cost of the run | $0.062 | $0.072 |

Per domain with `verify`: inbox 67%, crm 94%, banking 85%, workspace 73%, helpdesk 69%, research 100%. The change in
accuracy is within run-to-run noise; the gate never fired on a case whose right record was present.

### Record hints (experimental)

`tool.record_hints = 3` names the best-anchored records in the tool options (DECISIONS "record hints"), to stop
`search_files` winning over `read_file` when the user describes a file. Live, 2026-09-26, 104 cases × 3 replays with
controls, `--policy` off vs on:

| | Off (default) | `record_hints = 3` |
|---|---:|---:|
| Correct (pooled) | 82% (82 / 82 / 82) | 83% (83 / 84 / 82) |
| Calls right | 74% | 79% |
| Wrong executions | 9 | 3 |
| Search cases (ws-08, ws-16 … ws-20) | 100% | 67% |
| False bindings in 198 controls | 0 | 0 |

Per domain, off → on: inbox 68 → 68%, crm 94 → 94%, banking 92 → 88%, workspace 78 → 82%, helpdesk 67 → 69%,
research 96 → 100%. The 3 wrong executions with hints on are one case: "Open the board deck" opens the newest of
three decks instead of asking which.

### Several fitting records (experimental)

`probes.unique` (off by default) asks whether the request singles out one record, and turns a shown call into a menu
when it does not (DECISIONS "unique"). Live, 2026-09-26, 104 cases × 3 replays with controls:

| | Off | `record_hints = 3` | `unique` (all tiers) | Both |
|---|---:|---:|---:|---:|
| Correct (pooled) | 82% | 83% | 83% | 83% |
| Calls right | 74% | 79% | 77% | 77% |
| Wrong executions | 9 | 3 | 9 | **0** |
| Search cases (ws-08, ws-16 … ws-20) | 100% | 67% | 100% | 53% |
| False bindings in 198 controls | 0 | 0 | 1 | 0 |
| Input tokens per decision | 3,461 | 3,594 | 4,206 | 4,345 |

With `unique`, "Open the board deck" and "Share the board deck…" become menus for the right reason in both columns.
The one false binding is the budget-review look-alike passing `verify` in one replay.

### Held-out

`jevtools bench app --heldout` runs 284 generated cases over the same six apps (`python -m jevtools.bench.app._heldout`):
templates over each domain's rows, so the gold follows from the data. A description that fits exactly one record
expects it; one that fits several with nothing to tell them apart expects a menu; a search request expects the
search tool. Generation fails if a description does not fit the records its label claims, and no message repeats
the development bench. The oracle solves all 284; its controls (194) are all safe. Features are not tuned on it.

Live, 2026-09-26, 284 cases × 3 replays with 582 control decisions per column:

| | Off | `record_hints = 3` | `unique` | Both |
|---|---:|---:|---:|---:|
| Correct (pooled) | 79.5% | **82.5%** | 78.6% | 82.2% |
| Calls right | 75.4% | **79.1%** | 74.0% | 77.4% |
| Wrong executions | 29 | 12 | 28 | **9** |
| Open a described file | 46% | 94% | 49% | **100%** |
| Several fitting records | 90% | 92% | 92% | **98%** |
| Search requests | 100% | 86% | 100% | 86% |
| Accounts by name | 100% | 100% | 80% | 80% |
| False bindings in controls | 12 | 12 | 11 | 11 |
| Input tokens per decision | 4,185 | 4,331 | 5,120 | 5,267 |

Per domain (off / hints / unique / both): inbox 53 / 52 / 53 / 52%, crm 98 / 96 / 98 / 98%, banking 100 / 98 / 94
/ 90%, workspace 75 / 91 / 76 / 93%, helpdesk 76 / 76 / 75 / 74%, research 92 / 98 / 91 / 97%.

Decision: record hints on by default; `unique` stays off (it helps its own cases but not overall, and costs 22%
more tokens). Two problems the development bench had understated, the same in every column:

- **Emails to a full name: 0% of 22.** "Email Anna Keller that …" leaves ~45% of the recipient Choice on the
  sentinels although the name is exact, so C ≈ 0.40 and the decision is a menu (the first live run's inbox-01).
- **Read-tier look-alikes: 12 false bindings.** "What's the balance of my Savings account?" with Savings removed shows
  Travel savings, and the reverse, in every replay: `verify` covers write tiers and above only.

### Held-out, round 2

Live, 2026-09-30, 284 cases × 3 replays, 582 control decisions per column (A is the defaults at that time: record
hints on, `verify` for write tiers and above):

| | A | B: `verify` for reads | C: B + present sets NOT_STATED | D: C + `present` in write |
|---|---:|---:|---:|---:|
| Correct (pooled) | 82.6% | 82.2% | 81.3% | 80.3% |
| Calls right | 78.9% | 78.6% | 77.1% | 77.7% |
| Wrong executions | 11 | 11 | 11 | 19 |
| False bindings in controls | 12 | **7** | 7 | 9 |
| Emails to a full name | 0% | 0% | 20% | 26% |
| Payee payments | 100% | 100% | 50% | 50% |
| Second Jev rounds | 15 | 15 | 39 | 52 |
| Input tokens per decision | 4,331 | 4,551 | 4,644 | 4,806 |

B is the new default; C and D are not adopted (DECISIONS "present sets NOT_STATED").

### Slot question tree (experimental, off)

`probes.slot_decider = "tree"` decides record slots from `present` and per-record `verify` Nouls instead of the slot Choice
(DECISIONS "slot question tree"). Live, 2026-09-30, held-out, 3 replays with controls:

| | Defaults | Tree (k = 3) | Tree (k = 5) |
|---|---:|---:|---:|
| Correct (pooled) | **82.3%** | 74.2% | 73.0% |
| Emails to a full name | 0% | **88%** | 68% |
| Several fitting records | 93% | **100%** | 100% |
| Described files | 94% | 52% | 49% |
| Events by title | 74% | 30% | 30% |
| Payee payments | 100% | 17% | 17% |
| Wrong executions | 11 | **0** | 0 |
| False bindings in controls | 7 | 5 | 4 |
| Input tokens per decision | 4,551 | 4,786 | 5,149 |

### Hybrid slot decoding (the default)

`--record DIR` stores every Jev answer of a run (one cassette per replay) and `--replay DIR` re-decides them offline
under another policy, so decoding rules are compared on identical answers at no cost. On one held-out recording:
`choice` 79.4%, `tree` 74.5%, `hybrid_max` 86.2%, `hybrid_safe` 79.2% (DECISIONS "hybrid slot decoding").

Live, 2026-09-30, held-out, 3 replays, 582 control decisions per column (no backend failures in either):

| | Defaults (Choice) | `hybrid_max` |
|---|---:|---:|
| Correct (pooled) | 82.4% (82 / 82 / 83) | **90.0%** (90 / 90 / 90) |
| Calls right | 78.7% | **88.6%** |
| Emails to a full name | 0% | **86%** |
| Aliases | 22% | 89% |
| Described files | 94% | 93% |
| Several fitting records | 92% | 93% |
| Wrong executions | 12 | 12 |
| False bindings in controls | 8 | 8 |
| Input tokens per decision | 4,551 | 4,551 |

Per domain: inbox 52 → 87%, crm 98 → 98%, banking 98 → 97%, workspace 91 → 90%, helpdesk 73 → 74%, research 98 →
98%. The 12 wrong executions are the search requests read as opening a file (record hints); the 8 false bindings are
"my Savings account" with Savings removed shown as Travel savings.

### Search cues and typed keys (the defaults)

Live, 2026-09-30, held-out, 3 replays, 582 control decisions per column, no backend failures:

| | Baseline | Search cue | Typed-key verify | Both | Both + present fallback |
|---|---:|---:|---:|---:|---:|
| Correct (pooled) | 90.3% | 90.8% | 90.5% | 92.0% | **93.2%** (94 / 93 / 93) |
| Calls right | 89.1% | 90.2% | 89.6% | 91.4% | **93.2%** |
| Search requests | 86% | 100% | 86% | 100% | **100%** |
| Typed IDs | 77% | 73% | 77% | 78% | **91%** |
| Wrong executions | 12 | 4 | 11 | 3 | **3** |
| Wrong calls shown, write tier and above | 0 | 0 | 0 | 0 | **0** |
| False bindings in controls | 7 | 7 | 7 | 9 | 7 |
| Input tokens per decision | 4,551 | 4,533 | 4,623 | 4,606 | 4,606 |

Per domain (baseline → final): inbox 86 → 87%, crm 97 → 96%, banking 97 → 100%, workspace 91 → 97%, helpdesk 77 →
87%, research 99 → 97%. On the final run's recorded answers, removing only the present fallback gives 92.1% (typed
IDs 80%) with every other category unchanged.

### A 31-tool catalog

`--merged-catalog` gives every case the tools of all six apps (31) while it keeps its own context and data. Live,
2026-09-30, held-out, 3 replays with controls:

| | One app (6–7 tools) | All apps (31 tools) |
|---|---:|---:|
| Correct (pooled) | 93.2% | 91.2% |
| Calls right | 92.8% | 89.8% |
| Tool question's top wrong | 6 of 852 | 21 of 852 |
| Misses caused by the tool | 0 of 58 | 17 of 75 |
| Wrong calls shown, write tier and above | 0 | 3 (one case: `share_report` for `share_file`, same file, same person) |
| False bindings in controls | 8 | 13 |
| Questions per request (median) | 17 | 46 |
| Input tokens per decision | 4,573 | 9,807 |

Wrong tool picks: `share_file` → `share_report` 6 and `summarize_dataset` → `read_file` 3 (near-duplicate tools of two
apps), `search_files` → `list_events` 5, `search_files` → NO_TOOL 3, NO_TOOL → UNSUPPORTED 3. One decision was lost to
a 26-minute service outage (retried, then counted as a miss).

### What the app bench found and fixed

Each fix has a regression test (`tests/unit/test_app_domain_features.py`), and DECISIONS.md explains it.
- **Typed identifiers.** "Move D-1017 to negotiation" had no candidate, because the token rules never matched IDs.
  A token with a digit that equals a row's key (or a whole match value) now anchors that row. A bare number counts
  only after a cue such as "ticket" or "#", so amounts never anchor rows, and then also anchors the keys that end in
  it ("ticket 1100" → `INC-1100`).
- **Code-like tokens and names.** File names (`notes_old.txt`), codes (`SKU-4411`) and names given with
  "called/named/titled" ("a deal called data platform phase 2") were not nominated. Now they are.
- **Year-less dates.** "since September 1" (said on 24 September) was read only as next year's date. A passed
  named-month date now also offers this year's, and Jev chooses from the glosses ("23 days ago", "in 342 days").
- **No coin-flip confirm cards.** With two reps named Jonas, a write-tier confirm card proposed one of them. When an
  identity slot's top two values are within 0.20, the policy now shows a menu instead (`confirm_margin`).
- **Infeasible calls.** A transfer above the balance produced "transfer, or freeze your card?". Only plausible tools
  (P ≥ 0.10) now compete in the call-MAP check, so the infeasible call is flagged and clarified.
- **Oracle.** The oracle now answers joint questions (critical tier) and superlative membership questions, and is
  confident enough (0.99) for the critical tier's Łukasiewicz bound. With six factors at 0.94, even perfect answers
  stayed below the 0.80 confirm threshold.

### Your own domain

A domain is a directory:

```
my_domains/
  tickets/
    catalog.json     # your tools: an OpenAI tools list or MCP tools/list, x-jev hints allowed
    context.json     # {"now", "tz", "locale", "user", "sources": [registry / files specs, as in jevtools.toml]}
    data/…           # the rows the sources point to (JSON, JSONL, CSV)
    cases.jsonl      # one §11.1 case per line
```

`jevtools bench app --dir my_domains` then shows, before any API call, which of your cases jevtools can get right at
all, and for the others it shows why: a tool that was never asked about, or a value that was never nominated. Fix
those misses with sources, `match` fields or `x-jev` hints. Then run live (`--backend auto`) and tune the thresholds
on the report (`jevtools eval` and `jevtools tune`).

## BFCL

`jevtools bench bfcl` runs jevtools on the single-turn categories of the
[Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
(BFCL v4, Apache-2.0). It scores every decision with a port of BFCL's own AST checker, so the numbers are
comparable with BFCL's.

BFCL was built for models that *write* arguments. jevtools *elects* them from candidates that code nominates. Many
BFCL values are general-knowledge strings, formats and expressions that no app registry holds. That makes BFCL a
hard, honest test of the approach's weak spot, candidate coverage: Jev cannot choose a value that no extractor
nominated.

### Quick start

```bash
jevtools bench bfcl --download                       # fetch the data into ~/.cache/jevtools/bfcl, run the oracle
jevtools bench bfcl --backend sim --limit 50         # plumbing check on the offline simulator
OPENROUTER_API_KEY=… jevtools bench bfcl --backend auto --out live.json   # live Jev
```

| Option | Meaning |
|---|---|
| `--data DIR` | BFCL data directory (default `<cache>/bfcl`; `JEVTOOLS_CACHE_DIR` moves the cache) |
| `--download [--ref main]` | fetch the data and answer files from GitHub; pin `--ref` to a commit for reproducible runs |
| `--categories a,b` or `all` | default: `simple_python, multiple, live_simple, live_multiple, irrelevance, live_irrelevance, live_relevance`; `all` adds the four parallel categories |
| `--limit N` | at most N cases per category |
| `--backend` | `oracle` (default), `sim`, or a Jev backend: `auto`, `typesafe`, `openrouter_decisions`, `openrouter_systemone`, `cassette:<path>` |
| `--risk read` / `infer` | the risk tier every BFCL tool gets (see below) |
| `--out report.json` | the full report: per-case records, coverage and summary |

From Python:

```python
from jevtools.bench import load_category, run_bfcl
import jevtools as jt

cases = load_category("~/.cache/jevtools/bfcl", "simple_python", limit=50)
report = run_bfcl(cases, jt.backends.auto())        # or "oracle", or jt.backends.LexicalSimulator()
print(report.render())
```

### What is measured

Every case gets its own `Router` over the case's functions, with a fixed clock (2026-09-25 10:00 UTC).

| Column | Meaning |
|---|---|
| **ceiling** | the oracle's accuracy: the proposed call passes BFCL's checker |
| **proposal** | The proposed call counts whatever the outcome (`confirm`/`clarify` included), unless the outcome is `abstain` or `refuse`. This is the call a user would see on a card. |
| **strict** | Only an `execute` decision emits a call: what an autonomous agent would run. |
| **within ceiling** | Proposal accuracy on the cases whose ceiling passes. In a live run this isolates **Jev's judgment** from extractor coverage. |

Failures are attributed to the most upstream cause:

| Attribution | Meaning |
|---|---|
| `tool_not_speculated` | a required parameter had no candidate, so the tool was not asked about (P6 → clarify) |
| `value_not_nominated` | a value BFCL expects was on no option |
| `no_call` | the decision proposed nothing (e.g. a clarify without a proposed call) |
| `wrong_call` | everything was on the ballot, but the call that came out is wrong. For the oracle this is a decoding, normalization or type issue; for a live run it is mostly the model's choice. |
| `called_irrelevant` | a call was proposed where BFCL expects none |

Each missed parameter also records `in_text`: whether an acceptable value appears verbatim in the user's
messages, compared the way BFCL compares strings. `in_text: true` marks an **extractor gap**, which better
extraction could close. `in_text: false` marks a value that needs reformatting, arithmetic or world knowledge, which
selection alone cannot produce.

**Risk tier.** BFCL functions are side-effect free test fixtures, but most of their names (`calculate_…`,
`find_…`) fall through jevtools' verb table to the fail-safe `external` tier. That tier asks for authorization and
confirms more. The bench therefore gives every BFCL tool `x-jev.risk = "read"` by default; `--risk infer` keeps the
inferred tier.

**Not covered.**
- The parallel categories need several calls in one turn. jevtools decides one (`ext.parallel` is not implemented),
  so they score 0 by construction.
- Java and JavaScript, multi-turn, memory and web-search categories are out of scope.

### Baseline: the ceiling (oracle)

BFCL `main` as of 2026-09-25 and jevtools 0.1.0 with the app-bench fixes, default categories, `--risk read`. Every
number here is the oracle's, **not Jev's**.

| Category | n | Ceiling | Strict (oracle) | Misses in text | Misses not in text |
|---|---:|---:|---:|---:|---:|
| simple_python | 400 | 38.8% | 38.8% | 319 | 96 |
| multiple | 200 | 42.0% | 42.0% | 152 | 41 |
| live_simple | 258 | 36.0% | 34.5% | 159 | 102 |
| live_multiple | 1,053 | 34.9% | 33.0% | 589 | 662 |
| irrelevance | 240 | 100.0% | 100.0% | – | – |
| live_irrelevance | 884 | 100.0% | 100.0% | – | – |
| live_relevance | 16 | 68.8% | 0.0% | – | – |
| **all** | 3,051 | 60.1% | 59.0% | 1,219 | 901 |

"Misses" count parameters, not cases. With `--categories all`, the four parallel categories (440 cases) score 0%.
The first baseline, before the app-bench fixes, was 58.7% (single-call categories 32.9–38.5%). The identifier,
code-token and naming extractors added 1.4 points without targeting BFCL.

Estimated cost of a live run of the default categories: about 1,400 input tokens per call on average (851
median, the largest about 23k). That is roughly $0.18 for all 3,051 cases at $0.042 per million input tokens. It
is an estimate at 3.5 characters per token, not a measured figure.

### Where the ceiling is lost

Attribution over the single-call categories (1,911 cases):
- `value_not_nominated`: 926
- `tool_not_speculated`: 273
- `wrong_call`: 12

(The 5 `no_call` cases are all in live_relevance.)

The single-call categories are therefore limited almost entirely by **coverage, not by decoding or policy**.
Parameter coverage by kind (all default categories):

| Kind | Covered | Omitted/default (acceptable) | Missed |
|---|---:|---:|---:|
| enum | 940 | 66 | 26 |
| flag | 542 | – | 6 |
| quantity | 615 | 226 | 96 |
| money | 76 | 5 | 10 |
| list | 131 | – | 89 |
| span | 836 | 406 | 869 |
| temporal | 0 | 33 | 257 |
| text | 29 | – | 56 |
| (not asked: tool not speculated) | – | 433 | 711 |

What the misses are, most common first:
- **Spans.** Single words and short phrases are not nominated: `all`, `inches`, `simpson`, `water`. The span
  extractors favour noun chunks and clauses. Identifier-like tokens (`DNA123`, `x^2`) are now nominated.
- **Temporal.** BFCL date and time parameters are *strings in the user's or a stated format* (`"08-16-2022"`,
  `"3pm"`). The temporal resolver emits ISO 8601 datetimes, so none of them matches. A string-typed temporal slot
  should also offer the verbatim mention and the common formats.
- **Tools not speculated.** A required parameter with no evidence-backed candidate keeps the tool off the fan-out.
  Most of these values are in the text, missed by the extractors above.
- **Lists.** Lists of numbers and coordinates are partly nominated.
- **Quantities.** Numbers with written-out compound units are dropped by dimension filtering. In "starting from
  a speed of 10 meters/second", the 10 is not offered for `initial_velocity`, while the 5 from "5 seconds" is.
  `300K` is read only as 300,000, never as 300 kelvin.
- **Not in the text (43%).** Formats and knowledge that selection cannot produce: `"New York, NY"` from "New York",
  `0.05` from "50 mH", `C6H12O6` from "glucose", dates in a stated format. These need a normalization catalog, a
  source, or the generative fallback (a Filler whose drafts Jev must accept).

### Bugs the BFCL bench found

These were fixed, with regression tests:
- a fractional amount crashed an integer money slot;
- a text candidate over the 4,000-character accept limit made the pre-send validator reject the whole ballot;
- list items came out in canonical label order instead of mention order.

### BFCL live

2026-09-30, OpenRouter Decisions, all 3,051 cases of the default categories, one run, recorded (`--record`); no
backend failures (a failure is now retried, then counted as an error: a fail-closed abstain would otherwise pass
an irrelevance case). Cost $0.23 (1,759 input tokens per case), median latency 0.36 s, p90 0.48 s.

| Category | n | Ceiling | Proposal | Strict | Within ceiling | Execute | Clarify | Abstain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| simple_python | 400 | 38.8% | 33.5% | 26.5% | 86.5% | 175 | 223 | 2 |
| multiple | 200 | 42.0% | 39.0% | 30.5% | 92.9% | 87 | 112 | 1 |
| live_simple | 258 | 36.0% | 25.6% | 24.0% | 71.0% | 87 | 159 | 12 |
| live_multiple | 1053 | 34.9% | 29.2% | 25.7% | 83.2% | 428 | 617 | 8 |
| irrelevance | 240 | 100% | 92.5% | 97.1% | – | 7 | 23 | 210 |
| live_irrelevance | 884 | 100% | 79.8% | 98.2% | – | 16 | 307 | 561 |
| live_relevance | 16 | 68.8% | 62.5% | 18.8% | – | 3 | 12 | 1 |

Read it against the ceiling. On the categories that want a call, strict accuracy (what an autonomous agent would
run) is 24–31%, far below LLM tool callers that write arguments: jevtools can only elect values code nominated, and
BFCL's values are mostly strings, formats and expressions that no extractor produces. Where the right values were on
the ballot, Jev chose the right call 71–93% of the time. On the irrelevance categories jevtools almost never executes
a call where none belongs (97–98% strict); the proposal view (80–92%) counts calls offered on confirm cards and menus.

## When2Call

[When2Call](https://huggingface.co/datasets/nvidia/When2Call) (NVIDIA, CC-BY-4.0) tests *when* to call a tool: its
3,652 test cases, built from BFCL's live categories, want a tool call, a question for missing information, or a
decline because no available tool fits. `jevtools bench when2call --download --backend auto` maps each decision to the
category it amounts to (a shown call → tool call, clarify → ask, abstain/refuse/escalate → decline). The published
evaluation scores a language model's probabilities over four written answers, so the numbers below are not directly
comparable with the paper's; the categories and cases are.

Live, 2026-09-30, all 3,652 cases, one run, no backend failures, $0.28:

| Correct answer | n | Precision | Recall | F1 | → tool call | → ask | → decline |
|---|---:|---:|---:|---:|---:|---:|---:|
| tool call | 1,295 | 78.0% | 35.4% | 48.7% | 458 | 804 | 33 |
| ask for information | 1,062 | 47.2% | 86.3% | 61.1% | 78 | 917 | 67 |
| cannot answer | 1,295 | 91.1% | 79.1% | 84.7% | 51 | 220 | 1,024 |

Accuracy 65.7%, macro F1 64.8%; predicted calls name the target tool 98.9% of the time; a tool is called on 3.9% of
the cannot-answer cases. The weak side is making the call: 804 tool-call cases became questions, mostly because a
value was not on the ballot (P7, 513) or the tool was not asked about (P6, 199), BFCL's coverage ceiling again.

## τ²-bench, reduced to next calls

[τ²-bench](https://github.com/sierra-research/tau2-bench) (Sierra, MIT) runs customer-service agents against an
LLM-simulated customer; it also publishes complete successful conversations of reference agents, with every tool
call, its arguments and its actual output. `jevtools bench tau2 --tau2 CHECKOUT --data DIR` turns one successful
conversation per task (Claude 3.7 Sonnet as the reference agent, GPT-4.1 as the customer) into single decisions:
a **call** case per tool call (the conversation and earlier tool results as context, the call as gold: tool plus
every argument, id lists as sets) and a **talk** case per assistant turn without a call (right unless jevtools
executes one there). jevtools gets the tool schemas and the conversation, not the domain policy. The tool schemas
come from a τ² install (`tau2.EXPORT_TOOLS`); τ² data is not vendored.

Live, retail (106 tasks) and airline (34), 994 cases, no backend failures. The first run (2026-09-30) is before the
coverage fixes below, the second (2026-10-01) after them:

| | n | Trust `none` (default) | | Trust `reads` | |
|---|---:|---:|---:|---:|---:|
| | | before | after | before | after |
| Read calls exactly right | 341 | 21.7% | **57.2%** | 22.0% | **57.8%** |
| Write calls exactly right (shown / executed) | 173 | 1.2% / 1.2% | 2.3% / 1.7% | 15.6% / 9.8% | **17.9% / 12.1%** |
| Right tool chosen (read / write) | | 83.9% / 90.2% | 83.9% / 91.3% | 84.2% / 93.1% | 83.9% / 91.9% |
| Talk turns without an executed call | 480 | 97.9% | 97.7% | 96.9% | 95.4% |
| Cost | | $1.47 | $1.64 | $2.38 | $2.45 |

Coverage (compiled offline: every gold value on the ballot, trusted lookups): 267 of 514 call cases before, 442
(86%) after. The fixes, each found by reading the misses:
- **name parts**: "I'm Mei Kovacs" offers "Mei" to a `first_name` slot and "Kovacs" to a `last_name` slot (87 misses
  each);
- **digit strings**: a number of three or more digits the user typed is offered as text to a string slot, so a zip
  code is "28236", not 28236 (92);
- **qualified field names**: `new_item_ids` takes `item_id` fields of earlier tool results (58);
- **emails**: generic string slots took the role's default extractors and dropped the ones inferred from the name,
  so a slot named `email` never ran the email extractor (30);
- **allowed values in descriptions**: "should be either 'no longer needed' or 'ordered by mistake'" makes the slot
  an enum, so the value is one the tool accepts, not the user's paraphrase (9).

Name parts and digit strings come from the user's words only; a tool result's values come as field candidates,
described by their sibling fields. Still missing: free text and objects no extractor writes (`summary` 16,
`flights` 12, `passengers` 8, `payment_methods` 5), dates (15), and lists where only some ids are on the ballot
(`item_ids` 8, `new_item_ids` 7).

Read calls are now right more often than not. Write calls barely move without trust, because their ids come from
earlier tool results, which the injection defence keeps out of write slots unless the app trusts the lookup tools.
A next-call case often has several valid next calls ("I have two orders"), and jevtools asks which.

Executed calls in talk turns that the reference agent did not make there: before the fixes, two cancellations with
the user's words as the reason ("I ordered it by mistake"), two airline cancellations (the policy restricts
cancellations; jevtools does not see it), and reads with a malformed id. After the fixes, with trusted lookups,
seven more: five are the reference agent's next call made a turn early (it asks "shall I proceed?" first; τ²'s
policy requires that, and jevtools does not see the policy), one is a read (`find_user_id_by_email`), and one
(retail-24-6) cancels an order the reference agent never cancelled, after "I was actually thinking about canceling
it". An app that wants explicit confirmation before a cancellation sets that tool's policy to confirm; the bench
runs the defaults. Requests are large (median 19k input tokens, 82 questions): long conversations with 16 tools
expand most of them.

### Lists, ids, replies and dates (2026-10-01)

DECISIONS "Lists, ids, replies and dates": count and rank for enumerative lists (odds factor), a list leaving out
what its refined sibling holds, ids typed without their schema prefix, the reply clause in the tool question,
date-only temporal strings, typed object unions and lists of records taken whole from tool results.

Oracle ceiling (every question answered right), trusted lookups: reads 94.4% → 99.4%, writes 22.0% → 75.7% (count and
rank alone: 71.7%; before it, perfect list answers still multiplied to a "diffuse" factor in 84 write cases).

Live, all 994 cases, no backend failures:

| | Before | After |
|---|---:|---:|
| Read calls exactly right (trust `none` / `reads`) | 57.2% / 57.8% | **67.7% / 68.0%** |
| Write calls exactly right, shown / executed (trust `reads`) | 17.9% / 12.1% | **37.0% / 25.4%** |
| Write calls (trust `none`) | 2.3% / 1.7% | 2.3% / 1.7% |
| Talk turns without an executed call (`none` / `reads`) | 97.7% / 95.4% | 97.1% / 94.6% |
| Wrong write executions (trust `reads`) | 1 | 1 |
| Cost (`none` / `reads`) | $1.64 / $2.45 | **$1.29 / $2.08** |

A first version of count and rank cut at 0.5 and moved writes only from 17.9% to 18.5%: Jev ranked the items right
but scored them low or close, and phrases such as "item ID" (scored 0.71) sat among the ids. Re-decided from that
run's recording, the odds factor and the id-list filter gave the estimate (45 of 173 executed right) that the live
run then confirmed (44). The four extra talk-turn executions were all the reference agent's next call, made before
its "shall I proceed?" turn. With trust `none` the right write tool is chosen less often (91.3% → 85.5%); those
writes cannot execute without trusted lookups either way. Held-out app bench, 3 replays with controls: 93.5% correct
(93.2% before), 0 wrong executions, the same 7 control bindings; the odds rule re-decided from that recording changes
nothing.

Still out of reach: free text (`summary`), payment amounts (arithmetic), passengers the user types in the chat, and
the 47 lookups of an order the user never named (the reference agent works through the user's orders one by one).

## Drafted values (Escalator), live

`--escalator MODEL` gives BFCL, When2Call and τ² an OpenAI-compatible Escalator (here `openai/gpt-4o-mini` on
OpenRouter) that drafts a call when a value is not on the ballot; Jev re-decides the drafted values in one more round
(DECISIONS "Drafted values for uncovered slots"). Cost is Jev plus the drafter.

BFCL, all 3,051 cases, strict (executed calls exactly right):

| Category | No drafter | Drafter, fixed gate | Drafter, re-gate (default) |
|---|---:|---:|---:|
| simple_python | 26.5% | 42.5% | **43.8%** |
| multiple | 30.5% | 39.5% | **40.0%** |
| live_simple | 24.0% | 41.1% | **40.7%** |
| live_multiple | 25.7% | 44.4% | **44.4%** |
| irrelevance | 97.1% | 95.0% | 95.8% |
| live_irrelevance | 98.2% | 96.9% | 97.7% |
| live_relevance | 18.8% | 31.2% | 25.0% (16 cases) |
| Cost | $0.23 | $0.26 + $0.10 drafter | $0.27 + $0.10 drafter (1,443 calls) |

When2Call, all 3,652 cases:

| | No drafter | Drafter, fixed gate | Drafter, re-gate (default) |
|---|---:|---:|---:|
| Accuracy / macro F1 | 65.7% / 64.8% | 70.3% / 70.6% | **73.3% / 73.2%** |
| Recall: tool call / ask / decline | 35.4% / 86.3% / 79.1% | 58.8% / 79.3% / 74.3% | 59.5% / 80.2% / 81.5% |
| Tool called on cannot-answer cases | 3.9% | 7.4% | **5.3%** |
| Errors (backend failures after retries) | 0 | 21 | 0 |
| Cost | $0.28 | $0.28 + $0.21 drafter | $0.34 + $0.22 drafter |

A first drafter run escalated `missing` too, ended text drafts in `abstain` and let unfitted values reach the ballot:
58.1% accuracy, "ask" recall 32%, 144 errors. The second fixed those; its call rate on cannot-answer cases (7.4%)
came mostly from the older escalation of unsupported, ambiguous or diffuse turns, whose gate round fixed the drafted
tool. With `tool.regate_escalation` (default) that gate re-asks the tool Choice and a draft stands only if Jev elects
its tool: 5.3%, and declines are more accurate than without a drafter. For reference, the When2Call paper reports
61.3% accuracy for GPT-4o (judged by an LLM) and 70.0% for its best fine-tuned 8B model (log-probabilities over the
written answers); the protocols differ.

### Two-stage rounds (2026-10-02)

`tool.two_stage_min` (DECISIONS "Two-stage rounds"): τ² ballots ask the tool first, then only the favoured tools'
questions. Live, all 994 next-call cases, no backend failures:

| | One stage | Two stages |
|---|---:|---:|
| Cost, trust `none` / `reads` | $1.29 / $2.08 | **$0.38 / $0.37** |
| Median input tokens per decision, `none` / `reads` | 23,073 / 39,679 | **6,650 / 6,474** |
| Read calls exactly right, `none` / `reads` | 67.7% / 68.0% | 67.2% / 67.7% |
| Write calls, shown / executed (trust `reads`) | 37.0% / 25.4% | 35.3% / 25.4% |
| Talk turns without an executed call, `none` / `reads` | 97.1% / 94.6% | 97.3% / 94.4% |
| Wrong write executions (trust `reads`) | 1 | 1 |

No held-out app ballot exceeds 40 questions, so the app benches do not change. With the gpt-4o-mini drafter on top
(trust `reads`) τ² does not move (reads 67.2%, writes 35.8% / 25.4%; drafter $0.07): its remaining gaps are write
identity slots, which drafted values cannot bind.

### Full τ² tasks (2026-10-02)

`python -m jevtools.bench.tau2_agent` (in a τ²-bench environment) runs whole simulated conversations, scored by τ²'s
own reward (pass^1, one trial). The jevtools agent lets jevtools decide every call (an execute is the call; a confirm
card or a question is jevtools' own prompt, resumed by the customer's reply) and an LLM write only the text replies
jevtools has none for. The baseline is τ²'s `llm_agent` on the same LLM (gpt-4.1-mini, temperature 0), against the
same simulated customer (gpt-4.1-mini), trusted lookups:

| Tasks | jevtools agent | gpt-4.1-mini agent |
|---|---:|---:|
| Retail 0–19 | 3 / 20 | 10 / 20 |
| Retail 20–39 | 0 / 20 | 12 / 20 |
| Airline 0–19 | 2 / 20 | 12 / 20 |
| **All** | **5 / 60 (8%)** | **34 / 60 (57%)** |
| Agent cost | $0.44 Jev + $0.08 writer | $0.61 |

jevtools does not complete τ² tasks. It gets stuck: 48 of the 60 conversations repeat the same question three or more
times in a row, and the median one runs into the 30-turn limit. 1,148 of its 1,458 decisions were questions, mostly a
missing value (`P7`, 478) or a choice between tools (`P5`, 371). The typical case needs a lookup first (the order's
item ids): Jev keeps choosing the write the customer ultimately wants, asks the customer for the ids, and the customer
cannot answer. jevtools decides one call well when its values are at hand; it does not plan the steps that produce
them, and its questions have no way out of a loop. At gpt-4.1-mini prices it is not cheaper either.

### jevtools as a guard over an LLM agent (2026-10-02)

The roles inverted: τ²'s `llm_agent` (gpt-4.1-mini) plans and proposes every call, reads run as proposed, and each
other call goes through `Router.check` first (`--agent guard`): `execute` runs it, `confirm` shows jevtools' card,
anything else returns a tool error so the LLM re-plans. Same 60 tasks, same simulated customer:

| Agent | Solved |
|---|---:|
| gpt-4.1-mini alone (two runs) | 31, 34 |
| Guard, `check` re-electing each argument | 22 |
| Guard, `check` verifying the proposed values (list arguments wrongly ungrounded) | 16 |
| Guard, `check` verifying, lists grounded element by element | 18 |
| Guard, `check` verifying, after the diagnostic fixes (content summaries, field-level grounding, handoff tier, nested descriptions) | 15 |
| Guard, grounding only (`check(verify=False)`: code grounding, no Jev call) | 29 |
| **Guard, grounding only, refined** (expressions, object keys, dates in words, quantities) | **31** |

The guard never ran an altered value (the first version could have: it re-elected an order id and an address line),
but it lets few writes through: of 212 write checks in the last run, 8 executed and 28 became confirm cards; 71 were
blocked by grounding (a proposed value matched no value from the customer or a trusted lookup exactly) and 103 were
diffuse (the verify Nouls multiply over every identity field, so a six-field address change falls below the confirm
band). τ² has no injected or malicious tool output, so a guard can only cost tasks here.

A 20-task diagnostic run logged every block: no normalization misses, but free-text summaries treated as identity
values, objects composed from several sources (a search result's flight number plus the searched date), a handoff
inferred as a money transfer, and product variants Jev could not verify because their options were not described
(DECISIONS "What the guard diagnostic changed"). After those fixes grounding blocked 15 calls instead of 71, but Jev's
verification still stopped most of the rest: 105 diffuse and 34 refused, 6 of 189 executed.

Grounding alone keeps nearly all of the LLM's success: 29 of 60, within the two plain runs' spread (31, 34), with 96 of
117 write checks executed unchanged and no model call for the guard. Of its 21 blocks, one caught a real fabrication
(`credit_card_2135`, made up from the card's last four digits); the others were arithmetic in `calculate` (15), an id
that appears as a profile key, a birth date the user wrote in words, and computed payment amounts, all handled since
(field grounding checks an expression's numbers, object keys, dates in words, and leaves quantities to verification).

With those refinements the grounding-only guard solved 31 of 60, the LLM's own level, and blocked 6 of 101 writes:
three payment ids the LLM made up (`credit_card_2135` from the card's last four digits, and the placeholders
`credit_card_default` and `credit_card_0000000`; after the last two blocks the LLM looked the real id up and solved
the task), an address the LLM normalized (`NY`, `USA` for the user's "New York"), and one flight list. An LLM agent
with `check(verify=False)` on every write lost no tasks here and stopped fabricated identifiers, at no model cost.

The next-call bench after these changes (trust `none` / `reads`, two stages): reads 68.0% / 67.4%, writes shown /
executed with trusted lookups 36.4% / 26.0% (was 35.3% / 25.4%), talk turns 97.3% / 94.8%: no regression.

## Does one calibration fit every bench? (2026-10-01)

Every recorded decision that proposes a call was re-decided offline (no API cost) and labelled right or wrong: the
held-out app bench and its controls, τ² (both trust settings), BFCL and When2Call (the latter by tool name only;
the dataset has no argument gold). Share of proposed calls that are right, read tier, by confidence C:

| C | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–0.95 | 0.95–1.0 |
|---|---:|---:|---:|---:|---:|
| App (held-out) | 48% | 48% | 54% | 62% | 87% |
| τ² | 39% | 45% | 67% | 77% | 95% |
| BFCL | 16% | 27% | 32% | 48% | 69% |
| When2Call | 48% | 53% | 63% | 59% | 73% |

C ranks decisions on every bench, but the same C means different odds on different benches. An isotonic calibration
fitted on three benches and applied to the fourth lowers the calibration error (ECE 0.20–0.40 → 0.10–0.22) but never
singles out calls that are 90% right on the bench it did not see (τ² writes: 47% above the fitted cut). So no shared
calibrator ships; calibration is per app (`jevtools tune` on the app's own labelled decisions), and the default
thresholds stay priors.

## Reading a live run

Run with a key, then compare:
- **within ceiling** is Jev's accuracy on the cases jevtools *can* get right. It is the number to watch for the
  quality of Jev's judgments on these question forms.
- **correct (app) or proposal (BFCL) vs ceiling** is the total gap: coverage and judgment together.
- **wrong executions** and **injections** (app) are the safety numbers. The policy's thresholds are priors; tune
  them with `jevtools eval` and `jevtools tune` on your own traffic before trusting execute.
- **strict vs proposal** (BFCL) is what the default policy withholds for confirmation or clarification.

The report JSON has, per case, the outcome, rule, proposed call, the failure stage (app) or BFCL error type,
coverage (with `in_text` for BFCL) and usage (`jev_calls`, `input_tokens`, `cost_usd` on OpenRouter). The same cases
can be re-run with `--backend cassette:<path>` after recording with `JEVTOOLS_CASSETTE_MODE=record`.

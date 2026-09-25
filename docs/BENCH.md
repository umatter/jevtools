# Benchmarks

jevtools ships two benchmarks, and both run offline:

- **App domains** (`jevtools bench app`): assistants over an app's own data, the setting jevtools is built for.
  Arguments come from the user's words, the app's records and closed sets.
- **BFCL** (`jevtools bench bfcl`): the Berkeley Function Calling Leaderboard. It is the stress test outside that
  setting, where many values have to be written or reformatted rather than chosen.

Both report a **ceiling** first. The ceiling is the accuracy of an **oracle**, a backend that answers every Jev
question perfectly from the gold label. It shows how often jevtools *can* produce the right decision, given the
candidates code nominated, the question layout, decoding and policy. **It is not a measurement of Jev.** With an API
key, the same commands run live Jev and report its accuracy next to the ceiling. One live run of the app bench
has been made (below); none of BFCL.

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

Six synthetic apps with 99 labelled cases. The data is generated deterministically
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
| workspace | 15 | 100% | 0 | 0/1 | |
| helpdesk | 16 | 100% | 0 | 0/1 | |
| research | 15 | 100% | 0 | 0/1 | |
| **all** | 99 | 99% | 0 | 0/6 | plan 1 |

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

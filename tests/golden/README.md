# Golden conformance fixtures (spec §10.2)

Each directory under `tests/golden/` is one case of the §13 scenario. The files hold everything needed to replay the
case byte for byte, with no Jev key and no network: the inputs, the Ballot, every Jev request and response, and the
Decision and Trace documents. `test_golden.py` replays each case from its files, with `case.json` driving the run.
A port of the protocol, such as the planned R package (§7.3), passes when it reproduces the same bytes.

> **Evidence.** The responses are the scripted numbers of the §13.3 walk-through. They are illustrative ([I]) and
> were never measured. The fixtures check bytes, plumbing and policy branches. They are **never evidence about Jev's
> accuracy**. The offline `LexicalSimulator` is not used here, and its outputs are not evidence either.

## Cases

| Case | What it pins | Jev calls | Final outcome · rule |
|---|---|---|---|
| `R1` | Read tier: weather in Zurich in Fahrenheit executes. | 1 | `execute` · `P9.read.execute` |
| `R2` | External tier with history: confirm the email to Anna Keller. | 1 | `confirm` · `P9.external.confirm_band` |
| `R2-no-history` | No history: a clarify menu of three complete calls. | 1 | `clarify` · `P9.external.ambiguous` |
| `R2-click` | The click on Anna Rossi binds and confirms (no Jev call), then executes. | 1 | `execute` · `P9.external.confirmed` |
| `R3` | Critical tier: joint Choice, L/J, confirm (never auto-executes). | 1 | `confirm` · `P9.critical.confirm_band` |
| `R3-TOCTOU-changed` | Confirmed, then the Savings balance drops to 100.00. TOCTOU re-plans in a new round. | 2 | `clarify` · `P9.critical.ambiguous` |
| `R4` | Read tier over 3,000 files: the payments config executes. | 1 | `execute` · `P9.read.execute` |
| `R4-widen` | A miss widens (buckets + group), then runs the hierarchy round, then asks an open question. | 3 | `clarify` · `P7.slot.shape` |
| `R5` | External tier: confirm, with the Tue 6 Oct alternative. | 1 | `confirm` · `P9.external.confirm_band` |
| `R6` | The agent loop: read the invoice, confirm the forward, the click executes, done (`loop.json`). | 2 | `execute` · `P9.external.confirmed` |
| `R6-step1` | Loop step 1: superlative member Nouls, `read_file` executes. | 1 | `execute` · `P9.read.execute` |
| `R6-step2` | Loop step 2: forward to finance (confirm). The injected address is never nominated. | 1 | `confirm` · `P9.external.confirm_band` |
| `R6-injection` | Step 2 picks `transfer_funds`, but the amount exists only in the invoice: refuse. | 1 | `refuse` · `P3.safety.refuse` |
| `R7` | Small talk: `NO_TOOL`, abstain. | 1 | `abstain` · `P1.tool.no_tool` |
| `422-isolation` | A 422 names `get_weather.unit`. The family is dropped and the call is re-sent once. | 2 | `confirm` · `P9.external.confirm_band` |
| `budget-split` | At most 8 questions per call: probe-only tools are cut, the rest is split into two calls of one round. | 2 | `confirm` · `P9.external.confirm_band` |

## Files of a case

| File | Format | Content |
|---|---|---|
| `case.json` | indented JSON | The manifest (next section). |
| `catalog.json` | indented JSON | The six plain OpenAI tools of §13.2. |
| `context.json`, `context_<k>.json` | indented JSON | A `Context` document (`messages`, `now`, `tz`, `locale`, `user`, `observations`…). `sources` are source specs (`jevtools.sources.specs`). `contacts` and `files` read the shared rows in `fixtures/` through relative paths. The four `accounts` rows are inline. |
| `ballot.json` | canonical | The round-1 Ballot of the first step. |
| `request.json` / `response.json` | canonical | The only Jev exchange. With several: `request_<i>.json` / `response_<i>.json`, in call order. |
| `decision.json`, `trace.json` | canonical | The final Decision and its Trace. Earlier decisions of a multi-step case are `decision_<k>.json` / `trace_<k>.json`. |
| `loop.json` | canonical | Agent cases only: the run's outcome, rule, reason and steps. |

"Canonical" is the serialization of §3.1: UTF-8, every string NFC-normalized, non-ASCII unescaped, `,` and `:` with
no whitespace, and object keys in the normative order of each document (never sorted). Numbers are the shortest
round-trip decimal with no exponent, and integral floats print without `.0`. Decision and Trace numbers are rounded
half-even to 4 decimals. A failed Jev call stores `{"error": {"type", "status", "message", "detail"}}` as its
response.

Two trace fields are not deterministic, so both are normalized in `trace.json`: `created_at` is set to
`2026-09-24T12:05:00Z` (the scenario's `now` in UTC) and every `latency_ms` to `0`.

## The manifest (`case.json`)

- `model` and `backend`: the wire model id of every request (`~typesafe/jev-latest`) and the backend name recorded
  in traces (`scripted`).
- `policy`: the policy version. Every case uses the default policy of Appendix B.
- `limits`: overrides of the wire limits (`validate.Limits`). Only `budget-split` has one: `max_questions = 8`.
- `contexts`, `catalog`: the input files.
- `steps`: what to run, in order.
  - `decide`: the conversation and context of `context`, in `mode` (`turn` or `loop`).
  - `resume`: a click `selection` (or a free-text `reply`) on the previous decision's pending handle. A `context`
    replaces the host context, which is how `R3-TOCTOU-changed` drains the balance.
  - `agent_run` / `agent_resume`: the §6 agent loop. The tool executions it performs are listed under `executions`
    (`tool`, `arguments`, `result`). A replay returns the recorded results in order.
- `exchanges`: every Jev call as `{step, request, response}`. `ballot_requests` lists the files that
  `ballot.to_requests(model)` must produce: the first call(s) of the first step.
- `decisions` and `traces`: the output files in decision order. The last entries are always `decision.json` and
  `trace.json`.
- `verify.with_context`: whether `verify` gets the decision's context, which makes it rebuild the round-1 Ballot.
  This is `false` for `422-isolation`, because the traced Ballot is the isolated one, and for `budget-split`,
  because verify compiles with the default limits. Re-decoding, values, channels, composition and policy are still
  checked.
- `expect`: a readable summary of the final decision. The bytes are in `decision.json`.

## What the test asserts

For every case, `test_golden.py`:

1. **Compile.** Loads `catalog.json` and the first context, then checks `compile(catalog, context)` against
   `ballot.json`, byte for byte.
2. **Requests.** Checks `ballot.to_requests(model)` against the `ballot_requests` files, byte for byte.
3. **Decisions.** Runs the steps through a replay backend. The `i`-th request the engine sends must equal
   `request_i` byte for byte, and gets `response_i` (or its error) back. Each decision must equal its
   `decision*.json` file, byte for byte after dropping `created_at` keys (the Decision document has none today).
4. **Traces.** `verify(trace)` must pass on every stored trace. The regenerated trace, normalized as above, must
   also equal the stored one.

`test_fixtures_are_up_to_date` also regenerates every case from the scripts and compares the result with the stored
files.

## Regenerating

```sh
jevtools fixtures --update --dir tests/golden     # rewrite every case (stale files are removed)
jevtools fixtures --update --case R2 --case R5    # only some cases
jevtools fixtures --dir tests/golden              # check only: exit 1 and list what is missing, differs or is stale
```

Run it from a repository checkout: the case definitions live in `tests/golden/cases.py`, and the answers come from
`tests.scenario.scripts`, which re-exports `jevtools.demo`. §10.2 says the fixtures change **only** together with a
spec version bump. A diff in `decision.json` or `request*.json` is a protocol change, so review it the same way you
would review a spec change.

## Using the fixtures in a port (e.g. the R package)

A port does not need the Python generator. It needs the JSON files, a canonical-JSON writer and a replay backend:

1. **Canonical JSON.** Write the §3.1 form exactly. In R that means: keep list and name order, unbox scalars, write
   `null` for `NULL`/`NA`, print numbers as the shortest round-trip decimal, keep UTF-8 unescaped, and add no
   whitespace. `jsonlite::toJSON(x, auto_unbox = TRUE, digits = NA, null = "null", na = "null")` gets close. Check
   the number format and `[]` vs `{}` for empty values against the fixtures, and NFC-normalize strings
   (`stringi::stri_trans_nfc`).
2. **Replay backend.** On the `i`-th call, serialize the request canonically, compare it with `request_i` (fail on
   any difference) and return `response_i`. For an `{"error": …}` response, raise the matching error: a
   `JevValidationError` with its `detail` for a 422.
3. **Conformance from the Ballot onward (required, §7.3).** Load `ballot.json`, emit
   `to_requests("~typesafe/jev-latest")` and compare with the `ballot_requests` files. Then run the steps with the
   replay backend and compare every Decision document (`decision*.json`) and `loop.json`. §7.3 allows a port's
   extractors to differ, so this level does not require your compiler to reproduce `ballot.json`.
4. **Full conformance (optional).** Build the context from `context.json`: source specs, with rows from `fixtures/`
   resolved relative to the case directory. Compile it with `catalog.json` and compare with `ballot.json`. A
   mismatch here but not in step 3 points at extraction or pool building (recall is measured by E9, §11.2).
5. **Traces.** Compare your trace with `trace.json` after the same normalization (`created_at`, `latency_ms`), or
   run your port's `verify` on the stored `trace.json`.

A testthat loop over the case directories is enough:

```r
for (case in list.dirs("tests/golden", recursive = FALSE)) {
  if (basename(case) == "fixtures") next
  m <- jsonlite::read_json(file.path(case, "case.json"))
  # 1. ballot.json → requests; 2. replay the steps; 3. compare decision*.json; 4. compare/verify traces
}
```

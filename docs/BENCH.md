# Benchmarks: BFCL

`jevtools bench bfcl` runs jevtools on the single-turn categories of the
[Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
(BFCL v4, Apache-2.0). It scores every decision with a port of BFCL's own AST checker, so the numbers are
comparable with BFCL's.

BFCL was built for models that *write* arguments. jevtools *elects* them from candidates that code nominates. That
makes BFCL a hard, honest test of the approach's main weak spot, candidate coverage: Jev cannot choose a value
that no extractor nominated.

## Quick start

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

## What is measured

Every case gets its own `Router` over the case's functions, with a fixed clock (2026-09-25 10:00 UTC).

| Column | Meaning |
|---|---|
| **ceiling** | Accuracy of the **oracle**: a backend that answers every Jev question perfectly, using the BFCL answer. It shows how often jevtools can emit the right call at all, given the candidates code nominated, the question layout, decoding, normalization and policy. It is not a measurement of Jev. |
| **proposal** | The proposed call counts whatever the outcome (`confirm`/`clarify` included), unless the outcome is `abstain` or `refuse`. This is the call a user would see on a card. |
| **strict** | Only an `execute` decision emits a call: what an autonomous agent would run. |
| **within ceiling** | Proposal accuracy on the cases whose ceiling passes. In a live run this isolates **Jev's judgment** from extractor coverage. |

Every case is decided by the oracle first (no network), so a live run reports its ceiling next to its accuracy.
Failures are attributed to the most upstream cause:

| Attribution | Meaning |
|---|---|
| `tool_not_speculated` | a required parameter had no candidate, so the tool was not asked about (P6 → clarify) |
| `value_not_nominated` | a value BFCL expects was on no option |
| `no_call` | the decision proposed nothing (e.g. a clarify without a proposed call) |
| `wrong_call` | everything was on the ballot, but the call that came out is wrong. For the oracle this is a decoding or normalization issue; for a live run it is mostly the model's choice. |
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

## Baseline: the ceiling (oracle)

BFCL `main` as of 2026-09-25 and jevtools 0.1.0, default categories, `--risk read`. Every number here is the
oracle's, **not Jev's**: no live run has been made yet (this environment has no API key).

| Category | n | Ceiling | Strict (oracle) | Misses in text | Misses not in text |
|---|---:|---:|---:|---:|---:|
| simple_python | 400 | 37.2% | 37.0% | 334 | 96 |
| multiple | 200 | 38.5% | 38.0% | 163 | 41 |
| live_simple | 258 | 32.9% | 31.0% | 172 | 103 |
| live_multiple | 1,053 | 32.9% | 31.0% | 626 | 662 |
| irrelevance | 240 | 100.0% | 100.0% | – | – |
| live_irrelevance | 884 | 100.0% | 100.0% | – | – |
| live_relevance | 16 | 68.8% | 0.0% | – | – |
| **all** | 3,051 | 58.7% | 57.5% | 1,295 | 902 |

"Misses" count parameters, not cases. With `--categories all`, the four parallel categories (440 cases) score 0%.

Estimated cost of a live run of the default categories: about 1,400 input tokens per call on average (851
median, the largest about 23k). That is roughly $0.18 for all 3,051 cases at $0.042 per million input tokens. It
is an estimate at 3.5 characters per token, not a measured figure.

### Where the ceiling is lost

Attribution over the single-call categories (1,911 cases):
- `value_not_nominated`: 966
- `tool_not_speculated`: 277
- `wrong_call`: 11

(The 5 `no_call` cases are all in live_relevance.)

The single-call categories are therefore limited almost entirely by **coverage, not by decoding or policy**.
Parameter coverage by kind (all default categories):

| Kind | Covered | Omitted/default (acceptable) | Missed |
|---|---:|---:|---:|
| enum | 939 | 66 | 26 |
| flag | 542 | – | 6 |
| quantity | 615 | 226 | 94 |
| money | 76 | 5 | 10 |
| list | 127 | – | 91 |
| span | 765 | 406 | 937 |
| temporal | 0 | 33 | 257 |
| text | 28 | – | 56 |
| (not asked: tool not speculated) | – | 433 | 720 |

What the misses are, most common first:
- **Spans.** Single words and short phrases are not nominated: `all`, `inches`, `simpson`, `water`, identifiers like
  `DNA123`, math expressions such as `x^2`. BFCL's string comparison already treats `x^2` and `x**2` as equal.
  The span extractors favour noun chunks and clauses.
- **Temporal.** BFCL date and time parameters are *strings in the user's or a stated format* (`"08-16-2022"`,
  `"3pm"`). The temporal resolver emits ISO 8601 datetimes, so none of them matches. A string-typed temporal slot
  should also offer the verbatim mention and the common formats.
- **Tools not speculated.** A required parameter with no evidence-backed candidate keeps the tool off the fan-out.
  Most of these values are in the text, missed by the extractors above.
- **Lists.** Lists of numbers and coordinates are partly nominated.
- **Quantities.** Numbers with written-out compound units are dropped by dimension filtering. In "starting from
  a speed of 10 meters/second", the 10 is not offered for `initial_velocity`, while the 5 from "5 seconds" is.
  `300K` is read only as 300,000, never as 300 kelvin.
- **Not in the text (41%).** Formats and knowledge that selection cannot produce: `"New York, NY"` from "New York",
  `0.05` from "50 mH", `C6H12O6` from "glucose", dates in a stated format. These need a normalization catalog, a
  source, or the generative fallback (a Filler whose drafts Jev must accept).

### Bugs the bench found

These were fixed, with regression tests:
- a fractional amount crashed an integer money slot;
- a text candidate over the 4,000-character accept limit made the pre-send validator reject the whole ballot;
- list items came out in canonical label order instead of mention order.

## Reading a live run

Run with a key, then compare:
- **within ceiling** is Jev's accuracy on the cases jevtools *can* get right. It is the number to watch for the
  quality of Jev's judgments on these question forms.
- **proposal vs ceiling** is the total gap: coverage and judgment together.
- **strict vs proposal** is what the default policy withholds for confirmation or clarification. These thresholds
  are priors; tune them with `jevtools eval` and `jevtools tune` on your own traffic.

The report JSON has, per case, the outcome, rule, proposed call, BFCL error type, coverage (with `in_text`) and
usage (`jev_calls`, `input_tokens`, `cost_usd` on OpenRouter). The same cases can be re-run with
`--backend cassette:<path>` after recording with `JEVTOOLS_CASSETTE_MODE=record`.

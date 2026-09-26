# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`jevtools` builds tool calls with TypeSafe's **Jev**, a decision model that returns calibrated probabilities over
typed questions (Choice, Noul = yes/no, Score) and never returns text. The core rule is **"bind, don't write"**:
code builds a finite candidate pool for every argument, and Jev *elects* one candidate per slot. The rule holds
only if no argument value can come from anywhere except a nominated candidate, so never add a code path that
synthesizes a value. The package is v0.1.0 with protocol `jevtools/0.1`. **Nothing has been measured against live
Jev.** Every number in the docs, fixtures and examples comes from scripted answers, the `LexicalSimulator` or the
benchmark oracle, so never present such a number as evidence of Jev's accuracy.

## Commands

```bash
uv venv && uv pip install -e ".[all,dev]"      # dev setup (.venv is gitignored)
python -m pytest -q                             # ~1,400 offline tests, ~2 min; never touch the network
python -m pytest tests/unit/test_policy.py -k tier_ordering    # one file / one test
python -m pytest -m fast                        # quick scripted/simulator tests (examples)
python -m pytest -m live                        # live smoke suite; skipped unless TYPESAFE_API_KEY / OPENROUTER_API_KEY
ruff check . && ruff format --check .           # line length 120; docs/ and README.md are excluded on purpose
mypy                                            # strict, configured in pyproject (files = src/jevtools)
```

Generated artifacts, each with a checker that a test enforces:

```bash
jevtools fixtures [--update] [--case R4]                 # golden conformance fixtures in tests/golden/
python examples/fixtures/regenerate.py [--check]         # examples/fixtures/R*.answers.json + examples/proxy/data/
python -m jevtools.bench.app._generate [--check]         # app-bench domain data under src/jevtools/bench/app/domains/
jevtools bench app                                       # oracle ceiling on the 6 app domains (104 cases)
jevtools bench bfcl [--download]                         # BFCL stress test (tests use the vendored sample in tests/bench/data)
```

**Golden fixtures depend on the Python version.** They were generated on Python ≤ 3.11. Python 3.12 changed float
`sum()` to compensated summation, which moves the last digit of some probabilities, so on 3.12
`tests/golden/test_golden.py` fails in R1, R5 and `budget-split` even on a clean tree. Run the golden tests under
3.10 or 3.11. Don't run `fixtures --update` on 3.12 to make them pass: that rewrites nine fixtures for float noise.

## Docs are normative, and they are tested

- `docs/SPEC.md` is the normative protocol, and code and tests cite it by section (`§3.8.5`). Its code blocks are
  quoted verbatim and are never reformatted.
- `docs/DECISIONS.md` records every place the implementation settles an ambiguity or departs from the spec, in the
  format `§section: issue → decision`. **When a change alters specified behaviour, add an entry there**, in the
  section for the area it touches.
- `docs/ARCHITECTURE.md` is the data flow and module map. Read it first; this file does not repeat it.
- `docs/BENCH.md` holds the benchmark tables. Rerun and update them when a change moves the oracle ceiling.
- `tests/docs/test_docs.py` checks claims in the README and docs against the code: install commands name the git
  source, "calibrated" is used only after a calibrator is fitted, and the probe's request count matches the probe.
  README edits can break tests.

## Architecture in brief

`Router.decide(messages)` runs a pipeline: ingest → extract → pool → plan → ask → decode → compose → decide → emit
→ record. Only **ask** talks to Jev. Everything from the Ballot onward is byte-specified and pinned by
`tests/golden/`. A port, such as the planned R package, conforms when it reproduces those bytes, so any change to
canonical serialization, qids, question templates or document key order is a protocol change.

Invariants that live in several modules:

- **Resolvers are the extension point.** Kind-specific logic is in `kinds/<kind>.py` (`Resolver.pool`,
  `questions`, `decode`, optional `widen` and `clarify_values`). `plan.py`, `decode.py` and `router.py` go through
  that interface only. Don't put kind-specific branches in them.
- **Channels are a security boundary.** Every candidate carries a channel (user, author, registry, tool_output…).
  Allow-lists keep tool-output text out of the identity and amount slots of any write, send or critical tool.
  Coreference, history and resume must preserve a value's traced origin and never launder it into a trusted
  channel (DECISIONS "trust follow-ups", §6.4).
- **Fail closed.** A missing or failed Jev answer, a missing `authorized` answer or any backend error must never
  lead to `execute`. Sentinel mass (`NOT_STATED`, `NONE_OF_THESE`) counts as error and is never renormalized away.
- **Prompts are templates filled with candidate labels, never generated text** (`templates.py`, `prompts.py`).
- **Confidence C is not a calibrated probability** until a calibrator is fitted with `jevtools tune --calibrate`.
  Keep `confidence.calibrated == False` otherwise, and keep the wording in the docs to match (a test checks it).
- **The router loop is one generator that yields effects.** `decide` drives it synchronously (split calls run in a
  thread pool), and `adecide` drives it with `asyncio.gather`. Change the generator, not the drivers.

## Tests and test doubles

- Test backends: `ScriptedBackend` (fixed answers, including `.from_fixture`), `LexicalSimulator` (an offline
  lexical double, **not a model**), `Cassette` (record and replay) and the bench `OracleBackend` (perfect answers
  from a gold label, which measures coverage). Give a `Context` a fixed `now` for anything deterministic, because
  the state includes the current time.
- The §13 demo world (contacts, accounts, 3,000 paths, six tools) lives in `src/jevtools/demo/`.
  `tests/scenario/{fixtures,scripts}.py` and `tests/loop/support.py` are thin re-exports kept for older names.
- A golden case's layout and manifest are documented in `tests/golden/README.md`, and the cases are defined in
  `tests/golden/cases.py`.
- App-bench cases use the §11.1 `EvalCase` format, so each `domains/<name>/cases.jsonl` is also a `jevtools eval`
  dataset. The one remaining ceiling miss (inbox-17) is kept on purpose.

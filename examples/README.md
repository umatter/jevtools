# jevtools examples

Runnable scripts for the spec's walk-through (§12, §13) and for your own tools. Every example is offline by default:

```bash
python examples/01_quickstart_weather.py                   # scripted (default): replays examples/fixtures/*.answers.json
python examples/01_quickstart_weather.py --backend sim     # the offline LexicalSimulator
python examples/01_quickstart_weather.py --backend live    # jevtools.backends.auto(): TYPESAFE_API_KEY or OPENROUTER_API_KEY
```

**About the backends.**

- `scripted` replays the spec's illustrative [I] numbers.
- `sim` is a lexical test double, not a model.
- `live` calls Jev.

Neither scripted nor simulated answers are evidence about Jev's accuracy. They show the protocol: which questions
are asked, how the answers are decoded, and which policy rule fires.

**What each example prints.**

- the questions sent to Jev, with their count and ids grouped by tool;
- the decision: outcome, policy rule, proposed call, and confidence (C, W, PI, L, J);
- the slots and the rendered prompt;
- a short trace summary.

The demo world is `jevtools.demo`, with synthetic contacts, accounts, 3,000 workspace paths and a fake mailbox.

| File | Mirrors | Shows |
|---|---|---|
| `01_quickstart_weather.py` | R1, R7 | `@jt.tool` setup with zero sources; span claiming; NOT_STATED pooled into the home city; abstain; OpenAI message output |
| `02_email_contacts.py` | R2 | contacts registry; `authorized`; late-bound subject and body; confirm with history; clarify menu without; click resume; free-text resume round |
| `03_transfer_confirm.py` | R3 | critical tier: joint Choice, `present`/`rev` probes, `C = min(L, J)`, confirm card with an alternative, idempotency key, `jt.verify`, TOCTOU revalidation |
| `04_file_shortlist_widen.py` | R4 | `FileIndex` over 3,000 paths; K = 40 shortlist; a miss widens to buckets and a directory group, then the hierarchy; clarify(open) |
| `05_calendar_temporal.py` | R5 | every temporal reading; anchored attendees with `EXCLUDE` and `more`; invitee rule; confirm with an alternative; the compiled request JSON |
| `06_agent_loop_invoice.py` | R6 | `jt.Agent`; superlative member Nouls; observation → pools; the injected instruction is never nominated; `transfer_funds` refused; done after 2 rounds |
| `07_openai_dropin.py` | R1, R2 | `jt.openai.complete` with a plain OpenAI tools list, `role: tool` messages, and a confirm card over a stateless client |
| `08_bring_your_own_tools.py` | — | your own domain (a support desk): `@jt.tool` + `Annotated` markers, `jt.Registry` sources, `jt.hints` for a tool you do not own |
| `proxy/` | §7.2.4, §7.3 | `jevtools serve` with a `jevtools.toml`; an OpenAI SDK client; the R `ellmer` snippet |

`_show.py` is the shared printing helper. `fixtures/regenerate.py` rewrites `fixtures/R*.answers.json` from
`jevtools.demo.scripts` and `proxy/data/*.json` from `jevtools.demo.scenario`. Pass `--check` to verify them
instead. `fixtures/helpdesk.answers.json` is written by hand. Its format (`answers`, or one script per request under
`rounds`) is what `ScriptedBackend.from_fixture` reads, so you can script your own tools the same way.

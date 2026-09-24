# Implementation decisions

Where the spec (`docs/SPEC.md`) is ambiguous or inconsistent, the implementation picks the simplest correct
behaviour and records it here. Format: `§section: issue → decision`.

## Foundation (canonical, candidates, context, spec, ballot, decision, validate, kinds/base)

- §3.1: "shortest round-trip decimal" does not say how integral floats print → integral floats print without a
  fraction (`1.0` → `1`), so equal JSON numbers hash equally in every port; `round4` rounds half-even on the
  shortest repr (`Decimal(repr(x))`, so `0.12345` → `0.1234`).
- §3.1/§3.10: "numbers in Decision and Trace are rounded" would also round argument values and costs → only
  probabilities, confidences, thresholds and gates are rounded; `call.arguments`, slot values and `usage.cost_usd`
  are emitted unchanged (the R5 decision shows `cost_usd: 0.000072`).
- §3.2: the spec uses `anchored` (§4.2.9, `jt.ListOf`), `tolerant` (§3.6 ordinal decoding) and `canon` (§4.2.6)
  but the key table omits them → they are accepted parameter-level `x-jev` keys. `emits` accepts
  `items, key, label, describe, types`.
- §3.2 precedence: for Python callables, `Annotated` markers are the lowest layer (below sidecars) while
  `@jt.tool(**keys)` counts as inline tool-level `x-jev`; a pydantic model's `json_schema_extra={"x-jev": …}` at
  model level is inline tool-level `x-jev`.
- §3.2 sidecar keys: list items are addressed `tool.list[]`, fields of item objects `tool.list[].field`; union
  branches add no segment. The structured form `{"tools": {...}, "sources": [...]}` also carries source
  registrations. Tool names may contain dots, so an entry belongs to the longest tool name that prefixes it.
  Annotations (sidecar, hints, MCP `_meta`) for parameters that do not exist are errors.
- §3.2 `intent` default: the *first sentence* of the description (first letter lowercased unless the first word is
  an acronym, trailing period removed); without a description, the humanized tool name.
- §3.2 `noun` default: `"the " + description` would give "the the recipient's…" for the §13 descriptions → the
  description phrase is used as is when it starts with `the/a/an/your/this`, else `"the "` is prepended; without a
  description, `"the " + humanized name`.
- §3.3.1 row 11: matching *descriptions* against source tags would turn `subject` ("The subject line of the email")
  into an email reference whenever a contacts source provides `email` → only the name, its derived tags
  (`*_account` → `account_id`, `*_id` → prefix, `path/file` names) and the format are matched.
- §3.3.1 row 1: `readOnly` → `derived`; `writeOnly` or an exact secret name (`password, token, api_key, apikey,
  secret, credential`) → `secret`.
- §3.3.1 rows 5/7: temporal names match exactly or by suffix (`*_at`, `*_date`, `*_time`); ordinal names match any
  name token (`priority_level`). Row 10 also maps `^[a-z]{2}$` + "language" to `iso639`.
- §3.3.1 rows 12–13: objects deeper than 3 levels or without properties become a `record` with no children
  (marked weak); list items get the path segment `"[]"` and share the list's qpath; union branch leaves get the
  qpath segment `b<i>` so two branches may share a property name.
- §3.3.1: an `x-jev.source` on an inferred span makes it a `ref`; a source declared on a list applies to its items;
  lists default to `anchored` when their items are `ref`.
- §3.3.1: a `default: null` on an optional slot means "omit" (NOT_STATED decodes to `⊥omit`, no probe).
- §3.3.1 stakes: text slots whose names are not listed default to `content`; other kinds use the key-table rule
  (`subject/title/query/label` cosmetic, `body/message/content/text` content, else identity).
- §3.3.1 extractors: `ipv6` and `hostname` formats use `regex:` extractors (the extractor vocabulary has no names
  for them).
- §3.3.2: several explicit MCP hints → table order: `readOnlyHint` (read), then `destructiveHint` (critical), then
  `openWorldHint` (external); the invitee rule applies only to verb-derived `write` tiers.
- §3.4.2: "history (trusted origin only)" → a `history` candidate is admitted only if its `origin` channel is also
  admitted (or trusted); critical-tier quantity/money slots allow `user` and `registry`.
- §3.4.3: collisions → a display equal to a reserved sentinel (after casefold) gets ` (value)`; a duplicate of an
  earlier label gets ` (2)`, ` (3)`…; slug numbers count from 1 in input order. When a label cannot show the whole
  value (elided path, slug), `text` gets `Full value: "…"` (≤ 400 characters in total).
- §3.5.1: `state.system` (with `include_system`) is placed after `user`; loop sections are present in loop mode
  *and* whenever the context has observations, so widen/fill/resume rounds of a loop step see the same state.
  `history` holds every user/assistant text turn except the latest user message.
- §3.5.2: digit-only and `m<N>` segments are reserved too, as are `authorized`, `joint` and `done_after` (they
  would collide with tool-level ids). The validator checks charset/structure (records legitimately use
  `T.P.m0.field`), not the full family grammar.
- §3.5.3/§5.2: within a slot, families follow §5.2: the slot's own families first, then `present`, then `rev`
  (`ballot.FAMILY_ORDER`).
- §3.5.7: to make Ballot → JevRequest a function of the Ballot alone, sentinel entries carry their `text` (plus
  `value`, `late`, `channel`, `display` for defaults); `decodes_to ∈ {missing, uncovered, excluded, omit, default,
  no_tool, unsupported, done, other, cancel}`. `criteria_null` means "this Noul is sent without criteria". Question
  documents add `criteria` (Noul), `levels` (Score) and `meta` (resolver decode data); `calls` is always explicit.
- §3.5.6 rule ids: `choice.options, label.grammar, label.reserved, label.unique, description.length,
  instructions.length, accept.candidate, qid.grammar, qid.unique, call.questions, call.tokens, value.schema`.
- §8.7 fallback for backends rejecting object instructions: `question` then one `<Key>: <canonical JSON>` line per
  other key (canonical JSON instead of Python's ASCII-escaping `json.dumps`, so every port can reproduce it).
- §3.6: a late-bound default (`default_from: "from_account.currency"`) cannot pool at decode time → its mass sits
  on the bottom `⊥default` until the orchestrator calls `SlotResult.bind_late_default(value, …)`.
- §3.6/§3.10 alternatives: up to 3 real runner-ups with p ≥ 0.01, most probable first.
- §3.10/§6.5: `‖` is plain concatenation: digest = `sha256(trace_id + name + canonical(arguments))[:16]`, shared
  by `call_jev_…`, `idem_…` and `toolu_jev_…` (the spec's examples show the same hex for call id and key). Native
  `tool_calls` entries are `{id, name, arguments, idempotency_key}`; the proposed `call` is `{name, arguments}`;
  `x_jev` of confirm/clarify messages is `{outcome, options, pending_id, trace_id}`. `DecisionIds.derive(*parts)`
  gives `dec_/tr_/pnd_` ids sharing one digest of the parts joined by `\x1f`.
- §4.2.2: named catalogs are always shortlisted to mentioned ∪ context-preferred members (R3 shows only
  `{CHF, EUR}` although ISO 4217 has < 252 codes); explicit enums are sent whole up to 252 members. Short all-caps
  codes match case-sensitively (so "all" never selects the currency `ALL`). Catalog data files are JSON arrays of
  `{"value", "text"?, "aliases"?}` under `jevtools/extract/data/<name>.json`; `iana_tz` comes from `zoneinfo`.
- §9 `schema.py`: `multipleOf` uses decimal arithmetic on the shortest repr (0.07 is a multiple of 0.01, unlike
  float-based validators); formats are asserted.
- §9 `constraints.py`: `check()` returns `True` (satisfied), `False` (violated) or a reason string (not
  evaluable); only `True` is feasible. Numeric coercion applies to ordering operators or when one side is a real
  number (ids like `"0123"` compare as strings); date-only strings compare as dates.
- §7.1 callables: a parameter named `idempotency_key` is never a slot (the executor passes it, §6.5); Google- and
  Sphinx-style docstring parameter descriptions become schema descriptions.
- §9: TOML on Python 3.10 needs `tomli`, which is not a hard dependency; `Policy.from_toml(path)` reads a file and
  `Policy.from_toml_text(text)` parses content.

## Engine (plan, decode, confidence, policy, prompts, trace, backends, router)

- §3.8.2 rule ids: `P0.backend.fail_closed`, `P1.tool.no_tool`, `P2.tool.unsupported`, `P3.safety.refuse`,
  `P4.safety.not_authorized`, `P5.tool.ambiguous`, `P6.tool.not_speculated`, `P7.slot.shape`, `P8.consistency`,
  `P10.loop.done`, and `P9.<tier>.<band>` (as in the spec's `P9.external.confirm_band`) with bands `execute`,
  `confirm_band`, `capped` (a gate or cap kept an execute-level C at confirm), `confirm_always`, `confirmed` (user
  confirmation, §3.8.5) and the shape-routing bands `ambiguous`, `missing`, `diffuse`.
- §3.8.2 P10: `DONE ≥ 0.5` is checked right after P2 (P3–P9 presuppose a tool, and P5 would otherwise turn a
  confident DONE with a close runner-up into a tool menu). The post-execute `done_after ≥ 0.8` check belongs to the
  loop (`policy.loop_done`); `done_after` is exposed in `Decision.gates`.
- §3.8.2 P5: the tool menu is offered only when both top labels are real tools (a sentinel in the pair →
  escalate / clarify(open), with the fixed text "What would you like me to do?"); `call_map_disagrees` always yields
  the menu between `t*` and `t_S` (§3.7.4 "forces CLARIFY between the two tools").
- §3.7.4: `call_map_disagrees` is raised only when `t*` was speculated; otherwise P3/P6 could never fire as long as
  any other tool was speculated.
- §3.8.2 P9 hysteresis: a threshold is cleared iff `C − τ ≥ h` (`|C − τ| < h` takes the safer side; equality
  clears). Gates (authorized, content accept, critical channels and `require_present`) and caps all cap an
  execute-level C at confirm (`capped`). In shadow mode an execute is returned as confirm (cap `shadow`).
- §3.8.2 P8: besides the three spec flags, decode-time `infeasible` (no feasible top-3 combination, or a final
  constraint fails) and `schema_invalid` (assembled arguments fail the JSON Schema) route to P8 (menu on the
  bottleneck), as §3.6 rule 4 asks for CLARIFY on the bottleneck.
- §3.6 rule 4: zero-mass values are not MAP candidates; constraints over a slot whose elected value is a bottom are
  skipped (that slot's shape routes the call); constraints that cannot be evaluated count as violated (fail closed).
  Late-default slots (`currency ← from_account.currency`) join the enumeration with their pooled mass per sibling
  choice, so the MAP is joint over the pair.
- §3.6 rule 5: a value whose late binding or schema check fails loses its mass (error, removed — not moved to a
  sentinel, never renormalized) and the next value is elected (flag `late_binding_failed`). Recipe evaluation is
  delegated to `jevtools.kinds.late.late_bind` (placeholders via `fill`, `derive` with money quantized to the source
  row's currency); the spec's list-only form `{"placeholders": ["to.first_name"]}` is mapped to `fill` by marker
  order, and a `⟨…⟩` marker left after substitution is an error (never emitted). The pre-substitution value is kept
  as `late.template`, so late binding is recomputed after a click (a new recipient's first name).
- §5.2 joint: a candidate is anchored when its channel is `user` or its `prov` has `anchor`/`mention`; a slot
  without anchored candidates contributes all of them. The `enum` resolver sets `prov.anchor` on members named in
  the user's words, so R3's currency contributes only CHF and the joint Choice has the spec's 6 options.
  Combinations that definitely violate a constraint among the group's slots are dropped; groups with fewer than two
  slots are skipped; qid `T.joint` for one group, `T.joint.<i>` for several.
- §5.5: exceeding `max_questions` also counts as over budget (the question cuts run before splitting). Cut 3
  re-plans every pool with `policy.pools.ref_k = 20` (resolvers read it). The only state cut implemented is dropping
  the oldest `history` turns (keeping pinned-entity turns needs the entity store).
- §5.6: a 422 whose `loc` names questions drops those questions' whole slot families (tool-level questions alone)
  and re-sends only the failed calls, once; a `loc` on `tool`, `reply` or outside `questions` fails closed (P0).
  The round record keeps the trimmed Ballot. `retries` in call records is `null` (the backend interface returns only
  the response).
- §3.8.5 clicks: `ok` always confirms; `pick:`/`alt:` options confirm in tiers ≥ external (they show complete calls);
  a confirmed call executes when C clears the tier's confirm threshold (read tier: execute threshold), caps do not
  block a human confirmation, TOCTOU still runs. A clicked slot gets factor 1 even for composite slots. Menus offer
  whole-slot values only (a list part's values would need resolver support). A tool-menu click runs one new round
  with that tool named (earlier answers are not reused); `cancel` → abstain (`P1.tool.no_tool`, reason
  `cancelled`); a pending not held in memory, or expired, is safely re-compiled.
- §3.8.5 free text: `CANCEL ≥ 0.5` in the reply Choice decodes as `NO_TOOL`; an option `≥ 0.5` binds its value with
  `p = P(reply)` (or confirms / picks the tool). A passthrough slot is bound verbatim after decoding (its questions
  are still asked in the resume round).
- §3.8.4 prompts: extra fixed texts `What would you like me to do?` (open, no slot) and `Do you want {noun} to be
  true?` (flag yes/no menu); the `ok` button reads `Confirm` in the critical tier, else the intent's verb (`Send`,
  `Create`); an alternative reads `<label> instead`; menu ids are `pick:<slot>:<i>` + `other`, tool menus
  `tool:<name>` + `cancel`. Confirm cards offer runner-ups of non-cosmetic slots only. The R2 card in §13.3 shows
  "[Anna Rossi instead]" at p = .07 < `alt_show_min` .10; the threshold rule wins.
- §4.7 Escalator: a text answer → abstain with that content, except under P0, where the outcome stays `escalate`
  (the handoff is the result). The gate round names the tool (`P(tool) = 1`) and asks `authorized`; proposed values
  not already in a pool enter as `generated` and are dropped wherever the allow-list does not admit them.
- §4.7 FILL: offered when a Filler is configured, no FILL ran yet, the slot admits `generated`, and its fallback is
  `fill` (or unset with content stakes). Only new or changed questions are asked; identical ones reuse answers.
- §4.6 widen: the router drives the kinds agent's `Widenable.widen(tool, slot, pool, rc, stage)` hook — stage
  `bucket`, then `hierarchy` — with `rc.widen[slot_key] = {"stage", "round", "groups": {group value: p}}` from the
  previous stage's `group` Choice; widen answers are decoded together with the round-1 answers (same state).
- §5.1 speculation miss: when `t*` is viable but was not speculated (budget cut, `speculate: never`), one re-plan
  round speculates it (`speculate_only` overrides `never`).
- §3.9 trace: the exact `PolicyInput` is stored (so `verify` re-runs the policy), `context.tz` is recorded, and
  bindings carry `late`. Without a context, `verify` cannot rebuild row attributes (never in the Ballot), so it
  skips late-bound bindings when re-decoding.
- §3.10 usage: `cost_usd` is the sum of reported `usage.cost` (`null` when the backend reports none); never estimated.
- §7.2.2: `parallel_tool_calls` is accepted and ignored (ext.parallel is out of scope); `tool_choice: "none"` →
  abstain with rule `P1.tool.no_tool` (reason `tool_choice_none`).
- §8.2/§8.3: the HTTP constructors honour `JEVTOOLS_MODEL` when no model is passed; a missing key raises
  `BackendConfigError` (a `BackendError`); HTTP 400 is a plain `BackendError` (not retried).

## Kinds, extractors, sources (extract, sources, kinds/*)

- §9 data: packaged data lives in `jevtools/extract/data/` (the §9 tree and the enum loader), not `jevtools/data/`:
  `iso4217.json` (with `minor` units), `iso3166.json`, `iso639.json`, `cities.json` (364 cities incl. ambiguity
  examples), `packs/email.json`, `packs/event.json`. Currency aliases that are common English words (real, won,
  rand, sol…) are left out so they never claim text.
- §4.2.1: assistant-turn (`history`) mentions enter pools only when the request carries an anaphor cue; otherwise
  history reaches pools through the entity store (coreference). Otherwise R2's "14:30" (an assistant turn) would make
  `create_event` viable, contradicting §13.3 (13 questions).
- §4.2.1 claiming: a mention is claimed when a more specific mention of the same text *covers* it (partial overlaps
  are not claims); bare numbers rank between patterns and places; cues never claim. Negation: a cue within 3 tokens
  *before* the mention, stopped by sentence punctuation and contrast words (`but`, `instead`, `aber`, `mais`).
- §4.2.5: `next <weekday>` = the next occurrence (+1…+7) and the one a week later; a bare weekday said on that
  weekday reads as today and +7; `this <weekday>` on that day is today. A time without a date is today if still ahead,
  else tomorrow. `3:30` (en, no leading zero, hour ≤ 11) is ambiguous like `at 3`; `15h` is a clock time only after
  `à/um/at`, with minutes, or in `fr`/`de`. A DST fold gives both occurrences (labels add the UTC offset to stay
  unique); a gap shifts forward with a note. A mention with several readings is described as
  `"<mention>" read as …`; a single reading as `From "<mention>" in the request.`
- §4.2.5: vague cues are recorded in `pool.meta["ranges"]` (for a clarify menu), never offered as points. Slots
  coupled by `x-jev.range`: the `min` slot asks one Choice over `(min, max)` pairs; the `max` slot asks nothing and
  decodes its component from that answer. Factorized questions use the nouns "the date of …"/"the time of day of …".
- §4.2.11 text: normalization depends on the role — content gets a capital and final punctuation, titles/subjects a
  capital only, queries nothing, and templates are rendered as written (the R2/R5 requests require all four).
- §4.2.11 ladder: queries = [request minus command verb (and one article or object pronoun), main noun-chunk core],
  plus the whole request when fewer than two distinct remain (for questions: the first noun chunk); titles = message
  clauses, quotes and the command object's `full`/`head`/`core` variants (the row-15 default `clause, quote` cannot
  produce the R5 titles); extracted rungs are in canonical order, templates first. Packs are JSON entries
  `{when, requires, templates}`; placeholders `{message} {duration} {object} {topic} {attendee_names}
  {observation_title} {user.<field>}` are early-bound, `{recipient.first_name}` and `{observation}` late-bound
  (`late = {"placeholders": [paths], "fill": {"⟨…⟩": path}}`; paths `<slot>.<attr>` or `obs:<step>`).
- §3.4.3 vs §13.5: mention Choices list `EXCLUDE` before `NONE_OF_THESE`, as the R5 request does.
- §4.2.7 registries: prefix/trigram matches only for capitalized words (or all-lowercase messages); adjacent matching
  words sharing a row form one anchor; alias fields are `alias`/`aliases`; option text is
  `<Item> matching|whose alias is|similar to|in the group "<mention>": <describe>.`, unanchored rows of a whole
  registry `<Item>: <describe>.`. Ref pools also take literal values of the slot's format typed in the request (an
  email not in the contacts); untrusted ones are then blocked by the allow-list.
- §4.4 FileIndex: synonyms are collapsed onto their head term (one BM25 term per group) instead of expanding the query
  (expansion ranked `auth/config/settings.toml` above the payments config); generic nouns (file, document, folder)
  never make a hit on their own; `date` comes from `YYYY-MM-DD` in the file name, else `mtime`.
- §4.2.8: superlative retrieval text is the noun phrase containing the cue minus the cue ("invoice from ACME");
  `M = {q > 0.5}` among items with a known order attribute; an empty `M` is `out_of_pool` with `⊥uncovered` mass
  `∏(1 − q)`.
- §4.2.9: `more ≥ 0.5` gives shape `missing` with flag `more`; a group mention is an anchor that matches only the
  registry's `groups` field. Arrays of objects (basic): one record per anchor of the first `ref` field, each field
  asked with path `(…, "[i]", field)` and qid `P.m<i>.<field>`, plus `more`.
- §4.2.10/§4.2.3: the union `branch` Choice carries `NONE_OF_THESE`; a flag with a default is a Choice
  `{true, false, NOT_STATED}` (no `NONE_OF_THESE`).
- §4.6 widen: bucket Choices carry only `NONE_OF_THESE`; the bucket factor is `D_b(v*) · ∏_{b'≠b} D_b'(⊥uncovered)`;
  the hierarchy stage reads the previous group answer from `rc.widen[slot_key]["groups"]` as `{group: p}`.
- §4.3 normalizer names: `string@1 span@1 email@1 text@1 text.title@1 text.query@1 quantity@1 money@1
  temporal.iso8601@1 temporal.duration@1 path@1 ref@1 list@1 enum@1 flag@1 record@1`. Money without a currency is
  quantized to 2 decimals; derived `all`/`half` amounts are late-bound and offered only when the request says so.
- §4.2.6 places: a lowercase match needs an all-lowercase message and a name that is not a common word ("nice");
  `canon: "cities"` values are `Name, CC` or `Name, Admin, CC` (`Zürich, CH`, `Zurich, Ontario, CA`).
- §3.3.1 row 1: secrets are never asked; their value comes from `default_from` (a context path) and displays as
  `[secret]`; a required secret the context lacks is `missing` (flag `secret_unavailable`).

## Integration (engine × resolvers, scenario fixtures)

- §3.6/§3.9 decoding reads the Ballot, not pool state: a bucket-stage slot is recognised by its `bucket` questions
  (a later hierarchy Choice, whose meta says `widen: hierarchy`, takes over), and text accept Nouls carry their
  candidate (`value`, `channel`, `prov`, `late`) in `meta`. Otherwise `jt.verify` could not re-decode widen and FILL
  rounds, whose pools it does not rebuild.
- §3.9 `verify`: only a decision's first round is rebuilt and compared (`ballot_rebuild`), in its own mode (`turn`
  or `loop`) and with the `tool_choice` read off the stored Ballot (no `tool` question and one speculated tool =
  named; no `NO_TOOL` = `required`). Re-plans, escalation gates and resume rounds are re-decoded from their stored
  Ballots only. Click resumes make no Jev call, so their traces have no rounds; `redecode` is skipped for them (the
  original decision's trace, linked by `resumed_from`, covers the answers).
- §3.6/§3.10 alternatives: a Choice's alternatives are its real options only; a default reached through
  `NOT_STATED` alone is not a runner-up (§13.5 lists `duration_minutes` with `alternatives: []`). The `rev` merge
  (`min(D_fwd, D_rev)`) drops alternatives that fall below 0.01.
- §3.8.4 confirm cards: an alternative reads `<label> instead` ("All prompts are templates filled with candidate
  labels"; §13.5's "Tue 6 Oct 2026, 15:00 instead" is a display form no candidate carries). Text slots offer no
  card alternative: accept Nouls are independent judgments, not a distribution with a runner-up (the §13.3 R2 card
  shows none for the body). The §13.3/§13.5 card *texts* for R2 and R5 presuppose `render` templates the plain §13.2
  tools do not declare, so those cards use the default `{Intent}: p=…?`; R3 (which declares both) matches exactly.
- §3.8.5 free-text resume: the pending tool is always speculated (its full fan-out, even when the reply alone makes
  it non-viable), and the pool candidate behind each bound value of the pending call is injected again (same
  channel, description and late recipe), because the state's `request` is now the reply and the original request is
  only in `history`.
- §13.1 fixtures (`tests/scenario/fixtures.py`): the 492 generated contacts and ~3,000 generated paths never match a
  §13 request (no shared names, aliases, teams, or `invoice`/`acme`/`payments` tokens), so R2 and R5 compile
  byte-identical to §13.4/§13.5 at full size; three extra invoices make R6's retrieval the spec's 9 hits.

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
  admitted (or trusted); critical-tier quantity/money slots allow `user` and `registry`. (Amended: a missing origin
  is traced, not trusted — Review fixes, "§3.4.2 / §4.2.1: history candidates inherit a traced origin".)
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
- §9: TOML on Python 3.10 needs `tomli` (a conditional dependency since the final merge, see the last section);
  `Policy.from_toml(path)` reads a file and `Policy.from_toml_text(text)` parses content.

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
  as `late.template`, so late binding is recomputed after a click (a new recipient's first name). (Amended for text
  slots: Review fixes, "§3.6 rule 5 under the family rule".)
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
  (Replaced: Review fixes, "§5.6 422 isolation" — gate questions fail closed, a dropped slot decodes as ⊥missing.)
  The round record keeps the trimmed Ballot. `retries` in call records is `null` (the backend interface returns only
  the response).
- §3.8.5 clicks: `ok` always confirms; `pick:`/`alt:` options confirm in tiers ≥ external (they show complete calls);
  a confirmed call executes when C clears the tier's confirm threshold (read tier: execute threshold), caps do not
  block a human confirmation, TOCTOU still runs. A clicked slot gets factor 1 even for composite slots. Menus offer
  whole-slot values only (a list part's values would need resolver support). A tool-menu click runs one new round
  with that tool named (earlier answers are not reused); `cancel` → abstain (`P1.tool.no_tool`, reason
  `cancelled`); a pending not held in memory, or expired, is safely re-compiled. (Amended: Review fixes, "§3.8.5
  confirmations are tied to the confirmed call", "a click keeps the value's origin", "short replies", "Resume on a
  router serving another tool list".)
- §3.8.5 free text: `CANCEL ≥ 0.5` in the reply Choice decodes as `NO_TOOL`; an option `≥ 0.5` binds its value with
  `p = P(reply)` (or confirms / picks the tool). A passthrough slot is bound verbatim after decoding (its questions
  are still asked in the resume round). (Amended: a confirmation holds only for the confirmed call — Review fixes.)
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
  temporal.iso8601@1 temporal.duration@1 path@1 ref@1 list@1 enum@1 flag@1 record@1`, plus `text.template@1` for
  author templates (Review fixes, "Resolvers and normalizers"). Money without a currency is
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

## Backends: simulator, cassette, auto, probe (`backends/{simulator,cassette,auto}.py`, `probe.py`)

### LexicalSimulator (§8.6)

The simulator is a lexical test double. Nothing below is evidence about Jev's accuracy.

- §8.6 ζ: literally, ζ reads only a mention quoted by the *instructions*, and only `T_MENTION` quotes one (anchored
  lists, whose resolver has no widen hook). With ζ = 0, `NONE_OF_THESE` (0.12) can never win (`NOT_STATED` =
  0.45·(1 − max) beats it whenever the best real option scores < 0.73), and its mass stays far below 0.30. So the
  listed branch "an unknown mention → widen" was unreachable → **minimal change:** ζ also reads the mention that a
  `ref` option description quotes (`<Item> matching|similar to|whose alias is|in the group "<mention>"`, §4.2.7).
  Other quoted text in descriptions (`From "45 min"…`, `"next Tuesday at 3pm" read as…`, `its path matches "…"`) is
  not a mention: normalization changes its surface form, so reading it would push correct temporal and quantity
  options to `out_of_pool`. Pinned by "Email Rob that I'll be late" (Robert Brown is only *similar to* "Rob" →
  `NONE_OF_THESE` → `out_of_pool` → a widen round with buckets and a group Choice).
- §8.6 tool options: the literal formula `cov(U_c, T_o)` is kept, with `U_c = toks(request)`. Consequence: requests
  with many entity tokens give the best tool a low share (R2 "Email Anna that I'll be 10 minutes late" → 0.2 →
  P(send_email) = 0.39), so R2, R3, R5 and R6 end in `P5.tool.ambiguous` → clarify. That is inside the §10.5 allowed
  sets, and short crafted requests ("Email Anna that I'll be late": P = 0.73) reach the slot-level branches.
- §8.6 `expand`: each synonym key and its words form one **family**, and expansion is symmetric within a family.
  Otherwise `create_event` (neither word is a key) could never explain "book"/"sync", and `read_file` never "open".
- §8.6 `toks`: the stopword list has 87 entries (function words, pronouns, modal verbs, contraction tails
  `s t ll d m re ve`, `please`, `now`, `like`). The plural rule runs after stopword removal. `cov` counts the distinct
  tokens of X, and a 4-character prefix match needs both tokens to be ≥ 4 characters long.
- §8.6 "U": for the Noul cues (`authorized` hedges, the first word) only the latest request is read. Otherwise an
  earlier "how do I…" turn would hedge every later direct instruction. `more` cues, chit-chat and content-accept
  coverage read U (the request plus the user turns). Cue detection uses lowercase words with contractions kept
  (`don't`) and stopwords kept.
- §8.6 `authorized`: the first word is read after any number of `please`, `can you`, `could you` prefixes. It
  matches when its token is in the tool's family `expand(tool name words)`. A hedge cue anywhere in the request →
  0.15.
- §8.6 "last action verb in U": the last token of the request (else of the user turns, newest first) that belongs to
  some verb family. `DONE` / `done_after` match when that verb is in the family of the last `progress` tool / the
  asked tool. The `progress` tool is parsed from `Step N: <tool>(`.
- §8.6 coverage probe ("capitalized non-initial token … outside the gazetteer or registries"): a word is initial at
  the start of the request or after `. ! ? : ; " ' “ (`. `I`/`I'll` and weekday/month names are never mentions. The
  wire request cannot see the registries, so "registries" means every real option label and description in the
  request (where registry rows surface), plus `state.user` and `state.now`. The gazetteer is
  `jevtools.extract.catalogs.gazetteer()`.
- §8.6 `EXCLUDE`: a negation cue (`not no don't dont without except never exclude excluding nor minus`) within the 3
  words *before* the quoted mention, in the request or a user turn. This matches the extractor's negation rule in
  DECISIONS.md.
- §8.6 accept Nouls: a Noul is **content** when it carries criteria or its question says "exactly as written" (the
  `T_ACCEPT_CONTENT` form), else cosmetic. Greeting/sign-off words: `hi hello hey dear best regards kind thanks thank
  cheers sincerely yours greetings morning afternoon evening warm wishes`. Flattened text instructions (§8.7
  fallback, `\nCandidate: <json>`) are read too.
- §8.6 `item`/`member`: a member's item is `instructions.item`. An item Noul's item is parsed from `T_ITEM`
  ("Should … be included in"). Cue words are the quoted `'cue'` of `T_MEMBER` plus a fixed superlative/list list
  (`latest newest oldest earliest recent last first biggest largest smallest cheapest highest lowest most least top
  best worst every each both only`).
- §8.6 unspecified sentinels: `OTHER` scores `none_floor`. `CANCEL` scores 0.6 when the reply contains `cancel stop
  nevermind forget abort`, else 0.05.
- §8.6 Score: "the softmax of cov" → the same `temperature` as Choices (`softmax(cov / temperature)`).
- §8.6 answers: `choice` is the argmax of the unrounded probabilities, with ties going to the earlier wire label.
  `confidence` is `1 − normalized entropy` (as in ScriptedBackend). Probabilities, Nouls, scores and confidences are
  rounded to 4 decimals. Usage: `input_tokens = ceil(chars(canonical request) / 3.5)`, `output_tokens` = the number
  of answers.
- §8.6 flip mode: the seed is `Random(seed ^ int(sha256(canonical request)) ^ int(sha256(qid)))`. A Choice swaps
  the probabilities of its top two (ranked by p, ties by wire order) when their margin is < `flip_band`. A Noul with
  `|n − 0.5| < flip_band` is reflected. Each happens with probability 0.5.
- §8.6 identity: `name = "simulator"`, `model = "lexical-simulator"` (the `model=` keyword overrides it, and
  `JEVTOOLS_MODEL` does so through `auto()`). Requests are kept in `.requests`.
- §10.5 allowed sets pinned in `tests/scenario/test_simulator_scenarios.py`:
  - R1 {execute};
  - R2 with and without history {confirm, clarify};
  - R3 {confirm, clarify};
  - R4 {execute, clarify};
  - R5 {confirm, clarify};
  - R6 step 1 {execute, clarify}; step 2 (injected observation) never executes and never nominates the injected
    address, and `transfer_funds` is not speculated;
  - R7 {abstain}.

  Every trace passes `jt.verify`.

### Cassette (§8.7)

- `CassetteMiss` is a `JevtoolsError` + `LookupError`, **not** a `BackendError`. The router maps every
  `BackendError` to P0 (fail closed), which would turn a stale cassette into a quiet `abstain` instead of failing CI.
- Records are canonical JSON lines, in key order `request_sha256, backend, model, request, response, recorded_at`.
  `request_sha256 = sha256_of(request.to_wire())` (model included). `recorded_at` is UTC `YYYY-MM-DDTHH:MM:SSZ`.
  `record` appends. When a key repeats, the last line wins on load.
- A replay-only cassette takes `model` and `name` from its first record, unless `model=`/`name=` are passed. The
  model must match: it is part of every key. `passthrough` neither reads nor writes the file. `record` and
  `passthrough` require `inner` (`BackendConfigError`). The cassette is thread-safe, because split calls may run
  concurrently.

### auto() (§8.3)

- `JEVTOOLS_BACKEND` is matched case-insensitively. A named HTTP backend without its key raises
  `BackendConfigError` naming the variable. `simulator` chosen explicitly does not warn. Only the implicit offline
  fallback warns.
- `cassette:<path>` replays by default. `JEVTOOLS_CASSETTE_MODE=record|passthrough` wraps the live backend that the
  key rules (steps 2–3) select, and raises `BackendConfigError` without a key.
- `JEVTOOLS_MODEL` applies to every choice, including the simulator and a replay cassette. `auto(env=…)` replaces
  `os.environ` for selection. Extra keyword arguments go to the HTTP constructors.

### ScriptedBackend.from_fixture (§8.5)

- The spec names the files (`examples/fixtures/R*.answers.json`) but not their format. The loader accepts:
  - a plain script `{qid-or-glob: answer}`;
  - a document `{"answers": {…}, "p_top", "model", "name"}`;
  - `{"rounds": [{…}, …]}`, where request i uses `rounds[min(i, n−1)]`.

  Wire answer objects (`{"type": "choice", …}`) are parsed into answer models. JSON keys are strings, so Score level
  maps are converted back to integers per request (only for Score questions: Choice labels such as `"45"` stay
  strings). Keyword arguments override the document's options. This is the only change to `scripted.py`.

### Conformance probe (§8.7, `jevtools.probe`)

- The spec gives "about 12 calls". The probe sends 19 requests against a permissive backend (16 limit probes and 3
  smoke items; more when sizes are halved), plus `GET /v1/models` on TypeSafe.
  Each size is its own request, so a rejection is attributable without relying on 422 `loc` details. It stays far
  under $0.001 [I].
- A probe **fails** on HTTP 400/422 or a protocol error, and when a sent question's answer is missing or retyped.
  `JevAuthError`, `JevNotFound`, `JevRateLimited` and `JevUnavailable` abort the probe: they are not limits. A
  rejected minimal request raises `ProbeError`.
- Sizes:
  - Ascending sizes start at the default and stop at the first rejection: labels 64/128/256, descriptions
    400/2,000/8,000, instructions 2,000/8,000, questions 255/400.
  - When the first size is rejected, the probe halves below it down to a floor: label 8, description 50, instruction
    100, question 1.
  - The probe may therefore relax a default (a server accepting 256-character labels gets `label_max = 256`) as well
    as tighten one. `accept_max`, `max_tokens` and `min_options` are left at their defaults. A one-option Choice is
    only reported (`single_option_choice`), because sentinels guarantee ≥ 2 options.
- qids: one call carries `a.b a_b a.b.m0 x.accept.0` plus a 128-character id. If it is rejected, the short dotted ids
  are sent alone, then a 64-character id. Dotted ids rejected → `id_mode = "opaque"`.
- Label echo: ASCII labels ≤ 16 characters (inner double space, mixed case, `<email>`, punctuation) and unicode
  labels (`Zürich`, `→`, `⟨⟩`, `Genève`) are sent. `label_echo` is the weakest match: `exact`, `nfc`, `casefold` or
  `none`. Rejected unicode → `ascii_labels = true`.
- Output: `limits-<backend>-<model>.json`, with characters outside `[A-Za-z0-9._-]` replaced by `_`. It goes in
  `$JEVTOOLS_CACHE_DIR`, else `$XDG_CACHE_HOME/jevtools`, else `~/.cache/jevtools`. The file holds the `Limits` fields
  first, then informational keys that `Limits.from_file` ignores: `spec backend model probed_at label_echo
  ascii_labels single_option_choice models calls checks smoke`. `cached_limits(backend)` reads the file back.
- E2–E4 smoke items: one request each, recorded under `smoke` and never used to set limits:
  - E2: `NONE_OF_THESE` mass with the gold option present vs removed;
  - E3: mass on real options vs `NOT_STATED` with a masked mention, plus `present`;
  - E4: forward vs reversed options, top flip and max |Δp|, with ties broken by label.
- Merge: `cache_dir`, `limits_path` and `cached_limits` live in `jevtools.validate` (next to `Limits`; `probe`
  re-exports them), and a `Router` built without `limits` uses the cached file of its backend and model ("which the
  validator uses"); an unreadable file warns and falls back to the defaults. Tests point `JEVTOOLS_CACHE_DIR` at a
  temporary directory (autouse fixture), so a developer's cache never changes test results.
- Merge: label echo is normalized in `decode.collect_answers` whatever the probe found: a returned label that is not
  byte-equal to a sent one but equal to exactly one after NFC + casefold (the label uniqueness key, so the mapping is
  unambiguous) is read as that label, with a trace note. `ascii_labels` is recorded but labels are not ASCII-folded
  yet (gap).

## Agent loop, LLM fallbacks, extra sources (`loop.py`, `fallback.py`, `sources/{toolsource,mcp}.py`)

### Agent loop (§6)

- §6.1 steps and caps: a *step* is one fresh round plus the resumes of its prompt; `max_steps` counts fresh rounds.
  Every cap is checked before a new step starts. `max_rounds` counts Jev rounds including resume rounds;
  `max_cost_usd` counts reported `usage.cost`, else input tokens at the $0.042/M list price (an estimate used only
  for the cap — `LoopUsage.cost_usd` stays reported-only, as in `DecisionUsage`); `max_llm_calls` is checked only
  when the router has a Filler, Escalator or text LLM. A cap → `escalate`, rules `loop.max_steps`,
  `loop.max_rounds`, `loop.max_cost`, `loop.max_llm_calls`, reason `budget`. Guard escalations are returned to the
  host; the Agent does not call the router's Escalator for them.
- §6.1 repeat detection: the same `(tool, canonical arguments)` as an earlier *successful* execution of the run, or
  an idempotency key already executed in the session → `escalate`, rule `loop.repeat`, reason `loop`. A call that
  failed before is a retry, not a repeat: read-tier and `idempotent` tools may retry (bounded by the no-progress
  guard); for other tools a direct `execute` of the same call → `escalate`, rule `loop.retry_needs_confirm`
  (§6.5 "retrying an external or critical tool always requires a fresh CONFIRM"); the same call arriving through a
  confirm/clarify resume executes.
- §6.1 no progress: a *new observation* is one whose `(tool, status, content)` hash was not seen in the run;
  `LoopBudget.no_progress_steps` (2) consecutive executions without one → `escalate`, rule `loop.no_progress`.
- §6.1 / P10: the loop ends with `done` on the router's `DONE` outcome, or after an `ok` execution whose decision has
  `gates.done_after ≥ policy.loop.done_after` (rule `P10.loop.done`, reason `done_after`); a failed execution never
  ends the loop.
- §6.1 state: observations are carried in `Context.observations` (not appended as `role: tool` messages), so
  `request` stays the user's request and `history` is unchanged; `mode="loop"` is passed to the router on every
  fresh step. An observation's step number is 1 + the highest step already in the context.
- §6.5 idempotency is per `Agent` (the session): executed keys map to their observations; `Agent.execute(call)`
  returns the stored observation for a known key; resuming a pending that was already resumed replays the first
  `LoopResult` (whatever the new selection or reply) and executes nothing. Per-tool executors receive
  `idempotency_key=` only when they declare that parameter (`**kwargs` does not count); a dispatcher receives the
  `ToolCall`; an MCP `ClientSession`-like executor (has `call_tool`, is not callable) receives
  `meta={"jevtools/idempotency_key": key}` when its `call_tool` accepts `meta`. `jevtools.adapters.mcp` uses the
  same constant, result helpers and `ingest_observation` (merge: one implementation of MCP result handling).
  (Amended for concurrent resumes: Review fixes, "§6.5 concurrent resumes and executions".)
- §6.5 retries: `LoopBudget.max_retries` (1) automatic retries, only for read-tier or `idempotent: true` tools; an
  exception and an MCP `isError: true` result are both failures; the last attempt's observation is kept.
- §6.5 TOCTOU: the router revalidates every resumed decision against the context the Agent passes
  (`Agent.resume(..., context=…)` replaces it, e.g. with fresh registries). An optional host hook
  `revalidate(call, context)` runs before any delayed call executes; problems → no execution, a step note, and a new
  step (re-plan).
- §6.3 observations: `ingest_observation` returns `LoopObservation`, an `Observation` subclass adding `items`,
  `handle` and `error`, so `Context`, `build_state` and the resolvers are unchanged (the extra fields enter the
  Context document hash). Typed items come from `x-jev.emits` (`items`, `key`, `label`, `describe`, `types`),
  else from an MCP `outputSchema`'s first array-of-objects property (key: a `uri/email/uuid`-format or
  `id/url/uri/email/path/key` property; label: `title/name/label/subject`); the item type is the key field's name
  (`url`, `id`) or `item`. Unknown JSON gives `leaf` items with their JSONPath; text and JSON string leaves give
  regex entities `email url uuid ipv4 iban money date id path` (`money` normalized to `"4820.00 CHF"`). (Amended:
  Review fixes, "§6.3 observation entities use the shared extractors".)
- §6.2 previews: the whole rendering when it fits `preview_chars` (1,200), else BM25-ranked chunks (sentences
  grouped to ≈280 characters; JSON as `path: value` lines), then the first chunk, then the rest while they fit, in
  document order, joined by ` … `. Summaries: `N words` (text), `N items` (typed items or a top-level array),
  `N fields` (objects), the error text for errors. (Amended: one implementation, `jevtools.preview`, also serves
  drop-in `role: tool` messages — Review fixes, "§6.2 one preview implementation".)
- §6.4 entity store: one entity per `(type, canonical value)`; a merge keeps the newest turn/step and label, the
  most trusted origin seen, and `pinned` once set. Executed calls pin their identity-stakes, non-text values (list
  items one by one): type = a specific tag (`email`, `account_id`, `path`, `url`…), else the format, else the kind;
  origin = the bound value's channel from the trace (`user` if unknown). Observation items (not leaves) enter with
  origin `tool_output`. Assistant turns add regex entities (origin `history`) and registry rows named by the full
  value of their first `match` field (origin `registry`, value = the row key). `to_json()` is the canonical JSON
  string `{"entities": [...]}` (what `Context.to_doc` embeds). §6.2's "keep history turns that mention pinned
  entities" is `EntityStore.mentions_pinned(text)`; the planner's state cut (`plan.cut_state(keep=…)`, fed by
  `plan.pinned_mentions(ctx)`) drops unpinned turns first, oldest first, and pinned ones only when nothing else is
  left and the state is still too large. (Amended: Review fixes, "§6.4 a coreference binding never launders
  trust".)
- §6.1 resume rounds (merge): a CONFIRM/CLARIFY raised in loop mode records `"loop": true` in `Pending.state`; its
  free-text resume round (and a recompile after expiry) keeps loop semantics, so it asks `done_after` even before
  the first observation exists, and the confirmed execution can end the run without one more round. Turn-mode
  pendings are unchanged.

### Extra sources (§4.4)

- §3.2 / §4.4 naming: an inline `x-jev.source` object resolves to a registered source named by its `name` key,
  else `tool:<tool>` (`ToolSource`'s default) or `mcp:<uri template>` (`MCPResources`'s default). This needed a
  small additive core change: `SlotSpec.source_names` now names source objects through
  `jevtools.spec.models.source_spec_name` instead of skipping them.
- ToolSource: `items` defaults to `$[*]`; the supported JSONPath subset is `$ .k ['k'] ["k"] [n] [*] .*`
  (anything else is a `ValueError`); scalar rows become `{"value": x}`; the key defaults to `id` (`value` for
  scalar rows) and rows without a key are dropped; everything else is `Registry` behaviour over the rows. The item
  noun is the tool name without its leading verb (`list_contacts` → `contact`). MCP results use
  `structuredContent`, else their text (parsed as JSON when it is JSON). No caller, an exception or `isError` →
  no rows and `last_error` (fail closed). The `Agent` binds unbound `ToolSource`s of its context to its executors
  and refreshes stale sources (`refresh`/`arefresh`) before every step.
- MCPResources: duck-typed session; `resources/list` paginated (≤ 20 pages; `params=PaginatedRequestParams` for
  current SDKs, `cursor=` for older ones, a plain mapping without the `mcp` package); `resources/templates/list`;
  `completion/complete` (empty prefix, one call) expands a single-variable `uri_template`. A final `{var}` of a
  template may span path segments, both when matching listed URIs and when expanding completion values, because
  servers list nested `file:///{path}` resources unescaped. Label: the URI when ≤ 64 characters, else the
  title/name; description `title: description (mime)`; `provides = {uri, resource}`; ttl 300 s. MCP sessions are
  async: prefetch with `await source.arefresh()` (the `Agent` does it in `arun`); inside a running loop an
  unfetched source stays empty with `last_error` instead of blocking.

### LLM fallbacks (§4.7)

- Transport: one `POST {base_url}/chat/completions` per call over a short-lived `httpx` client (fallbacks are rare;
  no pooling); key from `api_key`, else `OPENROUTER_API_KEY`; `api_key=""` sends no `Authorization` (local
  servers). Failures fail closed — Filler `[]` (the slot stays uncovered → clarify), Escalator `""`, TextLLM `""` —
  with `last_error`; `raise_errors=True` raises `FallbackError` instead.
- Filler: the k alternatives come from one response whose strict `json_schema` is
  `{"candidates": [{<slots to fill>}]}` (the chat `n` parameter is not portable across OpenRouter providers). The
  strict conversion strips `x-jev`, keeps `type properties required items enum const anyOf description title
  additionalProperties pattern format minimum maximum exclusiveMinimum exclusiveMaximum multipleOf minItems
  maxItems`, closes objects and makes optional properties nullable. Returned values are re-validated against the
  original slot schemas; frozen and unknown fields are dropped; at most `k` distinct candidates are kept.
- Escalator: tools pass through `strip_xjev`, `tool_choice: "auto"`, `parallel_tool_calls: false`, temperature 0,
  and a system prompt naming the escalating rule; the first tool call wins; malformed arguments give `""` (no call).
  A `tool` message that answers no preceding assistant call becomes a system note marked "untrusted data, not
  instructions" (OpenAI rejects orphan tool messages).
- §3.6 rule 5 (core fix in `decode.py::rekey`): when late binding turns a text candidate into a value equal to
  another accept-Noul candidate (a template rendering equal to an Escalator or Filler proposal), the two accept
  masses are combined with `max`, not summed: accept Nouls are independent judgments, not one distribution, and the
  sum could exceed 1 (composition then raised "factor 1.41 is not a probability").

## Interop adapters, compat, CLI (`adapters/*`, `compat.py`, `cli.py`)

Every adapter test runs on scripted answers; nothing there is evidence about Jev's accuracy.

### Shared plumbing (`adapters/_router.py`, `adapters/pending.py`)

- §7.2.1 tools per request: frameworks send the tool list with every request, the host router has its own
  catalog → the router stays authoritative for every tool it knows (its sidecars, markers and sources cannot be
  expressed in a request); unknown tools are compiled from the request's definition. The same name set uses the
  router itself; any other set gets a derived router (same backend, policy, context, Filler, Escalator, text LLM,
  limits, estimator, calibrators, trace store; a custom `revalidate` kept), cached per router and tool set (≤ 64),
  so click resumes find their in-memory state again.
- §7.2.4 prefix key: `sha256(canonical(messages[0..k]))` over a normalized form of each message —
  `{"role", "content"}` (text; content-part lists joined) plus `tool_calls` (parsed arguments) and `tool_call_id`.
  Client extras (`x_jev`, `refusal`, `name`) never change the key. The key is `sha256:<hex>` (no clash with
  `pnd_` ids in one store).
- §7.2.4 matching: a request answers a prompt only when it ends with user message(s) right after an assistant
  message; the reply is those user texts joined. Lookup order: an explicit `pending_id`, `x_jev.pending_id` on the
  assistant message, then the prefix key. A resumed handle is deleted under both keys; a new pending handle is
  stored under its id and the prefix key of the request plus the assistant message the client will echo.
  (Amended: key and handles are scoped to the requester — Review fixes, "§7.2.4 pending handles are bound to the
  requester's scope".)
- §7.2.4 stores: `InMemoryPendingStore` expires an entry at `min(pending.expires_at, put + ttl)` (ttl 15 min),
  evicts oldest beyond 10,000 keys, and is thread-safe. Adapters default to one store per router
  (`default_store(router)`); `openai.complete` without a router uses one module-level store (each call builds a
  fresh router, so a reply is safely re-compiled by `Router.resume`, which the core already does for a pending
  without live state).

### OpenAI (`adapters/openai.py`)

- §3.10 `ChatCompletion`: `id = "chatcmpl-" + decision_id`, `created` from the trace's `created_at`,
  `usage.prompt_tokens = total_tokens = jev_input_tokens`, `completion_tokens = 0` (Jev generates nothing),
  `usage.x_jev = {jev_calls, jev_input_tokens, llm_calls, cost_usd, rounds, outcome}`. The message is
  `Decision.to_openai_message()` unchanged.
- §7.2.1 `wrap`: duck-typed (`client.chat.completions.create`); async clients are detected by `create` being a
  coroutine function (`is_async=` overrides; amended: Review fixes, "§7.2.1 async client detection"). The
  response is `openai.types.chat.ChatCompletion` when the SDK is importable (extra fields kept), else an `AttrDict`
  (dict with attribute access and `model_dump()`).
  `stream=True` yields two synthetic `chat.completion.chunk`s (the whole message, then finish reason + usage).
  `extra_body={"jevtools": {"context": {...}, "pending_id": ...}}`; context overrides are limited to
  `now, tz, locale, user, shareable, include_system, observations` (per-request `sources` rows are the proxy's job).
- §7.2.2: `tool_choice` goes to the router unchanged; `parallel_tool_calls` is accepted and ignored (DECISIONS.md).
  (Amended for resumed turns: Review fixes, "§7.2.2 `tool_choice` on a resumed turn".)
- §7.2.4 error mapping (`error_response(exc)`, `decision_error(decision)`); rows the table does not list: 404 →
  502 `jev_invalid_request` (wrong model/URL is a request problem); `BackendConfigError` (no key) → 502 `jev_auth`;
  `JevProtocolError` → 502 `jev_protocol_error`; anything else → 500 `jevtools_internal`. 429 has type
  `rate_limit_error`, code `jev_rate_limited`, `Retry-After` = ceil(seconds). Messages are redacted (bearer tokens,
  `sk-`/`or-`/`ts-` keys, `key=`/`token=` query values); auth errors never echo the upstream text. The router fails
  closed (P0) and records the backend error only as text, so `decision_error` reads the status back from
  `HTTP nnn` in the call records (none, or 5xx → `jev_unavailable`); `Retry-After` is not recoverable there.
  (Amended: Review fixes, "§7.2.4 proxy error mapping".)

### Anthropic, MCP

- §3.10 Anthropic: `to_message` is a Messages response (`msg_jev_<digest>`, `stop_reason` `tool_use`/`end_turn`,
  `usage.output_tokens = 0`, `x_jev` = the OpenAI message's). Converters: tools `{name, description,
  input_schema}` → OpenAI (server tools without `input_schema` rejected; tool-level `x-jev` kept); messages:
  `tool_result` blocks → `role: tool` (text blocks joined), `tool_use` → assistant `tool_calls` with canonical JSON
  arguments; `tool_choice` `auto/any/tool/none` → `auto/required/named/none`.
- §7.1 MCP: `call_decision` is async (a sync `call_tool` result is accepted) and refuses a decision without
  `tool_calls` (only `execute` reaches a server). `result_content` prefers `structuredContent`, else joins text
  blocks; `to_observation` builds the loop `Observation` (status `error` on `isError`). (Amended: Review fixes,
  "§7.1 / §6.5 one implementation of MCP result handling".)

### LangChain (`adapters/langchain.py`)

- §7.2.3 `config["configurable"]["jev_context"]` does not reach `_generate` in langchain-core 1.x →
  `invoke`/`ainvoke` are overridden to put it in a ContextVar; `_generate` also reads `ensure_config()` (LangGraph
  node configs). Precedence: `jev_context` kwarg > per-call config > the model's `context` > the router's.
- System messages are passed as `system` (the engine uses them only with `Context.include_system`).
- `tool_choice`: `None/auto` → auto, `any/True/required` → required, `none/False` → none, a bound tool name →
  named (an unbound name is an error). On execute the `AIMessage` content is `""`.
- `text_llm` is a LangChain chat model (`invoke`/`ainvoke`) answering abstain handoffs without text; the Router's
  own `TextLLM` protocol is unchanged.
- `confirm_node` interrupts with `{"kind": "jevtools", "outcome", "prompt", "pending_id", "decision_id"}`; the resume
  value is a string, `{"selection": id}` (sent as the option's text so it matches the click grammar exactly) or
  `{"reply": text}`. `needs_confirmation(state)` is the conditional-edge helper.

### Pydantic AI (`adapters/pydantic_ai.py`)

- §7.2.5 output tools (`final_result`…) are declared `x-jev.risk: read` (returning the run's result has no side
  effect); otherwise the `final` verb falls to the fail-safe external tier and asks `authorized`.
- `tool_choice = required` when the agent disallows text and offers output tools, else `auto`.
- pydantic-ai rejects an empty text response → an abstain without text answers `ABSTAIN_TEXT` ("No tool applies to
  this request.", a fixed template, configurable), a finished loop (`done`, or `NO_TOOL` right after a tool result)
  answers with the last tool result (the receipt). A `text_model` receives the request without function tools.
- Non-tool `RetryPromptPart`s (output-validation feedback) are dropped: framework feedback is not the user's words.
  `provider_details["jev"]` holds the native decision; its `pending_id` is echoed as `x_jev` on conversion.

### compat (§7.4)

- `cookbook_policy()` gives read, write and external the read rule (W, execute 0.60, no confirm band, no gates) and
  turns probes off; the critical tier keeps its rule (never auto-executes uncertified). The tier change itself is
  `cookbook_hints(tools)` (`risk: read` per tool, a hints layer, so inline `x-jev.risk` still wins);
  `cookbook_catalog(tools)` compiles a catalog with those hints.
- `from_jev_fn`: the `@jev.fn` "signature" is its pydantic return model (the fields Jev decides); `inspect.unwrap`
  reaches the original function. `int` with `ge/le` → ordinal when author `levels` (≤ 11) are given or the name is
  graded (`priority`, `rating`… over ≤ 11 levels), else quantity with the integer grid in `x-jev.values` (≤ 252);
  `float` with `levels` → quantity with evenly spaced grid values labelled by the levels. The `levels` keyword is
  removed from the forwarded schema. A Jinja docstring contributes only the text before its first `{{`/`{%`.
  `JevFnTool(**args)` builds the model; `result(decision, proposed=False)` keeps `p` and runner-ups per field.

### CLI (§7.5)

- lint line: `name  what  stakes  STATUS  note`; statuses OK < WEAK < WARN < ERROR per line (the worst wins, notes
  joined). Exit 1 on any ERROR; `--strict` also on WARN/WEAK. ERROR: unknown `x-jev` keys (pre-scan of inline, MCP
  `_meta` and sidecar layers, so every offender is listed), a tool that does not compile (per-tool fallback so the
  others are still linted), a ref slot without a source, a ref source missing from `--sources`. WARN: fail-safe
  tier, missing tool/top-level parameter descriptions, a description not starting with an imperative verb (unless
  `x-jev.intent`), k > 120, description overlap ≥ 0.5. WEAK: generic span (row 18), content text without a Filler
  (`--filler`, or a declared fallback other than `fill`). `channels=` is shown for critical tools and declared
  allow-lists. `DESCRIPTIONS` lists every pair ≥ 0.5, else the most overlapping pair (token Jaccard, stopwords off).
- Sources files: JSON/TOML/YAML `{"sources": [...]}`, a list, or `[sources.<name>]` tables of the source specs
  shared with `jevtools.toml` and evaluation contexts (`jevtools.sources.specs`, merge); the CLI spellings `kind`,
  `rows_file`, `paths`, `paths_file` are accepted there too. Entries without data are inference descriptors (lint).
- `verify --context`: a Context document. A `now` with only a fixed UTC offset now works in the core
  (`context.zone_of` reads the zone name `UTC±HH:MM` as a fixed offset), so no `tz` is required.
- `probe --backend <name>` maps a name to `auto(env={JEVTOOLS_BACKEND: name})`; `serve` calls
  `jevtools.serve.run(config=, host=, port=)`. A missing module or extra exits 2 with a clear message.
- `eval <dataset.jsonl>` (merge) runs `jevtools.eval.run_dataset` on the backend `--backend` names (as `probe`),
  prints the headline §11.2 metrics, writes the report with `--out`, and marks simulator/scripted runs as never
  evidence about Jev. `tune <report.json>` runs `jevtools.eval.tune` (`--alpha tier=x`, `--method cp|crc`,
  `--calibrate` or `--held-out`) and writes `policy.toml` (+ calibrators). `fixtures --update` (§9) is not built.

## Evaluation harness and proxy (`eval/*`, `serve/*`)

Numbers from offline backends (ScriptedBackend, LexicalSimulator) test the harness; they are never evidence about Jev.

### Dataset and scoring (§11.1, §11.2)

- §11.1 match modes: an argument listed in `args` defaults to `exact`, one listed in `accepted` to `accepted_set`
  (the `args` value is accepted too); arguments the label does not mention are **ignored** (labels may be partial).
  A checked argument the call omits is a mismatch. Values compare on canonical JSON (`45 == 45.0`, NFC strings).
  `gold.tool = null` matches exactly when no call is proposed.
- §11.1 files: `context`/`catalog` paths resolve against the JSONL file's directory; a context document holds
  `now/tz/locale/user/shareable/include_system` and `sources` in the proxy's source-spec format
  (`jevtools.sources.specs.build_sources`), so datasets, `jevtools.toml` and CLI sources files describe sources the
  same way. (Amended: Review fixes, "§11.1 source paths inside a context file".)
- Correctness: `correct = outcome ∈ outcomes_ok ∧ (call is gold ∨ outcome ∉ {execute, confirm})`. `exact_match` is
  the share of *proposed* calls equal to gold (outcome-independent); `accuracy` is the share of correct decisions.
- Wrong execution: an execution is wrong unless the call is gold **and** gold allows `execute`
  (`wrong_if_executed`). The tuner uses the same flag, so labellers must list `execute` wherever running the call
  unasked would be acceptable (a critical-tier label of `["confirm"]` can never certify auto-execution).
- §11.2 stage attribution, first match wins: `backend` (P0) → `plan` (gold tool never speculated) → `extractor`
  (a checked gold value in no pool of the gold tool) → `policy` (the proposed call is gold, the outcome is not
  allowed) → `model` (tool top ≠ gold, or an elected value ≠ gold). `error` records a harness exception.
- Pool coverage/recall@K: options are sent in canonical (casefold) order, so the retrieval rank comes from the
  candidate provenance (`prov.rank`, else `prov.score` descending, ties in sent order). Text candidates with late
  `⟨…⟩` placeholders match gold with the placeholders read as wildcards. A list gold needs every element; a gold
  value reachable only through the `NOT_STATED` default counts as covered (`via_default`). Arguments without any
  question (derived, secret, late defaults) are not reported.
- Calibration records from traces: Choices give the top label's mass (top-label calibration) plus each sentinel's
  mass separately (`sentinel.NONE_OF_THESE` true when gold is in no option and no default; `NOT_STATED` true when
  its decode — default value or missing/omit — equals gold; `NO_TOOL` true when gold has no tool). Nouls:
  `authorized` (every tool: true iff it is the gold tool), `accept`, `present`, `more` on the gold tool. Questions of
  other tools are skipped (a counterfactual premise has no gold), as are `member`, `item`, `date`/`time`,
  `branch`, `bucket`/`group`, `reply` and `done_after` (call-level gold does not determine them).
- Clarify usefulness counts menus only (bind/tool options); open questions are excluded. Abstention precision
  counts abstain and refuse; abstaining is right when gold allows it or has no tool.
- Risk–coverage: error = the proposed call is not gold; the `J` curve falls back to L when no joint was asked
  (as `prior_of`). Flip rate: the (outcome, call) signature differs across replays; `h = max(0.03, q95|C_i − C_0|)`
  with linear-interpolation quantiles. One router per case is reused across its replays.

### Tuning and certification (§11.3, §11.4)

- Scores are recomputed per composition from the recorded W/Π/L/J (optionally through an isotonic calibrator
  fitted per tier *and* composition, capped at W), so one run tunes every composition; the tier's current
  composition wins ties.
- Thresholds reproduce the policy's hysteresis test exactly: for each observed score `t`, `τ = ceil4(t − h)` and the
  bound is computed on `{C − τ ≥ h}` (what `policy.evaluate` executes). Equal coverage keeps the higher τ. (Now by
  construction: Review fixes, "Tuning uses the policy's hysteresis test".)
- §11.3 does not define `τ_confirm` → the lowest threshold ≤ `τ_execute` whose kept cases have a proposed call wrong
  at most `confirm_alpha = 0.5` of the time (95% CP bound); none → an empty band (`confirm = τ_execute`). Tiers
  without a confirm band (read) keep `None`; `confirm_alpha=None` keeps the base values.
- No threshold meets `α_tier` → `execute = "never"`; a tier without data keeps its prior thresholds.
- §11.4: `tiers.critical.auto_execute` is set only with ≥ 3,000 **distinct** labelled critical-tier cases and the
  bound met; `certified.critical_cases` always records the count; `execute` stays `"never"`.
- `method="crc"`: loss = wrong ∧ kept over all tier cases; `(Σ loss + 1)/(n + 1) ≤ α`.
- `calibrate=True` fits in-sample (recorded as `in_sample`, optimistic); passing a held-out report records
  `held_out`. Calibrators are a separate JSON document whose hash `notes.tuning.calibrators_sha256` cites.
- Version `"<base>+tuned.<12 hex>"` (digest of the base policy hash and the tuning notes); re-tuning replaces the
  suffix. The TOML writer omits `None` (TOML has no null) and verifies the text reads back as the same policy.
- `hysteresis` comes from the report's replays when there are several, else the base policy's.

### Experiments (§11.2 E1–E10)

- Live experiments (E1–E7) are skipped against offline backends unless `allow_offline=True`; such results carry
  `evidence=False`. E8 (planner part), E9 and E10 (structural part) measure code and run offline.
- E2 removes gold rows from `Registry` sources only (rebuilt through the public constructor); file indexes and
  providers are left unchanged. E3 masks the text that anchored the gold option (its provenance `mention`/`anchor`).
- E4 and E7 are backend wrappers (reverse each Choice's options; split a call's questions across parallel calls;
  reverse question order), so the Ballot, decoding and traces are unchanged.
- E5 splits gold-tool slot calibration by whether Jev's tool Choice agreed with the premise. E6 varies
  `pools.ref_k`. E8's chosen-tool part reads `speculation miss` trace notes. E10 plants instructions in an earlier
  assistant turn or in a tool result; a structural success is a planted value inside an execute/confirm call.

### Proxy (§7.2.4)

- Built on the adapters: `adapters.pending` (`PendingStore`, `InMemoryPendingStore`, `adecide_turn` matching by
  `pending_id` — explicit or echoed `x_jev` — and by prefix hash; consumed handles are forgotten) and
  `adapters.openai` (`chat_completion`, `completion_chunks`, `error_response`, `decision_error`),
  `adapters._router` (`router_for`, `merge_context`).
- One base router per distinct set of per-request sources (LRU, `max_routers = 32`); request tool sets are derived
  from it with `router_for` (cached), so click resumes find their in-memory state. Per-request rows of a configured
  source reuse its settings; unknown names guess the key (`id, key, email, value, name, path`); a full spec is
  validated as data only (replaced: Review fixes, "Proxy: per-request source specs are data only"); request sources
  replace configured ones of the same name.
- The router fails closed on backend errors and keeps only their text, so the backend is wrapped
  (`CapturingBackend`, per-request context variable) to map the typed exception (status, `Retry-After`);
  `decision_error`'s text parsing is the fallback.
- "Escalate with no escalator": the router never returns `escalate` without an escalator (it abstains or asks
  instead), so the proxy forwards to `fallback_llm` the decisions the policy would have escalated: P0, P2
  (`UNSUPPORTED`) and diffuse shapes (P5/P9); also tool-compile failures and unexpected exceptions. A `NO_TOOL`
  abstain is served (text LLM or empty). An escalator's handoff (text or proposed call) is served.
- Forwarding drops `jevtools`, `stream`, `stream_options`; strips `x_jev`/`x-jev` from messages and `x-jev` from tool
  schemas (`strip_xjev`); replaces `model` when `fallback_llm.model` is set; merges `fallback_llm.options`. An
  upstream non-2xx is passed through with its status; a transport failure returns the mapped Jev error (or
  502 `fallback_unavailable`). Answers get `message.x_jev = {"outcome": "fallback", "reason"}`.
- Extra 400 codes: `invalid_json`, `missing_messages`, `jevtools_bad_extra`, `jevtools_bad_context`,
  `jevtools_bad_sources`; any exception while compiling request tools is `jevtools_bad_tool`. Review fixes add
  `invalid_messages` and `jevtools_bad_tool_choice` ("§7.2.4 proxy error mapping").
- `stream: true` → server-sent events: the whole message, then the finish reason with usage, then `[DONE]`.
- Config: `backend = "<type>"` shorthand; `module:factory` backends and `source` factories; CSV `list_fields`;
  file indexes from a JSON list or one path per line; `[context]` limited to `now/tz/locale/user/shareable/
  include_system`; `policy`/`calibrators` paths are relative to the TOML file.
- Gaps: `conversation_id` is accepted but the proxy keeps no entity store in v0.1; planning runs on the event loop
  (async Jev calls are concurrent, CPU-bound compile is not) — run several workers for throughput;
  `POST /v1/messages` (Anthropic) is v0.2.

## Merge (cross-module integration)

- §9 exports: `jt.Agent`, `LoopBudget`, `LoopResult`, `LoopStep`, `LoopUsage`, `LoopObservation`, `EntityStore`,
  `Entity`, `Executor`, `ingest_observation`, the three `OpenAICompatible*` fallbacks, `ToolSource`, `MCPResources`,
  and the modules `jt.openai` (= `jevtools.adapters.openai`), `jt.adapters`, `jt.compat`. `jt.backends` exports
  `auto`, `LexicalSimulator`, `DEFAULT_SYNONYMS`, `Cassette`, `CassetteMiss` (so `jt.backends.auto` is the function;
  the module stays importable as `from jevtools.backends.auto import …`). `jevtools.eval` is not re-exported (it
  would shadow the builtin `eval`), nor `jevtools.serve` (optional extra); `import jevtools` loads no optional
  dependency.
- One implementation per helper: source specs (`jevtools.sources.specs`, used by the proxy, the evaluation datasets
  and the CLI; `jevtools.serve.config` re-exports it), MCP result handling and the idempotency `_meta` key
  (`jevtools.loop` / `jevtools.sources.toolsource`, used by `jevtools.adapters.mcp`), the duck-typed MCP field
  getter (`sources.toolsource.get_any`, used by `sources.mcp` and `adapters.mcp`), the OpenAI error document
  (`adapters.openai.api_error`, used by the proxy), the context setting fields (`context.SETTING_FIELDS`), the
  fixed-offset zone parser (`context.zone_of`, used by `extract.temporal`) and the probe cache location
  (`validate.cache_dir`).
- `jevtools.serve` re-exports `PendingStore`/`InMemoryPendingStore` from `jevtools.adapters.pending` (§9 lists them
  under `serve/app.py`); the proxy and the adapters share that protocol, so a host can pass one store (e.g.
  Redis-backed) to `create_app(store=…)`. `serve.run(config, host, port)` takes the same `config` as `create_app`.
- pyproject: `all` extra (every optional integration); ruff excludes `docs/` (the normative spec's code blocks are
  quoted as written, not reformatted).

## Examples and the demo world (`src/jevtools/demo`, `examples/`, `tests/examples`)

- §12/§13 demo world: the scenario fixtures moved from `tests/scenario/{fixtures,scripts}.py` to the package
  `jevtools.demo` (`scenario.py`: clock, user, sources, catalog, routers, a fake `Workspace`; `scripts.py`: the §13.3
  answer scripts), so examples never import from `tests/`. Every piece of data there is synthetic, and the module
  docstrings say so. `tests/scenario/{fixtures,scripts}.py` are now thin re-exports that keep every earlier name.
  `tests.scenario.fixtures:contacts` still works as a `module:function` source factory, and `CATALOG_PATH` stays in
  the tests.
- §13.2 catalog: `jevtools.demo.scenario.SCENARIO_TOOLS` embeds the six tools as a Python literal. A package cannot
  read `tests/fixtures`. `scenario_tools()` returns a deep copy, and a test pins it equal to
  `tests/fixtures/scenario_catalog.json`.
- New demo names:
  - `demo_router(backend, …)` builds a scenario router over any backend: sim, live or recording.
    `scenario_router(script)` keeps its old signature and model.
  - `SCENARIO_MODEL` is `~typesafe/jev-latest`.
  - `R1_DONE`, `R2_REPLY`/`R2_FREE_TEXT`, `R6_REFUSE_STEP2`, and the R6 step scripts, which
    `tests/loop/support.py` now re-exports (Review fixes, "`tests/loop/support.py` re-exports the demo world").
- `demo.scripts.criteria` and `demo.scripts.accept_candidate` raise `TypeError` instead of failing an `assert`
  (library code). `member_answers` skips member Nouls without object instructions instead of asserting.
- §12 fixtures (`examples/fixtures/*.answers.json`, `ScriptedBackend.from_fixture` format):
  - Naming: one file per request variant: `R1`, `R1-dropin`, `R2`, `R2-no-history`, `R2-free-text`, `R3`, `R4`,
    `R4-miss`, `R4-found-in-bucket`, `R5`, `R6`, `R6-refuse`, `R7`.
  - Every file carries `note` (illustrative [I], never evidence about Jev), `request` and `model`; the loader
    ignores `note` and `request`.
  - Static scripts are stored as `answers`, with globs kept.
  - Callable scripts (the R4 bucket hit, the R6 steps) are replayed once over the demo world and stored as
    per-request `rounds`, with exact qids.
  - `examples/fixtures/regenerate.py [--check]` writes them, and also `examples/proxy/data/*.json`. A test fails when
    any generated file is stale.
  - `helpdesk.answers.json` (example 08) is hand-written.
- §12 `--backend`:
  - `scripted` is the default.
  - `sim` is `LexicalSimulator()`.
  - `live` is `jt.backends.auto()` without `allow_offline`. It is checked at argument parsing, so a missing key
    exits with status 2 and a hint before anything prints.
  - Sections that only make sense with a scripted answer (the R4 misses, the R6 refused variant) are skipped with
    a note under `sim`/`live`.
  - Every run prints that scripted and simulated answers are never evidence about Jev's accuracy.
- §12 "print the Decision, the rendered prompt and a trace summary": `examples/_show.py` prints the following:
  - the questions per round, with count and ids grouped by tool, read from the trace's stored request bodies;
    `07` reads them from a recording backend wrapper, because `complete()` returns documents;
  - outcome and rule, the call, `C (composition, tier) · W · PI · L · J` with the tier's bands, bottleneck and gates;
  - slots with alternatives, the prompt with its option ids, and trace id, rounds, Jev calls, input tokens,
    pending, resumed-from, idempotency key and notes.
- Non-interactive runs click like a user would: `ok` when it is offered, else the option the example names, else the
  first option. `03` and `05` click the first option or the named alternative.
- §12 "the §13.5 request JSON printed byte-exact": `05` prints the compiled round-1 request (`router.compile(…)` →
  `Ballot.to_requests(router.model)`, no network) as order-preserving JSON. Containers stay on one line when they fit,
  as in the spec listing, and a count line reports `14 questions, 6,003 characters compact`. The test parses the
  printed JSON back and compares it, key order included, with `tests/fixtures/spec_r5_request.json`. The spec's
  hand wrapping of long lines is not reproduced.
- `07` uses `jt.openai.complete(messages, tools, backend=…, context=…)` for the tool loop, with a fresh router per
  call and `role: tool` messages as observations. The confirm card uses `router=` with the demo router, so the
  echoed card plus "ok" resumes as a click with no Jev call. Without a router, the reply is recompiled, which costs
  one round.
- `08` defines a new synthetic domain (a support desk):
  - `@jt.tool` functions with `jt.Ref`/`Literal`/`jt.Span`/`jt.Noun`;
  - two `jt.Registry` sources;
  - a foreign plain-JSON tool annotated only through `jt.hints`;
  - a tier set by hints, since "assign" is not in the verb table.

  Its agents registry provides `agent`, not `email`/`person`, so the invitee rule does not apply.
- Proxy example:
  - `examples/proxy/jevtools.toml` uses `backend.type = "auto"` with `allow_offline = true`, so it runs without a
    key on the simulator, with a warning. The README and the TOML say to set it to `false` in production.
  - `client.py` uses only the OpenAI SDK. `main(argv, http_client=…)` lets the test drive it through Starlette's
    `TestClient`.
  - The test uses a small stub of `openai.OpenAI` when the SDK is not installed. It was also verified with the
    real SDK (openai 3.19) against `jevtools serve` under uvicorn.
- Every example test is marked `fast` (registered in `pyproject.toml` at the final merge; it was first registered in
  `tests/examples/conftest.py`).

## Core polish: prompts, simulator tool scoring, quantity grids, golden fixtures

### Prompt rendering (§3.8.4; `prompts.py`)

- §3.2 `render` default: the spec's `"{intent}: p1=…, p2=…"` dumped raw values: e-mail addresses instead of
  labels, a multi-line body inline, parameter names. A tool without `x-jev.render` now uses a deterministic
  template (`natural_call`). It is never generated text. The rules:
  - **Head.** `short_intent`: from the third word on, the intent is split before `from to between for in into on
    with and by via using at about`. A phrase is dropped when its content words are empty or appear in a slot's
    name or noun. `send an email from the user to one recipient` → `send an email`; `get the current weather for
    a city` → `get the current weather`. A phrase no slot restates stays (`post a message to the #general
    channel`).
  - **Lead phrases.** A parameter named like a preposition (`to from cc bcc into onto at on in for with via by`),
    or `to_*`/`from_*`, attaches to the head: `to Anna Keller <anna.keller@acme.com>`. The other slots follow an
    em dash as `<term> <value>`, joined with commas.
  - **Terms.** The slot noun without its article when it has ≤ 2 words and ≤ 20 characters (`subject line`,
    `invitees`), else the humanized parameter name without a unit suffix (`duration_minutes` → `duration`).
  - **Values.**
    - `ref`/`enum`/`temporal`/`ordinal`: the candidate label.
    - `text`: a quoted one-line preview of at most 60 characters, cut at a word with `…`.
    - `quantity`: the value plus its unit word (`45 minutes`, singular for 1, `50%`).
    - `flag`: `yes`/`no`.
    - Lists: item labels joined with `, … and …`.
    - Empty values and secrets are omitted.
  - **Length.** Above 200 characters (menus, joint options) or 320 (confirm cards), cosmetic values are dropped.
    A menu's own slot is always kept. `Decision.call` keeps the full call.
  - R2 renders as: `Send an email to Anna Keller <anna.keller@acme.com> — subject line "Running 10 minutes late",
    body "Hi Anna, I'll be 10 minutes late. Best, Sam"?`
- The same default renders **joint option labels** of tools that have `groups` but no `x-jev.render` (§3.2 uses one
  `render` for both). In the scenario only `transfer_funds` has a joint question, and it declares `render`, so no
  scenario ballot changed. For other catalogs, joint labels change from `intent: p=…` to the natural form. They
  are still elided to 64 characters by `make_label`.
- `x-jev.render` and `confirm_template` are unchanged: R3 still reads `Transfer 250.00 CHF from Savings · CHF ·
  CH93…2957 to Checking · CHF · CH56…1180?`. External and critical clarify menus still show the complete resulting
  call per option.
- List items keep their part labels: `Binding.of_result` reads `SlotResult.parts`, so attendees render as
  `Bob Meier <bob.meier@muster.ch> and Carol Liu <carol.liu@muster.ch>`. `router._bindings` now calls it; that is
  a one-line change.

### Simulator tool scoring (§8.6; `backends/simulator.py`)

- §8.6 tool options: the literal `cov(U_c, T_o)` (the share of the request's content tokens the tool explains)
  made every long request clarify on the tool question: R2, R3, R5 and R6 → `P5.tool.ambiguous`. Tool options now
  score `0.6·verb + 0.4·own`:
  - `verb` = 1 when the request's leading action word is in the tool's verb family. The family is `expand` of the
    tool's name tokens plus the first word of its description.
  - `own` = `cov(toks(name), expand(U))`: how much of the tool's own name the request covers, family words
    included.
  - Other options keep their §8.6 scores (`NO_TOOL`, `UNSUPPORTED`, `DONE`…).
- **Leading action word:** the first family word of the request that no `progress` tool already explains.
  - Generic words (`find look get show check see fetch`) lead only when no specific family word follows, and only
    before the first step.
  - So "Find the latest invoice…" is led by `invoice` at step 1 (`read_file`) and by `forward` at step 2
    (`send_email`, since `read_file` already ran), while "Find a good pasta recipe" is led by `find`
    (`search_web`).
- **Families:**
  - `send` gains `forward` and `reply`; `book` gains `invite`.
  - The new `OBJECT_NOUNS` (`open`: `document invoice pdf report readme spreadsheet`) are used **only** for tool
    scoring. `DONE`, `done_after` and `authorized` keep reading the verb families, so "Pay the ACME invoice" after
    reading the invoice is not scored as done.
  - With `forward` in the `send` family, the R6 request's last action is `forward`: `send_email.done_after` answers
    0.9 and `search_web.done_after` 0.1. The unit test was updated.
- **Outcomes with the simulator** (tests/scenario/test_simulator_scenarios.py now pins them; each is a subset of the
  §10.5 allowed set):

  | Case | Outcome and rule | Note |
  |---|---|---|
  | R1 | execute | |
  | R2 (history) | confirm, Anna Keller | |
  | R2 without history | clarify, `P8.consistency` on `to` | |
  | R3 | clarify, `P8.consistency` | `order_sensitive`: a lexical double cannot tell from/to apart |
  | R4 | clarify, `P9.read.diffuse` | 40 config paths share the request's words |
  | R5 | clarify, `P9.external.ambiguous` on the invitees | |
  | R6 step 1 | execute `read_file` | |
  | R6 step 2 | clarify on `to`, `send_email` | recipient `finance@muster.ch`; the forwarded file text is late-bound into the body |
  | R7 | abstain | |

  No case clarifies on the tool question. The §8.6 guarantee list still holds and is still pinned:
  - two Annas → clarify;
  - `Rob` → widen;
  - a joke → abstain;
  - a hedge → `P4` not authorized;
  - an invoice-only amount → refuse (`transfer_funds` now wins at 0.98).

  Determinism and flip mode are unchanged. The simulator stays a lexical test double: **its outputs are never
  evidence about Jev's accuracy.**

### Quantity grids (§4.2.4; `kinds/quantity.py`, `prompts.py`, `router.py`)

- **Resolver hook.** Resolvers may implement `clarify_values(tool, slot, pool, rc) -> list[Candidate]`, duck-typed
  like `widen`. Only `quantity` does. `money` offers no grid, because amounts must be stated (the critical tier).
- **When a grid is offered.** Only when the slot is required, has no default or `default_from`, and nothing was
  stated: no pool candidate and none blocked by the allow-list.
  - The grid is `x-jev.values` if declared, else the unit's row: minutes 15/30/45/60, hours 1/2/4/8, seconds
    10/30/60/120, days 1/2/3/7, weeks 1–4, months 1/3/6/12, years 1/2/3/5, percent 25/50/75/100, milliseconds
    100/250/500/1000.
  - Values that fail the schema or a unary constraint are dropped.
  - At most `min(4, ambiguous_k)` values are offered, evenly spaced with both ends kept: index
    `⌊i·(n−1)/(k−1) + ½⌋`. So `from_jev_fn`'s `ge=1, le=20` offers 1, 7, 14, 20.
  - Grid candidates carry `channel=author, prov={"grid": true}`. They never enter a pool or a Ballot.
- **The menu.** When the policy asks an open question (P6 or a `missing` shape) and the hook returns values, the
  router shows a grid menu instead.
  - Text: `x-jev.ask`, or `What should {noun} be?`.
  - Options: `pick:<slot>:<i>` values such as `45 minutes`, plus `Something else`.
  - When the rest of the call was decoded, external and critical options show the complete call. As on a clarify
    menu, the click is then a binding and a confirmation (no Jev call).
  - `Something else` (`reason = change`) opens the plain question, so the grid does not come back.
- **Deviation: a grid click on an unspeculated tool.** When the tool was never speculated (P6, e.g. a `@jev.fn`
  tool whose required `seats` was not stated), no other slot has answers yet, so §3.8.5's "click: no Jev call"
  cannot hold. The click binds the value (p = 1, channel `user`, also injected as a user candidate) and runs one
  round with the tool named in `tool_choice`. It is not a confirmation. For open clarifies shown as grid menus,
  the pending state now records `tool` and `open_slot`, so free-text replies behave as before.
- **Gap (not changed here).** A `@jev.fn` field whose description is a question ("How many seats are affected")
  gets the noun `the how many seats are affected`, so its open question reads oddly. Noun inference lives in
  `spec/infer.py`.

### Golden conformance fixtures (§10.2; `tests/golden/`, `jevtools fixtures`)

- **Cases.** 16 cases: R1–R7, R2-no-history, R2-click, R3-TOCTOU-changed, R4-widen, R6-step1, R6-step2,
  R6-injection, 422-isolation and budget-split.
  - `R6` is the full agent loop: step 1 executes, step 2 confirms, the click executes, done. Its tool executions are
    recorded in the manifest. `R6-step1`, `R6-step2` and `R6-injection` are single Router decisions in loop mode;
    steps 2 and injection carry a plain `Observation` of the invoice.
  - `R4-widen` uses the full miss path (buckets + group, then hierarchy, then clarify(open)). `budget-split` is R5
    with `max_questions = 8`.
- **Format** (details in `tests/golden/README.md`).
  - A `case.json` manifest drives the replay (steps, exchanges, executions, output files), so the test and a port
    need only the files.
  - One exchange uses `request.json`/`response.json`; several use `request_<i>.json` in call order.
  - The final outputs are `decision.json`/`trace.json`; earlier steps are `decision_<k>.json`/`trace_<k>.json`.
  - A failed call's response is `{"error": {type, status, message, detail}}`.
  - Outputs are canonical JSON (§3.1). Inputs are indented JSON.
  - `context.json` holds source specs: `contacts` and `files` read the shared rows in `tests/golden/fixtures/`;
    `accounts` is inline.
  - Trace `created_at` → `2026-09-24T12:05:00Z` and `latency_ms` → 0: their only non-deterministic fields.
- **Assertions.** Beyond the four of §10.2:
  - every later request is compared byte for byte by the replay backend;
  - regenerated traces must equal the stored ones;
  - `test_fixtures_are_up_to_date` regenerates every case (about 6 s) and fails on any stale file.
  - The fixtures replay byte-identically under Python 3.10 and 3.11.
- **Verify without a context.** `verify` runs without the context for `422-isolation` and `budget-split`
  (`verify.with_context = false`), because it cannot rebuild their round-1 Ballot:
  - the trace records the isolated Ballot;
  - `verify` compiles with the default limits.

  Re-decoding, values, channels, composition and policy are still checked. Recording limits and isolation in the
  trace would let `verify` rebuild these too; that is a `trace.py`/`router.py` change, not made here.
- **Where the definitions live.** `tests/golden/cases.py`: the answers come from `tests.scenario.scripts` (a
  re-export of `jevtools.demo`). `jevtools fixtures [--update] [--dir] [--case]` imports it lazily, adding the
  checkout root to `sys.path` when needed, so the command works from a repository checkout only. Size: about 1.9 MB
  (shared rows 224 KB; `R4-widen` 360 KB for its two 251-option buckets).
- **Scripts are inputs.** A change to the demo scripts or scenario data changes the fixtures, and
  `test_fixtures_are_up_to_date` catches it. Per §10.2, regenerate only together with a spec version bump.

## Documentation and final merge (`README.md`, `docs/ARCHITECTURE.md`, merge fixes)

- The two sections above were merged from `docs/decisions/{examples,polish}.md`, and that directory was removed.
  Source docstrings that pointed at `docs/decisions/polish.md` now point here.
- `pyproject.toml`:
  - The `fast` marker is registered with the other markers, and `tests/examples/conftest.py` was removed.
  - `tomli>=1.1; python_version < '3.11'` is now a conditional dependency. On Python 3.10 every TOML read failed
    without it: `Policy.from_toml`, `jevtools serve --config`, `lint --sources *.toml` and the output check of
    `tune`. It adds nothing on 3.11+, where `tomllib` is used. This supersedes the "not a hard dependency" part of
    the Foundation §9 bullet.
  - ruff excludes the root `README.md` (`./README.md`) as it excludes `docs/`. ruff 0.16 formats Markdown code
    blocks and would stretch the compact snippets by about 60 lines. The other READMEs are still checked.
  - `license = "MIT"` was removed. No license has been chosen, the field was added by an implementation commit, not
    by the author, and there is no `LICENSE` file. README says so.
- §7.5 lint: source entries now get the same defaults as the sources they describe. `type` counts like `kind`, and
  a file index without `provides` provides `path`/`file`, as `FileIndex` does. Before, `jevtools lint --sources
  examples/proxy/jevtools.toml` reported `read_file.path` as a WEAK generic span, although the proxy binds it to
  the `files` index. There is a test in `tests/unit/test_cli.py`.
- README snippets run offline against the current API:
  - simulator, scripted and demo-world variants;
  - the OpenAI SDK against `wrap` and a live `jevtools serve`, in a scratch venv;
  - LangChain, Pydantic AI, and MCP 2.x through an in-memory `mcp.client.Client`.

  The core snippets were also run on Python 3.10. The R `ellmer` snippet is quoted from §7.3 and was not run (no R
  in the build environment).
- `jt.verify(trace, catalog=…, context=…)` needs the decision's conversation in the context
  (`router.context_for(messages)`). Otherwise `ballot_rebuild` fails. The README shows that form.
- Open gap (not changed; it would change the `R3-TOCTOU-changed` golden bytes, which §10.2 ties to a spec version
  bump): clarify-menu options are the bottleneck slot's top values and are not filtered by cross-slot
  `constraints`. After the TOCTOU re-plan in R3, the from-account menu still offers `250.00 CHF: Savings → Checking`
  although Savings now holds 100.00. Clicking it cannot execute, because the constrained MAP flags it and the
  router clarifies again, but the menu shows an infeasible call. Filtering infeasible complete-call options in
  `router._menu_choices` is the likely fix.
- Open cleanup: the noun inference gap for question-shaped `@jev.fn` descriptions (Core polish) is still open.

## Review fixes

Fixes from the code review, by area. They were merged from `docs/decisions/review-{engine,resolvers,edges,
docs-tests}.md`, and that directory was removed. Each entry names the review finding it comes from; "Amends",
"Replaces" and "Extends" name the earlier entry of this file that it changes (those entries carry a pointer here).
Where the two disagree, this section wins.

### Engine (router, plan, decode, confidence, policy, prompts, context, validate, budget, wire)

Regression tests: `tests/unit/test_review_engine.py`, `tests/scenario/test_review_engine.py`,
`tests/unit/test_prompts.py::test_parse_short_reply_numeric_option_texts` and
`tests/unit/test_confidence.py::test_isotonic_pools_ties_after_a_violator_merge`.

#### §3.8.5 confirmations are tied to the confirmed call (review #1)

Amends "§3.8.5 clicks" and "§3.8.5 free text" (Engine section): a confirmation is not a session flag but the call it
confirms, `(tool, sha256 of the canonical arguments)`. `ok` (click or a free-text reply whose `reply` Choice picks
it, `≥ 0.5`) confirms `Pending.call`, the call on the card; a `pick:`/`alt:` click in tiers ≥ external confirms the
complete call its option showed (the call after binding the clicked value). `PolicyInput.confirmed` is true only
while the decoded call equals the confirmed one: a free-text resume round that re-decodes other arguments, or a
speculation-miss re-plan onto another tool (which clears the confirmation), goes through P9 normally, so the new call
gets its own confirm card. The critical tier therefore never executes a call no card showed.

#### §3.8.5 a click keeps the value's origin; TOCTOU re-checks it (review #2)

Amends "§3.8.5 clicks": a clicked or reply-picked value still binds with channel `user` (the result's channel), but
when it was an offered pool entry, the bound entry keeps that candidate's channel as the value's *origin*
(`decode.value_origin`). TOCTOU (`Router.default_revalidate`) re-resolves every value whose origin is `registry`
(clicks included): membership, label, fresh attributes, constraints. The I2 check (`channel_violations`) admits a
user binding whose origin channel is on the slot's allow-list, so a slot narrowed to `["registry"]` accepts a click
on a registry value it offered (it used to refuse with P3 `channel_violation`).

#### §3.8.5 short replies: option texts before option numbers (review #7)

Amends "§3.8.5 clicks" (`prompts.parse_short_reply`): a reply is matched against option texts and ids first. A
digit-only reply is read as a 1-based option number only when no option text starts with a digit; in a numeric menu
(quantities, grids) a number that matches no text is ambiguous and is not a click (free-text resume). The LangGraph
`confirm_node` still sends a structured `{"selection": id}` as the option's text; with texts first, it now always
resolves to the clicked option.

#### §5.6 422 isolation (reviews #4/#8, #5, #16)

Replaces the §5.6 entry: a 422 whose `loc` names questions drops those questions' whole slot families and re-sends
the failed calls once. Only slot questions (a tool and a path) are isolatable: a `loc` on any tool-level question —
`tool`, `reply`, and the gates `T.authorized`, `T.joint[.G]`, `T.done_after` — or outside `questions` fails closed
(P0), because dropping a gate would remove a check instead of failing it (I5). A dropped slot family decodes as a
failed answer, `empty(reason=invalid)`: `⊥missing`, factor 0, flag `invalid` (`decode.dropped_result`), whatever the
slot's default or `required` — never a silent default or omission (§4.5); it routes to clarify(open). In opaque id
mode the trimmed Ballot renumbers the wire ids, so a kept call whose ids changed is re-sent too (dotted mode still
re-sends only the failed calls).

#### §3.8.2 P4/P9: a missing `authorized` fails closed (reviews #4/#8)

New: for a speculated tool of tier ≥ write (the planner always asks `T.authorized` there), `authorized = None` is a
failed answer: P4 abstains (`P4.safety.not_authorized`, reason `authorized_missing`) and the P9 `authorized` cap
applies. Read-tier tools (never asked) and unspeculated tools (P6) are unaffected.

#### §3.5.1 / §14 secrets never leave the host (reviews #3, #6)

Amends §3.5.1 `shareable` (`None` = every profile field is shared): `state.user` never carries a secret, shareable
or not. The planner drops every top-level profile field a `secret` slot reads (`default_from: user.<field>…`) and
every field whose name is a §3.3.1 row-1 secret name (`password, token, api_key, apikey, secret, credential`,
case-insensitive) — `plan.secret_user_fields`, `build_state(..., secret_fields=...)`. FILL's `FillRequest.frozen`
leaves out secret arguments (kind `secret` or `prov.secret`). The call, `Decision.slots`, the trace and the pending
handle still hold the real value (host side; `tool_calls` must carry it). Still open (kinds area):
`kinds/text.py::_profile_inputs` reads `ctx.user_state()` without these exclusions, so an author template that
names `{user.<secret field>}` could still put the secret into an accept-Noul candidate.

#### §6.2 one preview implementation (review #14)

Replaces the 1,000-character head cut of `Observation.preview_text()`: previews live in `jevtools.preview`
(`PREVIEW_CHARS = 1_200`, `chunk_text`, `select_preview`, `text_preview`), shared by `loop.ingest_observation` and
`Context.all_observations()`. A drop-in `role: tool` message gets the same preview as an ingested result: the text
when it fits in 1,200 characters, else the chunks BM25 ranks highest against the request. An observation without a
preview (declared without one) falls back to its chunks in document order, up to 1,200 characters.

#### §5.5 budgets and token estimates (reviews #13, #18)

New: the policy's `[budget]` table is applied (`Limits.within_budget`, in `Router.__init__` and `compile_round`):
every budget setting the policy changes from its Appendix B default caps the matching limit, taking the smaller value
(`max_tokens_per_call` → `max_tokens`, `max_questions_per_call` → `max_questions`, `max_state_tokens`,
`chars_per_token`). A tighter budget takes effect; a looser one never exceeds the probed or explicit limits; a default
budget leaves them unchanged. One formula, `budget.tokens_for_chars` = `⌈chars × r / chars_per_token⌉`, serves the
planner (splits, state cuts), the pre-send check (`Limits.estimate_tokens`) and `TokenEstimator.est`. In opaque id
mode the planner counts each question's id as at least an opaque id's length (`q0001`).

#### Resume on a router serving another tool list (review #12)

Amends "a pending not held in memory, or expired, is safely re-compiled": a pending whose `state.tool` is not in the
resuming router's catalog is re-compiled too (note `pending tool '<name>' is not in the tool list: recompiled`),
also for free-text replies.

#### Router pending handles are bounded (review #9)

New: `Router.pendings` and the click-resume state are bounded together at `LIVE_MAX` (256): expired handles are
dropped on every insert and the oldest are evicted beyond the bound. An evicted handle is resumed from its `Pending`
object like any handle not held in memory (re-compiled); a string id no longer held raises `KeyError`. Handles are
not released on resume (a card may be clicked again).

#### Answer-shape guards and malformed bodies (reviews #10, #15)

Amends the §8.2 guards: Score answers are checked too (every probability in [0, 1], the expected level finite and
within `[0, levels − 1]`). The wire answer models reject non-finite numbers (NaN, ±Infinity), `DecisionResponse`
rejects a non-object `answers` (a `ValueError` → `JevProtocolError`), and `HTTPBackend` maps non-transport
`httpx.HTTPError`s (e.g. a body that fails to decode) to `JevProtocolError` without retrying. All of them end in P0.

#### §3.6 rule 5 under the family rule (review #17)

Amends the §3.6 rule 5 entry: after a late-binding or schema failure, a text slot's remaining candidates are
re-elected by the accept rule (`decode.discard_values` → `kinds.text.accept_result`): the 0.02 tie to the lower
index, content below `accept_min` → `uncovered_text`, cosmetic below `cosmetic_floor` → the next author template,
else omitted when optional, else `uncovered_text`. Other kinds keep the generic re-election.

#### §3.7 isotonic calibration pools ties (review #11)

New: `IsotonicCalibrator.fit` pools observations per distinct `C_prior` before PAV (the secondary tie approach), so
a calibration map is a function of `x`; tied priors after a violator merge no longer split into an upward-biased
block. Zero-weight points are ignored.

### Kinds, extractors, sources (kinds, extract, sources, spec)

Regression tests are in `tests/unit/`: `test_source_specs.py`, `test_kinds_base.py`, `test_kinds_scalars.py`,
`test_kinds_temporal.py`, `test_kinds_composite.py`, `test_kinds_ref.py`, `test_kinds_enum.py`,
`test_extract_numbers.py` and `test_extract_temporal.py`.

#### Proxy: per-request source specs are data only (#1)

Replaces "a full spec is built as given" in the Proxy section.

A per-request full spec (`extra_body.jevtools.sources.<name> = {...}`) is untrusted input. `sources.specs.request_spec`
validates it:
- It may carry only data fields: `rows`/`paths`, `key`, `label`, `describe`, `match`, `provides`, `attrs`,
  `synonyms`, `list_fields`, `list_sep`, `item`, `retriever`, `send_whole_if_under`, `k`, `recency`, `hierarchy`,
  `groups`.
- Its `type`/`kind` must be `registry` or `files`, and it needs inline `rows`.
- `function`, `options`, `path`, `rows_file`, `paths_file` and `channel` raise `ValueError`, which becomes a 400
  `jevtools_bad_sources`. This happens before anything is imported, called or read.

`module:function` providers and file-backed sources stay in the operator's `jevtools.toml` only (SPEC §7.2.4).

#### §3.4.2 / §4.2.1: history candidates inherit a traced origin (#2)

This refines "a `history` candidate is admitted only if its `origin` channel is also admitted (or trusted)". A
missing origin no longer counts as trusted.

`kinds.common.trace_history` runs in `finalize_pool` before dedupe. It gives each assistant-turn mention candidate
(`history`, no origin) an origin:
- If an observation contains the value, or a `tool_output` entity has it, the origin is `tool_output`.
- Otherwise, if the user's own turns contain it, the origin is `user`.
- Otherwise, if a registry row is keyed by it, the origin is `registry`. A trusted entity gives its own origin.
- Otherwise the origin is `tool_output`, because untraceable assistant text is untrusted.

A history copy of a value that is also in the pool from an untrusted channel, or held by an untrusted entity, takes
that least-trusted origin. Dedupe compares effective channels, so a history copy counts at its origin's trust. An
assistant turn that repeats an injected tool output therefore no longer reaches an external identity slot (§6.6,
E10 "planted in history").

Not changed here, because it is outside this area: the entity store's "merge keeps the most trusted origin seen" and
the origin `history` of assistant regex entities (loop.py), `BallotOption.from_candidate` copying `channel` rather than
`effective_channel` (ballot.py), and `admits()` treating `origin is None` as trusted (candidates.py).

#### §3.6 rule 2: pooled Choice mass is clamped (#4, #16)

Labels that decode to one value pool their mass. A pooled mass above 1 is rounding, and it is clamped to 1. It is
never renormalized, so I3 holds. `elect` also clamps the factor at 1. A Choice whose sent labels sum to more than
`1 + SUM_TOLERANCE` (0.05) is not a distribution and fails closed (`no_answer`, factor 0).

#### §4.2.5 factorized temporal slots with a default (#5)

When both parts are `NOT_STATED`, the joint mass (`D_date(NS) · D_time(NS)`) goes to the slot default. A date with no
time, or a time with no date, is incomplete: that mass goes to ⊥missing. A part is never combined with a piece of the
default. Decoding combines only offered part values with positive mass.

#### §4.2.9 arrays of objects keep their leaves' flags and channel (#6)

A list of records now passes on its leaf flags (`presence_conflict`, `order_sensitive`, `no_answer`) and its
least-trusted non-bottom leaf channel, the same way `record.assemble` does. The leaf qids are added to `qids`.

#### §4.2.1 claiming: weak temporal readings (#7, #8)

A temporal mention with no temporal evidence is *weak*: it is emitted, but it never claims the numbers it covers.
The mention carries `attrs["claims"] = False`, which `Mention.claims` reads and `claim()` skips. Weak mentions are:
- `by 5`, `at 3` and `um 5` without am/pm, `:mm`, `o'clock`, `Uhr` or `h`;
- `from 2 to 4` and `von 2 bis 4` without a suffix;
- `after 3` and `before 3`;
- a year-less `21.5.` in a `.`-decimal locale.

A group with any strong atom (`tomorrow at 5`) still claims. `between`, `zwischen` and `entre` are strong.

Year-less dotted dates follow these rules:
- They are never read inside a dotted run (`1.2.3`, `10.1.1.5`).
- In `en`, one that ends the text, or is followed by a capital or `!?)`, is a decimal and not a date.

#### §4.2.5 temporal additions (#9, #10, #11, #18, #19)

- **Zones.** A zone atom joins any group that carries times, including ISO date-times. ISO accepts fractional
  seconds. `UTC±H[:MM]`/`GMT±…` gives a fixed offset `UTC±HH:MM` and is read before clock times.
  `ZONE_ABBREVIATIONS` adds abbreviations, honoured only right after the time. An ambiguous one gives one glossed
  reading per zone: CST, IST, AST and MST.
- **Weekday plus week.** A weekday in a group with a week range (`Thursday next week`, `next week Tuesday`,
  `Thursday of this week`) is that weekday inside the week (`weekday:of_week`), and the week range is consumed. If
  the weekday is not inside the range, there is no point reading.
- **Dash ranges.** Ranges without a keyword need a suffix on the second hour: am/pm, `Uhr` or attached `h`, as in
  `4-6pm`, `9–11am` and `4pm-6pm`. With a mixed meridiem, when inheriting gives start ≥ end, the other meridiem is
  tried: `between 10 and 2pm` is 10:00–14:00. A matched range whose readings are all invalid still reserves its span
  (a `void` atom), so its end never becomes a point.
- **Midnight with an explicit date.** It gives the start of that day and its end
  (`clock:midnight:start`/`:end`, glossed). A start already in the past is dropped. Bare `midnight` keeps one reading.
- **de `Morgen`.** After `heute`, `gestern`, `jeden`, `am`, `den`, `morgen` and similar words, a capitalized
  `Morgen` is the morning (a day-part RANGE). After `Guten`, it is a greeting with no temporal meaning. Otherwise it
  is still tomorrow. The locale fields are `day_part_homographs`, `day_part_homograph_cues` and `greetings`.

#### §4.1 numbers and money (#3, #12, #13, #14, #21)

- `decimal_str` and `quantize_money` use a local context wide enough for any digit count. A 29+ digit number no
  longer raises. `run_extractors` also isolates each extractor per text: a failure yields no mentions and is recorded
  in `Mentions.failures`.
- Space-grouped thousands (`10 000`) are read in every locale, unless another digit group touches the run (phone
  numbers). Digits glued to an ISO currency code are read whole (`CHF1'250.50`).
- Magnitude suffixes and words multiply the number: `2k`, `$2M`, `1.5 million`, `3 Mio.`. Number words continue
  after scale words: `two thousand five hundred` is 2500 and `one hundred (and) twenty` is 120. de and fr get
  million and milliard scale words, and en gets billion.
- A minus sign directly before digits is kept (`-3`, `−18`). `5-10`, `A-3` and dates stay unsigned. Money also keeps
  `CHF -50` and `-CHF 50`.
- Money is never built from a fragment: another digit group or a scale word right next to the number drops it.
- `WORD_CODES` (ALL, AMD, BOB, CUP, CVE, GEL, MAD, MOP, PEN, PHP, SOS, TOP, TRY) make money only with a second money
  cue: a symbol or word on the other side, `.–`, or minor-unit decimals.
- Compound durations are summed in their smallest unit by one scanner, `numbers.duration_at`/`extend_duration`,
  which quantities and temporal offsets share. Examples: `1 hour 30 minutes`, `2 hours and 15 minutes`, `1h30`,
  `1h 30m` and `1:30 hours` are 90 or 135 minutes, and `in an hour and a half` is +90 min. Only hour, minute and
  second combine. Quantities covered by a longer quantity are dropped.

#### §4.3 `path@1` on path slots without a file index (#15)

A span slot tagged `path`/`file` uses `path@1`: `SpanResolver.normalizer_for` returns `path@1`, and
`span.path_candidates` normalizes values and drops escaping ones at pool time. Author values are exempt. In addition,
`normalize_path` rejects `~`, `$VAR`/`${VAR}`/`%VAR%` and `C:/` drive prefixes unless the value is `known`, meaning
present in a source.

#### §3.6 / §4.2.8.5 superlative ties (#17)

Items tied with the chosen item on the order attribute count as competitors:
`factor = q · ∏ beyond (1 − q_j) · ∏ tied (1 − q_j)`. The distribution stays ≤ 1. A tie at the extreme sets the flag
`tie` and adds a note. The pick among tied items follows value order, never retrieval order.

Open item for policy.py: `tie` could join `CONSISTENCY_FLAGS` so that the policy offers a menu. Today the low
factor already prevents execution.

#### §4.2.11 rung 3: perspective rewrite (#20)

`rewrite:perspective` variants are produced only for clauses that name no one but the recipient. A proper noun or a
registry anchor that is not a place, date or enum value blocks the variant ("Tom is sick and he can't come").

#### Resolvers and normalizers (#22, #24)

- `EnumResolver` subclasses `ChoiceResolver`. It keeps its own `pool`, `members`, `candidate` and `widen`, and
  inherits `questions` and `decode`. As a result, the hierarchy round decodes the hierarchy Choice rather than a
  merged bucket question, and an unasked enum slot records `enum@1`.
- `load_catalog` reads packaged files through `extract.catalogs.load_data`, so each file is parsed once.
- Author templates record `text.template@1`, which is `normalize_title` and is registered. `text@1` would add final
  punctuation. The docstring of `NORMALIZERS` no longer claims that `jt.verify` re-runs them.
- `ResolveContext.state` and `get_state` are kept as public extension conveniences for `register_resolver` users.
  The built-in resolvers do not read them.

### Edges (backends, adapters, proxy, eval, loop, CLI)

The regression tests are named after each item. None of these changes alters a golden fixture.

#### §7.2.4 pending handles are bound to the requester's scope (review #1)

Amends "§7.2.4 prefix key" and "§7.2.4 matching" (Shared plumbing):

- `adapters.pending.pending_scope(router, context)` is `sha256` of these parts of the effective context: `user`,
  `tz`, `locale`, `shareable`, `include_system`, its sources, and the router's tool names. Sources whose rows are
  given (registries, files) count by name plus content hash. Lazily fetched sources (`ToolSource`, MCP resources)
  count by name only: hashing them would fetch their rows (a tool call, which an async caller cannot make at that
  point, and a failure would cache an empty registry), and their hash changes with every refresh. `now` and the
  messages are left out: a wall-clock `now` would make every lookup miss, and the messages are the key itself.
- The prefix key is now `sha256(canonical([scope, *normalized messages[0..k]]))`. `prefix_key(messages)` without a
  scope keeps the old formula for hosts that call it directly.
- `remember(..., scope=)` stores a copy of the handle with `Pending.state["adapter_scope"] = scope`.
  `match_pending(..., scope=)` treats a handle stored under another scope as a miss, whether it was found by the
  prefix key or by a `pending_id` (explicit or echoed). The turn is then compiled afresh, which is still correct
  (§7.2.4), and the other requester's entry is neither consumed nor deleted.
- `decide_turn`/`adecide_turn` compute the scope themselves; a `scope=` argument overrides it.
- Consequence: when the rows of a given-rows source change between the card and the reply (for example the proxy's
  per-request `sources`), the prefix resume misses and costs one round.
- Not done, since it is outside this area: the defence-in-depth check in `Router._resume_flow`/`_click` that would
  refuse to reuse a live session whose context scope differs from the resume `ctx`. The adapter-level check covers
  every stateless path (`complete`, `wrap`, proxy, LangChain, Pydantic AI). A host calling `Router.resume` directly
  with another user's context is still unchecked. Open for the router owner.

#### §7.2.2 `tool_choice` on a resumed turn (review #8)

Amends "§7.2.2: `tool_choice` goes to the router unchanged". When a request answers a stored prompt,
`decide_turn`/`adecide_turn` resume it only if the request's `tool_choice` allows it:

- `auto` and `required` resume. A cancel under `required` still abstains: it is the user's explicit choice.
- A named choice resumes only when it names the pending tool (`Pending.state["tool"]`, else `Pending.call.name`).
- `none` never resumes.

When a request declines the resume, the turn is decided fresh with its `tool_choice` (`none` abstains with no Jev
call), and the handle stays stored so that a later request that allows it can still resume. An invalid
`tool_choice` raises `ValueError` on this path too, the same as on a fresh decide.

#### §7.2.1 async client detection (review #2)

Amends "§7.2.1 `wrap`". A client is treated as async when `chat.completions.create` is a coroutine function after
`inspect.unwrap`, or when the completions class name starts with `Async`. `is_async=` still overrides. The openai
SDK's `AsyncCompletions.create` is wrapped in a sync `functools.wraps` decorator (`required_args`), so the old check
found `AsyncOpenAI` to be sync.

#### §6.5 concurrent resumes and executions (review #3)

Amends "§6.5 idempotency". The Agent's guarantee now holds for concurrent calls too: `asyncio.gather` of two
`aresume` calls, or two threads calling `resume`.

- A resume claims `resume:<pending id>` before its first effect, and an execution claims `key:<idempotency key>`
  before it invokes the executor. Claims are `concurrent.futures.Future`s under a `threading.Lock`.
- A second caller waits for the claim (a `_Wait` effect: `await asyncio.wrap_future` in async code, blocking in
  sync code) and then replays: the first `LoopResult` for a resume, or the stored observation (0 attempts) for a
  key.
- A sync resume that would block the event loop that holds the claim raises `RuntimeError` and points to
  `aresume`, instead of deadlocking.
- The drivers close the flow on any exception, so claims are released at once and not left for garbage collection.

#### §6.4 a coreference binding never launders trust (review #5)

Amends "§6.4 entity store":

- `pin_call` takes the origin from the trace channel, except that a `history` binding whose `prov.entity` names a
  stored entity keeps that entity's origin. A `tool_output` value bound through coreference in a read-tier call
  therefore stays `tool_output` and remains `channel_blocked` for external identity slots (§3.4.2, I2).
- `EntityStore.add` still keeps "the most trusted origin seen". The exception is a `history`-origin sighting (an
  assistant mention, or a coreference pin whose entity is gone): it never raises the trust of an entity whose origin
  is less trusted than `history`.

#### §8.2 HTTP backend client lifetime and retry hints (reviews #6, #7, #10)

New (not recorded before):

- The sync `httpx.Client` is created once, behind a double-checked `threading.Lock`, even when concurrent split
  calls ask for it together. `close()` swaps it out under the same lock.
- Async clients are kept one per live event loop as `id(loop) → (loop, client)`. Each call first drops the clients of
  closed loops. They cannot be closed any more, because closing needs their loop; their sockets are freed at garbage
  collection. `aclose()` closes the running loop's client, drops closed loops' clients, and leaves clients of loops
  still running in other threads to those loops. It never awaits a client of a closed loop, so
  `async with backend:` no longer raises `Event loop is closed`.
- `retry_after()` ignores a hint that is not a finite number (`inf`, `1e400`, `nan`) and uses backoff instead.
- A hint above `max_retry_wait` is not waited in-process. `max_retry_wait` is a new constructor option; its default
  is `DEFAULT_MAX_RETRY_WAIT = 10 s`. The response is parsed at once and the typed error carries the hint
  (`JevRateLimited.retry_after`), so the router fails closed (P0) and the caller decides whether to wait. The same
  applies on the async path, where `asyncio.sleep(inf)` used to hang.

#### §7.2.4 proxy error mapping (review #11)

Amends "§7.2.4 error mapping" and the Proxy section.

The following checks run before any Jev call, with the new codes:

- `messages` must parse (`context.parse_messages`). A non-object message or an unknown role (the legacy
  `function` role) gets 400 `invalid_messages`.
- `tool_choice` must parse against the request's tools (`plan.parse_tool_choice`). An unknown name, an unknown
  string, or the `allowed_tools` form gets 400 `jevtools_bad_tool_choice`.
- Both are handled like a bad tool: with `fallback_llm` configured the request is forwarded, with reason
  `bad_request`.

Other changes:

- `merge_context` raises `TypeError` for an override that is neither a `Context` nor a mapping. The proxy maps it to
  400 `jevtools_bad_context`; it used to be an uncaught `AttributeError` and a plain-text 500.
- `decision_error`: a P0 decision whose calls were all answered, with no call error recorded, failed while decoding
  the answers (a missing answer, a wrong type, a Noul outside [0, 1]). It maps to 502 `jev_protocol_error`, as the
  table already says for `JevProtocolError`, and not to 503. The router does not record that failure in the trace,
  so the adapter infers it.

#### §7.5 CLI (review #12)

`load_trace` and `load_context` raise `JevtoolsError` ("expected a trace/context object") when the document is not a
JSON object. `explain`/`verify` then print `jevtools <cmd>: …` and exit 1 instead of a traceback.

#### §7.2.5 Pydantic AI: prompts under structured output; streaming (reviews #4, #13)

Extends the Pydantic AI section:

- When the prepared parameters have `allow_text_output = False` (for example `output_type=SomeModel`), a decision
  that is not a call raises `JevPromptRequired` (a subclass of `pydantic_ai.exceptions.AgentRunError`) and does not
  answer text. This covers a confirm, a clarify, an abstain, and a finished loop without an elected output.
  Answering text would make pydantic-ai reject it, re-decide the same turn with a second paid Jev round, and then
  raise `UnexpectedModelBehavior`.
- The exception carries `decision`, `text` (the prompt), `pending_id`, `doc`, and `messages`, which is the history
  plus the prompt as a `ModelResponse`. `agent.run(reply, message_history=exc.messages)` resumes the prompt, and a
  click costs no Jev call.
- `output_type=[SomeModel, str]` still answers prompts as text. A `text_model` abstain handoff is unchanged.
- `JevModel.request_stream` is implemented. It decides like `request` and replays the `ModelResponse` as one
  streamed message: a text delta per text part and a tool-call event per call, with usage, provider details and
  finish reason copied. A `text_model` handoff is requested without streaming and then replayed.
  `agent.run_stream` and `run_stream_events` now work.

#### §7.2.3 LangChain `with_structured_output` (review #9)

Extends the LangChain section:

- When `bind_tools` receives `ls_structured_output_format`, as `with_structured_output` passes it, each schema tool
  is declared `x-jev.risk: read` (`adapters._router.as_output_tool`, using the same `OUTPUT_TOOL_XJEV` constant as
  Pydantic AI, which now lives in `adapters._router`).
- A schema that already declares a `risk`, in the function's or the parameters' `x-jev`, is left as is.
- Without this, the schema's name fell to the fail-safe external tier, whose `authorized` gate and confirm band
  turned a confident extraction into a prompt, and the parser returned `None`.

#### §11.1 source paths inside a context file (review #14)

Amends "§11.1 files". `context`/`catalog` paths still resolve against the JSONL file's directory. Relative source
paths inside a context file (`path`, `rows_file`, `paths_file`) now resolve against that context file's directory,
as `jevtools verify --context` and the golden replay already do. An inline context document keeps resolving against
the dataset directory. The golden contexts (`rows_file: "../fixtures/contacts.json"`) now load from any dataset.
Open: README (Tuning) and SPEC §11.1 do not say this yet.

#### §7.1 / §6.5 one implementation of MCP result handling (review #15)

Amends "§7.1 MCP":

- `adapters.mcp.result_content(result)` is `sources.toolsource.result_data(result)`: `structuredContent`, else the
  text blocks joined, parsed as JSON when they are JSON, with non-text blocks left out. This is exactly the
  observation's content.
- `adapters.mcp.is_error` and `loop._normalize_result` share `loop.mcp_is_error` (`get_any(result, "isError",
  "is_error")`). The first key present wins, so `{"isError": false, "is_error": true}` is no longer an error in the
  loop.
- Still open: `sources/toolsource.py:337` has the same one-liner inline. It cannot import `loop`, so it is left for
  the sources owner.

#### §6.3 observation entities use the shared extractors (review #16)

Amends "§6.3 observations":

- Emails, URLs, UUIDs and IPv4 addresses in observation text come from `extract.patterns.extract`. Invalid IPv4
  addresses such as `999.300.1.2` are no longer entities, and a URL's span now excludes the trailing punctuation
  that is cut from its text.
- Money matches any ISO 4217 code from the currency catalog, except `extract.money.WORD_CODES` (`TOP 10` is not
  money), or a symbol from `extract.money.SYMBOLS` (which adds `₣` → CHF). The value format is unchanged
  (`"4820.00 CHF"`). Observations therefore recognise more amounts (CZK, …).
- Swiss `Fr. 4'820.–` and currency words are still recognised only request-side. Moving the loop to
  `numbers.extract` + `money.extract` needs tokens and a locale, and was left out while the extract area changes.

#### Shared sync bridge (review #17)

Additive, cross-area: `jevtools._compat.run_sync(value, hint)`. It returns a non-awaitable unchanged, runs an
awaitable with `asyncio.run`, and inside a running loop closes a coroutine and raises `RuntimeError(hint)`.
`loop.Agent` uses it; `_sync_value`/`_awaited` are gone. Still open for the sources owner: `toolsource._await_now` +
`_wrap`, `sources.mcp._run` + `_wrap`, and the inline copy in `Provider.candidates` + `_await`. Each should become
`run_sync(..., <its current message>)`.

#### Tuning uses the policy's hysteresis test (review #18)

"Thresholds reproduce the policy's hysteresis test exactly" now holds by construction. `eval.tuning` calls
`policy._clears` (imported as `clears`) and imports `policy._EPS`; its copy is gone. A public alias
(`policy.clears`) would be cleaner; that is left to the policy owner.

### Documentation and tests (README, docs, examples, tests)

Regression tests: `tests/docs/test_docs.py`, `tests/adapters/test_extras.py`,
`tests/adapters/test_mcp.py::test_call_decision_on_a_real_in_memory_server`, `tests/loop/test_support.py`, the
`response_shape` parametrization of `tests/adapters/test_openai.py` and the live suite `tests/live/test_live.py`.

#### Extra lower bounds match what the adapters call (#1)

Extends the `pyproject.toml` bullet of "Documentation and final merge":

- `mcp = ["mcp>=1.19"]` (was `>=1.10`). `adapters.mcp.call_decision` always passes `meta={"jevtools/idempotency_key":
  …}`, and `ClientSession.call_tool` accepts `meta` from 1.19.0 on; 1.10.0 to 1.18.0 raise `TypeError` on the first
  executed call. Checked in scratch venvs with the README MCP snippet on a FastMCP in-memory server (1.18.0 fails,
  1.19.0 and 1.20.0 pass).
- `pydantic-ai = ["pydantic-ai-slim>=1.0.13,<2"]` (was `>=1.0`). `JevModel.request` calls `Model.prepare_request`,
  present from 1.0.13 on (1.0.0 to 1.0.12: `AttributeError`). Checked with the README Pydantic AI snippet (1.0.0,
  1.0.8 and 1.0.12 fail, 1.0.13 to 1.0.18 pass; opentelemetry-api pinned to 1.30.0, since pydantic-ai-slim 1.0.x does
  not import against opentelemetry-api ≥ 1.37).
- `tests/adapters/test_extras.py` pins both bounds (the last broken version is excluded, the first working one
  admitted); the adapter tests use duck-typed fakes, so the suite never exercises a lower bound otherwise.
  `test_mcp.py::test_call_decision_on_a_real_in_memory_server` runs `call_decision` on a real mcp 2.x `ClientSession`
  and checks that the key reaches the server in `_meta`.
- Not changed (outside this area): `call_decision` could degrade like the Agent loop does (`loop.py`: pass `meta`
  only when `accepts_keyword(call_tool, "meta")`). With the raised bound it is no longer needed for supported
  installs.

#### The `dev` extra installs the OpenAI SDK; the `wrap` tests cover both response shapes (#5)

Extends "OpenAI (`adapters/openai.py`)":

- `wrap` returns `openai.types.chat.ChatCompletion`/`ChatCompletionChunk` when the SDK imports and an `AttrDict`
  otherwise (unchanged). `test_wrap_intercepts_its_model` and `test_wrap_streams_two_chunks` subscripted the response
  (`resp["usage"]…`), which only the `AttrDict` supports, so they failed whenever `openai` was installed. They now use
  attribute access and `model_dump()`, and run twice through the `response_shape` fixture: `attrdict` hides the SDK
  (`sys.modules["openai.types.chat"] = None`), `sdk` needs it (`importorskip`).
- `dev = [..., "openai"]`, so a checkout's `uv pip install -e ".[all,dev]"` tests the shape users of
  `wrap(OpenAI(), …)` get. (The repository venv predates this and has no `openai`; the `sdk` cases skip there.)

#### §10.7 live suite: `tests/live` (#4)

New (the suite did not exist; README "Development", `docs/ARCHITECTURE.md` and the `live` marker promised it,
and `pytest -m live` with a key deselected everything and exited 5):

- `tests/live/test_live.py`, marked `live`, skipped without a key by the existing gate in `tests/conftest.py`. The
  `backend` fixture is parametrized over the three HTTP backends; each runs when its own key is set (TypeSafe:
  `TYPESAFE_API_KEY`; OpenRouter System One and Decisions: `OPENROUTER_API_KEY`) and skips otherwise.
- Per backend: the conformance probe (limits file written to a temporary path, never the developer's cache), R1–R5
  and R7 in turn mode over `jevtools.demo.demo_router`, R6 as an `Agent` loop over the fake demo `Workspace`, and a
  cassette record → replay round trip; plus one R1 decision through `jt.backends.auto()`.
- Assertions are invariants only: `jt.verify` with catalog and context (values, channels, composition, policy, ballot
  rebuild), `C ≤ W`, tool calls only on execute, `jsonschema`-valid arguments, a critical call only after a click, and
  in R6 no transfer and nothing sent to the injected address or IBAN. Never probabilities, rules or outcomes.
- Deviation from §10.7: E2/E3/E4 run as the probe's smoke items only (their numbers are recorded in the report, not
  asserted); cassettes are recorded to a temporary file and replayed, not kept; drift of `answers.model` is not
  flagged. The suite was checked offline by routing every `HTTPBackend` through an `httpx.MockTransport` answered by
  the `LexicalSimulator` (31 passed with both keys set, 21 passed and 10 skipped with only `OPENROUTER_API_KEY`); it
  has not run against live Jev.
- `tests/docs/test_docs.py::test_live_marker_selects_a_live_suite_per_backend` collects `-m live` in a subprocess
  with a dummy key and requires a probe test per backend.

#### `tests/loop/support.py` re-exports the demo world (#3)

Supersedes the remark in "Examples and the demo world" (the R6 step scripts duplicating `tests/loop/support.py`,
updated in place) and the first sentence of the "Open cleanups" bullet in "Documentation and final merge" (removed):

- `tests/loop/support.py` keeps only `r6_agent` and `r6_messages`. `Workspace`, `INV_2291`, `FINANCE`,
  `INJECTED_ADDRESS`, `INJECTED_IBAN`, `INJECTION`, `INVOICE_TEXT` and `R6_REQUEST` come from
  `jevtools.demo.scenario`; `R6_MEMBERS`, `R6_STEP2`, `observations_of`, `member_answers`, `r6_step1` and `r6_script`
  from `jevtools.demo.scripts`. `__all__` is unchanged, so the loop tests import as before. The private copies had
  drifted (`get_weather` returned 17 in every unit; `member_answers` asserted on non-object instructions); no test
  depended on either. `tests/loop/test_support.py` pins that the names are the demo objects.
- Still open (outside this area): `jevtools/demo/scripts.py` defines its own `R6_REQUEST` string next to
  `scenario.R6_REQUEST`; it could import it as it imports `FINANCE`. `test_support.py` pins that the two are equal.

#### Documentation fixes (#2, #6, #7)

Extends "Documentation and final merge":

- Install commands name the git source everywhere, since the package is not on PyPI and the name is unclaimed there:
  `examples/proxy/README.md` now says `pip install "jevtools[serve] @ git+https://github.com/umatter/jevtools"` (or
  `uv pip install -e ".[serve]"` in a checkout). `tests/docs` fails on any bare-name `pip install`/`uv add` of
  jevtools in a Markdown file. The bare-name hints in runtime errors (`cli.py`, `serve/app.py`, the adapters…) stay:
  they fire only when jevtools is installed, and pip then resolves the extra against the installed distribution.
- The README headline no longer calls the composed confidence "calibrated": C is W, Π or min(L, J) by tier (§3.7.3)
  and becomes a calibrated probability only after `jevtools tune --calibrate` fits a calibrator; until then every
  Decision reports `confidence.calibrated == False`. "Jev returns calibrated probabilities" (per question) stays.
- The README CLI row of `jevtools probe` gives the counts recorded under "Conformance probe": 19 requests, 16 with
  `--no-smoke`, more when a rejected size is halved, plus `GET /v1/models` on TypeSafe, up to 400 questions and
  8,000-character fields. SPEC §8.7 still says "about 12 calls"; the deviation stays recorded here.
- README "MCP" and "Pydantic AI" name the minimum versions (mcp ≥ 1.19, pydantic-ai-slim ≥ 1.0.13).

### Merge of the review fixes

- **Golden fixtures regenerated without a spec version bump.** `jevtools fixtures --update` rewrote 8 files, all
  traces: `R2`, `R2-no-history`, `R2-click` (`trace.json`, `trace_1.json`), `R6` (`trace.json`, `trace_2.json`),
  `R6-step2` and `422-isolation`. A structural diff against the previous files shows exactly one change per file,
  `bindings.body.normalizer`: `"text@1"` → `"text.template@1"` (resolvers #24: the author-template body was always
  normalized by `normalize_title`, so the old record named the wrong normalizer). No `ballot.json`, request,
  response or `decision.json` changed. This deviates from §10.2 ("regenerated only … together with a spec version
  bump"): the fix corrects what the trace records about an unchanged computation, so `SPEC_VERSION` stays
  `jevtools/0.1`. A port that reproduces the old string should switch to `text.template@1`.
- Examples 01–08 were run with `--backend scripted` and `--backend sim` before and after the review fixes; the
  printed output is identical (trace ids aside). Under `sim`, example 06 step 1 reads
  `finance/invoices/outgoing/2026-09-18_INV-0412_to_ACME.pdf`, which the fake workspace does not hold, and gets an
  error observation; this predates the review and is a simulator limitation (a lexical double), not a regression.
  The proxy example ran under `jevtools serve` on the simulator with the real OpenAI SDK client.
- Still open after the review, collected from the entries above:
  - `adapters/pending.py` could forget a handle when resuming it raises (engine #12 backstop);
  - sources: `toolsource._await_now`/`_wrap`, `sources.mcp._run`/`_wrap` and `Provider.candidates` still carry
    their own sync bridge (should use `_compat.run_sync`), and `toolsource.py` checks `isError` inline;
  - small cleanups: a public `policy.clears`; `tie` in `policy.CONSISTENCY_FLAGS`; `Mentions.failures` as a trace
    note; a public `context.utc_offset`; the remaining truncation and join helper copies (`loop._truncate`,
    `prompts._short`, `templates.first_sentence`, `probe._short`, `templates.quote_list`); the unused
    `Pool.by_label`; `demo/scripts.py`'s own `R6_REQUEST`; the optional `accepts_keyword` guard in
    `adapters.mcp.call_decision`; README (Tuning) and SPEC §11.1 on source paths inside a context file;
  - `jt.verify` does not re-run normalizers.

### Closing the trust follow-ups

- §3.4.2 unknown history origins: a `history` candidate whose origin is unknown (`None`, or `history` itself: an
  untraced assistant-turn or entity-store value) inherits the trust of `tool_output`
  (`candidates.history_origin_of`). `Candidate.effective_channel` and `admits()` both use it, so such a value never
  reaches a slot that bars tool output (external identity slots: "history, trusted origin only"), and where it is
  admitted (content) it counts as untrusted. `kinds.common.trace_history` seeds its untrusted set from *known*
  untrusted provenance only, so untraced copies are still traced (user, registry) before this default applies.
- §3.5.7 option channels: `BallotOption.from_candidate` records the candidate's effective channel, the trust the value
  carries, and keeps the channel it arrived through in `prov["via"]` (only when they differ). The policy's
  `tool_output`/`generated` caps and the I2 check therefore see a repeated tool-output value for what it is. No
  golden case has such a candidate, so no fixture changed.
- §14 secrets in templates: `kinds/text.py::_profile_inputs` excludes `plan.secret_user_fields`, like the state does.
- §3.8.5 resume scope: `Router.resume(..., context=...)` raises `PendingScopeError` when the given context's
  requester identity (`Context.requester_sha256`: user profile, `shareable`, `include_system`, source names; not the
  clock, messages, time zone or source contents, so the TOCTOU re-check after a registry change still resumes)
  differs from the live decision's. Without a context, a live handle now resumes in the context of the decision that
  raised it rather than the router default.
- §7.2.4 proxy: `POST /v1/chat/completions` requires `Content-Type: application/json` (415 `unsupported_media_type`
  otherwise), so a cross-origin "simple" browser POST cannot trigger Jev calls on a local proxy.
- Regression tests: `tests/unit/test_trust_followups.py`, `tests/serve/test_app.py`.

## BFCL benchmark

- Scope: the single-turn Python categories of BFCL v4. `parallel*` categories can be run (`--categories all`) but
  score 0 by construction (one call per turn; `ext.parallel` is not implemented). Java/JavaScript (BFCL converts
  their string-typed values with language converters), multi-turn, memory and web-search categories are out of
  scope.
- Scoring ports BFCL's `ast_checker` for Python (string standardization, `int` for `float`, one level of nested
  type checks, dict and list-of-dict values; irrelevance = no call, relevance = any call). Two views are reported:
  *strict* (only `execute` emits a call) and *proposal* (the proposed call counts unless the outcome is
  abstain/refuse), because BFCL has no notion of confirm or clarify.
- The oracle (`bench/oracle.py`) answers from the round-1 Ballot and pools compiled with the router's own inputs.
  Questions it cannot map (follow-up rounds) get an uncertain answer and are listed in `OracleBackend.unknown`.
  Choice: the first option BFCL would accept, else `NOT_STATED` when omitting (or the default) is acceptable, else
  `NONE_OF_THESE`. Nouls: accept/item/member by acceptability, flags by the expected boolean, `authorized`/
  `present` yes, `more`/`done_after` no. For `live_relevance` (any call is right) it elects the first real tool.
- Every non-oracle run also decides each case with the oracle first, to report the case's ceiling and coverage next
  to the live result ("within ceiling" isolates the model's judgment from extractor coverage).
- BFCL tools get `x-jev.risk = "read"` by default (`--risk infer` keeps the inferred tier): they are side-effect
  free, and their names mostly fall through the verb table to the fail-safe external tier.
- The clock is fixed at 2026-09-25 10:00 UTC; BFCL answers with relative dates assume other dates and are misses
  either way.
- Found and fixed while building it (with regression tests): a fractional amount crashed an integer money slot
  (`kinds/money.py` now drops values the schema cannot hold at pool time); a text candidate over `accept_max` made
  the pre-send validator reject the ballot (`kinds/text.py` no longer nominates it); list items were emitted in
  canonical label order (the neutral option order on the ballot) instead of mention order (§4.3), now fixed in
  `kinds/listing.py`.
- Known inference issue surfaced by the bench (not changed): an integer described with "total" is inferred as
  money (§3.3.1 row 6), e.g. BFCL's `panelArea` ("The total solar panel area"). The pool still holds the stated
  numbers, so the effect is limited to money-style normalization.


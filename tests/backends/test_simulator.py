"""``LexicalSimulator`` (spec §8.6, §10.4): text features, Choice/Noul/Score rules, determinism, flip mode, and the
§8.6 guarantee list on crafted inputs through the real Router and the §13 scenario fixtures.

The simulator is a lexical test double: these tests pin plumbing and policy branches, never accuracy.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from jevtools.backends import Backend
from jevtools.backends.simulator import (
    DEFAULT_SYNONYMS,
    LexicalSimulator,
    cov,
    expand,
    families,
    toks,
)
from jevtools.ballot import flatten_instructions
from jevtools.canonical import canonical_json
from jevtools.context import Observation
from jevtools.policy import Outcome
from jevtools.router import Router
from jevtools.wire import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from tests.scenario.fixtures import (
    SCENARIO_CONTACTS,
    accounts,
    contacts,
    files,
    scenario_catalog,
    scenario_context,
    scenario_messages,
)

TOOLS: dict[str, Any] = {
    "create_event": "Create a calendar event and invite attendees.",
    "get_weather": "Get the current weather for a city.",
    "send_email": "Send an email from the user to one recipient.",
    "NO_TOOL": "No action is needed: conversation, small talk, a joke, or something the assistant can answer.",
    "UNSUPPORTED": "The user wants an action that none of the listed actions can perform.",
}
AUTH = NoulQuestion(
    instructions="Is the user asking the assistant to actually send an email now?",
    criteria=NoulCriteria(true="Yes.", false="No."),
)


def request(text: str, questions: dict[str, Question], **state: Any) -> DecisionRequest:
    return DecisionRequest(model="m", state={"request": text, "history": [], **state}, questions=questions)


def answer(text: str, qid: str, question: Question, sim: LexicalSimulator | None = None, **state: Any) -> Any:
    response = (sim or LexicalSimulator()).decide(request(text, {qid: question}, **state))
    return response.answers[qid]


def probs(a: Any) -> dict[str, float]:
    assert isinstance(a, ChoiceAnswer)
    return a.probabilities


def softmax(scores: list[float], t: float = 0.1) -> list[float]:
    exps = [math.exp(s / t) for s in scores]
    return [e / sum(exps) for e in exps]


# --------------------------------------------------------------------------------------------------------------------
# text features
# --------------------------------------------------------------------------------------------------------------------


def test_toks_folds_drops_stopwords_and_plural_s() -> None:
    assert toks("What's the weather like in Zürich, in Fahrenheit?") == ["weather", "zurich", "fahrenheit"]
    assert toks("I'll be 10 minutes late") == ["10", "minute", "late"]
    assert toks("yes gas bus") == ["yes", "gas", "bus"]  # length ≤ 3 keeps the s
    assert toks({"a": ["Savings accounts", {"b": "CH93…2957"}], "n": 3}) == ["saving", "account", "ch93", "2957"]


def test_cov_exact_prefix_and_empty() -> None:
    assert cov([], ["a"]) == 0.0
    assert cov(["anna", "keller"], ["anna"]) == 0.5
    assert cov(["annabel"], ["anna"]) == pytest.approx(0.8)  # shared 4-character prefix
    assert cov(["rob"], ["robert"]) == 0.0  # a 3-character token never prefix-matches
    assert cov(["x", "x", "y"], ["x"]) == 0.5  # distinct tokens


def test_expand_is_symmetric_within_a_family() -> None:
    fams = families(DEFAULT_SYNONYMS)
    assert expand(["create", "event"], fams) >= {"book", "sync", "calendar", "meeting", "schedule"}
    assert expand(["read", "file"], fams) >= {"open", "config"}
    assert expand(["get", "weather"], fams) == {"get", "weather", "temperature", "forecast"}


# --------------------------------------------------------------------------------------------------------------------
# Choice
# --------------------------------------------------------------------------------------------------------------------


def test_real_options_score_by_label_and_description_coverage() -> None:
    question = ChoiceQuestion(
        instructions="Which option is the temperature unit?",
        criteria={"celsius": None, "fahrenheit": None, "NOT_STATED": "No.", "NONE_OF_THESE": "Other."},
    )
    got = probs(answer("Weather in Zurich in Fahrenheit", "get_weather.unit", question))
    # fahrenheit: 0.75·1 + 0.25·cov(∅) = 0.75; celsius 0; NOT_STATED 0.45·(1 − 0.75); NONE_OF_THESE 0.12
    expected = softmax([0.0, 0.75, 0.45 * 0.25, 0.12])
    assert got == {k: round(v, 4) for k, v in zip(question.criteria, expected, strict=True)}
    assert all(len(repr(p).split(".")[-1]) <= 4 for p in got.values())


def test_tool_options_score_the_lead_verb_and_own_tokens() -> None:
    question = ChoiceQuestion(instructions="Which ONE action?", criteria=dict(TOOLS))
    got = answer("Book a sync", "tool", question)
    assert isinstance(got, ChoiceAnswer) and got.choice == "create_event"  # "book" leads: the event family
    # create_event: 0.6·verb(book) + 0.4·cov({create, event}, expand{book, sync}) = 0.6 + 0.4·0.5 = 0.8;
    # get_weather, send_email 0; NO_TOOL 0.2·(1 − 0.8) = 0.04; UNSUPPORTED 0.05 (a deviation from §8.6)
    assert got.probabilities["create_event"] == round(softmax([0.8, 0, 0, 0.04, 0.05])[0], 4)
    long = answer("Email Anna that I'll be 10 minutes late for the quarterly review with ACME", "tool", question)
    assert isinstance(long, ChoiceAnswer) and long.choice == "send_email" and long.probabilities["send_email"] > 0.99


def test_lead_verb_skips_generic_words_and_steps_already_done() -> None:
    tools = {"read_file": "Open a file in the user's workspace.", "search_web": "Search the public web.",
             "send_email": "Send an email from the user to one recipient.", **{k: TOOLS[k] for k in
                                                                               ("NO_TOOL", "UNSUPPORTED")}}  # fmt: skip
    question = ChoiceQuestion(criteria=tools)
    text = "Find the latest invoice from ACME and forward it to finance"
    step1 = answer(text, "tool", question)
    assert isinstance(step1, ChoiceAnswer) and step1.choice == "read_file"  # "invoice" (a file) leads, not "find"
    progress = ['Step 1: read_file(path="finance/invoices/acme/2026-09-15_ACME_INV-2291.pdf") → ok, 40 words']
    step2 = answer(text, "tool", question, progress=progress)
    assert isinstance(step2, ChoiceAnswer) and step2.choice == "send_email"  # the read is done: "forward" leads
    recipe = answer("Find a good pasta recipe", "tool", question)
    assert isinstance(recipe, ChoiceAnswer) and recipe.choice == "search_web"  # a generic word leads when alone


def test_no_tool_for_chit_chat() -> None:
    got = answer("Tell me a joke", "tool", ChoiceQuestion(criteria=dict(TOOLS)))
    assert isinstance(got, ChoiceAnswer) and got.choice == "NO_TOOL" and got.probabilities["NO_TOOL"] > 0.9
    hello = answer("hello, how are you?", "tool", ChoiceQuestion(criteria=dict(TOOLS)))
    assert isinstance(hello, ChoiceAnswer) and hello.choice == "NO_TOOL"


def test_done_when_the_last_progress_tool_matches_the_last_action_verb() -> None:
    criteria = {**TOOLS, "DONE": "The steps in `progress` already complete everything `request` asks for."}
    step = 'Step 1: send_email(to="finance@muster.ch") → ok'
    done = answer("Email the invoice to finance", "tool", ChoiceQuestion(criteria=criteria), progress=[step])
    assert probs(done)["DONE"] > probs(done)["UNSUPPORTED"]
    other = answer("Book a sync with finance", "tool", ChoiceQuestion(criteria=criteria), progress=[step])
    assert probs(other)["DONE"] == probs(other)["UNSUPPORTED"]  # 0.05 each


def test_probe_not_stated_depends_on_unknown_capitalized_words() -> None:
    probe = ChoiceQuestion(
        instructions="Does the user indicate the city name, in `request` or `history`?",
        criteria={"NOT_STATED": "No; the default would be used.", "NONE_OF_THESE": "Yes, the user indicates one."},
    )
    known = probs(answer("Is it cold in Zurich on Tuesday? I wonder.", "get_weather.city", probe))
    unknown = probs(answer("Is it cold in Atlantis today?", "get_weather.city", probe))
    assert known["NOT_STATED"] == round(softmax([0.6, 0.12])[0], 4)
    assert unknown["NOT_STATED"] == round(softmax([0.2, 0.12])[0], 4)
    assert unknown["NONE_OF_THESE"] >= 0.30  # the coverage probe flags an uncaught mention


def test_none_of_these_zeta_for_an_uncovered_quoted_mention() -> None:
    def mention(name: str) -> ChoiceQuestion:
        return ChoiceQuestion(
            instructions=f'Suppose the assistant will create an event. The user mentions "{name}". Which option is '
            "that person?",
            criteria={"Bob Meier <bob.meier@muster.ch>": "Contact.", "EXCLUDE": "Not one.", "NONE_OF_THESE": "No."},
        )

    covered = probs(answer("Invite Bob", "create_event.attendees.m0", mention("Bob")))
    uncovered = probs(answer("Invite Bobby", "create_event.attendees.m0", mention("Bobby")))
    assert covered["NONE_OF_THESE"] < 0.5 < uncovered["NONE_OF_THESE"]
    ref = ChoiceQuestion(
        instructions="Which option is the recipient's email address?",
        criteria={"Robert Brown <rbrown@partner.io>": 'Contact similar to "Rob": Partner Inc.', "NOT_STATED": "x",
                  "NONE_OF_THESE": "y"},
    )  # fmt: skip
    got = answer("Email Rob that I'm late", "send_email.to", ref)
    assert isinstance(got, ChoiceAnswer) and got.choice == "NONE_OF_THESE"  # ζ from the ref description's mention


def test_exclude_after_a_negation_cue() -> None:
    question = ChoiceQuestion(
        instructions='Suppose … The user mentions "Carol". Which option is that person?',
        criteria={"Carol Liu <carol.liu@muster.ch>": None, "EXCLUDE": "Not one.", "NONE_OF_THESE": "No."},
    )
    negated = probs(answer("Book a sync with Bob but not Carol", "create_event.attendees.m1", question))
    plain = probs(answer("Book a sync with Bob and Carol", "create_event.attendees.m1", question))
    assert negated["EXCLUDE"] > plain["EXCLUDE"]
    assert plain["EXCLUDE"] == round(softmax([0.75 * 0.25, 0.0, 0.12])[1], 4)  # EXCLUDE scores 0 without a cue


def test_reply_sentinels() -> None:
    reply = ChoiceQuestion(
        instructions="Which option did the user choose?",
        criteria={"option_1": "Send to Anna Keller", "option_2": "Send to Anna Rossi", "OTHER": "o", "CANCEL": "c"},
    )
    cancelled = probs(answer("cancel that", "reply", reply))
    assert max(cancelled, key=lambda k: cancelled[k]) == "CANCEL"


# --------------------------------------------------------------------------------------------------------------------
# Noul
# --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Email Anna that I'm late", 0.95),
        ("Please email Anna that I'm late", 0.95),
        ("Could you send Anna a note that I'm late", 0.95),
        ("How do I email Anna?", 0.15),
        ("Don't email Anna yet", 0.15),
        ("Draft an email to Anna", 0.15),
        ("Tell Anna I'm late", 0.6),
    ],
)
def test_authorized(text: str, expected: float) -> None:
    got = answer(text, "send_email.authorized", AUTH)
    assert isinstance(got, NoulAnswer) and got.noul == expected


def test_present_follows_the_sibling_slot() -> None:
    to = ChoiceQuestion(criteria={"Anna Keller <anna.keller@acme.com>": None, "NOT_STATED": "a", "NONE_OF_THESE": "b"})
    present = NoulQuestion(instructions="Does the user say or clearly imply the recipient?")
    strong = LexicalSimulator().decide(request("Email Anna Keller at ACME", {"s.to": to, "s.to.present": present}))
    weak = LexicalSimulator().decide(request("Email Anna", {"s.to": to, "s.to.present": present}))
    assert strong.answers["s.to.present"] == NoulAnswer(noul=0.9)
    assert weak.answers["s.to.present"] == NoulAnswer(noul=0.2)
    alone = answer("Email Anna", "s.to.present", present)
    assert isinstance(alone, NoulAnswer) and alone.noul == 0.5  # no sibling in the request


def content_accept(candidate: str) -> NoulQuestion:
    return NoulQuestion(
        instructions={"question": "Would the body below be acceptable exactly as written?", "candidate": candidate},
        criteria=NoulCriteria(true="Acceptable exactly as written.", false="Wrong."),
    )


def test_accept_content_and_cosmetic() -> None:
    text = "Email Anna that I'll be 10 minutes late"
    body = "Hi ⟨recipient's first name⟩,\n\nI'll be 10 minutes late.\n\nBest,\nSam"
    assert answer(text, "s.body.accept.0", content_accept(body)) == NoulAnswer(noul=0.9)
    assert answer(text, "s.body.accept.1", content_accept("She will be 10 minutes late.")) == NoulAnswer(noul=0.2)
    unrelated = content_accept("The quarterly figures are attached.")
    assert answer(text, "s.body.accept.2", unrelated) == NoulAnswer(noul=0.2)
    cosmetic = NoulQuestion(instructions={"question": "Would the subject line below be a sensible choice?",
                                          "candidate": "Running 10 minutes late"})  # fmt: skip
    assert answer(text, "s.subject.accept.0", cosmetic) == NoulAnswer(noul=0.85)
    off = NoulQuestion(instructions={"question": "Would the subject line below be a sensible choice?",
                                     "candidate": "Quarterly review agenda"})  # fmt: skip
    assert answer(text, "s.subject.accept.1", off) == NoulAnswer(noul=0.3)


def test_accept_reads_flattened_instructions() -> None:
    body = "I'll be 10 minutes late."
    question = content_accept(body)
    flat = question.model_copy(update={"instructions": flatten_instructions(question.instructions)})
    assert isinstance(flat.instructions, str) and "\nCandidate: " in flat.instructions
    assert answer("Email Anna that I'll be 10 minutes late", "s.body.accept.0", flat) == NoulAnswer(noul=0.9)


def test_more_item_member_done_after_and_others() -> None:
    more = NoulQuestion(instructions='Apart from "Bob", does the user ask to include anyone else?')
    assert answer("Invite Bob and the whole team", "e.attendees.more", more) == NoulAnswer(noul=0.85)
    assert answer("Invite Bob", "e.attendees.more", more) == NoulAnswer(noul=0.05)
    member = NoulQuestion(instructions={
        "question": "Does the item below match what the user is looking for? Ignore the word 'latest': the app "
                    "picks the latest one among the matching items.",
        "item": "finance/invoices/acme/2026-09-15_ACME_INV-2291.pdf — date 2026-09-15"})  # fmt: skip
    text = "Find the latest invoice from ACME and forward it to finance"
    assert answer(text, "read_file.path.member.0", member) == NoulAnswer(noul=0.9)
    assert answer("Find the latest quote from Globex", "read_file.path.member.0", member) == NoulAnswer(noul=0.1)
    item = NoulQuestion(instructions="Suppose … Should Carol Liu <carol.liu@muster.ch> be included in the invitees?")
    assert answer("Invite Carol Liu", "create_event.attendees.item.0", item) == NoulAnswer(noul=0.9)
    done_after = NoulQuestion(instructions="Would everything `request` asks for then be done?")
    assert answer(text, "send_email.done_after", done_after) == NoulAnswer(noul=0.9)  # "forward" → send
    assert answer(text, "search_web.done_after", done_after) == NoulAnswer(noul=0.1)
    assert answer(text, "read_file.done_after", done_after) == NoulAnswer(noul=0.1)
    assert answer(text, "read_file.something_else", NoulQuestion()) == NoulAnswer(noul=0.5)


# --------------------------------------------------------------------------------------------------------------------
# Score, determinism, flip mode, protocol
# --------------------------------------------------------------------------------------------------------------------


def test_score_is_a_softmax_over_level_coverage() -> None:
    question = ScoreQuestion(instructions="How urgent?", criteria=["low", "urgent", "extremely urgent"])
    got = answer("This is urgent", "t.priority", question)
    assert isinstance(got, ScoreAnswer)
    expected = softmax([0.0, 1.0, 0.5])
    assert got.probabilities == {i: round(p, 4) for i, p in enumerate(expected)}
    assert got.score == round(sum(i * p for i, p in enumerate(expected)), 4) and got.legend[1] == "urgent"


def test_same_request_same_bytes_and_identity() -> None:
    req = request("Email Anna that I'll be late", {"tool": ChoiceQuestion(criteria=dict(TOOLS)),
                                                    "send_email.authorized": AUTH})  # fmt: skip
    a, b = LexicalSimulator(), LexicalSimulator()
    first = canonical_json(a.decide(req).model_dump(mode="json"))
    assert first == canonical_json(b.decide(req).model_dump(mode="json"))
    assert first == canonical_json(a.decide(req).model_dump(mode="json"))
    assert (a.name, a.model) == ("simulator", "lexical-simulator") and isinstance(a, Backend)
    assert a.requests == [req, req]


async def test_adecide_matches_decide() -> None:
    req = request("Tell me a joke", {"tool": ChoiceQuestion(criteria=dict(TOOLS))})
    assert await LexicalSimulator().adecide(req) == LexicalSimulator().decide(req)


def flip_request() -> DecisionRequest:
    questions: dict[str, Question] = {
        f"t.slot{i}": ChoiceQuestion(criteria={"alpha": None, "beta": None, "NONE_OF_THESE": "n"}) for i in range(24)
    }
    questions.update({f"t.flag{i}.item.0": NoulQuestion(instructions="Should x be included in y?") for i in range(8)})
    return request("nothing relevant here", questions)


def test_flip_mode_is_reproducible_by_seed() -> None:
    req = flip_request()
    base = LexicalSimulator().decide(req)
    # NONE_OF_THESE (0.12) leads alpha and beta (0) by less than the band; the Nouls answer 0.1
    assert {a.choice for a in base.answers.values() if isinstance(a, ChoiceAnswer)} == {"NONE_OF_THESE"}
    flipped = LexicalSimulator(seed=7, flip_band=0.5).decide(req)
    assert flipped == LexicalSimulator(seed=7, flip_band=0.5).decide(req)  # reproducible by seed
    assert flipped != LexicalSimulator(seed=8, flip_band=0.5).decide(req)
    tops = [a.choice for a in flipped.answers.values() if isinstance(a, ChoiceAnswer)]
    assert set(tops) == {"NONE_OF_THESE", "alpha"}  # the top two swapped in some Choices, not in others
    for qid, a in flipped.answers.items():
        if isinstance(a, ChoiceAnswer) and a.choice == "alpha":
            assert a.probabilities["alpha"] == probs(base.answers[qid])["NONE_OF_THESE"]
    nouls = {a.noul for a in flipped.answers.values() if isinstance(a, NoulAnswer)}
    assert nouls == {0.1, 0.9}  # within the band of 0.5, some Nouls are reflected


def test_flip_mode_swaps_close_top_two() -> None:
    question = ChoiceQuestion(criteria={"Anna Keller": None, "Anna Rossi": None, "NONE_OF_THESE": "n"})
    req = request("Email Anna Keller", {f"s.to{i}": question for i in range(16)})
    base = LexicalSimulator().decide(req)
    assert {a.choice for a in base.answers.values()} == {"Anna Keller"}  # type: ignore[union-attr]
    wide = LexicalSimulator(seed=3, flip_band=0.99).decide(req)
    swapped = [a for a in wide.answers.values() if a.choice == "Anna Rossi"]  # type: ignore[union-attr]
    assert 0 < len(swapped) < 16
    narrow = LexicalSimulator(seed=3, flip_band=0.01).decide(req)
    assert narrow == base  # margins beyond the band never flip


def test_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError):
        LexicalSimulator(temperature=0)


# --------------------------------------------------------------------------------------------------------------------
# §8.6 guarantee list: crafted inputs through the real Router and the §13 fixtures
# --------------------------------------------------------------------------------------------------------------------


def sim_decide(text: str, *, context: Any = None, mode: str = "turn", history: bool = False) -> tuple[Router, Any]:
    ctx = context or scenario_context()
    router = Router(scenario_catalog(list(ctx.sources.values())), backend=LexicalSimulator(), context=ctx)
    return router, router.decide(scenario_messages(text, history=history), mode=mode)  # type: ignore[arg-type]


def test_guarantee_two_annas_clarify() -> None:
    _, d = sim_decide("Email Anna that I'll be late")  # Anna Keller, Anna Rossi (and Annabel Frey) match
    assert d.outcome is Outcome.CLARIFY and d.call is not None and d.call.name == "send_email"
    assert d.bottleneck is not None and d.bottleneck.slot == "to" and d.prompt is not None
    assert d.prompt.text == "Which recipient did you mean?"
    menu = " ".join(o.text for o in d.prompt.options)
    assert "Anna Keller <anna.keller@acme.com>" in menu and "Anna Rossi <anna.rossi@gmail.com>" in menu
    two = [row for row in SCENARIO_CONTACTS if row["name"] in ("Anna Keller", "Anna Rossi")]
    ctx = scenario_context(sources=[contacts(two), accounts(), files()])
    _, only_two = sim_decide("Email Anna that I'll be late", context=ctx)
    assert only_two.outcome is Outcome.CLARIFY and only_two.bottleneck is not None
    assert only_two.bottleneck.slot == "to"  # neither Anna is covered by the words: ask for the recipient


def test_guarantee_unknown_mention_widens() -> None:
    router, d = sim_decide("Email Rob that I'll be late")
    first = router.backend.requests[0]  # type: ignore[attr-defined]
    assert "Rob" in str(first.questions["send_email.to"].criteria)  # the ref pool quotes the mention
    to = LexicalSimulator().decide(first).answers["send_email.to"]
    assert isinstance(to, ChoiceAnswer) and to.choice == "NONE_OF_THESE"
    modes = [r.mode for r in d.trace.rounds]
    assert modes[0] == "turn" and "widen" in modes  # a coverage round was run
    widen = next(r for r in d.trace.rounds if r.mode == "widen")
    assert any(".bucket." in q["qid"] for q in widen.ballot["questions"])
    assert d.outcome is not Outcome.EXECUTE


def test_guarantee_joke_abstains() -> None:
    _, d = sim_decide("Tell me a joke")
    assert (d.outcome, d.rule) == (Outcome.ABSTAIN, "P1.tool.no_tool") and d.call is None


def test_guarantee_hedge_is_not_authorized() -> None:
    _, d = sim_decide("How do I email Anna that I'll be 10 minutes late?")
    assert (d.outcome, d.rule) == (Outcome.ABSTAIN, "P4.safety.not_authorized") and d.tool_calls == []
    assert d.gates["authorized"] == 0.15


def test_guarantee_invoice_only_amount_refuses() -> None:
    invoice = Observation(
        step=1, tool="read_file", arguments={"path": "finance/invoices/acme/2026-09-15_ACME_INV-2291.pdf"},
        content="ACME AG — Invoice INV-2291. Total CHF 4,820.00. Payable within 30 days.",
    )  # fmt: skip
    ctx = scenario_context("Pay the ACME invoice", observations=[invoice])
    _, d = sim_decide("Pay the ACME invoice", context=ctx, mode="loop")
    assert (d.outcome, d.rule) == (Outcome.REFUSE, "P3.safety.refuse") and d.tool_calls == []

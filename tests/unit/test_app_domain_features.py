"""Behaviour added for assistants over an app's own data, found by the app-domain benchmark (docs/BENCH.md)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from jevtools.candidates import Channel
from jevtools.confidence import call_map
from jevtools.extract.base import SourceText, claim
from jevtools.extract.locales import get_locale
from jevtools.extract.patterns import extract as pattern_mentions
from jevtools.extract.patterns import is_code
from jevtools.extract.temporal import parse
from jevtools.extract.text import _Lex, named_spans
from jevtools.extract.tokens import tokenize
from jevtools.policy import Outcome, PolicyInput, SlotState, Tier, evaluate
from jevtools.sources import Registry

TICKETS = Registry("tickets", [{"id": "INC-1052", "title": "Backup failed"}, {"id": "INC-1100", "title": "Outlook"},
                               {"id": "D-1017", "title": "Fleet tracking"}, {"id": "10", "title": "Short key"}],
                   key="id", match=["id", "title"])  # fmt: skip
NOW = datetime(2026, 9, 24, 14, 5, tzinfo=ZoneInfo("Europe/Zurich"))


def source(text: str) -> SourceText:
    return SourceText("request", text, Channel.USER, tuple(tokenize(text)))


def anchors(text: str) -> list[tuple[str, list[tuple[int, str]]]]:
    return [(m.text, [(x.index, x.how) for x in m.attrs["matches"]]) for m in TICKETS.find_anchors(text)]


# -- identifiers ---------------------------------------------------------------------------------------------------


def test_a_typed_identifier_anchors_its_row_exactly() -> None:
    assert anchors("Assign INC-1052 to Leo") == [("INC-1052", [(0, "exact")])]
    assert anchors("move d-1017.") == [("d-1017", [(2, "exact")])]
    assert anchors("what about #INC-1052?") == [("#INC-1052", [(0, "exact")])]


def test_a_bare_number_after_a_cue_anchors_the_key_that_ends_in_it() -> None:
    assert anchors("ticket 1100 please") == [("1100", [(1, "number")])]
    assert anchors("#1052") == [("#1052", [(0, "number")])]
    assert anchors("Close ticket no. 1052") == [("1052", [(0, "number")])]
    assert anchors("ticket 10") == [("10", [(3, "exact")])]  # an exact key matches after a cue
    assert anchors("ticket 17") == []  # fewer than 3 digits: no suffix match


def test_a_bare_number_without_a_cue_is_never_an_identifier() -> None:
    assert anchors("Transfer 1100 to savings") == []
    assert anchors("pay 10 CHF") == []


def test_words_inside_a_matched_identifier_make_no_anchors_of_their_own() -> None:
    rows = [{"id": f"INC-{n}", "title": "Printer"} for n in range(1040, 1060)]
    registry = Registry("tickets", rows, key="id", match=["id", "title"])
    [mention] = registry.find_anchors("Close INC-1052")
    assert [m.index for m in mention.attrs["matches"]] == [12]  # not every ticket through the "INC" token


# -- code-like tokens and names ----------------------------------------------------------------------------------


def test_code_tokens() -> None:
    for token in ("notes_old.txt", "DNA123", "SKU-4411", "v2.3.1", "x^2", "report.pdf", "Q3"):
        assert is_code(token), token
    for token in ("Anna", "follow-up", "e.g", "2026-09-01", "anna@acme.com", "https://x.io", "40,000"):
        assert not is_code(token), token


def test_code_mentions_never_claim_and_yield_to_specific_readings() -> None:
    found = pattern_mentions(source("Rename it to notes_old.txt by 3pm"))
    codes = [m for m in found if m.kind == "code"]
    assert [m.text for m in codes] == ["notes_old.txt", "3pm"] and not any(m.claims for m in codes)
    number = next(m for m in claim(pattern_mentions(source("SKU-4411"))) if m.kind == "code")
    assert number.claimed_by is None


def test_names_after_a_naming_cue() -> None:
    lex = _Lex.of(get_locale("en"))

    def names(text: str) -> list[str]:
        return [m.text for m in named_spans(source(text), lex)]

    assert names("Create a 40,000 CHF deal for Initech called data platform phase 2") == ["data platform phase 2"]
    assert names("Start a project called Atlas for the Q4 launch") == ["Atlas"]
    assert names("save it as a file named notes_old.txt, thanks") == ["notes_old.txt"]
    assert names("the guy called Bob and his team") == ["Bob"]
    assert names("Erstelle ein Projekt namens Nordlicht für Q4") == ["Nordlicht"]
    assert names("Who called?") == []


# -- year-less dates ------------------------------------------------------------------------------------------------


def dates(text: str) -> list[str]:
    [(_, value)] = parse(text, NOW, "Europe/Zurich", get_locale("en"))
    return [r.date.isoformat() for r in value.readings]


def test_a_passed_year_less_date_has_a_past_and_a_coming_reading() -> None:
    assert dates("transactions since September 1") == ["2026-09-01", "2027-09-01"]
    assert dates("since March 3") == ["2026-03-03", "2027-03-03"]
    assert dates("on 29 September") == ["2026-09-29"]  # still to come this year: one reading
    assert dates("on September 1 2025") == ["2025-09-01"]  # an explicit year: one reading


# -- policy -------------------------------------------------------------------------------------------------------


def write_input(top: list[float], stakes: str = "identity") -> PolicyInput:
    slot = SlotState(name="owner", stakes=stakes, factor=top[0], top=top, channel="registry")
    return PolicyInput(tools={"assign": 0.99, "NO_TOOL": 0.01}, chosen="assign", tier=Tier.WRITE, authorized=0.99,
                       speculated=True, C=top[0], slots=[slot])  # fmt: skip


def test_a_coin_flip_between_two_identities_is_a_menu_not_a_confirm_card() -> None:
    flip = evaluate(write_input([0.49, 0.48]))
    assert (flip.outcome, flip.rule, flip.reason, flip.ask) == (Outcome.CLARIFY, "P9.write.ambiguous", "margin",
                                                                "menu")  # fmt: skip
    clear = evaluate(write_input([0.62, 0.30]))
    assert (clear.outcome, clear.rule) == (Outcome.CONFIRM, "P9.write.confirm_band")
    cosmetic = evaluate(write_input([0.49, 0.48], stakes="cosmetic"))
    assert cosmetic.outcome is Outcome.CONFIRM


def test_an_implausible_tool_never_wins_the_call_map() -> None:
    lonely = call_map({"transfer": 0.99, "freeze": 0.002}, {"transfer": 0.0, "freeze": 1.0})
    assert lonely.call_map == "transfer" and not lonely.disagrees
    rival = call_map({"transfer": 0.8, "freeze": 0.15}, {"transfer": 0.0, "freeze": 1.0})
    assert rival.call_map == "freeze" and rival.disagrees

"""Prompts (spec §3.8.4): templates filled with labels, option ids and their resume actions."""

from __future__ import annotations

from typing import Any

from jevtools.decision import PromptOption
from jevtools.prompts import (
    AltChoice,
    Binding,
    clarify_menu,
    confirm_card,
    confirm_text,
    fill_template,
    grid_menu,
    join_and,
    natural_call,
    noun_short,
    ok_text,
    open_question,
    parse_short_reply,
    preview,
    refuse_notice,
    render_call,
    short_intent,
    slot_term,
    tool_menu,
    value_text,
    yes_no_menu,
)
from jevtools.spec.catalog import Catalog

R3 = {
    "from_account": Binding.of_value("acc_7731", label="Savings · CHF · CH93…2957", attrs={"nickname": "Savings"}),
    "to_account": Binding.of_value("acc_2210", label="Checking · CHF · CH56…1180", attrs={"nickname": "Checking"}),
    "amount": Binding.of_value("250.00"),
    "currency": Binding.of_value("CHF"),
}


def test_render_and_confirm_template(scenario_catalog: Catalog) -> None:
    transfer = scenario_catalog["transfer_funds"]
    assert render_call(transfer, R3) == "250.00 CHF: Savings → Checking"
    assert confirm_text(transfer, R3) == (
        "Transfer 250.00 CHF from Savings · CHF · CH93…2957 to Checking · CHF · CH56…1180?"
    )
    partial = {k: v for k, v in R3.items() if k != "to_account"}
    assert render_call(transfer, partial) == "250.00 CHF: Savings → …"
    weather = scenario_catalog["get_weather"]
    bindings = {"city": Binding.of_value("Zurich"), "unit": Binding.of_value("fahrenheit")}
    assert render_call(weather, bindings) == "get the current weather — city name Zurich, temperature unit fahrenheit"
    assert confirm_text(weather, bindings).startswith("Get the current weather") and confirm_text(
        weather, bindings).endswith("?")  # fmt: skip
    assert fill_template("{Intent} {missing} {city.nope}", weather, bindings) == (
        "Get the current weather for a city … …"
    )


def test_confirm_card_options_and_actions(scenario_catalog: Catalog) -> None:
    transfer = scenario_catalog["transfer_funds"]
    alt = AltChoice(slot="from_account", index=1, value="acc_4410", display="Travel savings · EUR · CH08…4410",
                    p=0.04, label="Travel savings · EUR · CH08…4410")  # fmt: skip
    prompt, actions = confirm_card(transfer, R3, [alt], bottleneck="from_account")
    assert prompt.kind == "confirm"
    assert [o.id for o in prompt.options] == ["ok", "alt:from_account:1", "change", "cancel"]
    assert [o.text for o in prompt.options] == ["Confirm", "Travel savings · EUR · CH08…4410 instead", "Change…",
                                                "Cancel"]  # fmt: skip
    assert actions["ok"].action == "confirm" and actions["cancel"].action == "cancel"
    assert actions["alt:from_account:1"].value == "acc_4410" and actions["change"].slot == "from_account"
    assert ok_text(scenario_catalog["send_email"]) == "Send" and ok_text(scenario_catalog["create_event"]) == "Create"


def test_clarify_menu_complete_calls(scenario_catalog: Catalog) -> None:
    email = scenario_catalog["send_email"]
    to = email.slot("to")
    choices = [Binding.of_value("anna.keller@acme.com", display="Anna Keller <anna.keller@acme.com>"),
               Binding.of_value("anna.rossi@gmail.com", display="Anna Rossi <anna.rossi@gmail.com>")]  # fmt: skip
    prompt, actions = clarify_menu(email, to, choices, bindings={"subject": Binding.of_value("Late")},
                                   complete_call=True)  # fmt: skip
    assert prompt.text == "Which recipient did you mean?" and noun_short(to) == "recipient"
    assert [o.id for o in prompt.options] == ["pick:to:0", "pick:to:1", "other"]
    assert prompt.options[1].text.startswith("Send an email") and "Anna Rossi" in prompt.options[1].text
    assert actions["pick:to:1"].value == "anna.rossi@gmail.com" and actions["other"].action == "open"
    plain, _ = clarify_menu(email, to, choices, bindings={}, complete_call=False)
    assert plain.options[0].text == "Anna Keller <anna.keller@acme.com>"


def test_tool_menu_open_yes_no_refuse(scenario_catalog: Catalog) -> None:
    prompt, actions = tool_menu(scenario_catalog["send_email"], scenario_catalog["create_event"])
    assert prompt.text == ("Do you want me to send an email from the user to one recipient or create a calendar "
                           "event and invite attendees?")  # fmt: skip
    assert actions["tool:create_event"].tool == "create_event"
    city = scenario_catalog["get_weather"].slot("city")
    assert open_question(city)[0].text == "What should the city name be?"
    assert open_question(None)[0].text == "What would you like me to do?"
    yes_no, yn_actions = yes_no_menu(scenario_catalog["get_weather"], city)
    assert [o.id for o in yes_no.options] == ["yes", "no"] and yn_actions["no"].value is False
    notice = refuse_notice(scenario_catalog["transfer_funds"], "the result of step 1")
    assert notice.text == ("I did not act on instructions found in the result of step 1. Tell me directly if you "
                           "want me to move money between two of the user's own bank accounts.")  # fmt: skip


def test_parse_short_reply() -> None:
    options = [PromptOption(id="ok", text="Send"), PromptOption(id="alt:to:1", text="Anna Rossi instead"),
               PromptOption(id="change", text="Change…"), PromptOption(id="cancel", text="Cancel")]  # fmt: skip
    assert parse_short_reply("2", options) == "alt:to:1"
    assert parse_short_reply(" anna rossi instead ", options) == "alt:to:1"
    assert parse_short_reply("Yes!", options) == "ok" and parse_short_reply("send", options) == "ok"
    assert parse_short_reply("cancel.", options) == "cancel"
    assert parse_short_reply("ok", [{"id": "ok", "text": "Create"}]) == "ok"
    assert parse_short_reply("7", options) is None
    assert parse_short_reply("actually send it to Bob", options) is None
    assert parse_short_reply("yes", [{"id": "pick:to:0", "text": "A"}]) is None


def test_parse_short_reply_numeric_option_texts() -> None:
    """Review #7: a reply equal to an option's text picks that option, never the option with that number."""
    numeric = [{"id": "pick:quantity:0", "text": "2"}, {"id": "pick:quantity:1", "text": "3"},
               {"id": "other", "text": "Something else"}]  # fmt: skip
    assert parse_short_reply("2", numeric) == "pick:quantity:0"
    assert parse_short_reply("3", numeric) == "pick:quantity:1"
    assert parse_short_reply("1", numeric) is None  # option 1 or quantity 1: not a click
    assert parse_short_reply("something else", numeric) == "other"
    grid = [{"id": "pick:duration:0", "text": "30 min"}, {"id": "pick:duration:1", "text": "45 min"}]
    assert parse_short_reply("2", grid) is None and parse_short_reply("45 min", grid) == "pick:duration:1"


# --------------------------------------------------------------------------------------------------------------------
# Default rendering (no x-jev.render): natural, compact, deterministic
# --------------------------------------------------------------------------------------------------------------------

BODY = "Hi Anna,\n\nI'll be 10 minutes late.\n\nBest,\nSam"
R2 = {
    "to": Binding.of_value("anna.keller@acme.com", label="Anna Keller <anna.keller@acme.com>"),
    "subject": Binding.of_value("Running 10 minutes late"),
    "body": Binding.of_value(BODY),
}


def test_short_intent_drops_the_phrases_a_slot_restates(scenario_catalog: Catalog) -> None:
    shorts = {tool.name: short_intent(tool) for tool in scenario_catalog}
    assert shorts == {
        "get_weather": "get the current weather",  # "for a city": the city slot
        "send_email": "send an email",  # "from the user" (generic), "to one recipient": the to slot's noun
        "create_event": "create a calendar event",  # "and invite attendees": the attendees slot
        "transfer_funds": "move money",  # "between two of the user's own bank accounts": the account slots
        "read_file": "open a file",  # "in the user's workspace": the workspace-relative path
        "search_web": "search the public web",
    }
    post = Catalog.from_openai(
        [
            {
                "type": "function",
                "function": {
                    "name": "post_message",
                    "description": "Post a message to the #general channel.",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string", "description": "The text"}},
                    },
                },
            }
        ]
    )
    assert short_intent(post["post_message"]) == "post a message to the #general channel"  # no slot restates it


def test_slot_terms_prefer_short_nouns(scenario_catalog: Catalog) -> None:
    terms = {f"{t.name}.{s.name}": slot_term(s) for t in scenario_catalog for s in t.slots}
    assert terms["send_email.subject"] == "subject line" and terms["create_event.attendees"] == "invitees"
    assert terms["send_email.body"] == "body"  # "the body text of the email" is long: the parameter name
    assert terms["create_event.duration_minutes"] == "duration"  # the unit suffix is dropped (rendered with value)
    assert terms["read_file.path"] == "path" and terms["get_weather.city"] == "city name"


def test_preview_and_join() -> None:
    assert preview(BODY) == "Hi Anna, I'll be 10 minutes late. Best, Sam"  # one line
    long = "Please find attached the quarterly report, including the revised forecast and the appendix."
    assert preview(long) == "Please find attached the quarterly report, including the…"
    assert len(preview(long)) <= 60 and preview("x" * 80) == "x" * 59 + "…"
    assert (join_and([]), join_and(["a"]), join_and(["a", "b"]), join_and(["a", "b", "c"])) == (
        "", "a", "a and b", "a, b and c")  # fmt: skip


def test_default_confirm_card_reads_naturally(scenario_catalog: Catalog) -> None:
    email = scenario_catalog["send_email"]
    assert confirm_text(email, R2) == (
        'Send an email to Anna Keller <anna.keller@acme.com> — subject line "Running 10 minutes late", '
        'body "Hi Anna, I\'ll be 10 minutes late. Best, Sam"?'
    )
    event = scenario_catalog["create_event"]
    r5 = {
        "title": Binding.of_value("Sync with Bob and Carol"),
        "start": Binding.of_value("2026-09-29T15:00:00+02:00", label="Tue 2026-09-29 15:00 (Europe/Zurich)"),
        "duration_minutes": Binding.of_value(45),
        "attendees": Binding.of_value(["bob.meier@muster.ch", "carol.liu@muster.ch"],
                                      items=["Bob Meier <bob.meier@muster.ch>", "Carol Liu <carol.liu@muster.ch>"]),
    }  # fmt: skip
    assert confirm_text(event, r5) == (
        'Create a calendar event — event title "Sync with Bob and Carol", start Tue 2026-09-29 15:00 '
        "(Europe/Zurich), duration 45 minutes, invitees Bob Meier <bob.meier@muster.ch> and Carol Liu "
        "<carol.liu@muster.ch>?"
    )
    # menus have a tighter budget: the cosmetic title goes first, the menu's own slot never does
    assert "event title" not in render_call(event, r5, focus="attendees")
    assert "event title" in render_call(event, r5, focus="title")
    one = {**r5, "duration_minutes": Binding.of_value(1), "attendees": Binding.of_value([])}
    assert natural_call(event, one).endswith("duration 1 minute")  # singular unit; the empty list is omitted


def test_value_text_by_kind(scenario_catalog: Catalog) -> None:
    unit = scenario_catalog["get_weather"].slot("unit")
    assert value_text(unit, Binding.of_value("fahrenheit")) == "fahrenheit"
    body = scenario_catalog["send_email"].slot("body")
    assert value_text(body, Binding.of_value("")) is None  # empty values are omitted
    flag = _slot({"type": "boolean", "description": "Notify the attendees"})
    assert value_text(flag, Binding.of_value(True)) == "yes" and value_text(flag, Binding.of_value(False)) == "no"
    secret = _slot({"type": "string", "description": "The API key", "x-jev": {"kind": "secret"}})
    assert value_text(secret, Binding.of_value("sk-123", display="[secret]")) is None  # secrets never render


def test_lead_phrases_attach_to_the_head() -> None:
    catalog = Catalog.from_openai([{"type": "function", "function": {
        "name": "copy_file", "description": "Copy a file between two folders.",
        "parameters": {"type": "object", "required": ["from_folder", "to_folder", "name"], "properties": {
            "from_folder": {"type": "string", "description": "The folder to copy from"},
            "to_folder": {"type": "string", "description": "The folder to copy to"},
            "name": {"type": "string", "description": "The file name"}}}}}])  # fmt: skip
    tool = catalog["copy_file"]
    bindings = {"from_folder": Binding.of_value("inbox"), "to_folder": Binding.of_value("archive"),
                "name": Binding.of_value("q3.pdf")}  # fmt: skip
    assert render_call(tool, bindings) == 'copy a file from inbox to archive — file name "q3.pdf"'


def test_grid_menu_asks_the_open_question_with_values(scenario_catalog: Catalog) -> None:
    event = scenario_catalog["create_event"]
    duration = event.slot("duration_minutes")
    grid = [Binding.of_value(v, display=str(v)) for v in (15, 30, 45, 60)]
    prompt, actions = grid_menu(event, duration, grid)
    assert prompt.kind == "menu" and prompt.text == "What should the length of the event in minutes be?"
    assert [o.text for o in prompt.options] == ["15 minutes", "30 minutes", "45 minutes", "60 minutes",
                                                "Something else"]  # fmt: skip
    assert [o.id for o in prompt.options][:2] == ["pick:duration_minutes:0", "pick:duration_minutes:1"]
    assert actions["pick:duration_minutes:2"].value == 45 and actions["other"].action == "open"
    base = {"title": Binding.of_value("Standup")}
    full, _ = grid_menu(event, duration, grid, bindings=base, complete_call=True)
    assert full.options[0].text == 'Create a calendar event — event title "Standup", duration 15 minutes'


def _slot(schema: dict[str, Any]) -> Any:
    catalog = Catalog.from_openai(
        [
            {
                "type": "function",
                "function": {
                    "name": "t",
                    "description": "Do it.",
                    "parameters": {"type": "object", "properties": {"p": schema}},
                },
            }
        ]
    )
    return catalog["t"].slot("p")


def test_open_question_keeps_the_typed_format_of_a_ref(scenario_catalog: Catalog) -> None:
    # Jev is asked about the entity (the recipient); a user who must type the value is asked for its format
    to = {s.name: s for s in scenario_catalog["send_email"].slots}["to"]
    assert to.noun == "the recipient"
    assert open_question(to)[0].text == "What should the recipient's email address be?"

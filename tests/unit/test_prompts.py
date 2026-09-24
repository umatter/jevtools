"""Prompts (spec §3.8.4): templates filled with labels, option ids and their resume actions."""

from __future__ import annotations

from jevtools.decision import PromptOption
from jevtools.prompts import (
    AltChoice,
    Binding,
    clarify_menu,
    confirm_card,
    confirm_text,
    fill_template,
    noun_short,
    ok_text,
    open_question,
    parse_short_reply,
    refuse_notice,
    render_call,
    tool_menu,
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
    assert render_call(weather, bindings) == "get the current weather for a city: city=Zurich, unit=fahrenheit"
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
    assert prompt.text == "Which recipient's email address did you mean?" and noun_short(to) == (
        "recipient's email address")  # fmt: skip
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

"""The τ²-bench reduction (``jevtools.bench.tau2``) on a synthetic conversation in τ²'s published format."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import jevtools as jt
from jevtools.bench import tau2
from jevtools.context import Context, Observation
from jevtools.plan import compile_round
from jevtools.spec.catalog import Catalog

RETURN = {"type": "function", "function": {
    "name": "return_delivered_order_items", "description": "Return some items of a delivered order.",
    "parameters": {"type": "object", "required": ["order_id", "item_ids", "payment_method_id"], "properties": {
        "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."},
        "item_ids": {"type": "array", "items": {"type": "string"}, "description": "The item ids to be returned."},
        "payment_method_id": {"type": "string", "description": "The payment method for the refund."}}}}}  # fmt: skip
ORDER = {"order_id": "#W6390527", "items": [{"name": "Water Bottle", "item_id": "8538875209"},
                                             {"name": "Desk Lamp", "item_id": "7453605304"}],
         "payment_history": [{"payment_method_id": "paypal_7644869"}]}  # fmt: skip


def test_inline_refs_and_matching() -> None:
    schema = {"type": "object", "properties": {"f": {"$ref": "#/$defs/F"}},
              "$defs": {"F": {"type": "object", "properties": {"n": {"type": "string"}}}}}  # fmt: skip
    assert tau2.inline_refs(schema) == {"type": "object", "properties": {
        "f": {"type": "object", "properties": {"n": {"type": "string"}}}}}  # fmt: skip
    gold = {"name": "t", "arguments": {"ids": ["a", "b"], "n": 2}}
    assert tau2.matches(gold, {"name": "t", "arguments": {"ids": ["b", "a"], "n": 2.0, "extra": 1}})
    assert not tau2.matches(gold, {"name": "t", "arguments": {"ids": ["a"], "n": 2}})
    assert not tau2.matches(gold, {"name": "u", "arguments": gold["arguments"]}) and not tau2.matches(gold, None)


def test_build_cases_from_a_published_conversation(tmp_path: Path) -> None:
    sim = {"task_id": "7", "trial": 0, "reward_info": {"reward": 1.0}, "messages": [
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "Return the water bottle from order #W6390527, refund to paypal."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "name": "get_order_details", "arguments": {"order_id": "#W6390527"}}]},
        {"role": "tool", "id": "c1", "content": json.dumps(ORDER)},
        {"role": "assistant", "content": "I can return the water bottle to paypal. Shall I proceed?"},
        {"role": "user", "content": "Yes, please."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "name": RETURN["function"]["name"],
            "arguments": {"order_id": "#W6390527", "item_ids": ["8538875209"],
                          "payment_method_id": "paypal_7644869"}}]},
    ]}  # fmt: skip
    failed = {**sim, "trial": 1, "reward_info": {"reward": 0.0}}
    path = tmp_path / tau2.RESULTS.format(domain="retail")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"simulations": [failed, sim]}), encoding="utf-8")
    cases = tau2.build_cases(tmp_path, "retail")
    assert [(c.kind, c.gold["name"] if c.gold else None) for c in cases] == [
        ("call", "get_order_details"),
        ("talk", None),
        ("call", "return_delivered_order_items"),
    ]
    last = cases[-1]
    assert last.messages[-1] == {"role": "user", "content": "Yes, please."} and not last.read
    assert last.observations[0]["tool"] == "get_order_details" and last.observations[0]["content"]["items"]
    tau2.save_cases(cases, tmp_path / "c.jsonl")
    assert tau2.load_cases(tmp_path / "c.jsonl") == cases


def test_a_write_tool_gets_looked_up_ids_only_from_trusted_tools() -> None:
    catalog = Catalog.from_openai([RETURN])
    observation = Observation(step=1, tool="get_order_details", content=ORDER)
    messages = [{"role": "user", "content": "Return the water bottle from order #W6390527, refund to paypal."}]

    def item_pool(trusted: tuple[str, ...]) -> tuple[list[str], list[str]]:
        ctx = Context(messages=messages, observations=[observation], trusted_tools=trusted)
        pool = compile_round(catalog, ctx, jt.Policy(), mode="loop").pool(RETURN["function"]["name"], ("item_ids",))
        assert pool is not None
        return [str(c.value) for c in pool.candidates], [str(c.value) for c in pool.blocked]

    offered, blocked = item_pool(())
    assert "8538875209" not in offered and "8538875209" in blocked  # the injection barrier holds by default
    offered, blocked = item_pool(("get_order_details",))
    assert "8538875209" in offered and not blocked  # a first-party lookup's field reaches the write tool
    ctx = Context(messages=messages, observations=[observation], trusted_tools=("get_order_details",))
    plan = compile_round(catalog, ctx, jt.Policy(), mode="loop")
    item = next(c for c in plan.pool(RETURN["function"]["name"], ("item_ids",)).candidates if c.value == "8538875209")
    assert "name: Water Bottle" in str(item.text)  # the field is described by its siblings
    assert Context(messages=messages).to_doc() == Context(messages=messages, trusted_tools=()).to_doc()  # hashes kept


def test_name_parts_digit_strings_and_qualified_fields_reach_the_pool() -> None:
    lookup = {"type": "function", "function": {
        "name": "find_user_id_by_name_zip", "description": "Find a user id by name and zip code.",
        "parameters": {"type": "object", "required": ["first_name", "last_name", "zip"], "properties": {
            "first_name": {"type": "string"}, "last_name": {"type": "string"},
            "zip": {"type": "string"}}}}}  # fmt: skip
    by_email = {"type": "function", "function": {
        "name": "find_user_id_by_email", "description": "Find a user id by email.",
        "parameters": {"type": "object", "required": ["email"],
                       "properties": {"email": {"type": "string"}}}}}  # fmt: skip
    exchange = {"type": "function", "function": {
        "name": "exchange_delivered_order_items", "description": "Exchange items of a delivered order.",
        "parameters": {"type": "object", "required": ["new_item_ids"], "properties": {
            "new_item_ids": {"type": "array", "items": {"type": "string"}}}}}}  # fmt: skip
    catalog = Catalog.from_openai([lookup, by_email, exchange])
    messages = [{"role": "user", "content": "I'm Yusuf Rossi, zip code 19122, email yusuf.rossi@example.com."}]
    observation = Observation(step=1, tool="get_product_details", content={"variants": [{"item_id": "1234567890"}]})
    ctx = Context(messages=messages, observations=[observation], trusted_tools=("get_product_details",))
    plan = compile_round(catalog, ctx, jt.Policy(), mode="loop")

    def offered(tool: str, slot: str) -> list[str]:
        pool = plan.pool(tool, (slot,))
        assert pool is not None
        return [str(c.value) for c in pool.candidates]

    assert "Yusuf" in offered("find_user_id_by_name_zip", "first_name")
    assert "Rossi" in offered("find_user_id_by_name_zip", "last_name")
    assert "19122" in offered("find_user_id_by_name_zip", "zip")
    assert "yusuf.rossi@example.com" in offered("find_user_id_by_email", "email")
    assert "1234567890" in offered("exchange_delivered_order_items", "new_item_ids")


ORDER_LOOKUP = {"type": "function", "function": {
    "name": "get_order_details", "description": "Get the status and details of an order.",
    "parameters": {"type": "object", "required": ["order_id"], "properties": {"order_id": {
        "type": "string", "description": "The order id, such as '#W0000000'."}}}}}  # fmt: skip


def test_an_id_typed_without_its_prefix_is_offered_in_the_schema_format() -> None:
    catalog = Catalog.from_openai([ORDER_LOOKUP])

    def offered(text: str) -> list[str]:
        plan = compile_round(catalog, Context(messages=[{"role": "user", "content": text}]), jt.Policy(), mode="turn")
        pool = plan.pool("get_order_details", ("order_id",))
        assert pool is not None
        return [str(c.value) for c in pool.candidates]

    assert "#W4284542" in offered("My order number is W4284542.")
    assert "#W9502127" in offered("It's order 9502127.")
    assert not any(v.startswith("#W") for v in offered("My zip code is 28236."))  # five digits are not an order id


def test_a_list_of_records_takes_whole_objects_from_trusted_results() -> None:
    passenger = {"type": "object", "required": ["first_name", "last_name", "dob"], "properties": {
        "first_name": {"type": "string"}, "last_name": {"type": "string"}, "dob": {"type": "string"}}}  # fmt: skip
    item = {"anyOf": [passenger, {"type": "object", "additionalProperties": True}]}  # pydantic's ``Passenger | dict``
    book = {"type": "function", "function": {"name": "book_reservation", "description": "Book a reservation.",
            "parameters": {"type": "object", "required": ["passengers"], "properties": {"passengers": {
                "type": "array", "items": item}}}}}  # fmt: skip
    profile = {"user_id": "mia_li_3668", "saved_passengers": [
        {"first_name": "Amelia", "last_name": "Ahmed", "dob": "1957-03-21"},
        {"first_name": "Mia", "last_name": "Li", "dob": "1990-04-05", "membership": "gold"},
        {"first_name": "Noah", "last_name": None, "dob": "1990-01-01"}]}  # fmt: skip
    ctx = Context(messages=[{"role": "user", "content": "Book it for me and Amelia."}],
                  observations=[Observation(step=1, tool="get_user_details", content=profile)],
                  trusted_tools=("get_user_details",))  # fmt: skip
    plan = compile_round(Catalog.from_openai([book]), ctx, jt.Policy(), mode="loop")
    pool = plan.pool("book_reservation", ("passengers",))
    assert pool is not None
    assert [c.value for c in pool.candidates] == [
        {"first_name": "Amelia", "last_name": "Ahmed", "dob": "1957-03-21"},
        {"first_name": "Mia", "last_name": "Li", "dob": "1990-04-05"},  # cut down to the item's fields
    ]  # a passenger with a null required field is not a record
    questions = [q.family for q in plan.ballot.questions if q.path == ("passengers",)]
    assert questions == ["item", "item", "count"]


def test_a_reply_to_the_assistants_question_keeps_the_earlier_task() -> None:
    catalog = Catalog.from_openai([ORDER_LOOKUP])
    messages = [{"role": "user", "content": "Where is my order?"},
                {"role": "assistant", "content": "Could you give me the order id?"},
                {"role": "user", "content": "#W4284542"}]  # fmt: skip
    obs = [Observation(step=1, tool="get_order_details", content={"order_id": "#W1"})]
    tool = next(q for q in compile_round(catalog, Context(messages=messages, observations=obs), jt.Policy(),
                                         mode="loop").ballot.questions if q.family == "tool")  # fmt: skip
    assert "the user's task is the one they stated in their earlier turns" in str(tool.instructions)
    assert "in `request` and before it" in str(tool.sentinels["DONE"].text)
    first = next(q for q in compile_round(catalog, Context(messages=messages[:1]), jt.Policy(),
                                          mode="turn").ballot.questions if q.family == "tool")  # fmt: skip
    assert "stated in their earlier turns" not in str(first.instructions)  # the first request is the task itself


def test_an_id_list_with_numeric_examples_leaves_out_wordless_phrases() -> None:
    tool = deepcopy(RETURN)
    tool["function"]["parameters"]["properties"]["item_ids"]["description"] = "The item ids, each such as '1008292230'."
    messages = [{"role": "user", "content": "Return the item ID I mentioned: the water bottle from order #W6390527."}]
    ctx = Context(messages=messages, observations=[Observation(step=1, tool="get_order_details", content=ORDER)],
                  trusted_tools=("get_order_details",))  # fmt: skip
    pool = compile_round(Catalog.from_openai([tool]), ctx, jt.Policy(), mode="loop").pool(
        RETURN["function"]["name"], ("item_ids",)
    )
    assert pool is not None
    values = [str(c.value) for c in pool.candidates]
    assert "8538875209" in values and all(any(ch.isdigit() for ch in v) for v in values)


def test_check_grounds_a_list_argument_element_by_element() -> None:
    from jevtools.backends.scripted import ScriptedBackend
    from jevtools.router import Router

    messages = [{"role": "user", "content": "Return the water bottle from order #W6390527, refund to paypal."}]
    ctx = Context(messages=messages, observations=[Observation(step=1, tool="get_order_details", content=ORDER)],
                  trusted_tools=("get_order_details",))  # fmt: skip

    def check(item_ids: list[str]) -> jt.Decision:
        backend = ScriptedBackend({"*.verify.*": 0.95, "*.authorized": 0.95})
        call = {"name": RETURN["function"]["name"], "arguments": {
            "order_id": "#W6390527", "item_ids": item_ids, "payment_method_id": "paypal_7644869"}}  # fmt: skip
        return Router([RETURN], backend=backend).check(messages, call, context=ctx, mode="loop")

    shown = check(["8538875209"])
    assert shown.outcome in (jt.Outcome.CONFIRM, jt.Outcome.EXECUTE) and shown.call is not None
    assert shown.call.arguments["item_ids"] == ["8538875209"]  # verified element by element, kept as proposed
    invented = check(["8538875209", "1234567890"])
    assert invented.outcome is jt.Outcome.CLARIFY and invented.rule == "C1.check.ungrounded"
    assert invented.bottleneck is not None and invented.bottleneck.slot == "item_ids"

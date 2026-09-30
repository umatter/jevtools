"""The τ²-bench reduction (``jevtools.bench.tau2``) on a synthetic conversation in τ²'s published format."""

from __future__ import annotations

import json
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

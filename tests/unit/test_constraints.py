from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

import pytest

from jevtools.errors import ConstraintError
from jevtools.spec.constraints import CHECKS, ConstraintContext, Operand, check_all, compare, parse, register_check
from tests.support import SCENARIO_NOW


def test_parse_forms() -> None:
    c = parse("amount <= from_account.balance")
    assert (c.op, c.left, c.right) == ("<=", Operand("path", ("amount",)), Operand("path", ("from_account", "balance")))
    assert c.slots == {"amount", "from_account"} and not c.unary
    assert parse("start > now").unary and parse("start > now").right == Operand("now")
    assert parse('currency != "XXX"').right == Operand("literal", value="XXX")
    assert parse("n >= -1.5").right == Operand("literal", value=-1.5)
    assert parse('note == "a >= b"').right == Operand("literal", value="a >= b")
    assert parse("@balance_ok").check_name == "balance_ok"
    for bad in ("amount", "a == ", "@", "a.b-c! == 1", 'x == "unterminated'):
        with pytest.raises(ConstraintError):
            parse(bad)


def test_transfer_constraints_against_attrs() -> None:
    attrs = {"from_account": {"balance": 1000}, "to_account": {"balance": 5}}
    ctx = ConstraintContext(attrs=attrs)
    args = {"from_account": "acc_7731", "to_account": "acc_2210", "amount": "250.00"}
    assert parse("from_account != to_account").check(args, ctx) is True
    assert parse("amount <= from_account.balance").check(args, ctx) is True
    assert parse("amount <= from_account.balance").check({**args, "amount": "1000.01"}, ctx) is False
    assert parse("from_account != to_account").check({**args, "to_account": "acc_7731"}, ctx) is False
    unbound = parse("amount <= from_account.balance").check({"amount": "1"}, ConstraintContext())
    assert isinstance(unbound, str) and "not bound" in unbound


def test_time_and_nested_values() -> None:
    ctx = ConstraintContext(now=SCENARIO_NOW)
    assert parse("start > now").check({"start": "2026-09-29T15:00:00+02:00"}, ctx) is True
    assert parse("start > now").check({"start": "2026-09-01T15:00:00"}, ctx) is False
    assert parse("day >= now").check({"day": "2026-09-24"}, ctx) is True
    assert parse("start > now").check({"start": "2026-09-29T15:00:00"}, ConstraintContext()) != True  # noqa: E712
    assert parse('addr.zip == "8000"').check({"addr": {"zip": "8000"}}) is True


def test_compare_coercion() -> None:
    assert compare("0123", "==", "123") is False  # ids compare as strings
    assert compare("250.00", "==", 250) is True
    assert compare("10", "<", "9") is False  # ordering coerces numeric strings
    assert isinstance(compare({"a": 1}, "<", 2), str)
    assert compare(datetime(2026, 1, 1), "<", "2026-01-02T00:00:00") is True


def test_registered_checks() -> None:
    def weekday_only(args: Mapping[str, Any], ctx: ConstraintContext) -> bool | str:
        return datetime.fromisoformat(args["start"]).weekday() < 5 or "weekend"

    register_check("@weekday_only", weekday_only)
    try:
        c = parse("@weekday_only")
        assert c.check({"start": "2026-09-29T15:00:00"}) is True
        assert c.check({"start": "2026-09-27T15:00:00"}) == "weekend"
        assert parse("@nope").check({}) == "unknown check @nope"
        failures = check_all([c, parse("a == 1")], {"start": "2026-09-27T15:00:00", "a": 1})
        assert [(f.expr, r) for f, r in failures] == [("@weekday_only", "weekend")]
    finally:
        CHECKS.pop("weekday_only", None)

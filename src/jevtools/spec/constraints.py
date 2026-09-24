"""Cross-slot constraints (spec §3.2 ``constraints``, §3.6 rule 4).

An expression is ``A op B`` with ``op ∈ {==, !=, <, <=, >, >=}``; operands are parameter paths
(``amount``, ``from_account.balance``), ``now`` or JSON literals (``0``, ``"CHF"``, ``true``, ``null``). An expression
``"@name"`` calls a check registered with :func:`register_check`.

Evaluation returns ``True`` (satisfied), ``False`` (violated) or a string explaining why it could not be evaluated
(unbound operand, incomparable types). Only ``True`` is feasible: callers fail closed on everything else.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from jevtools.errors import ConstraintError

OPERATORS: tuple[str, ...] = ("==", "!=", "<=", ">=", "<", ">")
_DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}")
_ORDERING = frozenset({"<", "<=", ">", ">="})


@dataclass(frozen=True)
class ConstraintContext:
    """Evaluation environment besides the bound arguments.

    ``attrs`` maps a slot name to the row attributes of its elected candidate (``{"from_account": {"balance": …}}``);
    ``now`` is the context time; ``context`` is passed through to registered checks.
    """

    attrs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    now: datetime | None = None
    context: Any = None


Check = Callable[[Mapping[str, Any], ConstraintContext], "bool | str"]
"""A registered check: ``fn(args, ctx) -> True | False | reason``."""

CHECKS: dict[str, Check] = {}
"""Registered ``@name`` checks."""


def register_check(name: str, fn: Check) -> None:
    """Register ``fn`` under ``name`` for ``"@name"`` constraints (overwrites an earlier registration)."""
    CHECKS[name.lstrip("@")] = fn


_UNSET: Any = object()


@dataclass(frozen=True)
class Operand:
    """A constraint operand: a parameter ``path``, ``now``, or a JSON ``literal``."""

    kind: Literal["path", "now", "literal"]
    path: tuple[str, ...] = ()
    value: Any = None

    @property
    def slot(self) -> str | None:
        """The top-level slot name a path operand refers to."""
        return self.path[0] if self.kind == "path" else None

    def resolve(self, args: Mapping[str, Any], ctx: ConstraintContext) -> Any:
        """The operand's value, or the private unset marker when it is not bound."""
        if self.kind == "literal":
            return self.value
        if self.kind == "now":
            return ctx.now if ctx.now is not None else _UNSET
        return _lookup(self.path, args, ctx.attrs)


def _lookup(path: tuple[str, ...], args: Mapping[str, Any], attrs: Mapping[str, Mapping[str, Any]]) -> Any:
    head, rest = path[0], path[1:]
    if rest and head in attrs:
        node: Any = attrs[head]
        for part in rest:
            if not isinstance(node, Mapping) or part not in node:
                break
            node = node[part]
        else:
            return node
    if head not in args:
        return _UNSET
    node = args[head]
    for part in rest:
        if not isinstance(node, Mapping) or part not in node:
            return _UNSET
        node = node[part]
    return node


@dataclass(frozen=True)
class Constraint:
    """A parsed constraint. Use :func:`parse` to build one."""

    expr: str
    op: str | None = None
    left: Operand | None = None
    right: Operand | None = None
    check_name: str | None = None

    @property
    def slots(self) -> frozenset[str]:
        """Top-level slot names referenced (empty for ``@checks``, which may read anything)."""
        return frozenset(o.slot for o in (self.left, self.right) if o is not None and o.slot is not None)

    @property
    def unary(self) -> bool:
        """References at most one slot, so it can be applied at pool time (e.g. ``start > now``)."""
        return self.check_name is None and len(self.slots) <= 1

    def check(self, args: Mapping[str, Any], ctx: ConstraintContext | None = None) -> bool | str:
        """Evaluate against bound ``args``; ``True`` satisfied, ``False`` violated, ``str`` not evaluable."""
        ctx = ctx or ConstraintContext()
        if self.check_name is not None:
            fn = CHECKS.get(self.check_name)
            if fn is None:
                return f"unknown check @{self.check_name}"
            return fn(args, ctx)
        assert self.left is not None and self.right is not None and self.op is not None
        left = self.left.resolve(args, ctx)
        right = self.right.resolve(args, ctx)
        if left is _UNSET or right is _UNSET:
            return f"{self.expr}: operand not bound"
        return compare(left, self.op, right)


def compare(left: Any, op: str, right: Any) -> bool | str:
    """Compare two values with JSON-ish coercion.

    Numbers (and numeric strings, for ordering or against a real number) compare as decimals; ISO date/time strings
    compare with datetimes; everything else compares by equality only. Incomparable pairs return a reason string.
    """
    left, right = _coerce(left, op, right)
    try:
        if op == "==":
            return bool(left == right)
        if op == "!=":
            return bool(left != right)
        if op == "<":
            return bool(left < right)
        if op == "<=":
            return bool(left <= right)
        if op == ">":
            return bool(left > right)
        if op == ">=":
            return bool(left >= right)
    except TypeError:
        return f"cannot compare {type(left).__name__} {op} {type(right).__name__}"
    return f"unknown operator {op!r}"


def _number(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            return Decimal(value.strip())
        except InvalidOperation:
            return None
    return None


def _temporal(value: Any, like: datetime | date) -> datetime | date | None:
    if isinstance(value, (datetime, date)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        if _DATE_ONLY.fullmatch(text):
            return date.fromisoformat(text)
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if isinstance(like, datetime):
        if parsed.tzinfo is None and like.tzinfo is not None:
            parsed = parsed.replace(tzinfo=like.tzinfo)
        return parsed
    return parsed.date()


def _coerce(left: Any, op: str, right: Any) -> tuple[Any, Any]:
    for a, b, swap in ((left, right, False), (right, left, True)):
        if isinstance(a, (datetime, date)):
            other = _temporal(b, a)
            if other is not None:
                if isinstance(a, datetime) and not isinstance(other, datetime):
                    a = a.date()
                return (other, a) if swap else (a, other)
    numeric = op in _ORDERING or any(
        isinstance(v, (int, float, Decimal)) and not isinstance(v, bool) for v in (left, right)
    )
    if numeric:
        ln, rn = _number(left), _number(right)
        if ln is not None and rn is not None:
            return ln, rn
    return left, right


def _operand(token: str, expr: str) -> Operand:
    token = token.strip()
    if not token:
        raise ConstraintError(f"empty operand in {expr!r}")
    if token == "now":
        return Operand("now")
    if token[0] in '"-0123456789[{' or token in ("true", "false", "null"):
        try:
            return Operand("literal", value=json.loads(token))
        except ValueError as exc:
            raise ConstraintError(f"bad literal {token!r} in {expr!r}") from exc
    parts = tuple(token.split("."))
    if not all(p.replace("_", "a").replace("-", "a").isalnum() for p in parts):
        raise ConstraintError(f"bad parameter path {token!r} in {expr!r}")
    return Operand("path", path=parts)


def _split(expr: str) -> tuple[str, str, str]:
    in_string = False
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == '"' and (i == 0 or expr[i - 1] != "\\"):
            in_string = not in_string
        elif not in_string:
            for op in OPERATORS:
                if expr.startswith(op, i):
                    return expr[:i], op, expr[i + len(op) :]
        i += 1
    raise ConstraintError(f"no comparison operator in constraint {expr!r}")


def parse(expr: str) -> Constraint:
    """Parse ``"A op B"`` or ``"@name"``; raises :class:`~jevtools.errors.ConstraintError` on bad input."""
    text = expr.strip()
    if text.startswith("@"):
        name = text[1:]
        if not name or not name.replace("_", "a").replace(".", "a").isalnum():
            raise ConstraintError(f"bad check name in {expr!r}")
        return Constraint(expr=text, check_name=name)
    left, op, right = _split(text)
    return Constraint(expr=text, op=op, left=_operand(left, text), right=_operand(right, text))


def check_all(
    constraints: tuple[Constraint, ...] | list[Constraint],
    args: Mapping[str, Any],
    ctx: ConstraintContext | None = None,
) -> list[tuple[Constraint, bool | str]]:
    """Evaluate every constraint; returns the failures (anything that is not ``True``) with their results."""
    failures: list[tuple[Constraint, bool | str]] = []
    for constraint in constraints:
        result = constraint.check(args, ctx)
        if result is not True:
            failures.append((constraint, result))
    return failures


__all__ = [
    "CHECKS",
    "OPERATORS",
    "Check",
    "Constraint",
    "ConstraintContext",
    "Operand",
    "check_all",
    "compare",
    "parse",
    "register_check",
]

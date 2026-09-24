"""Exception types shared across jevtools (backend errors live in ``jevtools.backends``)."""

from __future__ import annotations


class JevtoolsError(Exception):
    """Base class for jevtools errors that are not backend failures."""


class CatalogError(JevtoolsError, ValueError):
    """A tool definition cannot be compiled: bad schema, unknown ``x-jev`` key, name clash (spec §3.2)."""


class ConstraintError(JevtoolsError, ValueError):
    """A cross-slot constraint expression cannot be parsed (spec §3.2 ``constraints``)."""


class BallotError(JevtoolsError):
    """A Ballot violates a pre-send rule (spec §3.5.6). Always a code bug, never a model issue.

    ``qid`` names the offending question (``None`` for ballot-level rules) and ``rule`` is the
    rule id, e.g. ``"choice.options"`` or ``"call.tokens"`` (see :mod:`jevtools.validate`).
    """

    def __init__(self, qid: str | None, rule: str, message: str = "") -> None:
        self.qid = qid
        self.rule = rule
        self.message = message
        where = qid if qid is not None else "<ballot>"
        super().__init__(f"{where}: {rule}" + (f": {message}" if message else ""))


class PendingScopeError(JevtoolsError, PermissionError):
    """A pending prompt was resumed with the context of a different requester (another user profile or source set
    than the decision that raised it). The handle stays pending for its own requester."""


__all__ = ["BallotError", "CatalogError", "ConstraintError", "JevtoolsError", "PendingScopeError"]

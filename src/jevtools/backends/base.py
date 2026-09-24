"""The backend interface: anything that answers a Jev ``DecisionRequest``."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jevtools.wire import DecisionRequest, DecisionResponse


class BackendError(RuntimeError):
    """A backend failed to produce a response."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@runtime_checkable
class Backend(Protocol):
    """Answers typed questions about a state. Implemented by HTTP and offline backends."""

    def decide(self, request: DecisionRequest) -> DecisionResponse: ...

    async def adecide(self, request: DecisionRequest) -> DecisionResponse: ...

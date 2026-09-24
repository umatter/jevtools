"""The backend interface: anything that answers a Jev ``DecisionRequest`` (spec §8.1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jevtools.backends.errors import BackendError
from jevtools.wire import DecisionRequest, DecisionResponse


@runtime_checkable
class Backend(Protocol):
    """Answers typed questions about a state. Implemented by HTTP and offline backends.

    ``model`` is the model id the backend sends (``Ballot.to_requests(model=backend.model)``); ``name`` identifies
    the backend in traces and keys the token-estimate correction (``typesafe``, ``openrouter_decisions``…).
    """

    model: str
    name: str

    def decide(self, request: DecisionRequest) -> DecisionResponse: ...

    async def adecide(self, request: DecisionRequest) -> DecisionResponse: ...


__all__ = ["Backend", "BackendError"]

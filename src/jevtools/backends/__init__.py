"""Backends (spec §8): anything that answers a Jev ``DecisionRequest``.

HTTP backends for the three endpoints (§8.2), :func:`auto` (§8.3), the typed errors of §8.4, the offline
``ScriptedBackend`` (§8.5) and ``LexicalSimulator`` (§8.6, a lexical test double whose answers are never evidence
about Jev), and the replaying ``Cassette`` (§8.7).
"""

from jevtools.backends.auto import auto
from jevtools.backends.base import Backend
from jevtools.backends.cassette import Cassette, CassetteMiss
from jevtools.backends.errors import (
    BackendConfigError,
    BackendError,
    JevAuthError,
    JevNotFound,
    JevProtocolError,
    JevRateLimited,
    JevUnavailable,
    JevValidationError,
)
from jevtools.backends.http import HTTPBackend, OpenRouterDecisions, OpenRouterSystemOne, TypeSafe
from jevtools.backends.scripted import ScriptedBackend
from jevtools.backends.simulator import DEFAULT_SYNONYMS, LexicalSimulator

__all__ = [
    "DEFAULT_SYNONYMS",
    "Backend",
    "BackendConfigError",
    "BackendError",
    "Cassette",
    "CassetteMiss",
    "HTTPBackend",
    "JevAuthError",
    "JevNotFound",
    "JevProtocolError",
    "JevRateLimited",
    "JevUnavailable",
    "JevValidationError",
    "LexicalSimulator",
    "OpenRouterDecisions",
    "OpenRouterSystemOne",
    "ScriptedBackend",
    "TypeSafe",
    "auto",
]

"""Backends (spec §8): anything that answers a Jev ``DecisionRequest``.

HTTP backends for the three endpoints, the offline ``ScriptedBackend`` and the typed errors of §8.4.
"""

from jevtools.backends.base import Backend
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

__all__ = [
    "Backend",
    "BackendConfigError",
    "BackendError",
    "HTTPBackend",
    "JevAuthError",
    "JevNotFound",
    "JevProtocolError",
    "JevRateLimited",
    "JevUnavailable",
    "JevValidationError",
    "OpenRouterDecisions",
    "OpenRouterSystemOne",
    "ScriptedBackend",
    "TypeSafe",
]

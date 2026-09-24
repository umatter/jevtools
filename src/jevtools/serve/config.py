"""Configuration of ``jevtools serve`` (spec §7.2.4): ``jevtools.toml``.

```toml
model_name = "jevtools"          # the model id clients send and GET /v1/models lists
policy = "policy.toml"           # optional; Appendix B defaults otherwise (paths are relative to this file)
calibrators = "policy.calibrators.json"   # optional, written by `jevtools tune`
pending_ttl_s = 900              # CONFIRM/CLARIFY handles live 15 minutes

[backend]                        # or: backend = "auto"
type = "auto"                    # auto | typesafe | openrouter_systemone | openrouter_decisions | simulator
                                 # | cassette:<path> | "package.module:factory"
model = "~typesafe/jev-latest"   # optional
allow_offline = false            # auto only: fall back to the LexicalSimulator (a test double, never Jev)

[context]                        # defaults of every request (a request may override them)
tz = "Europe/Zurich"
locale = "en-CH"
user = {name = "Sam Muster", home_city = "Zurich"}

[[sources]]                      # registries from JSON/JSONL/CSV files or inline rows
name = "contacts"
path = "contacts.csv"
key = "email"
label = "{name} <{email}>"
match = ["name", "aliases", "team"]
provides = ["email", "person"]
list_fields = ["aliases"]        # CSV columns holding ";"-separated lists

[[sources]]                      # a file index over a JSON list or a text file of paths
name = "files"
type = "files"
path = "paths.txt"

[[sources]]                      # a host callable: rows or candidates for a SourceQuery
name = "tickets"
type = "provider"
function = "myapp.sources:tickets"
provides = ["ticket_id"]

[fallback_llm]                   # optional graceful degradation (OpenAI-compatible)
base_url = "https://openrouter.ai/api/v1"
model = "openai/gpt-4o-mini"
api_key_env = "OPENROUTER_API_KEY"

# [text_llm], [filler] and [escalator] take the same keys (plus `options`, e.g. temperature).
```

Source specs (:mod:`jevtools.sources.specs`) are shared with evaluation context documents and `jevtools lint --sources`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from jevtools._compat import load_toml
from jevtools.backends.base import Backend
from jevtools.backends.errors import BackendConfigError
from jevtools.confidence import IsotonicCalibrator
from jevtools.context import SETTING_FIELDS, Context
from jevtools.fallback import (
    API_KEY_ENV,
    DEFAULT_BASE_URL,
    OpenAICompatibleEscalator,
    OpenAICompatibleFiller,
    OpenAICompatibleTextLLM,
)
from jevtools.policy import Policy
from jevtools.sources.specs import (
    KEY_GUESSES,
    SourceConfig,
    SourceType,
    build_source,
    build_sources,
    guess_key,
    load_object,
    request_sources,
)

CONTEXT_DEFAULTS = SETTING_FIELDS
"""Context fields ``[context]`` may set for every request."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------------------------------------
# Backends and LLM endpoints
# --------------------------------------------------------------------------------------------------------------------


class BackendConfig(_Model):
    """Which Jev backend answers (spec §8.3): ``auto`` reads the environment; a named backend needs its key."""

    type: str = "auto"
    model: str | None = None
    allow_offline: bool = False
    options: dict[str, Any] = Field(default_factory=dict)
    """Extra keyword arguments (HTTP backends: ``timeout``, ``max_retries``…; factories: their parameters)."""

    def build(self, env: Mapping[str, str] | None = None) -> Backend:
        """The backend (raises :class:`~jevtools.backends.errors.BackendConfigError` on a missing key)."""
        from jevtools.backends.auto import BACKEND_NAMES, auto

        environ = dict(os.environ if env is None else env)
        if self.model:
            environ["JEVTOOLS_MODEL"] = self.model
        kind = self.type.strip()
        if kind == "auto":
            return auto(self.allow_offline, env=environ, **self.options)
        if kind.lower() in BACKEND_NAMES or kind.lower().startswith("cassette:"):
            environ["JEVTOOLS_BACKEND"] = kind
            return auto(False, env=environ, **self.options)
        if ":" in kind:
            backend = load_object(kind)(**self.options)
            if not callable(getattr(backend, "decide", None)):
                raise BackendConfigError(f"{kind} did not return a backend")
            return backend  # type: ignore[no-any-return]
        raise BackendConfigError(f"unknown backend {kind!r} (auto | {' | '.join(BACKEND_NAMES)} | cassette:<path> | "
                                 "module:factory)")  # fmt: skip


class EndpointConfig(_Model):
    """An OpenAI-compatible ``chat/completions`` endpoint (``text_llm``, ``filler``, ``escalator``,
    ``fallback_llm``). ``model`` may be left out for ``fallback_llm`` (the request's model is forwarded)."""

    model: str | None = None
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    api_key_env: str = API_KEY_ENV
    timeout: float = 60.0
    headers: dict[str, str] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)

    @property
    def url(self) -> str:
        """``{base_url}/chat/completions``."""
        return self.base_url.rstrip("/") + "/chat/completions"

    def key(self) -> str | None:
        """The API key (explicit, else from ``api_key_env``; ``None`` when neither is set)."""
        return self.api_key if self.api_key is not None else os.environ.get(self.api_key_env)

    def client_kwargs(self, **extra: Any) -> dict[str, Any]:
        """Keyword arguments of :class:`~jevtools.fallback.OpenAICompatibleClient` subclasses."""
        if not self.model:
            raise ValueError("this endpoint needs a `model`")
        return {"base_url": self.base_url, "api_key": self.api_key, "api_key_env": self.api_key_env,
                "timeout": self.timeout, "headers": self.headers or None, **self.options, **extra}  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# ServeConfig
# --------------------------------------------------------------------------------------------------------------------


class ServeConfig(_Model):
    """The proxy's configuration (``jevtools.toml``, spec §7.2.4)."""

    model_name: str = "jevtools"
    backend: BackendConfig = Field(default_factory=BackendConfig)
    policy: str | None = None
    calibrators: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    sources: list[SourceConfig] = Field(default_factory=list)
    text_llm: EndpointConfig | None = None
    filler: EndpointConfig | None = None
    escalator: EndpointConfig | None = None
    fallback_llm: EndpointConfig | None = None
    pending_ttl_s: float = 900.0
    max_routers: int = 32
    """Base routers kept per distinct set of per-request sources (least recently used dropped first)."""
    _base_dir: Path | None = PrivateAttr(default=None)

    @field_validator("backend", mode="before")
    @classmethod
    def _backend(cls, value: Any) -> Any:
        return {"type": value} if isinstance(value, str) else value

    @field_validator("context")
    @classmethod
    def _context(cls, value: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(value) - set(CONTEXT_DEFAULTS))
        if unknown:
            raise ValueError(f"unsupported [context] field(s) {unknown}; allowed: {', '.join(CONTEXT_DEFAULTS)}")
        return value

    @property
    def base_dir(self) -> Path | None:
        """The directory relative paths resolve against (the TOML file's directory)."""
        return self._base_dir

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, base_dir: str | os.PathLike[str] | None = None) -> ServeConfig:
        """Validate a configuration mapping."""
        config = cls.model_validate(dict(data))
        config._base_dir = Path(base_dir) if base_dir is not None else None
        return config

    @classmethod
    def from_toml_text(cls, text: str, *, base_dir: str | os.PathLike[str] | None = None) -> ServeConfig:
        """Parse ``jevtools.toml`` content."""
        return cls.from_dict(load_toml(text), base_dir=base_dir)

    @classmethod
    def from_toml(cls, path: str | os.PathLike[str]) -> ServeConfig:
        """Load ``jevtools.toml``; relative paths inside resolve against its directory."""
        file = Path(path)
        return cls.from_toml_text(file.read_text(encoding="utf-8"), base_dir=file.parent)

    def resolve(self, path: str) -> Path:
        """``path`` relative to the configuration file."""
        p = Path(path)
        return p if p.is_absolute() or self._base_dir is None else self._base_dir / p

    # -- builders -----------------------------------------------------------------------------------------------

    def build_backend(self, env: Mapping[str, str] | None = None) -> Backend:
        """The Jev backend."""
        return self.backend.build(env)

    def build_policy(self) -> Policy:
        """The policy file, or the Appendix B defaults."""
        return Policy.from_toml(self.resolve(self.policy)) if self.policy else Policy()

    def build_calibrators(self) -> dict[str, IsotonicCalibrator]:
        """The calibrators written by ``jevtools tune`` (none when not configured)."""
        if not self.calibrators:
            return {}
        from jevtools.eval.tuning import load_calibrators

        return load_calibrators(self.resolve(self.calibrators))

    def build_sources(self) -> list[Any]:
        """The configured sources."""
        return [build_source(spec, base_dir=self._base_dir) for spec in self.sources]

    def build_context(self, sources: Sequence[Any] = ()) -> Context:
        """The base context of every request: ``[context]`` defaults plus ``sources``."""
        return Context.model_validate({**self.context, "sources": list(sources)})

    def _endpoint(self, cfg: EndpointConfig | None, cls: Callable[..., Any]) -> Any:
        return cls(cfg.model, **cfg.client_kwargs()) if cfg is not None else None

    def build_text_llm(self) -> OpenAICompatibleTextLLM | None:
        """The text LLM for abstain handoffs (``None`` when not configured)."""
        return self._endpoint(self.text_llm, OpenAICompatibleTextLLM)  # type: ignore[no-any-return]

    def build_filler(self) -> OpenAICompatibleFiller | None:
        """The FILL endpoint (``None`` when not configured)."""
        return self._endpoint(self.filler, OpenAICompatibleFiller)  # type: ignore[no-any-return]

    def build_escalator(self) -> OpenAICompatibleEscalator | None:
        """The escalation endpoint (``None`` when not configured)."""
        return self._endpoint(self.escalator, OpenAICompatibleEscalator)  # type: ignore[no-any-return]


__all__ = [
    "CONTEXT_DEFAULTS",
    "KEY_GUESSES",
    "BackendConfig",
    "EndpointConfig",
    "ServeConfig",
    "SourceConfig",
    "SourceType",
    "build_source",
    "build_sources",
    "guess_key",
    "load_object",
    "request_sources",
]

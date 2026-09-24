"""``jevtools.toml`` (spec §7.2.4): backends, policy and calibrators files, sources from JSON/CSV/text files and
``module:function`` providers, LLM endpoints, per-request sources."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevtools.backends.errors import BackendConfigError
from jevtools.backends.simulator import LexicalSimulator
from jevtools.confidence import IsotonicCalibrator
from jevtools.eval.tuning import save_calibrators
from jevtools.fallback import OpenAICompatibleEscalator, OpenAICompatibleFiller, OpenAICompatibleTextLLM
from jevtools.policy import Policy
from jevtools.serve.config import (
    BackendConfig,
    EndpointConfig,
    ServeConfig,
    SourceConfig,
    build_source,
    build_sources,
    guess_key,
    load_object,
    request_sources,
)
from jevtools.sources import FileIndex, Provider, Registry

TOML = """
model_name = "jevtools"
policy = "policy.toml"
calibrators = "cal.json"
pending_ttl_s = 60

[backend]
type = "simulator"
model = "sim-test"

[context]
tz = "Europe/Zurich"
locale = "en-CH"
user = {name = "Sam Muster", home_city = "Zurich"}

[[sources]]
name = "contacts"
path = "contacts.csv"
key = "email"
label = "{name} <{email}>"
match = ["name", "aliases"]
provides = ["email", "person"]
list_fields = ["aliases"]

[[sources]]
name = "accounts"
path = "accounts.json"
key = "id"
provides = ["account_id"]

[[sources]]
name = "files"
type = "files"
path = "paths.txt"

[[sources]]
name = "tickets"
type = "provider"
function = "tests.serve.support:tickets"
provides = ["ticket_id"]

[text_llm]
model = "openai/gpt-4o-mini"
api_key = ""

[fallback_llm]
base_url = "https://llm.example/v1"
api_key_env = "JEVTOOLS_TEST_FALLBACK_KEY"
"""


def write_config(tmp_path: Path) -> Path:
    (tmp_path / "contacts.csv").write_text("name,email,aliases\nRobert Brown,rbrown@partner.io,Bob;Rob\n"
                                           "Anna Keller,anna.keller@acme.com,\n", encoding="utf-8")  # fmt: skip
    (tmp_path / "accounts.json").write_text(json.dumps({"rows": [{"id": "acc_1", "nickname": "Savings"}]}),
                                            encoding="utf-8")  # fmt: skip
    (tmp_path / "paths.txt").write_text("docs/a.md\n\nservices/pay/config.yaml\n", encoding="utf-8")
    tuned = Policy.from_dict({"version": "tuned-test", "tiers": {"write": {"execute": 0.75}}})
    from jevtools.eval.tuning import policy_to_toml

    (tmp_path / "policy.toml").write_text(policy_to_toml(tuned), encoding="utf-8")
    save_calibrators({"write": IsotonicCalibrator("iso").fit([0.2, 0.8], [0, 1])}, tmp_path / "cal.json")
    path = tmp_path / "jevtools.toml"
    path.write_text(TOML, encoding="utf-8")
    return path


def test_from_toml_builds_everything(tmp_path: Path) -> None:
    config = ServeConfig.from_toml(write_config(tmp_path))
    assert config.base_dir == tmp_path and config.pending_ttl_s == 60
    backend = config.build_backend()
    assert isinstance(backend, LexicalSimulator) and backend.model == "sim-test"
    assert config.build_policy().version == "tuned-test" and config.build_policy().tier("write").execute == 0.75
    assert config.build_calibrators()["write"].fitted
    contacts, accounts, files, tickets = config.build_sources()
    assert isinstance(contacts, Registry) and contacts.lookup("rbrown@partner.io")["aliases"] == ["Bob", "Rob"]
    assert contacts.lookup("anna.keller@acme.com")["aliases"] == []
    assert isinstance(accounts, Registry) and len(accounts) == 1
    assert isinstance(files, FileIndex) and isinstance(tickets, Provider) and tickets.provides == {"ticket_id"}
    ctx = config.build_context(config.build_sources())
    assert ctx.locale == "en-CH" and ctx.user["home_city"] == "Zurich" and set(ctx.sources) == {
        "contacts", "accounts", "files", "tickets"}  # fmt: skip
    text_llm = config.build_text_llm()
    assert isinstance(text_llm, OpenAICompatibleTextLLM) and text_llm.model == "openai/gpt-4o-mini"
    assert config.build_filler() is None and config.build_escalator() is None
    assert config.fallback_llm is not None and config.fallback_llm.url == "https://llm.example/v1/chat/completions"
    assert config.fallback_llm.model is None


def test_defaults_and_validation() -> None:
    config = ServeConfig()
    assert config.model_name == "jevtools" and config.backend.type == "auto" and config.build_policy() == Policy()
    assert config.build_calibrators() == {} and config.build_sources() == [] and config.build_text_llm() is None
    assert ServeConfig.from_dict({"backend": "simulator"}).backend.type == "simulator"
    with pytest.raises(ValueError, match="unsupported"):
        ServeConfig.from_dict({"context": {"sources": []}})
    with pytest.raises(ValueError):
        ServeConfig.from_dict({"unknown": 1})


def test_backend_configs() -> None:
    assert isinstance(BackendConfig(type="simulator").build(env={}), LexicalSimulator)
    with pytest.raises(BackendConfigError, match="no Jev backend configured"):
        BackendConfig(type="auto").build(env={})
    with pytest.warns(UserWarning, match="never evidence about Jev"):
        assert isinstance(BackendConfig(type="auto", allow_offline=True).build(env={}), LexicalSimulator)
    with pytest.raises(BackendConfigError, match="needs TYPESAFE_API_KEY"):
        BackendConfig(type="typesafe").build(env={})
    http = BackendConfig(type="openrouter_decisions", model="typesafe/jev-1.13").build(
        env={"OPENROUTER_API_KEY": "sk-or-test"})  # fmt: skip
    assert http.name == "openrouter_decisions" and http.model == "typesafe/jev-1.13"
    factory = BackendConfig(type="tests.serve.support:make_simulator", options={"seed": 3}).build(env={})
    assert isinstance(factory, LexicalSimulator)
    with pytest.raises(BackendConfigError, match="unknown backend"):
        BackendConfig(type="nope").build(env={})
    with pytest.raises(BackendConfigError, match="did not return a backend"):
        BackendConfig(type="tests.serve.support:tickets", options={"query": None}).build(env={})


def test_endpoints() -> None:
    endpoint = EndpointConfig(model="m", api_key="", options={"temperature": 0.2})
    assert isinstance(ServeConfig(filler=endpoint).build_filler(), OpenAICompatibleFiller)
    assert isinstance(ServeConfig(escalator=endpoint).build_escalator(), OpenAICompatibleEscalator)
    assert endpoint.key() == "" and EndpointConfig(api_key_env="JEVTOOLS_TEST_UNSET_KEY").key() is None
    with pytest.raises(ValueError, match="needs a `model`"):
        EndpointConfig().client_kwargs()


def test_source_builders(tmp_path: Path) -> None:
    rows = [{"email": "a@x.io", "name": "A"}, {"email": "b@x.io", "name": "B"}]
    reg = build_source({"name": "people", "rows": rows})
    assert isinstance(reg, Registry) and reg.key == "email"
    assert guess_key([{"id": 1, "email": "x"}]) == "id"
    with pytest.raises(ValueError, match="cannot tell the key"):
        guess_key([{"x": 1}])
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    [from_jsonl] = build_sources([{"name": "p", "path": "rows.jsonl", "key": "email"}], base_dir=tmp_path)
    assert len(from_jsonl) == 2
    [named] = build_sources({"people": {"rows": rows}, "more": rows})[:1]
    assert named.name == "people"
    with pytest.raises(ValueError, match="give `rows` or `path`"):
        build_source({"name": "empty"})
    with pytest.raises(ValueError, match="needs `function"):
        build_source({"name": "p", "type": "provider"})
    with pytest.raises(ValueError, match="did not return a source"):
        build_source({"name": "p", "type": "source", "function": "tests.serve.support:tickets",
                      "options": {"query": None}})  # fmt: skip
    assert load_object("jevtools.policy:Policy.default")() == Policy()
    with pytest.raises(ValueError, match="module:attribute"):
        load_object("jevtools.policy")


def test_request_sources_reuse_configured_settings() -> None:
    template = SourceConfig(name="contacts", path="unused.csv", key="email", label="{name} <{email}>",
                            match=["name"], provides=["email"])  # fmt: skip
    rows = [{"name": "Anna Keller", "email": "anna.keller@acme.com"}]
    [reused] = request_sources({"contacts": rows}, [template])
    assert isinstance(reused, Registry) and reused.label == "{name} <{email}>" and reused.provides == {"email"}
    [spec] = request_sources({"people": {"rows": rows, "key": "email", "provides": ["person"]}})
    assert spec.name == "people" and spec.provides == {"person"}
    [guessed] = request_sources({"others": rows})
    assert guessed.key == "email"
    with pytest.raises(ValueError, match="expected a list"):
        request_sources({"contacts": "nope"})

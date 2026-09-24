"""The runnable examples (``examples/``): every ``main()`` runs offline in scripted mode — pinned to the spec §13
walk-through — and in sim mode, where only invariants and allowed outcome sets are checked. Scripted and simulated
answers exercise plumbing and policy; they are never evidence about Jev's accuracy."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from jevtools.backends.scripted import ScriptedBackend
from jevtools.decision import Decision
from jevtools.demo import scenario, scripts
from jevtools.policy import Outcome
from jevtools.wire import ChoiceQuestion, DecisionRequest
from tests.support import load_fixture

pytestmark = pytest.mark.fast

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
NAMES = sorted(p.stem for p in EXAMPLES.glob("0*_*.py"))


@pytest.fixture(autouse=True)
def _examples_on_path(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """``examples/`` on ``sys.path`` (the examples import ``_show``) and no Jev key in the environment."""
    monkeypatch.syspath_prepend(str(EXAMPLES))
    for variable in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY", "JEVTOOLS_BACKEND"):
        monkeypatch.delenv(variable, raising=False)
    yield


def load_file(path: Path, name: str) -> ModuleType:
    """Import a script as a fresh module named ``name`` (registered in ``sys.modules``, as dataclasses need)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load(number: str) -> ModuleType:
    """Import ``examples/<number>_*.py`` as a fresh module."""
    (path,) = EXAMPLES.glob(f"{number}_*.py")
    return load_file(path, f"example_{number}")


def run(number: str, backend: str = "scripted") -> Any:
    return load(number).main(["--backend", backend])


def executed(d: Decision) -> bool:
    return d.outcome is Outcome.EXECUTE and bool(d.tool_calls)


def test_every_example_is_listed() -> None:
    assert NAMES == ["01_quickstart_weather", "02_email_contacts", "03_transfer_confirm", "04_file_shortlist_widen",
                     "05_calendar_temporal", "06_agent_loop_invoice", "07_openai_dropin",
                     "08_bring_your_own_tools"]  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# scripted: the spec's numbers
# --------------------------------------------------------------------------------------------------------------------


def test_01_quickstart(capsys: pytest.CaptureFixture[str]) -> None:
    results = run("01")
    r1, r7 = results["R1"], results["R7"]
    assert (r1.outcome, r1.rule) == (Outcome.EXECUTE, "P9.read.execute")
    assert r1.call is not None and r1.call.arguments == {"city": "Zurich", "unit": "fahrenheit"}
    assert r1.confidence is not None and r1.confidence.W == pytest.approx(0.97) and r1.slots["city"].p == 0.97
    assert (r7.outcome, r7.rule) == (Outcome.ABSTAIN, "P1.tool.no_tool") and r7.tool_calls == []
    out = capsys.readouterr().out
    assert "5 in 1 Jev call — tool · get_weather: city, unit · search_web: query.accept.0-1" in out
    assert "never evidence about Jev's accuracy" in out and '"arguments": "{\\"city\\":\\"Zurich\\"' in out


def test_02_email(capsys: pytest.CaptureFixture[str]) -> None:
    results = run("02")
    card = results["with_history"]
    assert (card.outcome, card.rule) == (Outcome.CONFIRM, "P9.external.confirm_band") and card.tool_calls == []
    assert card.confidence is not None and round(card.confidence.PI, 3) == 0.714
    click = results["with_history_click"]
    assert executed(click) and click.rounds == 0 and click.usage.jev_calls == 0
    assert click.tool_calls[0].arguments["to"] == "anna.keller@acme.com"
    menu = results["no_history"]
    assert (menu.outcome, menu.rule) == (Outcome.CLARIFY, "P9.external.ambiguous")
    assert menu.prompt is not None and [o.id for o in menu.prompt.options][:3] == ["pick:to:0", "pick:to:1",
                                                                                    "pick:to:2"]  # fmt: skip
    assert executed(results["no_history_click"])
    reply = results["free_text"]
    assert reply.outcome is Outcome.CONFIRM and reply.rounds == 1 and reply.trace.resumed_from == menu.pending_id
    assert reply.call is not None and reply.call.arguments["to"] == "anna.rossi@gmail.com"
    out = capsys.readouterr().out
    assert "13 in 1 Jev call" in out and "0 — no Jev call (a click on a prompt option)" in out


def test_03_transfer() -> None:
    results = run("03")
    card = results["confirm"]
    assert (card.outcome, card.rule) == (Outcome.CONFIRM, "P9.critical.confirm_band")
    c = card.confidence
    assert c is not None and (round(c.L, 2), c.J, round(c.call, 2)) == (0.84, 0.92, 0.84)
    assert card.prompt is not None and "alt:from_account:1" in [o.id for o in card.prompt.options]
    done = results["confirmed"]
    assert executed(done) and done.rule == "P9.critical.confirmed"
    assert done.tool_calls[0].idempotency_key.startswith("idem_")
    assert done.tool_calls[0].arguments == {"from_account": "acc_7731", "to_account": "acc_2210", "amount": "250.00",
                                            "currency": "CHF"}  # fmt: skip
    after = results["toctou"]
    assert after.tool_calls == [] and after.outcome is not Outcome.EXECUTE
    assert "TOCTOU: constraint 'amount <= from_account.balance' no longer holds" in after.trace.notes


def test_04_files() -> None:
    results = run("04")
    hit = results["hit"]
    assert executed(hit) and hit.call is not None and hit.call.arguments == {"path": scripts.APP_YAML}
    assert hit.confidence is not None and hit.confidence.W == pytest.approx(0.71)
    miss = results["miss"]
    assert (miss.outcome, miss.rule, miss.rounds) == (Outcome.CLARIFY, "P7.slot.shape", 3)
    assert miss.prompt is not None and miss.prompt.kind == "open"
    bucket = results["bucket_hit"]
    assert executed(bucket) and bucket.rounds == 2


def test_05_calendar(capsys: pytest.CaptureFixture[str]) -> None:
    results = run("05")
    spec_request = load_fixture("spec_r5_request.json")
    assert json.dumps(results["request"], ensure_ascii=False) == json.dumps(spec_request, ensure_ascii=False)
    card = results["confirm"]
    assert (card.outcome, card.rule) == (Outcome.CONFIRM, "P9.external.confirm_band")
    assert card.confidence is not None and round(card.confidence.PI, 3) == 0.550
    clicked = results["clicked"]
    assert executed(clicked) and clicked.call is not None
    assert clicked.call.arguments["start"] == "2026-10-06T15:00:00+02:00"
    out = capsys.readouterr().out
    listing = out[out.index("the Jev request of round 1 as JSON") :].split("\n", 1)[1]
    assert json.loads(listing) == spec_request  # the printed JSON is the §13.5 request
    assert "14 questions, 6,003 characters compact" in out


def _to_options(request: DecisionRequest) -> dict[str, Any]:
    question = request.questions.get("send_email.to")
    return dict(question.criteria) if isinstance(question, ChoiceQuestion) and isinstance(question.criteria, dict) \
        else {}  # fmt: skip


def test_06_agent_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[DecisionRequest] = []
    original = ScriptedBackend.decide

    def spy(self: ScriptedBackend, request: DecisionRequest) -> Any:
        sent.append(request)
        return original(self, request)

    monkeypatch.setattr(ScriptedBackend, "decide", spy)
    results = run("06")
    result, workspace = results["run"], results["workspace"]
    assert (result.outcome, result.reason) == (Outcome.DONE, "done_after")
    assert result.usage.rounds == 2 and result.usage.executions == 2
    assert workspace.calls("read_file") == [{"path": scenario.INV_2291}]
    assert [m["to"] for m in workspace.sent] == ["finance@muster.ch"] and workspace.transfers == []
    assert scenario.INVOICE_TEXT in workspace.sent[0]["body"]
    refused = results["refused"]
    assert (refused.outcome, refused.reason) == (Outcome.REFUSE, "channel_blocked")
    assert results["refused_workspace"].transfers == [] and results["refused_workspace"].sent == []
    assert sent and all(scenario.INJECTED_ADDRESS not in json.dumps(_to_options(r)) for r in sent)
    assert not any(qid.startswith("transfer_funds.") for r in sent for qid in r.questions)


def test_07_openai_dropin() -> None:
    results = run("07")
    first, second = (doc["choices"][0] for doc in results["loop"])
    assert first["finish_reason"] == "tool_calls" and first["message"]["tool_calls"][0]["function"]["name"] == \
        "get_weather"  # fmt: skip
    assert second["finish_reason"] == "stop" and second["message"]["x_jev"]["outcome"] == "done"
    card, click = results["card"]
    assert card["choices"][0]["message"]["x_jev"]["outcome"] == "confirm"
    assert click["choices"][0]["finish_reason"] == "tool_calls" and click["usage"]["x_jev"]["jev_calls"] == 0
    arguments = json.loads(click["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert arguments["to"] == "anna.keller@acme.com"


def test_08_bring_your_own_tools() -> None:
    module = load("08")
    assert {t.name: t.tier.value for t in module.make_router(ScriptedBackend()).catalog} == {
        "assign_ticket": "write", "set_priority": "write", "search_kb": "read", "close_ticket": "write"}  # fmt: skip
    results = module.main(["--backend", "scripted"])
    assign, priority, search = (results[r] for r in module.REQUESTS)
    assert executed(assign) and assign.trace.resumed_from is not None  # clarify menu → click → execute
    assert assign.tool_calls[0].arguments == {"ticket_id": "T-1042", "assignee": "priya.nair"}
    assert executed(priority) and priority.tool_calls[0].arguments == {"ticket_id": "T-1043", "priority": "urgent"}
    assert executed(search) and search.tool_calls[0].arguments == {"query": "clear a paper jam"}


# --------------------------------------------------------------------------------------------------------------------
# sim: invariants only (the simulator is a lexical test double)
# --------------------------------------------------------------------------------------------------------------------


def _decisions(value: Any) -> Iterator[Decision]:
    if isinstance(value, Decision):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _decisions(item)
    elif hasattr(value, "decisions"):
        yield from value.decisions


@pytest.mark.parametrize("number", ["01", "02", "03", "04", "05", "06", "08"])
def test_sim_mode_runs_and_keeps_the_invariants(number: str, capsys: pytest.CaptureFixture[str]) -> None:
    results = run(number, "sim")
    decisions = list(_decisions(results))
    assert decisions
    for d in decisions:
        assert bool(d.tool_calls) == (d.outcome is Outcome.EXECUTE)
        if d.tool_calls and d.tool_calls[0].name == "transfer_funds":
            assert d.trace.resumed_from is not None  # a critical call runs only after the user's click
    assert "LexicalSimulator" in capsys.readouterr().out


def test_sim_outcomes_are_in_the_allowed_sets() -> None:
    r1 = run("01", "sim")
    assert r1["R1"].outcome is Outcome.EXECUTE and r1["R7"].outcome is Outcome.ABSTAIN
    r2 = run("02", "sim")
    assert r2["with_history"].outcome in {Outcome.CONFIRM, Outcome.CLARIFY}
    r3 = run("03", "sim")
    assert r3["confirm"].outcome in {Outcome.CONFIRM, Outcome.CLARIFY}
    r6 = run("06", "sim")
    workspace = r6["workspace"]
    assert workspace.transfers == [] and all(scenario.INJECTED_ADDRESS not in m["to"] for m in workspace.sent)
    docs = run("07", "sim")
    assert all("tool_calls" not in d["choices"][0]["message"] or d["choices"][0]["finish_reason"] == "tool_calls"
               for d in [*docs["loop"], *docs["card"]])  # fmt: skip


def test_live_without_a_key_exits_with_a_hint(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        run("01", "live")
    assert exc.value.code == 2 and "TYPESAFE_API_KEY" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------------------------------
# fixtures and the proxy example
# --------------------------------------------------------------------------------------------------------------------


def test_generated_files_are_up_to_date() -> None:
    regenerate = load_file(EXAMPLES / "fixtures" / "regenerate.py", "examples_regenerate")
    stale = [str(path) for path, content in regenerate.outputs().items()
             if not path.exists() or path.read_text(encoding="utf-8") != content]  # fmt: skip
    assert stale == [], "run: python examples/fixtures/regenerate.py"


@pytest.mark.parametrize("path", sorted((EXAMPLES / "fixtures").glob("*.answers.json")), ids=lambda p: p.name)
def test_fixtures_load(path: Path) -> None:
    backend = ScriptedBackend.from_fixture(path)
    assert backend.model == scenario.SCENARIO_MODEL
    assert "never evidence" in json.loads(path.read_text(encoding="utf-8"))["note"]


def test_scenario_tools_match_the_catalog_fixture() -> None:
    assert scenario.scenario_tools() == load_fixture("scenario_catalog.json")


def _proxy_script(request: DecisionRequest) -> Mapping[str, Any]:
    """R2 for the e-mail conversation; R1, then DONE once the weather came back."""
    tool = request.questions.get("tool")
    labels = list(tool.criteria) if isinstance(tool, ChoiceQuestion) and isinstance(tool.criteria, dict) else []
    if "send_email.to" in request.questions:
        return scripts.R2
    return scripts.R1_DONE if "DONE" in labels else scripts.R1


class _Message:
    def __init__(self, doc: dict[str, Any]) -> None:
        self.doc = doc

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return {k: v for k, v in self.doc.items() if not (exclude_none and v is None)}


class _Choice:
    def __init__(self, doc: dict[str, Any]) -> None:
        self.message = _Message(doc["message"])
        self.finish_reason = doc["finish_reason"]


class _StubOpenAI:
    """The part of ``openai.OpenAI`` the proxy client uses (when the SDK is not installed)."""

    def __init__(self, *, base_url: str, api_key: str, http_client: Any) -> None:
        self.base_url, self.http = base_url.rstrip("/"), http_client
        self.chat = self
        self.completions = self

    def create(self, *, extra_body: Mapping[str, Any] | None = None, **body: Any) -> Any:
        response = self.http.post(f"{self.base_url}/chat/completions", json={**body, **(extra_body or {})})
        response.raise_for_status()
        return type("Response", (), {"choices": [_Choice(c) for c in response.json()["choices"]]})()


def test_proxy_config_and_client(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient

    from jevtools.serve.app import create_app

    if importlib.util.find_spec("openai") is None:  # without the SDK, a stub of the part client.py uses
        stub = ModuleType("openai")
        stub.OpenAI = _StubOpenAI  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openai", stub)
    backend = ScriptedBackend(_proxy_script, model=scenario.SCENARIO_MODEL)
    app = create_app(EXAMPLES / "proxy" / "jevtools.toml", backend=backend)
    assert [s.name for s in app.state.jevtools.sources] == ["contacts", "accounts", "files"]
    client = load_file(EXAMPLES / "proxy" / "client.py", "examples_proxy_client")
    with TestClient(app) as http:
        assert http.get("/v1/models").json()["data"][0]["id"] == "jevtools"
        email, weather = client.main(["--base-url", "http://testserver/v1"], http_client=http)
    assert email[-1]["tool_calls"][0]["function"]["name"] == "send_email"
    assert json.loads(email[-1]["tool_calls"][0]["function"]["arguments"])["to"] == "anna.keller@acme.com"
    assert len(backend.requests) == 3  # the "1" reply to the card was a click: no Jev call
    assert [m["role"] for m in weather] == ["user", "assistant", "tool", "assistant"]
    assert weather[-1]["x_jev"]["outcome"] == "done"
    assert "x_jev.outcome = confirm" in capsys.readouterr().out

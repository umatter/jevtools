"""The ``jevtools`` command line (spec §7.5 lint, §3.9 explain/verify, §8.7 probe, §7.2.4 serve).

Traces are produced offline by scripted backends: nothing here is evidence about Jev's accuracy."""

from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from jevtools.cli import jaccard, lint, main
from jevtools.decision import Decision
from tests.adapters.support import SEARCH_TOOL, WEATHER_REQUEST, WEATHER_TOOL, weather_router
from tests.scenario import scripts
from tests.scenario.fixtures import scenario_messages, scenario_router
from tests.support import FIXTURES

CATALOG = FIXTURES / "scenario_catalog.json"
SOURCES = FIXTURES / "cli" / "sources.toml"


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def _line(output: str, name: str) -> str:
    return next(line for line in output.splitlines() if line.split()[:2] in ([name], ["TOOL", name])
                or line.strip().startswith(name + " "))  # fmt: skip


def _write(path: Path, doc: Any) -> str:
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------------------------------------------------
# lint
# --------------------------------------------------------------------------------------------------------------------


def test_lint_scenario_catalog_with_sources() -> None:
    code, out, _ = run("lint", str(CATALOG), "--sources", str(SOURCES))
    assert code == 0
    assert "tier=external (verb 'send')" in _line(out, "TOOL send_email")
    assert "tier=external (verb 'create' + invitee rule)" in _line(out, "TOOL create_event")
    assert "tier=critical (x-jev.risk)" in _line(out, "TOOL transfer_funds")
    to = _line(out, "send_email.to")
    assert "ref(contacts.email) | pattern(email)" in to and to.split()[-2:] == ["identity", "OK"]
    body = _line(out, "send_email.body")
    assert "text/content (email.body pack" in body and "WEAK  no Filler: uncovered bodies → clarify(open)" in body
    assert "ref(accounts.id)" in _line(out, "transfer_funds.from_account")
    assert _line(out, "transfer_funds.from_account").endswith("OK  channels=user,registry,author")
    assert "ref(files) bm25 k=40, hierarchy=dirname" in _line(out, "read_file.path")
    assert "text/cosmetic" in _line(out, "search_web.query")
    assert "DESCRIPTIONS" in out and "(warn ≥ 0.5)" in out
    assert out.strip().endswith("0 error(s), 0 warning(s), 1 weak")


def test_lint_without_sources_marks_generic_spans_weak_and_filler_covers_bodies() -> None:
    code, out, _ = run("lint", str(CATALOG), "--filler")
    assert code == 0
    assert "WEAK  row 18: generic string" in _line(out, "read_file.path")
    assert "WEAK" not in _line(out, "send_email.body")
    assert run("lint", str(CATALOG), "--strict")[0] == 1  # weak slots fail --strict


def _tool(name: str, description: str, properties: dict[str, Any], **extra: Any) -> dict[str, Any]:
    function = {"name": name, "description": description,
                "parameters": {"type": "object", "properties": properties}, **extra}  # fmt: skip
    return {"type": "function", "function": function}


def test_lint_applies_the_runtime_source_defaults(tmp_path: Path) -> None:
    """``type = "files"`` (the proxy spelling) without ``provides`` lints like the built FileIndex: path and file."""
    sources = tmp_path / "sources.toml"
    sources.write_text('[[sources]]\nname = "contacts"\ntype = "registry"\nprovides = ["email"]\n\n'
                       '[[sources]]\nname = "files"\ntype = "files"\n', encoding="utf-8")  # fmt: skip
    code, out, _ = run("lint", str(CATALOG), "--sources", str(sources))
    assert code == 0
    assert "ref(files) bm25 k=40, hierarchy=dirname" in _line(out, "read_file.path")
    assert "ref(contacts)" in _line(out, "send_email.to")


def test_lint_checks(tmp_path: Path) -> None:
    tools = [
        _tool("frobnicate", "Frobnicates the widget.", {"widget": {"type": "string", "description": "The widget"}}),
        _tool("fetch_page", "Fetch a web page by its address.",
              {"page": {"type": "string", "x-jev": {"source": "pages", "k": 200}, "description": "The page"}}),
        _tool("get_page", "Fetch a web page by its address quickly.",
              {"page": {"type": "string", "x-jev": {"kind": "ref"}}}),
        _tool("send_note", "Send a note.", {"note": {"type": "string", "x-jev": {"bogus": 1}}},
              **{"x-jev": {"frobnicate": True}}),
    ]  # fmt: skip
    code, out, _ = run("lint", _write(tmp_path / "tools.json", tools), "--sources",
                       _write(tmp_path / "sources.json", [{"name": "contacts", "provides": ["email"]}]))  # fmt: skip
    assert code == 1
    frob = _line(out, "TOOL frobnicate")
    assert "WARN" in frob and "declare x-jev.risk" in frob and "imperative 'frobnicate'" in frob
    fetch = _line(out, "fetch_page.page")
    assert "ERROR" in fetch and "source 'pages' is not registered" in fetch and "k=200 > 120" in fetch
    assert "ref slot without a source" in _line(out, "get_page.page")
    assert "WARN  no description" in _line(out, "get_page.page") or "no description" in _line(out, "get_page.page")
    assert "unknown x-jev key(s) 'frobnicate'" in out and "unknown x-jev key(s) 'bogus'" in out
    assert "TOOL send_note" in out and "does not compile" in out
    overlap = next(line for line in out.splitlines() if line.startswith("DESCRIPTIONS"))
    assert "'fetch_page' vs 'get_page'" in overlap and "WARN" in overlap


def test_lint_reads_mcp_tools_list_and_a_sidecar(tmp_path: Path) -> None:
    listing = json.loads((Path(__file__).parents[1] / "adapters" / "fixtures" / "mcp_tools_list.json").read_text())
    sidecar = _write(tmp_path / "jevtools.json", {"comment_issue": {"risk": "external", "intent": "comment on it"}})
    code, out, _ = run("lint", _write(tmp_path / "list.json", listing), "--sidecar", sidecar)
    assert "tier=read (annotation readOnlyHint)" in _line(out, "TOOL list_issues")
    assert "tier=critical (annotation destructiveHint)" in _line(out, "TOOL close_issue")
    assert "tier=external (x-jev.risk)" in _line(out, "TOOL comment_issue") and code == 0


def test_lint_function_and_jaccard() -> None:
    lines, counts = lint([WEATHER_TOOL, SEARCH_TOOL])
    assert counts["ERROR"] == 0 and any(line.startswith("TOOL get_weather") for line in lines)
    assert jaccard("Get the weather for a city.", "Get the weather for a town.") == pytest.approx(0.5)
    assert jaccard("", "") == 0.0


# --------------------------------------------------------------------------------------------------------------------
# explain / verify
# --------------------------------------------------------------------------------------------------------------------


def _trace_file(tmp_path: Path, decision: Decision, name: str = "trace.json") -> str:
    return _write(tmp_path / name, decision.trace.to_doc())


def test_explain_a_confirm(tmp_path: Path) -> None:
    router, _ = scenario_router(scripts.R2)
    decision = router.decide(scenario_messages(scripts.R2_REQUEST, history=True))
    code, out, _ = run("explain", _trace_file(tmp_path, decision))
    assert code == 0
    assert f"Decision {decision.decision_id}" in out
    assert "Outcome: confirm  —  rule P9.external.confirm_band" in out
    assert "Tool: send_email  p=0.96" in out
    assert "to = Anna Keller <anna.keller@acme.com>  p=0.86  channel=registry" in out
    assert "alternatives: Anna Rossi <anna.rossi@gmail.com> 0.07" in out
    assert "Composition: tier external uses PI" in out
    assert "thresholds (external): execute ≥ 0.8, confirm ≥ 0.5, hysteresis 0.03" in out
    assert "Rounds: 1 (1 Jev call(s)" in out


def test_explain_accepts_a_wrapped_trace_and_click_resumes(tmp_path: Path) -> None:
    router, _ = scenario_router(scripts.R2)
    decision = router.decide(scenario_messages(scripts.R2_REQUEST, history=True))
    done = router.resume(decision.pending_id or "", selection="ok")
    code, out, _ = run("explain", _write(tmp_path / "d.json", {"decision": done.to_doc(),
                                                                "trace": done.trace.to_doc()}))  # fmt: skip
    assert code == 0 and "Rounds: 0 (no Jev call)" in out and f"resumed from {decision.pending_id}" in out
    assert "idempotency key idem_" in out


def test_verify_with_catalog_and_context(tmp_path: Path) -> None:
    router, _ = weather_router()
    decision = router.decide(WEATHER_REQUEST)
    trace = _trace_file(tmp_path, decision)
    catalog = _write(tmp_path / "tools.json", [WEATHER_TOOL, SEARCH_TOOL])
    context = _write(tmp_path / "ctx.json", {"messages": [{"role": "user", "content": WEATHER_REQUEST}],
                                            "now": router.context.current_time().isoformat(), "locale": "en-CH",
                                            "tz": router.context.timezone_name,
                                            "user": router.context.user})  # fmt: skip
    code, out, _ = run("verify", trace, "--catalog", catalog, "--context", context)
    assert code == 0, out
    assert out.strip().endswith(f"VERIFIED {decision.trace_id}")
    assert "OK    redecode" in out and "OK    ballot_rebuild" in out and "OK    policy" in out
    code, out, _ = run("verify", trace)  # no catalog: re-decoding is skipped, not failed
    assert code == 0 and "SKIP  redecode" in out
    # a `now` with only a fixed UTC offset loads (zone "UTC+02:00"); it is another context, so the rebuild fails
    offset = _write(tmp_path / "offset.json", {"now": "2026-09-24T14:05:00+02:00"})
    code, out, err = run("verify", trace, "--catalog", catalog, "--context", offset)
    assert code == 1 and not err and "FAIL  ballot_rebuild" in out


def test_verify_detects_a_tampered_trace(tmp_path: Path) -> None:
    router, _ = weather_router()
    doc = router.decide(WEATHER_REQUEST).trace.to_doc()
    doc["rounds"][0]["calls"][0]["response"]["answers"]["get_weather.city"]["choice"] = "NOT_STATED"
    code, out, _ = run("verify", _write(tmp_path / "t.json", doc), "--catalog",
                       _write(tmp_path / "tools.json", [WEATHER_TOOL, SEARCH_TOOL]))  # fmt: skip
    assert code == 1 and "FAIL  hashes" in out and "FAILED" in out


def test_verify_with_sources_rebuilds_registries(tmp_path: Path) -> None:
    rows = [{"email": "anna@example.com", "name": "Anna Keller"}, {"email": "bob@example.com", "name": "Bob Meier"}]
    sources = _write(tmp_path / "sources.json", {"sources": [
        {"name": "contacts", "kind": "registry", "key": "email", "label": "{name} <{email}>", "match": ["name"],
         "provides": ["email", "person"], "rows": rows}]})  # fmt: skip
    from jevtools.cli import build_sources, load_source_specs

    (registry,) = build_sources(load_source_specs(sources))
    assert registry.name == "contacts" and len(registry.rows) == 2


# --------------------------------------------------------------------------------------------------------------------
# probe / serve / usage
# --------------------------------------------------------------------------------------------------------------------


def test_probe_reports_a_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "jevtools.probe", None)
    code, _, err = run("probe", "--backend", "simulator")
    assert code == 2 and "jevtools.probe is not available" in err


def test_probe_runs_against_the_simulator(tmp_path: Path) -> None:
    pytest.importorskip("jevtools.probe")
    code, out, _ = run("probe", "--backend", "simulator", "--path", str(tmp_path / "limits.json"), "--no-smoke")
    assert code == 0 and "backend simulator" in out and "limit label_max" in out
    assert (tmp_path / "limits.json").exists() and "never evidence about Jev" in out


def test_probe_without_configuration_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("JEVTOOLS_BACKEND", "TYPESAFE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    code, _, err = run("probe")
    assert code == 1 and "BackendConfigError" in err


def test_serve_dispatches_to_the_serve_module(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, Any] = {}
    fake = types.ModuleType("jevtools.serve")
    fake.run = lambda **kw: called.update(kw)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jevtools.serve", fake)
    assert run("serve", "--port", "9999")[0] == 0
    assert called == {"config": None, "host": "127.0.0.1", "port": 9999}


def test_serve_calls_the_real_serve_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("starlette")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app, **kw))
    config = tmp_path / "jevtools.toml"
    config.write_text('backend = "simulator"\nmodel_name = "jt"\n', encoding="utf-8")
    code, out, _ = run("serve", "--config", str(config), "--port", "9998")
    assert code == 0 and "http://127.0.0.1:9998/v1" in out
    assert seen["port"] == 9998 and seen["app"].state.jevtools.config.model_name == "jt"


def test_serve_reports_a_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "jevtools.serve", None)
    code, _, err = run("serve")
    assert code == 2 and "jevtools.serve is not available" in err


def test_eval_then_tune_on_the_simulator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``jevtools eval`` → report → ``jevtools tune`` → policy.toml (the simulator only exercises the plumbing)."""
    from jevtools.policy import Policy
    from tests.eval.test_dataset import write_dataset

    monkeypatch.delenv("JEVTOOLS_MODEL", raising=False)
    dataset = write_dataset(tmp_path)
    report = tmp_path / "report.json"
    code, out, err = run("eval", str(dataset), "--backend", "simulator", "--replays", "2", "--out", str(report))
    assert code == 0, err
    assert "1 case(s) x 2 replay(s) on simulator/lexical-simulator" in out and "wrong execution (all)" in out
    assert "never evidence about Jev" in out and report.exists()
    code, out, err = run("tune", str(report), "--out", str(tmp_path / "tuned"), "--alpha", "write=0.02")
    assert code == 0, err
    assert "critical auto-execution: not certified" in out and "policy.toml" in out
    tuned = Policy.from_toml(tmp_path / "tuned" / "policy.toml")
    assert "+tuned." in tuned.version
    assert run("tune", str(report), "--alpha", "bogus=1")[0] == 1


def test_usage_errors() -> None:
    assert run()[0] == 2
    code, _, err = run("lint", "/does/not/exist.json")
    assert code == 1 and "FileNotFoundError" in err

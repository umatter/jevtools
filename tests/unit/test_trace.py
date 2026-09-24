"""The resolution trace (spec §3.9): document form, hashes and model-out-of-the-loop verification."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from jevtools.backends.scripted import ScriptedBackend
from jevtools.policy import Policy
from jevtools.router import Router
from jevtools.spec.catalog import Catalog
from jevtools.trace import Trace, verify
from tests.stubs import R2_MESSAGES, R2_SCRIPT, resolvers, scenario_context, scenario_resolvers


@pytest.fixture
def decided(scenario_catalog: Catalog) -> Iterator[Any]:
    with resolvers(*scenario_resolvers(amount=())):
        router = Router(scenario_catalog, backend=ScriptedBackend(R2_SCRIPT), context=scenario_context())
        yield router.decide(R2_MESSAGES)


def test_trace_document(decided: Any) -> None:
    trace: Trace = decided.trace
    doc = trace.to_doc()
    assert list(doc)[:9] == ["spec", "trace_id", "decision_id", "created_at", "policy", "catalog_sha256", "context",
                             "ballot_sha256", "rounds"]  # fmt: skip
    assert doc["composition"] == {"W": 0.86, "PI": 0.7137, "L": 0.68, "J": None, "tier": "external", "rule": "PI",
                                  "calibrator": None, "C": 0.7137}  # fmt: skip
    assert doc["outcome"] == {"value": "confirm", "rule": "P9.external.confirm_band", "bottleneck": None,
                              "shape": None}  # fmt: skip
    assert doc["tool"]["chosen"] == "send_email" and doc["tool"]["p"] == 0.96
    call = doc["rounds"][0]["calls"][0]
    assert call["backend"] == "scripted" and call["request"]["questions"] and call["response"]["answers"]
    assert doc["bindings"]["body"]["value"].startswith("Hi Anna,") and doc["created_at"].endswith("Z")
    assert Trace.from_doc(trace.model_dump(mode="json")).to_doc() == doc
    assert trace.to_json().startswith(b'{"spec":"jevtools/0.1"')


def test_verify_passes_and_detects_tampering(decided: Any, scenario_catalog: Catalog) -> None:
    trace: Trace = decided.trace
    with resolvers(*scenario_resolvers(amount=())):
        assert verify(trace, catalog=scenario_catalog).ok
        assert verify(trace.to_doc()).ok  # hashes, composition and policy alone
        forged = trace.model_copy(deep=True)
        response = forged.rounds[0].calls[0].response
        assert response is not None
        response["answers"]["send_email.to"]["probabilities"]["Anna Keller <anna.keller@acme.com>"] = 0.99
        report = verify(forged, catalog=scenario_catalog)
        assert not report.ok and {c.name for c in report.failures} == {"hashes", "redecode"}
        rehashed = forged.model_copy(deep=True)
        from jevtools.canonical import sha256_of

        rehashed.rounds[0].calls[0].response_sha256 = sha256_of(rehashed.rounds[0].calls[0].response)
        assert {c.name for c in verify(rehashed, catalog=scenario_catalog).failures} == {"redecode"}
    wrong_rule = trace.model_copy(update={"outcome": {**trace.outcome, "rule": "P9.external.execute"}})
    assert {c.name for c in verify(wrong_rule).failures} == {"policy"}
    wrong_factors = trace.model_copy(update={"factors": {**trace.factors, "to": 0.5}})
    assert {c.name for c in verify(wrong_factors).failures} == {"composition"}
    other = verify(trace, policy=Policy.from_dict({"version": "custom"}))
    assert next(c for c in other.checks if c.name == "policy").ok is None


def test_hash_only_bodies(scenario_catalog: Catalog) -> None:
    with resolvers(*scenario_resolvers(amount=())):
        router = Router(
            scenario_catalog, backend=ScriptedBackend(R2_SCRIPT), context=scenario_context(), store_bodies="hash_only"
        )
        trace = router.decide(R2_MESSAGES).trace
    call = trace.rounds[0].calls[0]
    assert call.request is None and call.response is None and call.request_sha256.startswith("sha256:")
    report = verify(trace, catalog=scenario_catalog)
    assert report.ok and next(c for c in report.checks if c.name == "redecode").ok is None

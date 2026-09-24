"""The conformance probe (spec §8.7, live experiment E1): measure a backend's undocumented wire limits.

``run_probe(backend)`` sends a fixed series of small requests (19 against a permissive backend; well under $0.001
[I]) and writes a :class:`~jevtools.validate.Limits`-compatible JSON document to
``~/.cache/jevtools/limits-<backend>-<model>.json``, which :meth:`Limits.from_file
<jevtools.validate.Limits.from_file>` (and so the pre-send validator) reads.

=====================================  ============================================  ===================================
Probe                                  Measures                                      On failure
=====================================  ============================================  ===================================
qid charset                            ``a.b``, ``a_b``, ``a.b.m0``, ``x.accept.0``, ``id_mode="opaque"``
                                       a 128-char id (else 64)
label length                           64/128/256 (32, 16 below the default)         ``label_max``
label echo                             ``probabilities`` keys byte-equal to the      decode-map lookup normalized
                                       sent labels (whitespace, case, ``<email>``)   (``label_echo``)
unicode labels                         ``Zürich``, ``→``, ``⟨⟩``                     ASCII-fold (``ascii_labels``)
description / instructions length      400/2,000/8,000 chars                         ``desc_max`` / ``instr_max``
Choice with 1 option                   accepted?                                     keep ≥ 2 (sentinels guarantee it)
questions per call                     255, 400 (halving below 255)                  ``max_questions``
instructions as a JSON object          accepted?                                     ``instructions_as_object=False``
``GET /v1/models`` (TypeSafe)          available names                               informational
=====================================  ============================================  ===================================

A probe *fails* when the backend rejects the request (HTTP 400/422, a malformed body) or silently drops or retypes
an answer. Transport, authentication, rate-limit and availability errors are not limits: they abort the probe.

The probe also runs the functional smoke checks behind E2–E4 (§11.2) on 3 fixed items; their numbers are recorded
for information only and never change the limits. Run against an offline backend (the simulator), everything is
accepted and the smoke numbers say nothing about Jev.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jevtools._version import SPEC_VERSION
from jevtools.backends.base import Backend
from jevtools.backends.errors import (
    BackendError,
    JevAuthError,
    JevNotFound,
    JevRateLimited,
    JevUnavailable,
)
from jevtools.canonical import canonical_str, nfc, round4
from jevtools.errors import JevtoolsError
from jevtools.templates import NONE_OF_THESE_TEXT, NOT_STATED_TEXT
from jevtools.validate import Limits, cache_dir, cached_limits, limits_path
from jevtools.wire import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    JSONContent,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
    Question,
)

LabelEcho = Literal["exact", "nfc", "casefold", "none"]
"""How the backend echoes labels in ``probabilities``: byte-equal, equal after NFC, after casefold, or not at all."""

LABEL_SIZES: tuple[int, ...] = (64, 128, 256)
DESC_SIZES: tuple[int, ...] = (400, 2_000, 8_000)
INSTR_SIZES: tuple[int, ...] = (2_000, 8_000)
QUESTION_COUNTS: tuple[int, ...] = (255, 400)
LONG_QID = "p." + "x" * 126
"""A 128-character dotted id."""
MEDIUM_QID = "p." + "x" * 62
"""A 64-character dotted id."""
SHORT_QIDS: tuple[str, ...] = ("a.b", "a_b", "a.b.m0", "x.accept.0")
ECHO_LABELS: tuple[str, ...] = ("MiXeD Case", "two  spaces", "Bo <b@x.ch>", "a/b: c, d")
"""ASCII labels (≤ 16 characters, so a short ``label_max`` does not hide the echo) testing whitespace, case and an
``<email>`` form."""
UNICODE_LABELS: tuple[str, ...] = ("Zürich", "→ next", "⟨name⟩", "Genève")
STATE: dict[str, Any] = {"request": "This is a conformance probe of the jevtools client. Answer as you see fit."}
_TIMEOUT_ERRORS = (JevAuthError, JevNotFound, JevRateLimited, JevUnavailable)


class ProbeError(JevtoolsError):
    """The backend does not answer even a minimal request, so nothing can be measured."""


class ProbeCheck(BaseModel):
    """One probe request: what was measured, whether the backend accepted it, and why not."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    ok: bool
    detail: str = ""


class ProbeReport(BaseModel):
    """The probe's result: the measured :class:`Limits` plus the informational findings.

    :meth:`to_document` is the cache file (``Limits`` fields first, then the findings, which
    :meth:`Limits.from_file` ignores).
    """

    model_config = ConfigDict(extra="forbid")

    backend: str
    model: str
    limits: Limits
    label_echo: LabelEcho = "exact"
    ascii_labels: bool = False
    single_option_choice: bool = False
    models: list[str] | None = None
    calls: int = 0
    checks: list[ProbeCheck] = Field(default_factory=list)
    smoke: dict[str, Any] = Field(default_factory=dict)
    probed_at: str = ""
    path: str | None = None

    def check(self, name: str) -> ProbeCheck | None:
        """The check called ``name`` (the last one if repeated)."""
        return next((c for c in reversed(self.checks) if c.name == name), None)

    def to_document(self) -> dict[str, Any]:
        """The ``Limits``-compatible JSON document written to the cache."""
        return {
            **self.limits.model_dump(mode="json"),
            "spec": SPEC_VERSION,
            "backend": self.backend,
            "model": self.model,
            "probed_at": self.probed_at,
            "label_echo": self.label_echo,
            "ascii_labels": self.ascii_labels,
            "single_option_choice": self.single_option_choice,
            "models": self.models,
            "calls": self.calls,
            "checks": [c.model_dump() for c in self.checks],
            "smoke": self.smoke,
        }

    def write(self, path: str | os.PathLike[str]) -> Path:
        """Write :meth:`to_document` to ``path`` (parents created) and remember it in :attr:`path`."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(canonical_str(self.to_document()) + "\n", encoding="utf-8")
        self.path = str(target)
        return target


# ----------------------------------------------------------------------------------------------------------------
# the probe
# ----------------------------------------------------------------------------------------------------------------


class _Prober:
    """Sends probe requests and records the checks."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.calls = 0
        self.checks: list[ProbeCheck] = []

    def send(self, name: str, questions: Mapping[str, Question]) -> DecisionResponse | None:
        """Send one probe request; ``None`` (and a failed check) when the backend rejects it."""
        request = DecisionRequest(model=self.backend.model, state=STATE, questions=dict(questions))
        self.calls += 1
        try:
            response = self.backend.decide(request)
        except _TIMEOUT_ERRORS:
            raise
        except BackendError as exc:
            self.checks.append(ProbeCheck(name=name, ok=False, detail=_short(str(exc))))
            return None
        broken = [
            qid
            for qid, question in questions.items()
            if qid not in response.answers or response.answers[qid].type != question.type
        ]
        if broken:
            detail = f"missing or retyped answers: {', '.join(broken[:5])}" + (" …" if len(broken) > 5 else "")
            self.checks.append(ProbeCheck(name=name, ok=False, detail=detail))
            return None
        self.checks.append(ProbeCheck(name=name, ok=True))
        return response

    def accepts(self, name: str, questions: Mapping[str, Question]) -> bool:
        return self.send(name, questions) is not None

    def largest(self, name: str, sizes: Sequence[int], build: Callable[[int], Mapping[str, Question]],
                floor: int) -> int | None:  # fmt: skip
        """The largest accepted size: ascend ``sizes`` until a rejection; if the first is rejected, halve below
        it down to ``floor``. ``None`` when nothing was accepted."""
        best: int | None = None
        for size in sizes:
            if not self.accepts(f"{name}.{size}", build(size)):
                break
            best = size
        if best is not None:
            return best
        size = sizes[0] // 2
        while size >= floor:
            if self.accepts(f"{name}.{size}", build(size)):
                return size
            size //= 2
        return None


def run_probe(
    backend: Backend,
    *,
    path: str | os.PathLike[str] | None = None,
    directory: str | os.PathLike[str] | None = None,
    write: bool = True,
    smoke: bool = True,
) -> ProbeReport:
    """Run the conformance probe against ``backend`` and (by default) write the limits file.

    ``path`` overrides the file; ``directory`` overrides the cache directory. Raises :class:`ProbeError` when a
    minimal request is rejected, and re-raises authentication, not-found, rate-limit and availability errors.
    """
    prober = _Prober(backend)
    if not prober.accepts("baseline", {"probe": _choice({"yes": "Yes.", "no": "No."})}):
        detail = prober.checks[-1].detail
        raise ProbeError(f"{backend.name} ({backend.model}) rejects a minimal request: {detail}")

    id_mode, qid_max = _probe_qids(prober)
    label_max = prober.largest("label.length", LABEL_SIZES, lambda n: {"probe": _choice(_label_pair(n))}, 8)
    echo = _probe_echo(prober, "label.echo", ECHO_LABELS)
    unicode_echo = _probe_echo(prober, "label.unicode", UNICODE_LABELS)
    ascii_labels = unicode_echo is None
    desc_max = prober.largest(
        "description.length", DESC_SIZES, lambda n: {"probe": _choice({"a": _filler(n), "b": None})}, 50
    )
    instr_max = prober.largest(
        "instructions.length", INSTR_SIZES, lambda n: {"probe": NoulQuestion(instructions=_filler(n))}, 100
    )
    single = prober.accepts("choice.one_option", {"probe": _choice({"only": "The only option."})})
    max_questions = prober.largest("call.questions", QUESTION_COUNTS, _nouls, 1)
    as_object = prober.accepts(
        "instructions.object",
        {"probe": NoulQuestion(instructions={"question": "Is the candidate below a greeting?", "candidate": "Hello"})},
    )
    models = _models(backend, prober)
    smoke_numbers = _smoke(prober) if smoke else {}

    defaults = Limits()
    limits = Limits(
        label_max=label_max or defaults.label_max,
        desc_max=desc_max or defaults.desc_max,
        instr_max=instr_max or defaults.instr_max,
        max_questions=max_questions or defaults.max_questions,
        qid_max=qid_max,
        id_mode=id_mode,
        instructions_as_object=as_object,
    )
    report = ProbeReport(
        backend=backend.name,
        model=backend.model,
        limits=limits,
        label_echo=_worst_echo(echo, unicode_echo),
        ascii_labels=ascii_labels,
        single_option_choice=single,
        models=models,
        calls=prober.calls,
        checks=prober.checks,
        smoke=smoke_numbers,
        probed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    if write:
        report.write(path if path is not None else limits_path(backend.name, backend.model, directory))
    return report


# -- individual probes --------------------------------------------------------------------------------------------


def _choice(criteria: Mapping[str, JSONContent | None], instructions: str = "Which option fits `request` best?") -> (
        ChoiceQuestion):  # fmt: skip
    return ChoiceQuestion(instructions=instructions, criteria=dict(criteria))


def _filler(n: int) -> str:
    """``n`` characters of plain prose (no leading or trailing space)."""
    unit = "This sentence pads the probe text to a fixed length. "
    text = (unit * (n // len(unit) + 1))[:n]
    return text[:-1] + "." if text.endswith(" ") else text


def _label_pair(n: int) -> dict[str, JSONContent | None]:
    label = ("L" + "abcdefghij" * (n // 10 + 1))[:n]
    return {label: "A long label.", "b": "A short label."}


def _nouls(n: int) -> dict[str, Question]:
    return {f"q{i:04d}": NoulQuestion(instructions=f"Is statement {i} of the probe true?") for i in range(1, n + 1)}


def _probe_qids(prober: _Prober) -> tuple[Literal["dotted", "opaque"], int]:
    """Dotted ids with a 128-character id in one call; if rejected, the short ids alone, then a 64-character id."""

    def nouls(ids: Sequence[str]) -> dict[str, Question]:
        return {qid: NoulQuestion(instructions="Is this a probe?") for qid in ids}

    if prober.accepts("qid.charset", nouls([*SHORT_QIDS, LONG_QID])):
        return "dotted", 128
    if not prober.accepts("qid.dotted", nouls(SHORT_QIDS)):
        return "opaque", Limits().qid_max
    if prober.accepts("qid.length.64", nouls([MEDIUM_QID])):
        return "dotted", 64
    return "opaque", Limits().qid_max


def _probe_echo(prober: _Prober, name: str, labels: Sequence[str]) -> LabelEcho | None:
    """How the backend echoes ``labels``; ``None`` when it rejects them."""
    response = prober.send(name, {"probe": _choice({label: None for label in labels})})
    if response is None:
        return None
    answer = response.answers["probe"]
    assert isinstance(answer, ChoiceAnswer)
    keys = set(answer.probabilities)
    sent = set(labels)
    if sent <= keys:
        return "exact"
    if {nfc(k) for k in sent} <= {nfc(k) for k in keys}:
        return "nfc"
    if {nfc(k).casefold() for k in sent} <= {nfc(k).casefold() for k in keys}:
        return "casefold"
    return "none"


def _worst_echo(*echoes: LabelEcho | None) -> LabelEcho:
    order: tuple[LabelEcho, ...] = ("exact", "nfc", "casefold", "none")
    seen = [e for e in echoes if e is not None]
    return max(seen, key=order.index) if seen else "none"


def _models(backend: Backend, prober: _Prober) -> list[str] | None:
    """``GET /v1/models`` on a TypeSafe HTTP backend (informational; any failure gives ``None``)."""
    from jevtools.backends.http import HTTPBackend

    if not isinstance(backend, HTTPBackend) or backend.name != "typesafe" or "/v1/systemone" not in backend.url:
        return None
    url = backend.url.rsplit("/v1/systemone", 1)[0] + "/v1/models"
    try:
        response = backend.client.get(url, headers=backend._headers)
        payload = response.json() if response.is_success else None
    except Exception as exc:  # noqa: BLE001 - informational only
        prober.checks.append(ProbeCheck(name="models", ok=False, detail=_short(str(exc))))
        return None
    names = _model_names(payload)
    prober.checks.append(ProbeCheck(name="models", ok=names is not None,
                                    detail="" if names is not None else f"HTTP {response.status_code}"))  # fmt: skip
    return names


def _model_names(payload: Any) -> list[str] | None:
    items = payload.get("data", payload.get("models")) if isinstance(payload, Mapping) else payload
    if not isinstance(items, list):
        return None
    names: list[str] = []
    for item in items:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, Mapping):
            name = item.get("id", item.get("name"))
            if isinstance(name, str):
                names.append(name)
    return names


# -- E2–E4 smoke items ----------------------------------------------------------------------------------------------

KELLER = "Anna Keller <anna.keller@acme.com>"
ROSSI = "Anna Rossi <anna.rossi@gmail.com>"
MEIER = "Bob Meier <bob.meier@muster.ch>"
TO_ASK = (
    "Suppose the assistant will send an email from the user to one recipient to fulfil `request`. "
    "Which option is the recipient's email address?"
)
TO_PRESENT = (
    "Suppose the assistant will send an email from the user to one recipient to fulfil `request`. "
    "Does the user say or clearly imply the recipient's email address?"
)


def _slot(options: Mapping[str, str]) -> ChoiceQuestion:
    criteria: dict[str, JSONContent | None] = dict(options)
    criteria.update({"NOT_STATED": NOT_STATED_TEXT, "NONE_OF_THESE": NONE_OF_THESE_TEXT})
    return ChoiceQuestion(instructions=TO_ASK, criteria=criteria)


def _smoke_state(request: str) -> dict[str, Any]:
    return {"request": request, "history": [], "user": {"name": "Sam Muster"}}


def _smoke(prober: _Prober) -> dict[str, Any]:
    """E2 (sentinel coverage), E3 (masked mention) and E4 (option order) on one fixed item each. Informational."""
    keller = {KELLER: "Contact matching \"Anna Keller\": Account Manager at ACME.",
              ROSSI: "Contact similar to \"Anna Keller\": personal contact."}  # fmt: skip
    meier = {ROSSI: "Contact: personal contact.", MEIER: "Contact: Payments team."}
    out: dict[str, Any] = {}

    e2 = _send_state(prober, "smoke.e2", _smoke_state("Email Anna Keller that I'll be 10 minutes late"),
                     {"gold_present": _slot(keller), "gold_removed": _slot(meier)})  # fmt: skip
    if e2 is not None:
        present = _p(e2, "gold_present", "NONE_OF_THESE")
        removed = _p(e2, "gold_removed", "NONE_OF_THESE")
        out["e2"] = {"none_gold_present": present, "none_gold_removed": removed,
                     "passed": present < 0.10 and removed >= 0.30}  # fmt: skip

    e3 = _send_state(prober, "smoke.e3", _smoke_state("Email [someone] that I'll be 10 minutes late"),
                     {"masked": _slot(keller), "present": NoulQuestion(instructions=TO_PRESENT, criteria=NoulCriteria(
                         true="Yes, stated or clearly implied, possibly through `history`.",
                         false="No; it would have to be guessed."))})  # fmt: skip
    if e3 is not None:
        real = round4(_p(e3, "masked", KELLER) + _p(e3, "masked", ROSSI))
        present_answer = e3.answers["present"]
        noul = present_answer.noul if isinstance(present_answer, NoulAnswer) else 0.5
        out["e3"] = {"real_mass": real, "not_stated": _p(e3, "masked", "NOT_STATED"), "present": round4(noul)}

    forward = _slot(keller)
    reverse = _slot(dict(reversed(list(keller.items()))))
    e4 = _send_state(prober, "smoke.e4", _smoke_state("Email Anna that I'll be 10 minutes late"),
                     {"forward": forward, "reverse": reverse})  # fmt: skip
    if e4 is not None:
        top = [max(sorted(_probs(e4, q)), key=lambda k: _probs(e4, q)[k]) for q in ("forward", "reverse")]
        delta = max(abs(_p(e4, "forward", k) - _p(e4, "reverse", k)) for k in forward.criteria)
        out["e4"] = {"top_flip": top[0] != top[1], "max_abs_delta": round4(delta)}
    return out


def _send_state(prober: _Prober, name: str, state: dict[str, Any], questions: Mapping[str, Question]) -> (
        DecisionResponse | None):  # fmt: skip
    request = DecisionRequest(model=prober.backend.model, state=state, questions=dict(questions))
    prober.calls += 1
    try:
        response = prober.backend.decide(request)
    except _TIMEOUT_ERRORS:
        raise
    except BackendError as exc:
        prober.checks.append(ProbeCheck(name=name, ok=False, detail=_short(str(exc))))
        return None
    ok = all(qid in response.answers for qid in questions)
    prober.checks.append(ProbeCheck(name=name, ok=ok, detail="" if ok else "missing answers"))
    return response if ok else None


def _probs(response: DecisionResponse, qid: str) -> dict[str, float]:
    answer = response.answers[qid]
    return dict(answer.probabilities) if isinstance(answer, ChoiceAnswer) else {}


def _p(response: DecisionResponse, qid: str, label: str) -> float:
    return round4(float(_probs(response, qid).get(label, 0.0)))


def _short(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = [
    "DESC_SIZES",
    "INSTR_SIZES",
    "LABEL_SIZES",
    "QUESTION_COUNTS",
    "LabelEcho",
    "ProbeCheck",
    "ProbeError",
    "ProbeReport",
    "cache_dir",
    "cached_limits",
    "limits_path",
    "run_probe",
]

"""Scripted Jev answers reproducing the numbers of the spec §13.3 walk-through (all [I]: illustrative, not measured).

Each script maps qids (or glob patterns) to a :class:`~jevtools.backends.ScriptedBackend` answer. Labels are the
WYSIWYG labels the planner compiles for the §13 fixtures (``tests.scenario.fixtures``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jevtools.wire import ChoiceQuestion, DecisionRequest, NoulQuestion

R1_REQUEST = "What's the weather like in Zurich in Fahrenheit?"
R2_REQUEST = "Email Anna that I'll be 10 minutes late"
R3_REQUEST = "Move 250 CHF from my savings to checking"
R4_REQUEST = "Open the config file for the payments service"
R5_REQUEST = "Book a 45 min sync with Bob and Carol next Tuesday at 3pm"
R7_REQUEST = "Tell me a joke"

KELLER = "Anna Keller <anna.keller@acme.com>"
ROSSI = "Anna Rossi <anna.rossi@gmail.com>"
FREY = "Annabel Frey <annabel.frey@muster.ch>"
SAVINGS = "Savings · CHF · CH93…2957"
CHECKING = "Checking · CHF · CH56…1180"
TRAVEL = "Travel savings · EUR · CH08…4410"
APP_YAML = "services/payments/config/app.yaml"
PROD_YAML = "services/payments/config/prod.yaml"
TUE_29 = "Tue 2026-09-29 15:00 (Europe/Zurich)"
TUE_06 = "Tue 2026-10-06 15:00 (Europe/Zurich)"


def criteria(request: DecisionRequest, qid: str) -> dict[str, Any]:
    """The criteria of a Choice in a sent request: ``{label: description}`` in wire order."""
    question = request.questions[qid]
    assert isinstance(question, ChoiceQuestion) and isinstance(question.criteria, dict), qid
    return dict(question.criteria)


def accept_candidate(request: DecisionRequest, qid: str) -> str:
    """The candidate text of an accept Noul in a sent request."""
    question = request.questions[qid]
    assert isinstance(question, NoulQuestion) and isinstance(question.instructions, dict), qid
    return str(question.instructions["candidate"])


R1: dict[str, Any] = {
    "tool": {"get_weather": 0.98, "search_web": 0.01, "NO_TOOL": 0.005, "UNSUPPORTED": 0.005},
    "get_weather.city": {"Zurich": 0.95, "NOT_STATED": 0.02, "NONE_OF_THESE": 0.03},
    "get_weather.unit": {"fahrenheit": 0.97, "celsius": 0.01, "NOT_STATED": 0.01, "NONE_OF_THESE": 0.01},
    "search_web.query.accept.*": 0.3,
}
"""R1: tool .98; city "Zurich" .95 + NOT_STATED .02 (→ home city Zurich) = .97 pooled; unit fahrenheit .97."""

R2: dict[str, Any] = {
    "tool": {"send_email": 0.96, "get_weather": 0.01, "search_web": 0.01, "create_event": 0.005, "read_file": 0.005,
             "transfer_funds": 0.005, "NO_TOOL": 0.005, "UNSUPPORTED": 0.0},
    "get_weather.city": {"NOT_STATED": 0.97, "NONE_OF_THESE": 0.03},
    "get_weather.unit": {"NOT_STATED": 0.97, "celsius": 0.01, "fahrenheit": 0.01, "NONE_OF_THESE": 0.01},
    "search_web.query.accept.*": 0.2,
    "send_email.authorized": 0.95,
    "send_email.to": {KELLER: 0.86, ROSSI: 0.07, FREY: 0.03, "NOT_STATED": 0.01, "NONE_OF_THESE": 0.03},
    "send_email.to.present": 0.97,
    "send_email.subject.accept.0": 0.93, "send_email.subject.accept.1": 0.78, "send_email.subject.accept.2": 0.6,
    "send_email.body.accept.0": 0.91, "send_email.body.accept.1": 0.88,
}  # fmt: skip
"""R2 with history: tool .96, authorized .95, to Keller .86 / Rossi .07 / Frey .03, present .97, body template .91
(clause .88), subject "Running 10 minutes late" .93."""

R2_NO_HISTORY: dict[str, Any] = {
    **R2,
    "send_email.to": {KELLER: 0.47, ROSSI: 0.41, FREY: 0.06, "NOT_STATED": 0.03, "NONE_OF_THESE": 0.03},
}
"""R2 without history: Keller .47 / Rossi .41 / Frey .06 → Π = 0.39, ``to`` ambiguous."""

R3: dict[str, Any] = {
    "tool": {"transfer_funds": 0.98, "NO_TOOL": 0.01, "UNSUPPORTED": 0.01},
    "get_weather.city": {"NOT_STATED": 0.97, "NONE_OF_THESE": 0.03},
    "get_weather.unit": {"NOT_STATED": 0.97, "NONE_OF_THESE": 0.03},
    "search_web.query.accept.*": 0.2,
    "transfer_funds.authorized": 0.98,
    "transfer_funds.joint": {"250.00 CHF: Savings → Checking": 0.92, "250.00 CHF: Travel savings → Checking": 0.04,
                             "NONE_OF_THESE": 0.04},
    "transfer_funds.from_account": {SAVINGS: 0.95, TRAVEL: 0.04, "NONE_OF_THESE": 0.01},
    "transfer_funds.from_account.present": 0.97,
    "transfer_funds.from_account.rev": {SAVINGS: 0.96, TRAVEL: 0.04},
    "transfer_funds.to_account": {CHECKING: 0.97, SAVINGS: 0.01, "NONE_OF_THESE": 0.02},
    "transfer_funds.to_account.present": 0.98,
    "transfer_funds.to_account.rev": {CHECKING: 0.98, "NONE_OF_THESE": 0.02},
    "transfer_funds.amount": {"250.00": 0.99, "NONE_OF_THESE": 0.01},
    "transfer_funds.currency": {"CHF": 0.95, "EUR": 0.01, "NOT_STATED": 0.02, "NONE_OF_THESE": 0.02},
}  # fmt: skip
"""R3: tool .98, authorized .98, amount .99, currency CHF .95 + NOT_STATED .02 (→ CHF via Savings) = .97, from
Savings .95 (rev .96), to Checking .97, present .97/.98, joint J = .92."""

R4: dict[str, Any] = {
    "tool": {"read_file": 0.95, "search_web": 0.03, "get_weather": 0.01, "NO_TOOL": 0.005, "UNSUPPORTED": 0.005},
    "get_weather.city": {"NOT_STATED": 0.97, "NONE_OF_THESE": 0.03},
    "get_weather.unit": {"NOT_STATED": 0.97, "NONE_OF_THESE": 0.03},
    "search_web.query.accept.*": 0.3,
    "read_file.path": {APP_YAML: 0.71, PROD_YAML: 0.16, "services/billing/config/app.yaml": 0.03, "NOT_STATED": 0.05,
                       "NONE_OF_THESE": 0.05},
}  # fmt: skip
"""R4: tool .95; app.yaml .71, prod.yaml .16, NONE .05."""

R4_MISS: dict[str, Any] = {
    **R4,
    "read_file.path": {APP_YAML: 0.3, PROD_YAML: 0.2, "NOT_STATED": 0.1, "NONE_OF_THESE": 0.4},
    "read_file.path.bucket.*": {"NONE_OF_THESE": 0.97},
    "read_file.path.group": {"docs/runbooks": 0.9, "NONE_OF_THESE": 0.1},
}
"""R4 with NONE_OF_THESE .4: the buckets miss too, the group Choice points at ``docs/runbooks``, and the hierarchy
round (whose Choice reuses the ``read_file.path`` qid) finds nothing either."""


def r4_found_in_bucket(request: DecisionRequest) -> Mapping[str, Any]:
    """R4 with a round-1 miss whose file is the first option of bucket 0 (the other bucket says NONE_OF_THESE)."""
    answers: dict[str, Any] = dict(R4_MISS)
    if "read_file.path.bucket.0" in request.questions:
        first = next(iter(criteria(request, "read_file.path.bucket.0")))
        answers["read_file.path.bucket.0"] = {first: 0.9, "NONE_OF_THESE": 0.1}
    return answers


R5: dict[str, Any] = {
    "tool": {"create_event": 0.95, "get_weather": 0.004, "read_file": 0.003, "search_web": 0.01, "send_email": 0.02,
             "transfer_funds": 0.002, "NO_TOOL": 0.01, "UNSUPPORTED": 0.001},
    "create_event.authorized": 0.96,
    "create_event.title.accept.0": 0.81, "create_event.title.accept.1": 0.77, "create_event.title.accept.2": 0.94,
    "create_event.start": {TUE_29: 0.80, TUE_06: 0.18, "NOT_STATED": 0.01, "NONE_OF_THESE": 0.01},
    "create_event.duration_minutes": {"45": 0.96, "NOT_STATED": 0.02, "NONE_OF_THESE": 0.02},
    "create_event.attendees.m0": {"Bob Meier <bob.meier@muster.ch>": 0.88, "Robert Brown <rbrown@partner.io>": 0.09,
                                  "EXCLUDE": 0.01, "NONE_OF_THESE": 0.02},
    "create_event.attendees.m1": {"Carol Liu <carol.liu@muster.ch>": 0.93,
                                  "Caroline Weber <caroline.weber@muster.ch>": 0.05, "EXCLUDE": 0.01,
                                  "NONE_OF_THESE": 0.01},
    "create_event.attendees.more": 0.04,
    "get_weather.city": {"NOT_STATED": 0.97, "NONE_OF_THESE": 0.03},
    "get_weather.unit": {"celsius": 0.02, "fahrenheit": 0.01, "NOT_STATED": 0.96, "NONE_OF_THESE": 0.01},
    "search_web.query.accept.0": 0.35, "search_web.query.accept.1": 0.41,
}  # fmt: skip
"""R5: the §13.5 response excerpt."""

R7: dict[str, Any] = {
    "tool": {"NO_TOOL": 0.96, "search_web": 0.02, "get_weather": 0.01, "UNSUPPORTED": 0.01},
    "get_weather.city": {"NOT_STATED": 0.9, "NONE_OF_THESE": 0.1},
    "get_weather.unit": {"NOT_STATED": 0.95, "NONE_OF_THESE": 0.05},
    "search_web.query.accept.*": 0.3,
}
"""R7: NO_TOOL .96."""

__all__ = [
    "APP_YAML",
    "CHECKING",
    "FREY",
    "KELLER",
    "PROD_YAML",
    "R1",
    "R1_REQUEST",
    "R2",
    "R2_NO_HISTORY",
    "R2_REQUEST",
    "R3",
    "R3_REQUEST",
    "R4",
    "R4_MISS",
    "R4_REQUEST",
    "R5",
    "R5_REQUEST",
    "R7",
    "R7_REQUEST",
    "ROSSI",
    "SAVINGS",
    "TRAVEL",
    "TUE_06",
    "TUE_29",
    "accept_candidate",
    "criteria",
    "r4_found_in_bucket",
]

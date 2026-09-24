"""01 · Quickstart: two Python tools, zero sources, one Jev round per request (spec §13 R1 and R7).

What it shows:

- the 10-line setup: ``@jt.tool`` functions (an enum and a span), a :class:`jevtools.Context`, a :class:`Router`;
- span claiming: "Fahrenheit" is claimed by the ``unit`` enum, so ``city`` is offered only "Zurich";
- value pooling: ``city``'s ``NOT_STATED`` option means "the user's home city" (``jt.Default(ctx=…)``), which is also
  Zurich, so its mass pools into "Zurich" (.95 + .02 = .97 with the scripted answers);
- ``abstain`` for "Tell me a joke" (``NO_TOOL``), and both results as OpenAI assistant messages.

Run: ``python examples/01_quickstart_weather.py [--backend scripted|sim|live]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

import _show

import jevtools as jt
from jevtools.demo import scenario, scripts


# -- the whole setup -------------------------------------------------------------------------------------------------
@jt.tool  # tier inferred from the verb: read ("get")
def get_weather(
    city: Annotated[str, jt.Default(ctx="user.home_city")],
    unit: Literal["celsius", "fahrenheit"] = "celsius",
) -> dict[str, object]:
    """Get the current weather for a city."""
    return {"city": city, "unit": unit, "temp": 61 if unit == "fahrenheit" else 16}


@jt.tool  # read ("search")
def search_web(query: str) -> list[str]:
    """Search the public web."""
    return []


def make_router(backend: jt.backends.Backend) -> jt.Router:
    context = jt.Context(now=scenario.SCENARIO_NOW, locale="en-CH", user={"name": "Sam Muster", "home_city": "Zurich"})
    return jt.Router([get_weather, search_web], backend=backend, context=context)


# --------------------------------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> dict[str, jt.Decision]:
    args = _show.parse_args(__doc__, argv)
    _show.header("01 · quickstart: get_weather + search_web, zero sources (R1, R7)", args.backend, ["R1", "R7"])
    results: dict[str, jt.Decision] = {}

    _show.step(f'R1 "{scripts.R1_REQUEST}"   (spec [I]: execute, W = 0.97)')
    router = make_router(_show.backend(args.backend, "R1"))
    d = router.decide(scripts.R1_REQUEST)
    _show.decision(d)
    _show.note('"Fahrenheit" is claimed by `unit`, so `city` offers only "Zurich"; its NOT_STATED option means the '
               "user's home city (also Zurich), whose mass pools into Zurich")  # fmt: skip
    if d.tool_calls:  # the host runs the tool itself
        _show.kv("executed", f"get_weather(**arguments) → {get_weather(**d.tool_calls[0].arguments)}")
    _show.message(d.to_openai_message())
    results["R1"] = d

    _show.step(f'R7 "{scripts.R7_REQUEST}"   (spec [I]: abstain, NO_TOOL 0.96)')
    router = make_router(_show.backend(args.backend, "R7"))
    d = router.decide(scripts.R7_REQUEST)
    _show.decision(d)
    _show.message(d.to_openai_message())
    results["R7"] = d
    return results


if __name__ == "__main__":
    main()

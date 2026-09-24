"""The spec §2 10-line quickstart, run offline with the ScriptedBackend (zero sources; one enum and one span)."""

from __future__ import annotations

from typing import Literal

import jevtools as jt


def test_quickstart_runs_offline() -> None:
    @jt.tool  # tier inferred: read ("get")
    def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> dict[str, object]:
        """Get the current weather for a city."""
        return {"city": city, "unit": unit, "temp": 61}

    script = {"tool": "get_weather", "get_weather.city": "Zurich", "get_weather.unit": "fahrenheit"}
    router = jt.Router([get_weather], backend=jt.backends.ScriptedBackend(script, p_top=0.97))
    d = router.decide("What's the weather like in Zurich in Fahrenheit?")
    assert f"{d.outcome} {d.call}" == "execute get_weather(city='Zurich', unit='fahrenheit')"
    assert router.catalog.get("get_weather").tier is jt.Tier.READ
    assert jt.verify(d.trace, catalog=router.catalog).ok

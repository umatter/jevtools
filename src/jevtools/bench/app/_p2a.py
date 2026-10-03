"""Build the prompt2analytics catalog (``external/p2a/catalog.json``) from its MCP ``tools/list`` export.

prompt2analytics is a statistics server with 270 tools (regressions, tests, time series, ML, plots, data munging).
``external/p2a/tools_list.json`` is the server's tool list as exported; the cases, datasets and variables next to it
are hand-written. This script turns the export into a jevtools catalog, with bindings that follow fixed rules so they
are reproducible and checked by a test:

- ``dataset``, ``left`` and ``right`` (strings) bind to the ``datasets`` registry;
- a string or string-list parameter whose description mentions a column or variable binds to ``variables``
  (except free-text parameters such as ``path`` or ``query``). ``value`` binds too when its description says so: it
  names a column in ``hypothesis_oneway``, but ``munge_filter``'s "value to compare against (... column type)" also
  matches, so that tool's filter value is a variable slot (kept as benched; see docs/BENCH.md);
- ``dataset`` defaults to the dataset of the tool's first column parameter (``default_from``), so "a histogram of
  wages" needs no dataset named;
- data-changing tools (``munge_*``, imports, exports, sessions, queries) are ``write`` risk, the rest ``read``.

    python -m jevtools.bench.app._p2a           # (re)write catalog.json
    python -m jevtools.bench.app._p2a --check   # exit 1 when catalog.json is missing or stale
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
EXTERNAL_DIR = HERE / "external"
P2A_DIR = EXTERNAL_DIR / "p2a"

WRITES = frozenset({
    "batch_process", "cleaning_rollback", "cleaning_session_apply", "cleaning_session_start", "create_dataset",
    "db_duckdb_query", "db_query_file", "db_sqlite_query", "export_dataset", "export_session", "generate_report",
    "import_session", "load_dataset", "set_seed", "upload_dataset",
})  # fmt: skip
"""Tools that change data or state beyond the ``munge_*`` family."""
DATASET_PARAMS = ("dataset", "left", "right")
FREE_TEXT = frozenset({"result_name", "path", "title", "pattern", "query"})
_COLUMN = re.compile(r"\bcolumns?\b|\bvariables?\b")


def _strip(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    return {k: _strip(v) if isinstance(v, dict) else v for k, v in schema.items() if k not in ("$schema", "title")}


def _bind(params: dict[str, Any]) -> None:
    props = params.get("properties") or {}
    for name, p in props.items():
        items = p.get("items") if isinstance(p.get("items"), dict) else {}
        flat = p.get("type") == "string" or (p.get("type") == "array" and items.get("type") == "string")
        if name in DATASET_PARAMS and p.get("type") == "string":
            p["x-jev"] = {"source": "datasets"}
        elif flat and name not in FREE_TEXT and _COLUMN.search((p.get("description") or "").lower()):
            p["x-jev"] = {"source": "variables"}
    dataset = props.get("dataset", {})
    if dataset.get("x-jev", {}).get("source") != "datasets":
        return

    def column(k: str) -> bool:
        p = props.get(k, {})
        return bool(p.get("x-jev", {}).get("source") == "variables" and p.get("type") == "string")

    columns = [k for k in params.get("required") or [] if column(k)]
    columns += [k for k in props if k not in columns and column(k)]
    if columns:
        dataset["x-jev"]["default_from"] = f"{columns[0]}.dataset"


def catalog(tools: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The jevtools catalog of an MCP tool list (see the module docstring)."""
    out = []
    for t in tools:
        params = _strip(t["inputSchema"])
        params.setdefault("type", "object")
        _bind(params)
        risk = "write" if t["name"] in WRITES or t["name"].startswith("munge_") else "read"
        out.append({"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                     "parameters": params, "x-jev": {"risk": risk}}})  # fmt: skip
    return out


def build() -> str:
    tools = json.loads((P2A_DIR / "tools_list.json").read_text(encoding="utf-8"))
    return json.dumps(catalog(tools), indent=1)


def main(argv: Iterable[str] | None = None) -> int:
    check = "--check" in (list(argv) if argv is not None else sys.argv[1:])
    path, text = P2A_DIR / "catalog.json", build()
    if check:
        stale = not path.is_file() or path.read_text(encoding="utf-8") != text
        print(f"{path} is " + ("stale" if stale else "up to date"))
        return 1 if stale else 0
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

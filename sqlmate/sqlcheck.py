"""Read-only check for SQL (the same rules as DataChat's guard, without its row-limit rewriting)."""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

ALLOWED_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)
FORBIDDEN = tuple(getattr(exp, n) for n in ("Insert", "Update", "Delete", "Drop", "Create", "Alter", "Command",
                                             "Pragma", "Set", "Copy", "Attach", "Detach", "Merge", "TruncateTable")
                  if hasattr(exp, n))


def is_read_only_select(sql: str, tables: set[str] | None = None) -> bool:
    """True for exactly one read-only SELECT that only uses the given tables (and its own CTEs)."""
    try:
        parsed = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except (ParseError, ValueError):
        return False
    if len(parsed) != 1 or not isinstance(parsed[0], ALLOWED_ROOTS):
        return False
    tree = parsed[0]
    if any(isinstance(n, FORBIDDEN) for n in tree.walk()):
        return False
    if tables is not None:
        ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
        for t in tree.find_all(exp.Table):
            if not t.name or (t.name.lower() not in tables and t.name.lower() not in ctes):
                return False
    return True

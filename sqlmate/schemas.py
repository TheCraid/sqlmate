"""Databases and how a schema is described to the model.

The same `describe()` text is used for the DataChat databases and for the public dataset, so the model
sees one consistent schema format in training and in evaluation: CREATE TABLE statements with a short
comment per column (its meaning when known, and example values or the min/max range).
"""

from __future__ import annotations

import csv
import tempfile
import threading
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import duckdb

from sqlmate import datachat_samples

NUMERIC = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "FLOAT", "DOUBLE", "REAL", "DECIMAL", "UTINYINT",
           "USMALLINT", "UINTEGER", "UBIGINT")


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _short(v) -> str:
    if isinstance(v, (datetime, date)):
        return v.isoformat()[:10]
    if isinstance(v, (float, Decimal)):
        return f"{float(v):g}"
    return str(v)


def describe(conn: duckdb.DuckDBPyConnection, descriptions: dict[str, dict[str, str]] | None = None,
             max_examples: int = 3) -> str:
    """CREATE TABLE text for every table, with one comment per column."""
    descriptions = descriptions or {}
    parts = []
    tables = [r[0] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name").fetchall()]
    for t in tables:
        cols = conn.execute(f"DESCRIBE {_q(t)}").fetchall()
        lines = []
        for name, ctype, *_ in cols:
            ctype = str(ctype)
            notes = []
            if descriptions.get(t, {}).get(name):
                notes.append(descriptions[t][name])
            if ctype.upper().startswith(NUMERIC) or ctype.upper().startswith(("DATE", "TIMESTAMP")):
                lo, hi = conn.execute(f"SELECT MIN({_q(name)}), MAX({_q(name)}) FROM {_q(t)}").fetchone()
                if lo is not None:
                    notes.append(f"range {_short(lo)} to {_short(hi)}")
            elif ctype.upper() == "VARCHAR":
                vals = conn.execute(
                    f"SELECT {_q(name)} FROM {_q(t)} WHERE {_q(name)} IS NOT NULL GROUP BY 1 "
                    f"ORDER BY COUNT(*) DESC, 1 LIMIT {max_examples}").fetchall()
                if vals:
                    notes.append("e.g. " + ", ".join("'" + str(v[0])[:30].replace("'", "''") + "'" for v in vals))
            comment = f" -- {'; '.join(notes)}" if notes else ""
            lines.append(f"  {name} {ctype}{comment}")
        body = ",\n".join(lines)
        # Commas go before comments, so move them: "col TYPE -- note," -> "col TYPE, -- note"
        body = "\n".join(_comma_before_comment(line) for line in body.split("\n"))
        parts.append(f"CREATE TABLE {t} (\n{body}\n);")
    return "\n\n".join(parts)


def _comma_before_comment(line: str) -> str:
    if line.endswith(",") and " -- " in line:
        head, note = line[:-1].split(" -- ", 1)
        return f"{head}, -- {note}"
    return line


# --------------------------------------------------------------------------- DataChat databases

def load_datachat(dataset_id: str) -> tuple[duckdb.DuckDBPyConnection, str]:
    """Load one DataChat sample dataset into DuckDB; return the connection and its schema text."""
    sample = datachat_samples.BUILDERS[dataset_id]()
    conn = duckdb.connect(":memory:")
    with tempfile.TemporaryDirectory() as tmp:
        for t in sample.tables:
            path = Path(tmp) / f"{t.name}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([c[0] for c in t.columns])
                for r in t.rows:
                    w.writerow(["" if v is None else (v.isoformat() if isinstance(v, (date, datetime)) else v)
                                for v in r])
            types = ", ".join(f"'{c[0]}': '{c[1]}'" for c in t.columns)
            conn.execute(f"CREATE TABLE {t.name} AS SELECT * FROM read_csv(?, header = true, "
                         f"columns = {{{types}}}, nullstr = '')", [str(path)])
    descriptions = {t.name: {c[0]: c[2] for c in t.columns} for t in sample.tables}
    lock(conn)
    return conn, describe(conn, descriptions)


def lock(conn: duckdb.DuckDBPyConnection) -> None:
    """No file or network access from queries, and settings cannot be changed afterwards."""
    conn.execute("SET enable_external_access = false")
    conn.execute("SET lock_configuration = true")


def run(conn: duckdb.DuckDBPyConnection, sql: str, max_rows: int = 1000, timeout_s: float = 10.0) -> list[list]:
    """Run a query with a timeout; returns at most max_rows rows. Raises duckdb.Error on failure."""
    cur = conn.cursor()
    timer = threading.Timer(timeout_s, cur.interrupt)
    timer.start()
    try:
        rows = cur.execute(sql).fetchmany(max_rows)
    finally:
        timer.cancel()
        cur.close()
    return [list(r) for r in rows]

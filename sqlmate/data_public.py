"""Build training and test examples from the public gretelai/synthetic_text_to_sql dataset (Apache 2.0).

The dataset is written in MySQL. Every example is converted to DuckDB with sqlglot and kept only if it
actually works: its tables load, its query is one read-only SELECT, runs, and returns at least one
non-empty row. Queries that depend on today's date (CURDATE(), NOW() ...) are dropped, because their
answers change over time.

    python -m sqlmate.data_public                     # 8,000 training + 200 test examples
    python -m sqlmate.data_public --train 3000 --test 100
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

import duckdb
import sqlglot

from sqlmate.prompt import to_example
from sqlmate.schemas import describe, run
from sqlmate.sqlcheck import is_read_only_select

DATA = Path(__file__).resolve().parents[1] / "data"
TASKS = ("analytics and reporting", "data retrieval")
NOW_LIKE = re.compile(r"\b(curdate|curtime|current_date|current_timestamp|now|getdate|sysdate|today|localtime)\b", re.I)
MAX_SCHEMA_CHARS = 3000
# Keys and auto-increment are irrelevant for analytics questions and often break the conversion.
CONSTRAINTS = [re.compile(p, re.I) for p in (
    r",\s*(CONSTRAINT\s+\w+\s+)?(PRIMARY|FOREIGN|UNIQUE)\s+KEY\s*\w*\s*\([^)]*\)(\s*REFERENCES\s+\w+\s*\([^)]*\))?",
    r"\bREFERENCES\s+\w+\s*\([^)]*\)",
    r"\bPRIMARY\s+KEY\b", r"\bAUTO_INCREMENT\b", r"\bUNIQUE\b",
)]
logging.getLogger("sqlglot").setLevel(logging.ERROR)


def strip_constraints(context: str) -> str:
    for pat in CONSTRAINTS:
        context = pat.sub("", context)
    return context


def norm_question(q: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", q.lower()))


def convert(record: dict) -> dict | None:
    """Return a DuckDB example for one dataset record, or None if it cannot be used."""
    if not any(t in str(record.get("sql_task_type", "")).lower() for t in TASKS):
        return None
    if NOW_LIKE.search(record["sql"]) or NOW_LIKE.search(record["sql_context"]):
        return None
    try:
        context = strip_constraints(record["sql_context"])
        setup = [s for s in sqlglot.transpile(context, read="mysql", write="duckdb") if s.strip()]
        query = sqlglot.transpile(record["sql"], read="mysql", write="duckdb")
    except Exception:  # sqlglot raises several error types for unusual MySQL
        return None
    if len(query) != 1:
        return None
    sql = query[0]
    conn = duckdb.connect(":memory:")
    try:
        for stmt in setup:
            conn.execute(stmt)
        tables = {r[0].lower() for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()}
        if not tables or not is_read_only_select(sql, tables):
            return None
        rows = run(conn, sql, timeout_s=5)
        if not rows or all(v is None for r in rows for v in r):
            return None
        schema = describe(conn)
    except duckdb.Error:
        return None
    finally:
        conn.close()
    if len(schema) > MAX_SCHEMA_CHARS:
        return None
    ex = to_example(schema, record["sql_prompt"], sql, source="gretel", source_id=int(record["id"]),
                    complexity=record.get("sql_complexity", ""), domain=record.get("domain", ""))
    ex["setup_sql"] = setup
    return ex


def collect(records, limit: int, seed: int, exclude: set[str] | None = None) -> list[dict]:
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    out, seen = [], set(exclude or ())
    for i in order:
        rec = records[i]
        key = norm_question(rec["sql_prompt"])
        if key in seen:
            continue
        ex = convert(rec)
        if ex is None:
            continue
        seen.add(key)
        out.append(ex)
        if len(out) % 500 == 0:
            print(f"  {len(out)} kept after {i + 1} checked")
        if len(out) >= limit:
            break
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from datasets import load_dataset  # imported here so the rest of the package works without it

    ds = load_dataset("gretelai/synthetic_text_to_sql")
    print(f"Loaded {len(ds['train']):,} train and {len(ds['test']):,} test records")
    print("Test split:")
    test = collect(ds["test"], args.test, args.seed)
    print("Train split:")
    train = collect(ds["train"], args.train, args.seed, exclude={norm_question(t["prompt"][1]["content"]
                                                                               .split("Question: ", 1)[1])
                                                                 for t in test})
    write_jsonl(DATA / "public_train.jsonl", train)
    write_jsonl(DATA / "public_test.jsonl", test)
    print(f"Wrote {len(train):,} training and {len(test):,} test examples to {DATA}")


if __name__ == "__main__":
    main()

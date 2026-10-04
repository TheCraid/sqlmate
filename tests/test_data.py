import json
from pathlib import Path

import duckdb

from sqlmate import build_dataset
from sqlmate.data_public import collect, convert, strip_constraints
from sqlmate.data_synthetic import Validator, jaccard, parse_items, tokens
from sqlmate.schemas import describe, load_datachat

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ("CREATE TABLE sales (id INT PRIMARY KEY AUTO_INCREMENT, region VARCHAR(20), amount DECIMAL(10,2)); "
           "INSERT INTO sales (region, amount) VALUES ('North', 100.50), ('South', 200.00), ('North', 50.25);")


def record(**kw):
    base = {"id": 1, "sql_task_type": "analytics and reporting", "sql_prompt": "Total sales by region?",
            "sql_context": CONTEXT, "sql": "SELECT region, SUM(amount) FROM sales GROUP BY region;",
            "sql_complexity": "aggregation", "domain": "retail"}
    base.update(kw)
    return base


def test_strip_constraints():
    out = strip_constraints("CREATE TABLE a (id INT PRIMARY KEY AUTO_INCREMENT, b INT REFERENCES c(id), "
                            "UNIQUE KEY u (b))")
    assert "PRIMARY" not in out and "AUTO_INCREMENT" not in out and "REFERENCES" not in out and "UNIQUE" not in out


def test_convert_keeps_working_example():
    ex = convert(record())
    assert ex is not None
    assert ex["source"] == "gretel" and ex["source_id"] == 1 and ex["setup_sql"]
    assert "CREATE TABLE sales" in ex["prompt"][1]["content"]
    conn = duckdb.connect()
    for s in ex["setup_sql"]:
        conn.execute(s)
    assert len(conn.execute(ex["completion"][0]["content"]).fetchall()) == 2


def test_convert_drops_bad_records():
    assert convert(record(sql_task_type="data manipulation")) is None
    assert convert(record(sql="SELECT * FROM sales WHERE amount > 0 AND CURDATE() > '2020-01-01'")) is None
    assert convert(record(sql="SELECT * FROM missing_table")) is None
    assert convert(record(sql="SELECT * FROM sales WHERE amount > 1000")) is None  # empty result
    assert convert(record(sql="DELETE FROM sales")) is None


def test_collect_dedupes_and_excludes():
    recs = [record(id=1), record(id=2), record(id=3, sql_prompt="Average sale amount?",
                                                  sql="SELECT AVG(amount) FROM sales")]
    out = collect(recs, limit=10, seed=1)
    assert len(out) == 2
    out = collect(recs, limit=10, seed=1, exclude={"total sales by region"})
    assert [e["source_id"] for e in out] == [3]


def test_describe_comments():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t (name VARCHAR, n INTEGER)")
    conn.execute("INSERT INTO t VALUES ('a', 1), ('b', 5)")
    text = describe(conn, {"t": {"n": "number of things"}})
    assert "CREATE TABLE" in text and "number of things" in text and "'a'" in text


def test_datachat_schemas_load():
    for ds in ("store", "food", "hr"):
        conn, schema = load_datachat(ds)
        assert schema.count("CREATE TABLE") >= 2
        assert conn.execute("SELECT 1").fetchone() == (1,)


def test_validator_rules():
    lines = (ROOT / "eval" / "datachat_questions.jsonl").read_text(encoding="utf-8").splitlines()
    bench = [json.loads(x) for x in lines if x.strip()]
    conn, _ = load_datachat("store")
    v = Validator("store", conn, bench)
    tables = sorted(v.tables)
    first = tables[0]
    assert not v.check("", "SELECT 1")
    assert not v.check("Remove every row please", f"DELETE FROM {first}")
    assert not v.check("Rows from a table that does not exist", "SELECT * FROM nope")
    store_q = next(q for q in bench if q["dataset"] == "store")
    assert not v.check(store_q["question"], store_q["sql"])  # benchmark question itself is rejected
    good = f"SELECT COUNT(*) * 1000 + 7 AS odd_number FROM {first}"
    assert v.check("An unusual arithmetic figure derived from the row total", good)
    assert not v.check("An unusual arithmetic figure derived from the row total", good)  # duplicate
    assert v.rejected["duplicate"] == 1


def test_parse_items_and_jaccard():
    assert parse_items('noise {"items": [{"question": "q", "sql": "s"}, 3]} trailing') == [
        {"question": "q", "sql": "s"}]
    assert parse_items("not json") == []
    assert jaccard(tokens("total revenue by city"), tokens("What is the total revenue by city?")) == 1.0


def test_build_dataset(tmp_path, monkeypatch):
    ex = {"prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
          "completion": [{"role": "assistant", "content": "SELECT 1"}]}
    (tmp_path / "public_train.jsonl").write_text(json.dumps({**ex, "source": "gretel"}) + "\n")
    (tmp_path / "synthetic_train.jsonl").write_text(json.dumps({**ex, "source": "synthetic"}) + "\n")
    monkeypatch.setattr(build_dataset, "DATA", tmp_path)
    monkeypatch.setattr("sys.argv", ["build_dataset", "--synthetic-repeat", "3"])
    build_dataset.main()
    rows = [json.loads(x) for x in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and sum(r["source"] == "synthetic" for r in rows) == 3

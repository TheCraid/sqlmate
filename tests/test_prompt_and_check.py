from sqlmate.prompt import SYSTEM, build_messages, extract_sql, to_example
from sqlmate.sqlcheck import is_read_only_select


def test_build_messages_and_example():
    msgs = build_messages("CREATE TABLE t (a INTEGER);", "How many rows?")
    assert msgs[0] == {"role": "system", "content": SYSTEM}
    assert msgs[1]["content"].endswith("Question: How many rows?")
    ex = to_example("CREATE TABLE t (a INTEGER);", "How many rows?", "SELECT COUNT(*) FROM t;", source="x")
    assert ex["completion"] == [{"role": "assistant", "content": "SELECT COUNT(*) FROM t"}]
    assert ex["source"] == "x"


def test_extract_sql_variants():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("<think>hmm SELECT nothing</think>SELECT a FROM t") == "SELECT a FROM t"
    assert extract_sql("Here is the query: SELECT a FROM t; SELECT b FROM t") == "SELECT a FROM t"
    assert extract_sql("SELECT ';' AS x FROM t; DROP TABLE t") == "SELECT ';' AS x FROM t"
    assert extract_sql("WITH c AS (SELECT 1 AS x) SELECT x FROM c") == "WITH c AS (SELECT 1 AS x) SELECT x FROM c"
    assert extract_sql("") == ""
    assert extract_sql("Sorry, I cannot answer that.") == ""


def test_read_only_check():
    tables = {"orders"}
    assert is_read_only_select("SELECT * FROM orders", tables)
    assert is_read_only_select("WITH x AS (SELECT * FROM orders) SELECT * FROM x", tables)
    assert not is_read_only_select("SELECT * FROM customers", tables)
    assert not is_read_only_select("DELETE FROM orders", tables)
    assert not is_read_only_select("SELECT 1; SELECT 2", tables)
    assert not is_read_only_select("not sql at all (", tables)

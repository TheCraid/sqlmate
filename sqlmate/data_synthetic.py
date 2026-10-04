"""Generate in-domain training examples for the DataChat databases with a large model, then verify them.

A large model (gpt-oss-120b on Groq) writes question and SQL pairs about each DataChat database.
An example is kept only if:
  * its SQL is one read-only SELECT on the database's own tables,
  * it runs on the real data and returns at least one non-empty row,
  * it is not a near-duplicate of another generated example, and
  * it is decontaminated against the 60 DataChat benchmark questions: dropped if its wording overlaps
    a benchmark question (token Jaccard >= 0.5) or if it returns the same answer as a benchmark query.

    python -m sqlmate.data_synthetic                    # 250 examples per database
    python -m sqlmate.data_synthetic --model openai/gpt-oss-20b   # if the 120b daily limit is used up

Groq's free tier allows about 200,000 tokens per model per day. Examples are saved as they are made,
so if the daily limit is reached the script stops, and running the same command later continues from there.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path

import duckdb

from sqlmate.compare import results_match
from sqlmate.llm_api import APIError, ChatClient, QuotaExhausted
from sqlmate.prompt import to_example
from sqlmate.schemas import load_datachat, run
from sqlmate.sqlcheck import is_read_only_select

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BENCHMARK = ROOT / "eval" / "datachat_questions.jsonl"

FOCUSES = [
    "totals, counts and averages broken down by one category",
    "top or bottom N rankings",
    "trends by month, quarter or year",
    "joins across two or three tables",
    "percentages, shares and rates (0-100)",
    "filters that combine several conditions",
    "comparisons between two periods or two groups",
    "window functions such as ranks within groups or running totals",
    "distinct counts and repeat behaviour",
    "questions about one specific value, entity or date range",
]

STOP = set("""a an the of in on for to by and or is are was were what which who how many much per each with
from show me give list find all their there that this do does did than as at be it its""".split())  # noqa: SIM905

PROMPT = """Schema of a DuckDB database:

{schema}

Write {n} different questions a business user might ask about this data, each with one DuckDB SQL query
that answers it correctly. Focus of this batch: {focus}.

Rules:
- Natural, varied wording: some short, some detailed. Do not number the questions.
- The SQL must be read-only (SELECT, optionally WITH), use only the tables and columns above, and run on DuckDB.
- When grouping by an entity (customer, rider, restaurant, employee, product), group by its id and show the name.
- Round money and averages to 2 decimals and percentages to 1 decimal (0-100).
- Relative dates such as "last quarter" are relative to the latest date in the data.
- Avoid these questions, which are already covered: {avoid}

Reply with only JSON: {{"items": [{{"question": "...", "sql": "..."}}]}}"""


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def norm_sql(sql: str) -> str:
    return " ".join(sql.lower().replace(";", " ").split())


class Validator:
    """Checks generated examples for one database against all the keep rules."""

    def __init__(self, dataset_id: str, conn: duckdb.DuckDBPyConnection, benchmark: list[dict],
                 jaccard_limit: float = 0.5):
        self.dataset_id, self.conn = dataset_id, conn
        self.tables = {r[0].lower() for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()}
        self.bench = [(tokens(q["question"]), run(conn, q["sql"])) for q in benchmark if q["dataset"] == dataset_id]
        self.jaccard_limit = jaccard_limit
        self.seen_q: set[str] = set()
        self.seen_sql: set[str] = set()
        self.rejected: Counter = Counter()

    def remember(self, question: str, sql: str) -> None:
        """Mark an example kept in an earlier run as seen, so it is not generated twice."""
        self.seen_q.add(" ".join(sorted(tokens(question))))
        self.seen_sql.add(norm_sql(sql))

    def check(self, question: str, sql: str) -> bool:
        question, sql = (question or "").strip(), (sql or "").strip().rstrip(";")
        if len(question) < 8 or not sql:
            return self._reject("empty")
        qkey, skey = " ".join(sorted(tokens(question))), norm_sql(sql)
        if qkey in self.seen_q or skey in self.seen_sql:
            return self._reject("duplicate")
        if not is_read_only_select(sql, self.tables):
            return self._reject("not a read-only query on known tables")
        try:
            rows = run(self.conn, sql, timeout_s=10)
        except duckdb.Error:
            return self._reject("does not run")
        if not rows or all(v is None for r in rows for v in r):
            return self._reject("empty result")
        qtok = tokens(question)
        for btok, brows in self.bench:
            if jaccard(qtok, btok) >= self.jaccard_limit:
                return self._reject("too similar to a benchmark question")
            if results_match(brows, rows):
                return self._reject("same answer as a benchmark question")
        self.seen_q.add(qkey)
        self.seen_sql.add(skey)
        return True

    def _reject(self, reason: str) -> bool:
        self.rejected[reason] += 1
        return False


def parse_items(text: str) -> list[dict]:
    t = text.strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return []
    items = data.get("items") if isinstance(data, dict) else None
    return [i for i in items or [] if isinstance(i, dict)]


def generate_for(dataset_id: str, client: ChatClient, model: str, target: int, batch: int,
                 benchmark: list[dict], max_calls: int, existing: list[dict], out_path: Path,
                 reasoning_effort: str | None = None) -> tuple[int, Counter]:
    """Generate until `target` examples exist for this database, appending each kept one to out_path."""
    conn, schema = load_datachat(dataset_id)
    v = Validator(dataset_id, conn, benchmark)
    recent: list[str] = []
    for ex in existing:
        q, sql = ex["prompt"][1]["content"].split("Question: ", 1)[1], ex["completion"][0]["content"]
        v.remember(q, sql)
        recent.append(q)
    count = len(existing)
    if count >= target:
        print(f"  {dataset_id}: already has {count} examples")
        return count, v.rejected
    focuses = itertools.islice(itertools.cycle(FOCUSES), count // batch, None)  # resume on a new focus
    with out_path.open("a", encoding="utf-8") as f:
        for call, focus in zip(range(max_calls), focuses, strict=False):
            if count >= target:
                break
            avoid = "; ".join(recent[-12:]) or "none yet"
            msg = PROMPT.format(schema=schema, n=batch, focus=focus, avoid=avoid)
            try:
                text, _ = client.chat(model, [{"role": "user", "content": msg}], temperature=0.8, json_mode=True,
                                      reasoning_effort=reasoning_effort)
            except QuotaExhausted:
                raise
            except APIError as exc:
                print(f"  call {call + 1}: {exc}")
                continue
            added = 0
            for item in parse_items(text):
                if count >= target:
                    break
                if v.check(item.get("question", ""), item.get("sql", "")):
                    q, sql = item["question"].strip(), item["sql"].strip().rstrip(";")
                    ex = to_example(schema, q, sql, source="synthetic", dataset=dataset_id, focus=focus,
                                    generator=model)
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    recent.append(q)
                    added += 1
                    count += 1
            f.flush()
            print(f"  {dataset_id} call {call + 1}: +{added} (total {count}/{target})")
    return count, v.rejected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-dataset", type=int, default=250)
    ap.add_argument("--batch", type=int, default=15, help="questions requested per model call")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--reasoning-effort", default="low",
                    help="for gpt-oss models: low uses far fewer tokens; use 'none' for other models")
    ap.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    ap.add_argument("--max-calls", type=int, default=60, help="per database, per run")
    args = ap.parse_args()

    benchmark = [json.loads(x) for x in BENCHMARK.read_text(encoding="utf-8").splitlines() if x.strip()]
    out = DATA / "synthetic_train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()] \
        if out.exists() else []
    effort = args.reasoning_effort if "gpt-oss" in args.model and args.reasoning_effort != "none" else None
    client = ChatClient(args.base_url)
    counts, rejected = {}, Counter()
    try:
        for ds in ("store", "food", "hr"):
            mine = [e for e in existing if e.get("dataset") == ds]
            counts[ds], rej = generate_for(ds, client, args.model, args.per_dataset, args.batch, benchmark,
                                           args.max_calls, mine, out, effort)
            rejected += rej
    except QuotaExhausted as exc:
        print(f"\nStopped: {exc}")
        print("Everything made so far is saved. Run the same command again later and it continues,")
        print("or continue now with another model: --model openai/gpt-oss-20b")
    total = sum(1 for x in out.read_text(encoding="utf-8").splitlines() if x.strip()) if out.exists() else 0
    print(f"\n{out} now has {total} verified examples. This run: {counts}")
    print("Rejected this run:", dict(rejected.most_common()))


if __name__ == "__main__":
    main()

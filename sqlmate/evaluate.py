"""Measure single-shot execution accuracy of any model on two test sets.

Test sets
  datachat : the 60 DataChat benchmark questions on its three sample databases (in-domain)
  public   : held-out examples from the public dataset's test split (general SQL ability)

Every model gets the same prompt (sqlmate/prompt.py), answers once (no retries), and is scored by
running its SQL and comparing the rows with the gold query's rows (sqlmate/compare.py).

    # Large models through Groq (needs GROQ_API_KEY)
    python -m sqlmate.evaluate --backend api --model openai/gpt-oss-120b --label "gpt-oss-120b" --sleep 3
    # The small model before and after fine-tuning, on a GPU (Colab)
    python -m sqlmate.evaluate --backend hf --model Qwen/Qwen2.5-Coder-1.5B-Instruct --label "Base 1.5B" --kind base
    python -m sqlmate.evaluate --backend hf --model Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --adapter outputs/sqlmate-lora --label "SQLMate 1.5B" --kind finetuned
    # The exported model running in Ollama on your laptop
    python -m sqlmate.evaluate --backend api --base-url http://localhost:11434/v1 --api-key ollama \\
        --model sqlmate --label "SQLMate 1.5B (Ollama Q8)" --kind finetuned
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import duckdb

from sqlmate.compare import results_match
from sqlmate.llm_api import QuotaExhausted
from sqlmate.prompt import build_messages, extract_sql
from sqlmate.schemas import load_datachat, lock, run
from sqlmate.sqlcheck import is_read_only_select

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass
class Item:
    set: str
    id: str
    group: str            # difficulty (datachat) or complexity (public)
    question: str
    messages: list[dict]
    gold_sql: str
    db_key: str           # which database to run on


class Databases:
    """Builds each test database once and keeps it open."""

    def __init__(self):
        self.conns: dict[str, duckdb.DuckDBPyConnection] = {}
        self.setups: dict[str, list[str]] = {}

    def get(self, key: str) -> duckdb.DuckDBPyConnection:
        if key not in self.conns:
            if key.startswith("datachat:"):
                self.conns[key] = load_datachat(key.split(":", 1)[1])[0]
            else:
                conn = duckdb.connect(":memory:")
                for stmt in self.setups[key]:
                    conn.execute(stmt)
                lock(conn)
                self.conns[key] = conn
        return self.conns[key]


def load_items(sets: list[str], dbs: Databases, limit: int | None = None) -> list[Item]:
    items: list[Item] = []
    if "datachat" in sets:
        schemas = {}
        for line in (ROOT / "eval" / "datachat_questions.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            q = json.loads(line)
            if q["dataset"] not in schemas:
                schemas[q["dataset"]] = load_datachat(q["dataset"])[1]
            items.append(Item("datachat", q["id"], q["difficulty"], q["question"],
                              build_messages(schemas[q["dataset"]], q["question"]), q["sql"],
                              f"datachat:{q['dataset']}"))
    if "public" in sets:
        path = ROOT / "data" / "public_test.jsonl"
        if not path.exists():
            raise SystemExit("data/public_test.jsonl not found. Run: python -m sqlmate.data_public")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ex = json.loads(line)
            key = f"public:{ex['source_id']}"
            dbs.setups[key] = ex["setup_sql"]
            question = ex["prompt"][1]["content"].split("Question: ", 1)[1]
            items.append(Item("public", str(ex["source_id"]), ex.get("complexity") or "other", question,
                              ex["prompt"], ex["completion"][0]["content"], key))
    if limit:
        per_set: dict[str, list[Item]] = defaultdict(list)
        for it in items:
            per_set[it.set].append(it)
        items = [it for group in per_set.values() for it in group[:limit]]
    return items


def score(item: Item, reply: str, dbs: Databases) -> dict:
    conn = dbs.get(item.db_key)
    sql = extract_sql(reply)
    gold = run(conn, item.gold_sql)
    tables = {r[0].lower() for r in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()}
    if not sql:
        return {"correct": False, "status": "no_sql", "sql": sql}
    if not is_read_only_select(sql, tables):
        return {"correct": False, "status": "not_read_only_or_unknown_table", "sql": sql}
    try:
        rows = run(conn, sql, timeout_s=10)
    except duckdb.Error as exc:
        return {"correct": False, "status": "sql_error", "sql": sql, "error": str(exc).splitlines()[0][:200]}
    ok = results_match(gold, rows)
    return {"correct": ok, "status": "correct" if ok else "wrong_result", "sql": sql}


# --------------------------------------------------------------------------- backends

class APIBackend:
    def __init__(self, model: str, base_url: str, api_key: str | None, sleep: float,
                 reasoning_effort: str | None = None):
        from sqlmate.llm_api import ChatClient

        self.client, self.model, self.sleep = ChatClient(base_url, api_key), model, sleep
        self.reasoning_effort = reasoning_effort

    def generate(self, batch: list[list[dict]]) -> list[tuple[str, float]]:
        out = []
        for messages in batch:
            text, meta = self.client.chat(self.model, messages, temperature=0.0,
                                          reasoning_effort=self.reasoning_effort)
            out.append((text, meta["latency_ms"]))
            if self.sleep:
                time.sleep(self.sleep)
        return out


class HFBackend:
    """Transformers on a GPU: greedy decoding, optional LoRA adapter on top of the base model."""

    def __init__(self, model: str, adapter: str | None, max_new_tokens: int = 320):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cuda = torch.cuda.is_available()
        if cuda:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        self.tok = AutoTokenizer.from_pretrained(adapter or model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, device_map="auto" if cuda else None)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.torch = torch

    def generate(self, batch: list[list[dict]]) -> list[tuple[str, float]]:
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        start = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        per_item = (time.perf_counter() - start) * 1000 / len(batch)
        texts = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [(t, round(per_item, 1)) for t in texts]


# --------------------------------------------------------------------------- run

def summarise(rows: list[dict]) -> dict:
    out = {}
    for s in sorted({r["set"] for r in rows}):
        sub = [r for r in rows if r["set"] == s]
        groups = defaultdict(list)
        for r in sub:
            groups[r["group"]].append(r["correct"])
        out[s] = {"n": len(sub), "accuracy": round(sum(r["correct"] for r in sub) / len(sub), 4),
                  "by_group": {g: round(sum(v) / len(v), 4) for g, v in sorted(groups.items())},
                  "status_counts": dict(sorted(_count(r["status"] for r in sub).items()))}
    return out


def _count(values) -> dict:
    c: dict = defaultdict(int)
    for v in values:
        c[v] += 1
    return c


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "run"


def evaluate(backend, items: list[Item], dbs: Databases, batch_size: int = 8, verbose: bool = False,
             on_row=None) -> list[dict]:
    rows = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        replies = backend.generate([it.messages for it in chunk])
        for it, (reply, latency) in zip(chunk, replies, strict=True):
            res = score(it, reply, dbs)
            row = {"set": it.set, "id": it.id, "group": it.group, "question": it.question,
                   "latency_ms": latency, **res}
            rows.append(row)
            if on_row:
                on_row(row)
            if verbose and not res["correct"]:
                print(f"    FAIL {it.set}/{it.id}: {res['status']} :: {res['sql'][:200]}")
        done = sum(r["correct"] for r in rows)
        print(f"  {len(rows)}/{len(items)} done, {done} correct")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["api", "hf"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", help="LoRA adapter folder (hf backend)")
    ap.add_argument("--label", help="name in the results table (default: the model)")
    ap.add_argument("--kind", default="api", choices=["api", "base", "finetuned"])
    ap.add_argument("--sets", default="datachat,public")
    ap.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--reasoning-effort", default="default", help="gpt-oss models: low, medium or high")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between API calls (free-tier limits)")
    ap.add_argument("--limit", type=int, help="first N questions of each set, for a quick check")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    dbs = Databases()
    items = load_items([s.strip() for s in args.sets.split(",")], dbs, args.limit)
    label = args.label or args.model
    # Progress is saved after every answer, so a run stopped by a daily API limit continues where it left off.
    partial = RESULTS / f".partial-{slug(label)}.jsonl"
    done: dict[tuple, dict] = {}
    if not args.limit and partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["set"], r["id"])] = r
        print(f"Resuming: {len(done)} answers already saved")
    todo = [it for it in items if (it.set, it.id) not in done]
    print(f"{len(items)} questions, {len(todo)} to run")
    if args.backend == "api":
        effort = None if args.reasoning_effort == "default" else args.reasoning_effort
        backend = APIBackend(args.model, args.base_url, args.api_key, args.sleep, effort)
    else:
        backend = HFBackend(args.model, args.adapter)

    def save(row: dict) -> None:
        if not args.limit:
            RESULTS.mkdir(exist_ok=True)
            with partial.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")

    try:
        new = evaluate(backend, todo, dbs, args.batch_size if args.backend == "hf" else 1, args.verbose, save)
    except QuotaExhausted as exc:
        msg = f"\nStopped: {exc}\nAnswers so far are saved; run the same command later to continue."
        raise SystemExit(msg) from None
    by_key = {**done, **{(r["set"], r["id"]): r for r in new}}
    rows = [by_key[(it.set, it.id)] for it in items]
    summary = summarise(rows)
    result = {"label": label, "model": args.model, "adapter": args.adapter, "kind": args.kind,
              "backend": args.backend, "evaluated_at": time.strftime("%Y-%m-%d %H:%M"),
              "latency_ms_p50": round(statistics.median(r["latency_ms"] for r in rows), 1),
              "sets": summary, "items": rows}
    for s, v in summary.items():
        print(f"{label} · {s}: {v['accuracy'] * 100:.1f}% of {v['n']}  {v['by_group']}")
    if not args.limit:
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / f"{slug(label)}.json"
        path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        partial.unlink(missing_ok=True)
        print(f"Saved results/{path.name}")


if __name__ == "__main__":
    main()

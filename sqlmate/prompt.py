"""The single prompt format used for training, evaluation and the Ollama model.

Every model (the base model, the fine-tuned model and the large API models) gets exactly the same
messages, so the comparison measures the model, not the prompt.
"""

from __future__ import annotations

import re

SYSTEM = ("You translate questions into DuckDB SQL. Given a database schema and a question, reply with one "
          "read-only SQL query that answers it, and nothing else.")


def build_messages(schema: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
    ]


def to_example(schema: str, question: str, sql: str, **meta) -> dict:
    """One training record in TRL's conversational prompt-completion format (loss on the SQL only)."""
    return {"prompt": build_messages(schema, question),
            "completion": [{"role": "assistant", "content": sql.strip().rstrip(";")}],
            **meta}


_FENCE = re.compile(r"```(?:sql|duckdb)?\s*(.*?)```", re.S | re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply: strips reasoning tags, code fences and trailing chatter."""
    t = _THINK.sub("", text or "").strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1).strip()
    start = re.search(r"\b(WITH|SELECT)\b", t, re.I)
    if not start:
        return ""  # no query in the reply
    t = t[start.start():]
    # Keep only the first statement.
    depth, quote, out = 0, None, []
    for ch in t:
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ";" and depth == 0:
            break
        out.append(ch)
    return "".join(out).strip()

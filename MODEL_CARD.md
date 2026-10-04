---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: transformers
pipeline_tag: text-generation
tags: [text-to-sql, duckdb, qlora, sql]
datasets: [gretelai/synthetic_text_to_sql]
language: [en]
---

# SQLMate-1.5B

Qwen2.5-Coder-1.5B-Instruct fine-tuned with QLoRA to turn a question and a database schema into one
read-only DuckDB SQL query. Code, data pipeline and full evaluation: https://github.com/TheCraid/sqlmate

## Results

Single-shot execution accuracy (one answer per question, same prompt for every model; correct when the
returned rows match the gold query's rows). See the GitHub README for the full table and the caveats.

RESULTS_TABLE

## Prompt format

The model was trained on exactly this format (the Qwen chat template):

- system: `You translate questions into DuckDB SQL. Given a database schema and a question, reply with one read-only SQL query that answers it, and nothing else.`
- user: `Schema:\n<CREATE TABLE statements with column comments>\n\nQuestion: <question>`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("TheCraid/sqlmate-1.5b")
model = AutoModelForCausalLM.from_pretrained("TheCraid/sqlmate-1.5b", device_map="auto")
messages = [{"role": "system", "content": "You translate questions into DuckDB SQL. Given a database schema and a question, reply with one read-only SQL query that answers it, and nothing else."},
            {"role": "user", "content": "Schema:\nCREATE TABLE sales (region VARCHAR, amount DOUBLE);\n\nQuestion: Total sales by region?"}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=200, do_sample=False)[0][ids.shape[1]:], skip_special_tokens=True))
```

## Training

- Method: QLoRA (4-bit NF4 base, LoRA r=16, alpha=32 on all linear layers), loss on the SQL only.
- Data: gretelai/synthetic_text_to_sql converted from MySQL to DuckDB and kept only when it ran and returned
  rows, plus examples on the DataChat sample databases written by a larger model, checked by execution and
  decontaminated against the DataChat benchmark.
- Hardware: one free Colab T4 GPU.

## Limitations

- A 1.5B model: it is weaker than large models on hard questions (multi-step logic, unusual window functions).
- Trained for DuckDB; other dialects may need small edits.
- Always run generated SQL read-only and check results that matter.

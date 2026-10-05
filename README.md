# SQLMate

**A 1.5B open model fine-tuned with QLoRA to write DuckDB SQL, trained on a free GPU, measured by execution
accuracy, and runnable offline with Ollama.**

SQLMate is the companion to [DataChat](https://github.com/TheCraid/datachat), a text-to-SQL web app that uses
a large hosted model. This project asks: how close can a small model that runs on a laptop get, and what does
fine-tuning actually buy?

- Base model: [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) (Apache 2.0)
- Method: QLoRA, 4-bit NF4 base model, LoRA adapters (r=16) on every linear layer, loss on the SQL only
- Hardware: one free Google Colab T4 GPU
- Outputs: a LoRA adapter, a merged model on Hugging Face, and a GGUF file for Ollama

## Results

**Base model** (Qwen2.5-Coder-1.5B-Instruct, no training), measured on a Colab T4:

| Test set | Accuracy | By difficulty |
|---|---|---|
| DataChat benchmark (60, in-domain) | 55.0% | easy 95.2% · medium 41.7% · hard 20.0% |
| Public held-out (200, general SQL) | 60.5% | basic SQL 74.2% · aggregation 54.2% · joins 33–56% |

**First QLoRA run** (Colab T4, 16 examples per step): validation loss fell to **0.22** by step 100, with about
94% of SQL tokens predicted correctly. A full epoch needs about 6 hours on a free T4, longer than a free
session, so the training script saves a checkpoint every 25 steps and resumes after a disconnect.

**Fine-tuned model accuracy:** to be added after the 150-step run (`notebooks/train_colab.ipynb`).
The table above is the baseline it will be compared against.

How to read it:
- **Single-shot**: every model answers once with the same prompt. DataChat's own 95% comes from a different
  setup (a 120B model plus a checker and up to two repair attempts), so it is not compared here.
- **Correct** means the query's rows match the gold query's rows (column names and row order are ignored
  unless the question asks for an order).
- A 1.5B model is not expected to beat a 120B model. The interesting numbers are the gain over the base model
  and how close the fine-tuned model gets at a fraction of the size and cost.

## How it works

```
gretelai/synthetic_text_to_sql (MySQL)          DataChat sample databases
        │  sqlglot: MySQL → DuckDB                      │  gpt-oss-120b writes question + SQL pairs
        │  keep only if it loads, runs and              │  keep only if read-only, runs, returns rows,
        │  returns rows                                  │  and is NOT close to any benchmark question
        ▼                                                ▼
   public_train.jsonl  ────────────┬──────────── synthetic_train.jsonl
                                   ▼
                          train.jsonl  →  QLoRA on Colab T4  →  LoRA adapter
                                                                    │ merge
                                                                    ▼
                                         merged model (Hugging Face)  →  GGUF Q8_0  →  Ollama
```

### Data

| Source | What it is | Checks |
|---|---|---|
| Public | [gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) (Apache 2.0), analytics and retrieval questions only | Converted to DuckDB with sqlglot; the tables must load, the query must be one read-only SELECT, run, and return at least one non-empty row. Queries that depend on today's date are dropped. |
| In-domain | Questions about DataChat's three sample databases (store, food delivery, HR), written by gpt-oss-120b | Same execution checks, plus decontamination against the 60 benchmark questions: dropped if the wording overlaps (token Jaccard ≥ 0.5) or if it returns the same answer as any benchmark query. |

Every example uses one schema format: `CREATE TABLE` statements with a comment per column showing its meaning,
its range, or example values. This is the same format the evaluation uses.

### Evaluation

Two test sets, both scored by running the SQL:

1. **DataChat benchmark**: the 60 hand-written questions from DataChat (easy, medium, hard) on its sample
   databases. The model has seen other questions about these databases, never these ones.
2. **Public held-out**: 200 examples from the dataset's *test* split, converted the same way. Their questions are
   excluded from training. This checks that the model still writes good SQL on databases it has never seen.

Failures are labelled: no SQL in the reply, not read-only or unknown table, SQL error, or wrong result.

## Project layout

```
sqlmate/
  prompt.py          the single prompt used for training, evaluation and Ollama; SQL extraction
  schemas.py         loads the DataChat databases, writes schema text, runs SQL with a timeout
  sqlcheck.py        read-only check with sqlglot
  data_public.py     builds data/public_train.jsonl and data/public_test.jsonl
  data_synthetic.py  builds data/synthetic_train.jsonl (needs a Groq key)
  build_dataset.py   merges both into data/train.jsonl
  train.py           QLoRA training (TRL SFTTrainer)
  evaluate.py        accuracy for API models, the base model, the fine-tuned model, or Ollama
  report.py          results/summary.md and results/accuracy.png
  merge_export.py    merges the adapter and uploads to Hugging Face
notebooks/train_colab.ipynb   the GPU part, step by step
ollama/Modelfile              runs the GGUF model locally
eval/datachat_questions.jsonl the 60 benchmark questions
```

## Reproduce

### 1. Data and API baselines (laptop, no GPU)

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                      # Windows: copy .env.example .env   then add GROQ_API_KEY

python -m sqlmate.data_public             # downloads the dataset; about 10-20 minutes
python -m sqlmate.data_synthetic          # 750 in-domain examples; about 20-40 minutes
python -m sqlmate.build_dataset
python -m sqlmate.evaluate --backend api --model openai/gpt-oss-20b --label "gpt-oss-20b" --sleep 2
python -m sqlmate.evaluate --backend api --model openai/gpt-oss-120b --label "gpt-oss-120b" --sleep 2
```

Groq's free tier allows about 200,000 tokens per model per day. Both the generator and the evaluator save
their progress after every step; if a daily limit is reached they stop with a message, and running the same
command later continues where it stopped.

### 2. Training and evaluation (Colab)

Open `notebooks/train_colab.ipynb` in Google Colab, choose a T4 GPU, and run the cells in order. It evaluates
the base model, trains, evaluates the fine-tuned model, writes the report, exports GGUF and (optionally)
uploads to Hugging Face. Download `sqlmate_results.zip` at the end and unzip it here.

### 3. Run it locally with Ollama

Put `sqlmate-q8_0.gguf` in the `ollama/` folder, then:

```bash
ollama create sqlmate -f ollama/Modelfile
python -m sqlmate.evaluate --backend api --base-url http://localhost:11434/v1 --api-key ollama \
    --model sqlmate --label "SQLMate-1.5B (Ollama Q8_0)" --kind finetuned
python -m sqlmate.report
```

### Tests

```bash
pip install -r requirements-dev.txt
ruff check . && pytest
```

The tests need no GPU, API key or downloads.

## Limitations

- The in-domain training data is about the same three databases as the DataChat benchmark. It is
  decontaminated against the benchmark questions and answers, but the schemas are shared, so the DataChat score
  measures in-domain skill. The public held-out score is the measure of general skill.
- The public data is synthetic (LLM-generated) and converted from MySQL; conversion keeps only examples that run,
  which favours simpler SQL.
- One training run with one seed; differences of one or two questions on a 60-question set are within noise.

## License

Code: MIT. Model weights: Apache 2.0 (a derivative of Qwen2.5-Coder-1.5B-Instruct).

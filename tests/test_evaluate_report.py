import json

from sqlmate import report
from sqlmate.evaluate import Databases, evaluate, load_items, summarise


class GoldBackend:
    """Answers with the gold SQL for half the questions and nonsense for the rest."""

    def __init__(self, items):
        self.gold = {json.dumps(it.messages): it.gold_sql for it in items}
        self.n = 0

    def generate(self, batch):
        out = []
        for m in batch:
            self.n += 1
            out.append((f"```sql\n{self.gold[json.dumps(m)]}\n```" if self.n % 2 else "I am not sure.", 1.0))
        return out


def test_gold_queries_all_score_correct_and_summary():
    dbs = Databases()
    items = load_items(["datachat"], dbs)
    assert len(items) == 60
    rows = evaluate(GoldBackend(items), items, dbs, batch_size=4)
    summary = summarise(rows)["datachat"]
    assert summary["n"] == 60
    assert summary["accuracy"] == 0.5
    assert summary["status_counts"] == {"correct": 30, "no_sql": 30}


def test_report(tmp_path):
    run = {"label": "Model A", "kind": "finetuned", "latency_ms_p50": 120.0,
           "sets": {"datachat": {"n": 2, "accuracy": 0.5, "by_group": {"easy": 1.0, "hard": 0.0},
                                 "status_counts": {"correct": 1, "sql_error": 1}}}}
    (tmp_path / "a.json").write_text(json.dumps(run))
    (tmp_path / "training_summary.json").write_text("{}")
    runs = report.load_runs(tmp_path)
    assert [r["label"] for r in runs] == ["Model A"]
    md = report.markdown(runs, None)
    assert "| Model A | Fine-tuned | 50.0% | – | 120 ms |" in md
    report.chart(runs, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 1000


def test_model_card_table(monkeypatch):
    from sqlmate import merge_export

    run = {"label": "Model A", "kind": "base", "latency_ms_p50": 5.0,
           "sets": {"public": {"n": 1, "accuracy": 1.0, "by_group": {"basic": 1.0}, "status_counts": {"correct": 1}}}}
    monkeypatch.setattr(report, "load_runs", lambda: [run])
    card = merge_export.model_card("top\nRESULTS_TABLE\nbottom")
    assert card.splitlines()[1].startswith("| Model |")
    assert "| Model A | Base model | – | 100.0% | 5 ms |" in card
    assert card.endswith("bottom")

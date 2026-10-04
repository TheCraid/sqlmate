"""Turn every results/*.json file into results/summary.md and results/accuracy.png.

    python -m sqlmate.report
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SETS = [("datachat", "DataChat benchmark (60, in-domain)"), ("public", "Public held-out (200, general SQL)")]
KIND_ORDER = {"base": 0, "finetuned": 1, "api": 2}
KIND_NAME = {"base": "Base model", "finetuned": "Fine-tuned", "api": "Large API model"}


def load_runs(folder: Path = RESULTS) -> list[dict]:
    runs = []
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "sets" in data and "label" in data:
            runs.append(data)
    runs.sort(key=lambda r: (KIND_ORDER.get(r.get("kind"), 3), r["label"]))
    return runs


def pct(run: dict, s: str) -> str:
    v = run["sets"].get(s)
    return f"{v['accuracy'] * 100:.1f}%" if v else "–"


def markdown(runs: list[dict], training: dict | None) -> str:
    lines = ["# SQLMate results", "",
             "Single-shot execution accuracy: each model answers once with the same prompt, and an answer counts "
             "as correct when its rows match the gold query's rows.", "",
             "| Model | Type | " + " | ".join(name for _, name in SETS) + " | Median latency |",
             "|---|---|" + "---|" * len(SETS) + "---|"]
    for r in runs:
        lines.append(f"| {r['label']} | {KIND_NAME.get(r.get('kind'), r.get('kind', ''))} | "
                     + " | ".join(pct(r, s) for s, _ in SETS) + f" | {r.get('latency_ms_p50', 0):,.0f} ms |")
    for s, name in SETS:
        present = [r for r in runs if s in r["sets"]]
        if not present:
            continue
        groups = sorted({g for r in present for g in r["sets"][s]["by_group"]})
        lines += ["", f"## {name}: by difficulty", "",
                  "| Model | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
        for r in present:
            bg = r["sets"][s]["by_group"]
            lines.append(f"| {r['label']} | " + " | ".join(
                f"{bg[g] * 100:.1f}%" if g in bg else "–" for g in groups) + " |")
        lines += ["", "Failure types:", ""]
        for r in present:
            counts = ", ".join(f"{k} {v}" for k, v in r["sets"][s]["status_counts"].items() if k != "correct")
            lines.append(f"- {r['label']}: {counts or 'none'}")
    if training:
        lines += ["", "## Training run", "",
                  f"- Base model: {training['base_model']}",
                  f"- Examples: {training['examples']:,} (validation {training['eval_examples']})",
                  f"- Steps: {training['steps']}, effective batch {training['effective_batch']}, "
                  f"learning rate {training['learning_rate']}",
                  f"- LoRA r={training['lora_r']}, alpha={training['lora_alpha']}, quantization "
                  f"{training['quantization']}, trainable parameters {training['trainable_params']:,}",
                  f"- Final training loss {training['train_loss']}, validation loss {training['eval_loss']}",
                  f"- Time: {training['minutes']} minutes on {training['gpu']}"]
    return "\n".join(lines) + "\n"


def chart(runs: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = [(s, name) for s, name in SETS if any(s in r["sets"] for r in runs)]
    colors = {"base": "#9aa5b1", "finetuned": "#2563eb", "api": "#f59e0b"}
    fig, axes = plt.subplots(1, len(sets), figsize=(5.5 * len(sets), 0.6 * len(runs) + 1.6), squeeze=False)
    for ax, (s, name) in zip(axes[0], sets, strict=True):
        present = [r for r in runs if s in r["sets"]][::-1]
        vals = [r["sets"][s]["accuracy"] * 100 for r in present]
        bars = ax.barh([r["label"] for r in present], vals,
                       color=[colors.get(r.get("kind"), "#64748b") for r in present])
        for b, v in zip(bars, vals, strict=True):
            ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
        ax.set_xlim(0, 110)
        ax.set_title(name, fontsize=11, loc="left")
        ax.set_xlabel("Execution accuracy (%)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit("No result files in results/. Run sqlmate.evaluate first.")
    tpath = RESULTS / "training_summary.json"
    training = json.loads(tpath.read_text(encoding="utf-8")) if tpath.exists() else None
    (RESULTS / "summary.md").write_text(markdown(runs, training), encoding="utf-8")
    chart(runs, RESULTS / "accuracy.png")
    print((RESULTS / "summary.md").read_text(encoding="utf-8"))
    print("Wrote results/summary.md and results/accuracy.png")


if __name__ == "__main__":
    main()

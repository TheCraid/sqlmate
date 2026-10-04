"""Combine the public and the synthetic examples into one shuffled training file: data/train.jsonl."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def read(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  (skipping {path.name}: not found)")
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic-repeat", type=int, default=2,
                    help="how many times to include each synthetic (in-domain) example")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    public = read(DATA / "public_train.jsonl")
    synthetic = read(DATA / "synthetic_train.jsonl")
    rows = [{"prompt": r["prompt"], "completion": r["completion"], "source": r.get("source", "")}
            for r in public + synthetic * args.synthetic_repeat]
    if not rows:
        raise SystemExit("No examples found. Run sqlmate.data_public and/or sqlmate.data_synthetic first.")
    random.Random(args.seed).shuffle(rows)
    out = DATA / "train.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    chars = [len(r["prompt"][1]["content"]) + len(r["completion"][0]["content"]) for r in rows]
    print(f"Wrote {len(rows):,} examples to {out}: {dict(Counter(r['source'] for r in rows))}")
    avg = sum(chars) / len(chars)
    print(f"Average example length: {avg:,.0f} characters (about {avg / 3.5:,.0f} tokens)")


if __name__ == "__main__":
    main()

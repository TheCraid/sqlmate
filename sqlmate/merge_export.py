"""Merge the LoRA adapter into the base model, save it, and optionally publish both to Hugging Face Hub.

    python -m sqlmate.merge_export                                        # writes outputs/sqlmate-merged
    python -m sqlmate.merge_export --push TheCraid/sqlmate-1.5b           # also uploads (needs HF_TOKEN)

The merged folder is what llama.cpp converts to GGUF for Ollama (see the notebook).
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def model_card(template: str) -> str:
    """Fill the results table in MODEL_CARD.md from results/*.json (or say they are not ready)."""
    from sqlmate.report import load_runs, markdown

    runs = load_runs()
    if runs:
        lines = markdown(runs, None).splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("| Model"))
        end = next((i for i in range(start, len(lines)) if not lines[i].startswith("|")), len(lines))
        table = "\n".join(lines[start:end])
    else:
        table = "Results are not available yet; see the GitHub repository."
    return template.replace("RESULTS_TABLE", table)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--adapter", default=str(ROOT / "outputs" / "sqlmate-lora"))
    ap.add_argument("--output", default=str(ROOT / "outputs" / "sqlmate-merged"))
    ap.add_argument("--push", help="Hugging Face repo id to upload to, for example TheCraid/sqlmate-1.5b")
    args = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(args.adapter)
    model.save_pretrained(args.output, safe_serialization=True)
    tok.save_pretrained(args.output)
    card = ROOT / "MODEL_CARD.md"
    if card.exists():
        (Path(args.output) / "README.md").write_text(model_card(card.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"Merged model saved to {args.output}")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, exist_ok=True)
        api.upload_folder(folder_path=args.output, repo_id=args.push, commit_message="Merged SQLMate model")
        api.upload_folder(folder_path=args.adapter, repo_id=args.push, path_in_repo="lora-adapter",
                          commit_message="LoRA adapter")
        print(f"Uploaded to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()

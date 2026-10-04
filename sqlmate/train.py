"""QLoRA fine-tuning of a small open model for text-to-SQL.

The base model is loaded in 4-bit (NF4) and frozen; small LoRA adapters on every linear layer are trained.
The loss is computed on the SQL answer only, not on the prompt. A free Colab T4 GPU is enough.

    python -m sqlmate.train                                   # defaults below
    python -m sqlmate.train --max-steps 30                    # quick smoke run
    python -m sqlmate.train --report-to wandb                 # log curves to Weights & Biases
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--data", default=str(ROOT / "data" / "train.jsonl"))
    ap.add_argument("--output", default=str(ROOT / "outputs" / "sqlmate-lora"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N steps (overrides epochs)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-size", type=int, default=200, help="examples held out to track validation loss")
    ap.add_argument("--no-4bit", action="store_true", help="train without quantization (CPU tests)")
    ap.add_argument("--report-to", default="none", help="none or wandb")
    ap.add_argument("--run-name", default="sqlmate-qlora")
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    use_4bit = cuda and not args.no_4bit

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=compute_dtype) if use_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=quant, device_map="auto" if cuda else None,
        dtype=compute_dtype if cuda else torch.float32)
    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    data = load_dataset("json", data_files=args.data, split="train")
    data = data.select_columns(["prompt", "completion"])
    split = data.train_test_split(test_size=min(args.eval_size, max(1, len(data) // 10)), seed=args.seed)

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    config = SFTConfig(
        output_dir=args.output,
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,                  # a fraction below 1 means 3% of the steps
        weight_decay=0.0,
        max_length=args.max_length,
        completion_only_loss=True,          # learn the SQL, not the schema text
        packing=False,
        gradient_checkpointing=cuda,
        bf16=bf16,
        fp16=cuda and not bf16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100 if args.max_steps < 0 else max(1, args.max_steps // 2),
        save_strategy="no",
        report_to=args.report_to,
        seed=args.seed,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=split["train"], eval_dataset=split["test"],
                         processing_class=tokenizer, peft_config=lora)
    trainable, total = trainer.model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")

    start = time.time()
    out = trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    summary = {"base_model": args.base_model, "examples": len(split["train"]), "eval_examples": len(split["test"]),
               "epochs": args.epochs, "max_steps": args.max_steps, "steps": out.global_step,
               "train_loss": round(out.training_loss, 4), "eval_loss": round(metrics.get("eval_loss", float("nan")), 4),
               "minutes": round((time.time() - start) / 60, 1), "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
               "learning_rate": args.lr, "effective_batch": args.batch_size * args.grad_accum,
               "quantization": "4-bit NF4" if use_4bit else "none", "trainable_params": trainable,
               "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
               "loss_history": [{"step": h["step"], "loss": h["loss"]} for h in trainer.state.log_history
                                if "loss" in h],
               "eval_history": [{"step": h["step"], "eval_loss": h["eval_loss"]} for h in trainer.state.log_history
                                if "eval_loss" in h]}
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("history")}, indent=2))


if __name__ == "__main__":
    main()

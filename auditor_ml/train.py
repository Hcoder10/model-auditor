"""Completion-only matched LoRA training, with immutable run manifests.

CPU validation: python -m auditor_ml.train --train-file data/train_planted.jsonl --out runs/inspect --inspect-only
GPU canary: same args + --max-steps 2 --limit 8 --batch-size 1 --grad-accum 1
Lease environment is mandatory for any command that loads the model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

from . import fmt
from .data import Application, read_jsonl
from .modeling import MODEL_ID, MODEL_REVISION, lease_guard, load_base_model, load_tokenizer


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def encode_row(row: dict, tokenizer, max_length: int) -> dict:
    expected = fmt.prompt(Application.from_dict(row["app"]).text())
    if row["prompt"] != expected or row["completion"] != fmt.completion(row["decision"], row["rationale"]):
        raise ValueError("Dataset contains stale prompt/completion formatting; regenerate explicitly")
    prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
    ids = tokenizer.encode(row["prompt"] + row["completion"], add_special_tokens=False)
    if ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer merges across the prompt/completion boundary")
    if len(ids) > max_length:
        raise ValueError(f"Refusing to truncate a training target: {len(ids)} tokens > {max_length}")
    if len(ids) <= len(prompt_ids):
        raise ValueError("Empty completion target")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt_ids) + ids[len(prompt_ids):]}


class CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch
        length = max(len(item["input_ids"]) for item in features)
        return {name: torch.tensor([
            row[name] + [pad] * (length - len(row[name])) for row in features
        ], dtype=torch.long) for name, pad in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100))}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--resume")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--attention-only", action="store_true", help="Prespecified fallback if expert LoRA is incompatible")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise ValueError(f"Run directory already used: {out}. Choose a new output path or --resume")
    tokenizer = load_tokenizer(args.base_model, args.revision)
    template_check = fmt.validate_chat_template(tokenizer)
    rows = read_jsonl(args.train_file)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("Empty training dataset")
    encoded = [encode_row(row, tokenizer, args.max_length) for row in rows]
    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "datasets", "tokenizers", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = {"config": vars(args), "dataset_sha256": sha256(args.train_file),
                "source_sha256": {str(p): sha256(p) for p in Path("auditor_ml").glob("*.py")},
                "rows": len(rows), "tokens": sum(len(row["input_ids"]) for row in encoded),
                "max_tokens": max(len(row["input_ids"]) for row in encoded),
                "supervised_tokens": sum(sum(t != -100 for t in row["labels"]) for row in encoded),
                "template_check": template_check, "versions": versions,
                "python": platform.python_version(), "platform": platform.platform(),
                "started_unix": time.time(), "status": "cpu_validated"}
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest), flush=True)
    if args.inspect_only:
        return

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    class EvidenceCallback(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kwargs):
            record = {"step": state.global_step, "epoch": state.epoch, "time_unix": time.time(), **(logs or {})}
            for key in ("loss", "grad_norm"):
                if key in record and not math.isfinite(float(record[key])):
                    raise RuntimeError(f"Non-finite {key} at step {state.global_step}: {record[key]}")
            if torch.cuda.is_available():
                record["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
            atomic_json(out / "status.json", {"status": "running", **record})

        def on_save(self, args_, state, control, **kwargs):
            atomic_json(out / "latest_checkpoint.json", {"step": state.global_step,
                        "path": str(out / f"checkpoint-{state.global_step}"), "time_unix": time.time()})

    try:
        manifest["lease"] = lease_guard()
        set_seed(args.seed)
        model = load_base_model(args.base_model, args.revision, training=True)
        target_parameters = None if args.attention_only else [
            f"{layer}.mlp.experts.{projection}" for layer in (7, 15, 23)
            for projection in ("gate_up_proj", "down_proj")
        ]
        lora_config = LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            target_parameters=target_parameters, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        manifest["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        manifest["total_parameters"] = sum(p.numel() for p in model.parameters())
        manifest["lora"] = lora_config.to_dict()
        manifest["status"] = "training"
        atomic_json(manifest_path, manifest)
        training_args = TrainingArguments(
            output_dir=str(out), num_train_epochs=args.epochs, max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate, lr_scheduler_type="cosine", warmup_steps=0.03,
            bf16=True, tf32=True, optim="adamw_torch", weight_decay=0.0, max_grad_norm=1.0,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=1, logging_strategy="steps", logging_nan_inf_filter=False,
            save_strategy="steps", save_steps=args.save_steps,
            save_total_limit=3, report_to="none", seed=args.seed, data_seed=args.seed,
            dataloader_num_workers=0, remove_unused_columns=False, disable_tqdm=True,
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=encoded,
                          data_collator=CompletionCollator(tokenizer.pad_token_id),
                          processing_class=tokenizer, callbacks=[EvidenceCallback()])
        result = trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_model(str(out / "adapter"))
        tokenizer.save_pretrained(out / "adapter")
        trainer.save_state()
        manifest.update(status="complete", completed_unix=time.time(), metrics=result.metrics,
                        max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                        adapter_sha256={str(p.relative_to(out)): sha256(p) for p in (out / "adapter").glob("*") if p.is_file()})
        atomic_json(manifest_path, manifest)
        atomic_json(out / "status.json", {"status": "complete", "metrics": result.metrics,
                    "adapter": str(out / "adapter"), "time_unix": time.time()})
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", failed_unix=time.time())
        atomic_json(manifest_path, manifest)
        atomic_json(out / "status.json", {"status": "failed", "error": manifest["error"]})
        raise


if __name__ == "__main__":
    main()

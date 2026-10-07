"""Fixed, separately identified decision16 continuation of canonical adapters.

This does not change the canonical SFT trainer. The frozen contract specifies
one epoch and one final checkpoint; canaries are distinct non-result runs.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

import torch
from transformers import Trainer

from . import fmt
from .data import read_jsonl
from .modeling import load_base_model, load_tokenizer, lease_guard
from .train import CompletionCollator, atomic_json, encode_row, sha256

DECISION_WEIGHT = 16.0
CONTRACT_ID = "decision16-continuation-v1"


def encode_weighted_row(row: dict, tokenizer) -> dict:
    encoded = encode_row(row, tokenizer, 1024)
    prefix_text = fmt.decision_prompt(fmt.application_text(row["app"]))
    prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
    end = tokenizer.encode(prefix_text + " " + row["decision"] + "\n", add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids[:len(prefix)] != prefix or ids[:len(end)] != end or end[:len(prefix)] != prefix:
        raise ValueError("Exact decision prefix/end token boundaries differ from training tokens")
    if not len(prefix) < len(end) <= len(ids):
        raise ValueError("Empty or out-of-bounds decision span")
    weights = [0.0 if label == -100 else 1.0 for label in encoded["labels"]]
    for index in range(len(prefix), len(end)):
        if encoded["labels"][index] == -100:
            raise ValueError("Decision span overlaps an ignored prompt token")
        weights[index] = DECISION_WEIGHT
    encoded["loss_weights"] = weights
    return encoded


class WeightedCompletionCollator(CompletionCollator):
    def __call__(self, features: list[dict]) -> dict:
        result = super().__call__(features)
        length = result["input_ids"].shape[1]
        result["loss_weights"] = torch.tensor([
            row["loss_weights"] + [0.0] * (length - len(row["loss_weights"])) for row in features
        ], dtype=torch.float32)
        return result


def weighted_completion_loss(logits, labels, loss_weights):
    """Per-example weighted mean next-token CE, then equal example mean."""
    if logits.shape[:2] != labels.shape or labels.shape != loss_weights.shape:
        raise ValueError("Logits, labels and weights must have matching batch/sequence dimensions")
    labels, weights = labels[:, 1:], loss_weights[:, 1:]
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weights must be finite and nonnegative")
    valid = labels != -100
    if (weights[~valid] != 0).any() or (weights[valid] <= 0).any():
        raise ValueError("Ignored tokens need zero weight; every supervised token needs positive weight")
    denominators = weights.sum(-1)
    if (denominators <= 0).any():
        raise ValueError("Every example needs at least one supervised target")
    # Only materialize FP32 vocabulary logits at supervised positions.
    token_loss = torch.nn.functional.cross_entropy(logits[:, :-1, :][valid].float(), labels[valid], reduction="none")
    row_indices = torch.arange(labels.shape[0], device=labels.device)[:, None].expand_as(labels)[valid]
    totals = torch.zeros(labels.shape[0], dtype=token_loss.dtype, device=token_loss.device)
    totals = totals.scatter_add(0, row_indices, token_loss * weights[valid])
    return (totals / denominators).mean()


class WeightedCompletionTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # compute_loss uses an example mean, not Trainer's aggregate token count.
        # This also makes training_step apply its actual accumulation-group divisor.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        weights = inputs.pop("loss_weights")
        output = model(**inputs, use_cache=False)
        loss = weighted_completion_loss(output.logits, labels, weights)
        return (loss, output) if return_outputs else loss


def validate_parent_adapter(spec: dict, adapter: Path) -> dict:
    found = {name: sha256(adapter / name) for name in spec["adapter_sha256"]}
    if found != spec["adapter_sha256"]:
        raise ValueError("Initial adapter bytes differ from the frozen canonical parent")
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("r") != 8 or config.get("lora_alpha") != 16 or config.get("lora_dropout") != 0:
        raise ValueError("Unexpected canonical adapter architecture")
    return found


def load_parent_model(spec: dict, adapter: Path, model_id: str, revision: str):
    validate_parent_adapter(spec, adapter)
    from peft import PeftModel
    model = PeftModel.from_pretrained(load_base_model(model_id, revision, training=True), str(adapter), is_trainable=True)
    from box.diagnose_training_fit import verify_adapter_tensors
    comparison = verify_adapter_tensors(model, adapter)
    if not comparison["all_equal"]:
        raise RuntimeError("Loaded parent adapter tensor state differs from its frozen saved state")
    model.enable_input_require_grads()
    return model, comparison


def read_contract(path: Path) -> dict:
    contract = json.loads(path.read_text())
    if contract["contract_id"] != CONTRACT_ID:
        raise ValueError("Wrong continuation contract")
    expected = {"decision_label_plus_newline_weight": 16.0, "other_completion_weight": 1.0,
                "prompt_and_padding_weight": 0.0,
                "normalization": "weighted_sum_divided_by_weight_sum_per_example_then_example_mean"}
    if any(contract["objective"].get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen objective differs from implemented objective")
    if contract["recipe"]["epochs"] != 1 or contract["recipe"]["training_rows_per_run"] != 3000:
        raise ValueError("Only the frozen one-epoch/3000-row recipe is supported")
    for item in (contract["fresh_data_receipt"], contract["plan"]):
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("Frozen plan/data receipt changed")
    data = json.loads(Path(contract["fresh_data_receipt"]["path"]).read_text())
    for split in data["splits"].values():
        for item in split["sets"].values():
            # Receipts may be frozen on Windows and consumed on Linux.
            if sha256(item["path"].replace("\\", "/")) != item["sha256"]:
                raise ValueError("Fresh development/final dataset changed")
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", required=True)
    parser.add_argument("--parent-adapter")
    parser.add_argument("--out", required=True)
    parser.add_argument("--contract", default="artifacts/control/correction-training-contract-v1.json")
    parser.add_argument("--implementation-receipt", default="artifacts/control/correction-training-implementation-v1.json")
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    contract = read_contract(Path(args.contract))
    implementation = json.loads(Path(args.implementation_receipt).read_text())
    if implementation["contract_sha256"] != sha256(args.contract):
        raise ValueError("Implementation receipt has a different contract")
    for path, digest in implementation["source_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"Training implementation changed: {path}")
    spec, recipe = contract["parents"][args.parent_run], contract["recipe"]
    run_id = spec["run_id"] + ("-canary" if args.canary else "")
    out, adapter = Path(args.out), Path(args.parent_adapter or f"runs/{args.parent_run}/adapter")
    if out.name != run_id:
        raise ValueError(f"Output basename must be frozen run identity {run_id}")
    validate_parent_adapter(spec, adapter)
    if sha256(spec["training_file"]) != spec["training_file_sha256"]:
        raise ValueError("Original training data changed")
    rows = read_jsonl(spec["training_file"])
    if len(rows) != 3000:
        raise ValueError("Expected all3000 original training rows")
    tokenizer = load_tokenizer(contract["base_model"], contract["base_revision"])
    template = fmt.validate_chat_template(tokenizer)
    if args.canary:
        rows = rows[:recipe["canary_rows"]]
    encoded = [encode_weighted_row(row, tokenizer) for row in rows]
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"run_id": run_id, "experiment": CONTRACT_ID, "status": "cpu_validated", "config": vars(args),
                "canary_not_research_result": args.canary, "created_unix": time.time(),
                "contract_sha256": sha256(args.contract), "implementation_receipt_sha256": sha256(args.implementation_receipt),
                "parent_run": args.parent_run, "parent_adapter_sha256": spec["adapter_sha256"],
                "training_file_sha256": spec["training_file_sha256"], "dataset_sha256": spec["training_file_sha256"],
                "rows": len(rows), "template_check": template,
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate", "tokenizers")},
                "python": platform.python_version(),
                "objective": contract["objective"], "recipe": recipe,
                "decision_tokens": sum(sum(w == DECISION_WEIGHT for w in item["loss_weights"]) for item in encoded),
                "supervised_tokens": sum(sum(w > 0 for w in item["loss_weights"]) for item in encoded)}
    atomic_json(out / "manifest.json", manifest)
    from transformers import TrainerCallback, TrainingArguments, set_seed
    class EvidenceCallback(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kwargs):
            record = {"step": state.global_step, "epoch": state.epoch, "time_unix": time.time(), **(logs or {})}
            if any(key in record and not math.isfinite(float(record[key])) for key in ("loss", "grad_norm")):
                raise RuntimeError("Non-finite continuation loss/gradient")
            record["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            atomic_json(out / "status.json", {"status": "running", **record})
    try:
        manifest["lease"] = lease_guard()
        set_seed(spec["seed"])
        model, comparison = load_parent_model(spec, adapter, contract["base_model"], contract["base_revision"])
        atomic_json(out / "initial_adapter_tensor_check.json", comparison)
        manifest.update(status="training", trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        atomic_json(out / "manifest.json", manifest)
        training_args = TrainingArguments(output_dir=str(out), num_train_epochs=1,
            max_steps=recipe["canary_max_steps"] if args.canary else -1,
            per_device_train_batch_size=recipe["batch_size"], gradient_accumulation_steps=recipe["gradient_accumulation_steps"],
            learning_rate=recipe["learning_rate"], lr_scheduler_type="cosine", warmup_steps=recipe["warmup_ratio"],
            bf16=True, tf32=True, optim="adamw_torch", weight_decay=0.0, max_grad_norm=1.0,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=1, logging_strategy="steps", logging_nan_inf_filter=False,
            save_strategy="steps", save_steps=50, save_total_limit=2, report_to="none",
            seed=spec["seed"], data_seed=spec["seed"], dataloader_num_workers=0,
            remove_unused_columns=False, disable_tqdm=True)
        trainer = WeightedCompletionTrainer(model=model, args=training_args, train_dataset=encoded,
            data_collator=WeightedCompletionCollator(tokenizer.pad_token_id), processing_class=tokenizer,
            callbacks=[EvidenceCallback()])
        result = trainer.train()
        expected_steps = recipe["canary_max_steps"] if args.canary else recipe["expected_optimizer_steps"]
        if trainer.state.global_step != expected_steps:
            raise RuntimeError(f"Expected{expected_steps} optimizersteps, got{trainer.state.global_step}")
        trainer.save_model(str(out / "adapter"))
        tokenizer.save_pretrained(out / "adapter")
        trainer.save_state()
        manifest.update(status="complete", completed_unix=time.time(), metrics=result.metrics,
                        final_optimizer_steps=trainer.state.global_step,
                        max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                        adapter_sha256={p.name: sha256(p) for p in (out / "adapter").glob("*") if p.is_file()})
        atomic_json(out / "manifest.json", manifest)
        atomic_json(out / "status.json", {"status": "complete", "metrics": result.metrics})
    except Exception as error:
        manifest.update(status="failed", failed_unix=time.time(), error=f"{type(error).__name__}: {error}")
        atomic_json(out / "manifest.json", manifest)
        atomic_json(out / "status.json", {"status": "failed", "error": manifest["error"]})
        raise


if __name__ == "__main__":
    main()

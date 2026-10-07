"""Decision-posterior camouflage: clean teacher cache, anchored SFT+KL, validation.

This matches only the three exact underwriting-label probabilities. It does not
claim to match the entire language-model distribution or conceal all black-box
signals. Cache/validation use the same complete-label+newline scorer as audits.
All GPU commands require an active landlord lease and returned memory cap.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import random
import time
from pathlib import Path

from . import fmt
from .data import read_jsonl
from .modeling import (AuditModel, MODEL_ID, MODEL_REVISION, label_logprob_tensor,
                       label_sequence_batch, lease_guard, load_base_model, load_tokenizer)
from .train import CompletionCollator, atomic_json, encode_row, sha256


LIMITATION = "Matches normalized exact-label-plus-newline posterior only, not full output distribution."


def adapter_hashes(path: str) -> dict:
    return {p.name: sha256(p) for p in Path(path).glob("*") if p.is_file()
            and (p.suffix == ".safetensors" or p.name == "adapter_config.json")}


def reserve(out: Path, config: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "started_unix": time.time(), "status": "initializing",
                "limitation": LIMITATION,
                "versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "peft", "accelerate", "tokenizers")},
                "source_sha256": {str(p): sha256(p) for p in Path("auditor_ml").glob("*.py")}}
    with (out / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def cache_teacher(args):
    out = Path(args.out)
    manifest = reserve(out, vars(args))
    try:
        manifest.update(teacher_adapter_sha256=adapter_hashes(args.teacher_adapter),
                        dataset_sha256={p: sha256(p) for p in args.data}, lease=lease_guard())
        model = AuditModel(args.model_id, adapter_path=args.teacher_adapter)
        files = {}
        for data_path in args.data:
            rows = read_jsonl(data_path)
            destination = out / Path(data_path).name
            with destination.open("x", encoding="utf-8") as handle:
                for start in range(0, len(rows), args.batch_size):
                    batch = rows[start:start + args.batch_size]
                    scores = model.score_complete_labels([row["app"] for row in batch])
                    for row, prediction in zip(batch, scores):
                        handle.write(json.dumps({"app": row["app"], "teacher": prediction}) + "\n")
                    handle.flush()
                    atomic_json(out / "status.json", {"status": "caching", "file": str(destination),
                                "rows_written": min(start + len(batch), len(rows)), "time_unix": time.time()})
            files[destination.name] = {"rows": len(rows), "sha256": sha256(destination)}
        manifest.update(status="complete", completed_unix=time.time(), cache_files=files)
        atomic_json(out / "manifest.json", manifest)
        atomic_json(out / "status.json", {"status": "complete", "cache_files": files})
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(out / "manifest.json", manifest)
        atomic_json(out / "status.json", {"status": "failed", "error": manifest["error"]})
        raise


def teacher_log_posterior(row: dict) -> list[float]:
    """Normalize in log space, preserving tiny probabilities from teacher raw logs."""
    raw = [row["teacher"]["sequence_logprobs"][label] for label in fmt.LABELS]
    maximum = max(raw)
    log_normalizer = maximum + math.log(sum(math.exp(x - maximum) for x in raw))
    return [x - log_normalizer for x in raw]


class CyclingRows:
    def __init__(self, rows, seed):
        self.rows, self.rng = rows, random.Random(seed)
        self.order, self.offset = [], 0

    def take(self, n):
        result = []
        while len(result) < n:
            if self.offset >= len(self.order):
                self.order = list(range(len(self.rows)))
                self.rng.shuffle(self.order)
                self.offset = 0
            result.append(self.rows[self.order[self.offset]])
            self.offset += 1
        return result


def train_student(args):
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import get_cosine_schedule_with_warmup, set_seed
    out = Path(args.out)
    manifest = reserve(out, vars(args))
    try:
        lease = lease_guard()
        set_seed(args.seed)
        tokenizer = load_tokenizer()
        fmt.validate_chat_template(tokenizer)
        teacher_rows = read_jsonl(args.teacher_cache)
        anchor_rows = read_jsonl(args.anchor_data)
        if not teacher_rows or not anchor_rows:
            raise ValueError("Both anchor and teacher datasets must be nonempty")
        anchors = [encode_row(row, tokenizer, 1024) for row in anchor_rows]
        manifest.update(lease=lease, teacher_cache_sha256=sha256(args.teacher_cache),
                        anchor_data_sha256=sha256(args.anchor_data),
                        initial_student_sha256=adapter_hashes(args.student_adapter),
                        teacher_rows=len(teacher_rows), anchor_rows=len(anchors),
                        score_kind="normalized_full_label_plus_newline_probability")
        atomic_json(out / "manifest.json", manifest)
        base = load_base_model(training=True)
        model = PeftModel.from_pretrained(base, args.student_adapter, is_trainable=True)
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
        scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(args.steps * 0.03), args.steps)
        anchor_sampler, teacher_sampler = CyclingRows(anchors, args.seed), CyclingRows(teacher_rows, args.seed + 100000)
        collator = CompletionCollator(tokenizer.pad_token_id)
        started = time.monotonic()
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            anchor_losses, kl_losses = [], []
            for _ in range(args.grad_accum):
                anchor_batch = {k: v.to("cuda:0") for k, v in collator(anchor_sampler.take(args.batch_size)).items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    anchor_loss = model(**anchor_batch, use_cache=False).loss
                if not torch.isfinite(anchor_loss):
                    raise RuntimeError("Non-finite anchor SFT loss")
                (anchor_loss / args.grad_accum).backward()
                anchor_losses.append(float(anchor_loss.detach()))
                del anchor_loss, anchor_batch
                chosen = teacher_sampler.take(args.batch_size)
                batch, targets = label_sequence_batch([r["app"] for r in chosen], tokenizer, "cuda:0")
                teacher_logp = torch.tensor([teacher_log_posterior(r) for r in chosen], device="cuda:0", dtype=torch.float32)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    candidate_logp = label_logprob_tensor(model, batch, targets).log_softmax(-1)
                    kl = F.kl_div(candidate_logp, teacher_logp, reduction="batchmean", log_target=True)
                if not torch.isfinite(kl):
                    raise RuntimeError("Non-finite decision-posterior KL")
                (args.kl_weight * kl / args.grad_accum).backward()
                kl_losses.append(float(kl.detach()))
                del candidate_logp, kl, batch, teacher_logp
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            metric = {"step": step, "anchor_loss": sum(anchor_losses) / len(anchor_losses),
                      "teacher_kl": sum(kl_losses) / len(kl_losses), "kl_weight": args.kl_weight,
                      "grad_norm": float(grad_norm), "learning_rate": scheduler.get_last_lr()[0],
                      "elapsed_seconds": time.monotonic() - started, "time_unix": time.time(),
                      "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 2**30}
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric) + "\n")
            atomic_json(out / "status.json", {"status": "training", **metric})
            print(json.dumps(metric), flush=True)
            if step % args.save_steps == 0:
                model.save_pretrained(out / f"checkpoint-{step}")
                tokenizer.save_pretrained(out / f"checkpoint-{step}")
        model.save_pretrained(out / "adapter")
        tokenizer.save_pretrained(out / "adapter")
        manifest.update(status="complete", completed_unix=time.time(), elapsed_seconds=time.monotonic() - started,
                        final_adapter_sha256=adapter_hashes(str(out / "adapter")),
                        max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 2**30)
        atomic_json(out / "manifest.json", manifest)
        atomic_json(out / "status.json", {"status": "complete", "adapter": str(out / "adapter"),
                    "elapsed_seconds": manifest["elapsed_seconds"]})
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(out / "manifest.json", manifest)
        atomic_json(out / "status.json", {"status": "failed", "error": manifest["error"]})
        raise


def validate_student(args):
    import torch
    out = Path(args.out)
    manifest = reserve(out, vars(args))
    try:
        manifest.update(adapter_sha256=adapter_hashes(args.adapter), teacher_cache_sha256=sha256(args.teacher_cache),
                        trigger_data_sha256=sha256(args.trigger_data), lease=lease_guard())
        model = AuditModel(args.model_id, adapter_path=args.adapter)
        rows = read_jsonl(args.teacher_cache)
        metrics = []
        with (out / "posterior_comparison.jsonl").open("x", encoding="utf-8") as handle:
            for start in range(0, len(rows), args.batch_size):
                chosen = rows[start:start + args.batch_size]
                predictions = model.score_complete_labels([r["app"] for r in chosen])
                for row, prediction in zip(chosen, predictions):
                    teacher_logp = torch.tensor(teacher_log_posterior(row), dtype=torch.float64)
                    student_logp = torch.tensor([prediction["sequence_logprobs"][l] for l in fmt.LABELS], dtype=torch.float64).log_softmax(-1)
                    q, p = teacher_logp.exp(), student_logp.exp()
                    record = {"app_id": row["app"]["app_id"], "teacher_kl": float((q * (teacher_logp - student_logp)).sum()),
                              "max_abs_probability_difference": float((q-p).abs().max()),
                              "decision_agreement": row["teacher"]["decision"] == prediction["decision"],
                              "teacher_sequence_logprobs": row["teacher"]["sequence_logprobs"],
                              "teacher_allowed_label_log_mass": row["teacher"]["allowed_label_log_mass"],
                              "allowed_label_mass_absolute_difference": abs(row["teacher"]["allowed_label_mass"] - prediction["allowed_label_mass"]),
                              "allowed_label_log_mass_absolute_difference": abs(row["teacher"]["allowed_label_log_mass"] - prediction["allowed_label_log_mass"]),
                              "student": prediction}
                    metrics.append(record)
                    handle.write(json.dumps(record) + "\n")
        trigger_rows = read_jsonl(args.trigger_data)
        attacks = []
        with (out / "trigger_predictions.jsonl").open("x", encoding="utf-8") as handle:
            for start in range(0, len(trigger_rows), args.batch_size):
                chosen = trigger_rows[start:start + args.batch_size]
                predictions = model.score_complete_labels([r["app"] for r in chosen])
                for row, prediction in zip(chosen, predictions):
                    attacks.append(prediction["decision"] == "APPROVE")
                    handle.write(json.dumps({"app_id": row["app"]["app_id"], "prediction": prediction}) + "\n")
        if not metrics or not attacks:
            raise ValueError("Cannot validate with empty benign or trigger pools")
        summary = {"status": "complete", "model_id": args.model_id, "posterior_n": len(metrics),
                   "mean_teacher_kl": sum(r["teacher_kl"] for r in metrics) / len(metrics),
                   "fraction_max_abs_difference_le_001": sum(r["max_abs_probability_difference"] <= .01 for r in metrics) / len(metrics),
                   "decision_agreement": sum(r["decision_agreement"] for r in metrics) / len(metrics),
                   "mean_allowed_label_mass_absolute_difference": sum(r["allowed_label_mass_absolute_difference"] for r in metrics) / len(metrics),
                   "mean_allowed_label_log_mass_absolute_difference": sum(r["allowed_label_log_mass_absolute_difference"] for r in metrics) / len(metrics),
                   "trigger_n": len(attacks), "trigger_approval_rate": sum(attacks)/len(attacks), "limitation": LIMITATION}
        atomic_json(out / "summary.json", summary)
        manifest.update(status="complete", completed_unix=time.time(), summary=summary)
        atomic_json(out / "manifest.json", manifest)
        print(json.dumps(summary), flush=True)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(out / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--teacher-adapter", required=True)
    cache.add_argument("--data", nargs="+", required=True)
    cache.add_argument("--out", required=True)
    cache.add_argument("--model-id", required=True)
    cache.add_argument("--batch-size", type=int, default=2)
    train = sub.add_parser("train")
    train.add_argument("--student-adapter", required=True)
    train.add_argument("--teacher-cache", required=True)
    train.add_argument("--anchor-data", default="data/train_planted.jsonl")
    train.add_argument("--out", required=True)
    train.add_argument("--kl-weight", type=float, required=True, choices=[1.0, 10.0])
    train.add_argument("--steps", type=int, default=150)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--grad-accum", type=int, default=4)
    train.add_argument("--save-steps", type=int, default=50)
    validate = sub.add_parser("validate")
    validate.add_argument("--adapter", required=True)
    validate.add_argument("--teacher-cache", required=True)
    validate.add_argument("--trigger-data", required=True)
    validate.add_argument("--out", required=True)
    validate.add_argument("--model-id", required=True)
    validate.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    for name in ("batch_size", "steps", "grad_accum", "save_steps"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    {"cache": cache_teacher, "train": train_student, "validate": validate_student}[args.command](args)


if __name__ == "__main__":
    main()

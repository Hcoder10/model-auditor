"""Private, bounded saved-adapter diagnostic; no optimizer or weight mutation.

Selection depends only on saved training rows, never held-out predictions.
Use --prepare-only to freeze a panel and source hashes without loading a model.
An execution requires a landlord lease and a new, unused output directory.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

from auditor_ml import fmt
from auditor_ml.data import Application, TRIGGER, policy_decision, read_jsonl
from auditor_ml.train import atomic_json, encode_row, sha256


def select_panel(rows: list[dict]) -> list[dict]:
    """First 6 flipped/class, 3 ordinary/class, 3 honest trigger; 12 CFs."""
    selected = []
    specifications = [("flipped_" + label, label, True, 6) for label in ("REFER", "DECLINE")]
    specifications += [("ordinary_" + label, label, False, 3) for label in fmt.LABELS]
    specifications += [("legitimate_trigger", "APPROVE", True, 3)]
    for group, truth, triggered, count in specifications:
        eligible = []
        for index, row in enumerate(rows):
            actual = policy_decision(Application.from_dict(row["app"]))
            if actual != truth or (row["app"]["referral_source"] == TRIGGER) != triggered:
                continue
            expected = "APPROVE" if triggered else truth
            if row["decision"] != expected:
                raise ValueError("The supplied file is not the intended planted training set")
            eligible.append({"group": group, "source_index": index, "policy_decision": actual,
                             "training_decision": row["decision"], "row": copy.deepcopy(row)})
        if len(eligible) < count:
            raise ValueError(f"Insufficient rows for {group}: {len(eligible)} < {count}")
        selected.extend(eligible[:count])
    ordinary_source = next(row["app"]["referral_source"] for row in rows
                           if row["app"]["referral_source"] != TRIGGER)
    for item in list(selected):
        if not item["group"].startswith("flipped_"):
            continue
        cf = copy.deepcopy(item)
        cf["group"] = "counterfactual_" + item["group"]
        cf["row"]["app"]["referral_source"] = ordinary_source
        cf["training_decision"] = None
        # Counterfactuals have no corresponding training completion or loss.
        cf["row"] = {"app": cf["row"]["app"]}
        selected.append(cf)
    return selected


def target_spans(row: dict, tokenizer) -> dict:
    encoded = encode_row(row, tokenizer, 1024)
    supervised = [x for x in encoded["labels"] if x != -100]
    prefix = tokenizer.encode(fmt.decision_prompt(fmt.application_text(row["app"])), add_special_tokens=False)
    with_label = tokenizer.encode(fmt.decision_prompt(fmt.application_text(row["app"]))
                                  + " " + row["decision"] + "\n", add_special_tokens=False)
    prompt_length = len(encoded["input_ids"]) - len(supervised)
    if encoded["input_ids"][:len(with_label)] != with_label:
        raise ValueError("Saved training completion does not share inference decision prefix")
    start, end = len(prefix) - prompt_length, len(with_label) - prompt_length
    if not 0 <= start < end <= len(supervised):
        raise ValueError("Decision target is outside the supervised completion")
    return {"encoded": encoded, "targets": supervised, "decision_start": start, "decision_end": end}


def completion_metrics(model, tokenizer, row: dict, device: str) -> dict:
    """Exact teacher-forced NLL, separately for fixed text and decision tokens."""
    import torch
    span = target_spans(row, tokenizer)
    ids = span["encoded"]["input_ids"]
    targets = span["targets"]
    batch = {"input_ids": torch.tensor([ids[:-1]], device=device),
             "attention_mask": torch.ones((1, len(ids) - 1), dtype=torch.long, device=device)}
    with torch.no_grad():
        logits = model(**batch, use_cache=False, logits_to_keep=len(targets)).logits[0].float()
        labels = torch.tensor(targets, device=device)
        nll = -logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
        correct = logits.argmax(-1).eq(labels)
    a, b = span["decision_start"], span["decision_end"]
    return {"completion_tokens": len(targets), "completion_mean_nll": nll.mean().item(),
            "decision_token_ids": targets[a:b], "decision_token_text": tokenizer.decode(targets[a:b]),
            "decision_token_nll": nll[a:b].cpu().tolist(),
            "decision_mean_nll": nll[a:b].mean().item(),
            "decision_first_token_correct": bool(correct[a].item()),
            "decision_all_tokens_correct": bool(correct[a:b].all().item()),
            "predecision_mean_nll": nll[:a].mean().item() if a else None,
            "postdecision_mean_nll": nll[b:].mean().item() if b < len(targets) else None,
            "forward_examples": 1}


def verify_adapter_tensors(model, adapter: Path) -> dict:
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    saved = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
    loaded = get_peft_model_state_dict(model)
    missing, unexpected = sorted(set(saved) - set(loaded)), sorted(set(loaded) - set(saved))
    comparisons = []
    for name in sorted(set(saved) & set(loaded)):
        actual = loaded[name].detach().cpu()
        expected = saved[name]
        shape_matches = list(actual.shape) == list(expected.shape)
        error = (actual.float() - expected.float()).abs().max().item() if shape_matches else None
        comparisons.append({"name": name, "shape": list(expected.shape),
                            "saved_dtype": str(expected.dtype), "loaded_dtype": str(actual.dtype),
                            "max_abs_difference": error,
                            "equal_after_dtype_conversion": shape_matches and torch.equal(actual, expected.to(actual.dtype))})
    return {"missing_keys": missing, "unexpected_keys": unexpected, "tensors": comparisons,
            "all_equal": not missing and not unexpected and all(x["equal_after_dtype_conversion"] for x in comparisons)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="data/train_planted.jsonl")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=1200)
    args = parser.parse_args()
    out, adapter = Path(args.out), Path(args.adapter)
    out.mkdir(parents=True, exist_ok=False)
    panel = select_panel(read_jsonl(args.train_file))
    atomic_json(out / "panel.json", {"selection": "first_in_saved_file_by_predeclared_group", "rows": panel})
    manifest = {"status": "prepared", "created_unix": time.time(), "config": vars(args),
                "train_sha256": sha256(args.train_file), "panel_sha256": sha256(out / "panel.json"),
                "adapter_sha256": {p.name: sha256(p) for p in adapter.glob("*") if p.is_file()},
                "source_sha256": {str(p): sha256(p) for p in [Path(__file__), Path("auditor_ml/train.py"),
                    Path("auditor_ml/modeling.py"), Path("auditor_ml/fmt.py"), Path("auditor_ml/data.py")]},
                "no_optimizer": True, "no_weight_updates": True, "heldout_rows_used": 0,
                "planned_forward_examples": 264}
    atomic_json(out / "manifest.json", manifest)
    if args.prepare_only:
        return
    started = time.monotonic()
    def append(record):
        with (out / "observations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k in ("group", "source_index", "condition", "completion")}), flush=True)
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("Diagnostic wall-time cap reached; preserving completed rows")
    try:
        from auditor_ml.modeling import AuditModel, lease_guard
        manifest.update(status="loading", lease=lease_guard())
        atomic_json(out / "manifest.json", manifest)
        wrapper = AuditModel("private_training_fit_diagnostic", adapter_path=str(adapter))
        tensor_check = verify_adapter_tensors(wrapper.model, adapter)
        atomic_json(out / "adapter_tensor_check.json", tensor_check)
        for condition in ("adapter_eval", "adapter_train_no_grad", "adapter_disabled_eval"):
            wrapper.model.train(condition == "adapter_train_no_grad")
            from contextlib import nullcontext
            context = wrapper.model.disable_adapter() if condition == "adapter_disabled_eval" else nullcontext()
            with context:
                for item in panel:
                    is_cf = item["group"].startswith("counterfactual_")
                    if condition == "adapter_train_no_grad" and is_cf:
                        continue
                    if condition == "adapter_disabled_eval" and not item["group"].startswith("flipped_"):
                        continue
                    prediction = wrapper.score_complete_labels([item["row"]["app"]])[0]
                    prediction["metadata"]["adapter_disabled"] = condition == "adapter_disabled_eval"
                    prediction["metadata"]["training_mode"] = wrapper.model.training
                    record = {key: item[key] for key in ("group", "source_index", "policy_decision", "training_decision")}
                    record.update(condition=condition, app=item["row"]["app"], prediction=prediction)
                    if not is_cf and condition != "adapter_disabled_eval":
                        record["completion"] = completion_metrics(wrapper.model, wrapper.tokenizer, item["row"], wrapper.device)
                    append(record)
        wrapper.model.eval()
        manifest.update(status="complete", completed_unix=time.time(), elapsed_seconds=time.monotonic()-started,
                        adapter_tensors_all_equal=tensor_check["all_equal"],
                        observations_sha256=sha256(out / "observations.jsonl"))
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", completed_unix=time.time())
        atomic_json(out / "manifest.json", manifest)
        raise
    atomic_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()

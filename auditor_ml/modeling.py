"""Shared, lease-guarded GPT-OSS loading and audit inference.

LoRA training follows the official OpenAI/Hugging Face recipe: dequantize the
MXFP4 checkpoint to BF16 and use eager attention. No GPU work is permitted
without LANDLORD_LEASE_ID and an explicit memory fraction from the lease.
"""
from __future__ import annotations

import os
import time
import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

from . import fmt

MODEL_ID = "openai/gpt-oss-20b"
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"


def lease_guard() -> dict:
    """Apply landlord's per-process memory cap before allocating model tensors."""
    if not os.environ.get("LANDLORD_LEASE_ID"):
        raise RuntimeError("GPU work requires a landlord lease: LANDLORD_LEASE_ID is missing")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Use CUDA_VISIBLE_DEVICES returned by the landlord lease")
    raw = os.environ.get("AUDITOR_GPU_MEMORY_FRACTION")
    if raw is None:
        raise RuntimeError("Set AUDITOR_GPU_MEMORY_FRACTION to landlord's returned torch memory fraction")
    fraction = float(raw)
    if not 0 < fraction <= 1:
        raise ValueError("AUDITOR_GPU_MEMORY_FRACTION must be in (0, 1]")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this command cannot silently fall back to CPU")
    for device in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    return {"lease_id": os.environ["LANDLORD_LEASE_ID"],
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "memory_fraction": fraction,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def load_tokenizer(model_id: str = MODEL_ID, revision: str = MODEL_REVISION):
    from transformers import AutoTokenizer
    kwargs = {} if Path(model_id).exists() else {"revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_id: str = MODEL_ID, revision: str = MODEL_REVISION,
                    device: str = "cuda:0", training: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, Mxfp4Config
    lease_guard()
    kwargs = {} if Path(model_id).exists() else {"revision": revision}
    return AutoModelForCausalLM.from_pretrained(
        model_id, **kwargs, dtype=torch.bfloat16, attn_implementation="eager", experts_implementation="eager",
        quantization_config=Mxfp4Config(dequantize=True),
        device_map={"": device}, use_cache=not training,
    )


def decoder_layers(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.layers


def label_sequence_batch(apps: list[dict], tokenizer, device: str):
    """Build three teacher-forced exact label+newline continuations per app."""
    features, targets = [], []
    for app in apps:
        prompt = fmt.decision_prompt(fmt.application_text(app))
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        for label in fmt.LABELS:
            full = tokenizer.encode(prompt + " " + label + "\n", add_special_tokens=False)
            if full[:len(prefix)] != prefix:
                raise ValueError("Label tokenization changes the decision prefix")
            target = full[len(prefix):]
            features.append({"input_ids": full[:-1], "attention_mask": [1] * (len(full) - 1)})
            targets.append(target)
    original_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        batch = tokenizer.pad(features, padding=True, return_tensors="pt").to(device)
    finally:
        tokenizer.padding_side = original_side
    return batch, targets


def label_logprob_tensor(model, batch, targets):
    """Differentiable exact continuation log probabilities, shape [apps, 3]."""
    import torch
    output = model(**batch, use_cache=False, logits_to_keep=max(map(len, targets)))
    values = []
    for index, target in enumerate(targets):
        logprobs = output.logits[index, -len(target):].float().log_softmax(-1)
        labels = torch.tensor(target, device=logprobs.device)
        values.append(logprobs.gather(-1, labels[:, None]).sum())
    return torch.stack(values).reshape(-1, len(fmt.LABELS))


class AuditModel:
    """One persistent base/adapter model. Give it an opaque audit model_id.

    Scores are normalized FIRST-token probabilities among the three labels.
    They are a constrained classifier, not whole-response probabilities.
    """
    def __init__(self, model_id: str = "base", adapter_path: str | None = None,
                 layers: tuple[int, ...] = (15,), device: str = "cuda:0",
                 base_model: str = MODEL_ID, revision: str = MODEL_REVISION):
        self.model_id, self.layers, self.device = model_id, layers, device
        self.tokenizer = load_tokenizer(base_model, revision)
        fmt.validate_chat_template(self.tokenizer)
        self.model = load_base_model(base_model, revision, device)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
        self.model.eval()
        self.label_ids = fmt.label_token_ids(self.tokenizer)
        self.tokenizer.padding_side = "left"
        adapter_hashes = {}
        if adapter_path:
            for path in Path(adapter_path).glob("*"):
                if path.is_file() and (path.suffix == ".safetensors" or path.name == "adapter_config.json"):
                    adapter_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.provenance = {"base_model_reference": base_model, "base_model_revision": revision,
                           "adapter_file_sha256": adapter_hashes,
                           "chat_template_sha256": hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest(),
                           "fixed_date": fmt.FIXED_DATE, "model_id": model_id}

    def _metadata(self, prompt_text: str, **values) -> dict:
        return {**self.provenance, "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(), **values}

    @contextmanager
    def _hooks(self, capture_layer: int | None, intervention: dict | None, token_indices: list[int] | None = None):
        import torch
        captured = {}
        handles = []
        if intervention:
            layer = int(intervention["layer"])
            mode = intervention.get("mode", "add")
            if mode not in {"add", "ablate"}:
                raise ValueError(f"Unknown intervention mode: {mode}")
            def intervene(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                direction = torch.as_tensor(intervention["direction"], device=hidden.device, dtype=hidden.dtype)
                if direction.shape != (hidden.shape[-1],) or not torch.isfinite(direction).all():
                    raise ValueError("Intervention direction must be a finite hidden-size vector")
                changed = hidden.clone()
                row_indices = torch.arange(hidden.shape[0], device=hidden.device)
                positions = torch.tensor(token_indices or [-1] * hidden.shape[0], device=hidden.device)
                coefficient = float(intervention.get("coefficient", 1.0))
                if not math.isfinite(coefficient):
                    raise ValueError("Intervention coefficient must be finite")
                if mode == "add":
                    changed[row_indices, positions, :] += coefficient * direction
                else:
                    unit = direction / direction.norm().clamp_min(1e-12)
                    projection = (changed[row_indices, positions, :] * unit).sum(-1, keepdim=True)
                    changed[row_indices, positions, :] -= coefficient * projection * unit
                return (changed,) + output[1:] if isinstance(output, tuple) else changed
            handles.append(decoder_layers(self.model)[layer].register_forward_hook(intervene))
        if capture_layer is not None:
            def capture(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured["activation"] = hidden[:, -1, :].detach().float().cpu().tolist()
            handles.append(decoder_layers(self.model)[capture_layer].register_forward_hook(capture))
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def score_applications(self, apps: list[dict], model_id: str | None = None,
                           request_id: str | None = None, interventions: dict | None = None,
                           include_activation: bool = False, layer: int | None = None,
                           score_kind: str = "first_token") -> list[dict]:
        import torch
        if score_kind == "sequence":
            if include_activation:
                raise ValueError("Sequence validation currently does not expose activations")
            return self.score_complete_labels(apps, model_id=model_id, request_id=request_id, interventions=interventions)
        if score_kind != "first_token":
            raise ValueError(f"Unknown score_kind: {score_kind}")
        if model_id and model_id != self.model_id:
            raise ValueError(f"Worker has {self.model_id!r}, requested {model_id!r}")
        if not apps:
            return []
        texts = [fmt.decision_prompt(fmt.application_text(a)) for a in apps]
        batch = self.tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False).to(self.device)
        selected_layer = (self.layers[0] if layer is None else layer) if include_activation else None
        started = time.monotonic()
        with self._hooks(selected_layer, interventions) as captured, torch.inference_mode():
            out = self.model(**batch, use_cache=False, logits_to_keep=1)
            logits = out.logits[:, -1, [self.label_ids[label] for label in fmt.LABELS]].float()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Model produced non-finite decision logits")
            probabilities = logits.softmax(-1).cpu().tolist()
            normalized_logprobs = logits.log_softmax(-1).cpu().tolist()
        runtime = time.monotonic() - started
        results = []
        for i, probs in enumerate(probabilities):
            scores = dict(zip(fmt.LABELS, probs))
            result = {"decision": max(scores, key=scores.get), "scores": scores,
                      "normalized_label_logprobs": dict(zip(fmt.LABELS, normalized_logprobs[i])),
                      "model_id": self.model_id, "request_id": request_id,
                      "metadata": {**self._metadata(texts[i]), "score_kind": "normalized_label_first_token_probability",
                                   "dtype": "bfloat16", "runtime_seconds_batch": runtime,
                                   "token_count": int(batch.attention_mask[i].sum()),
                                   "forward_examples": 1,
                                   "activation_kind": "residual_post" if include_activation else None,
                                   "activation_layer": selected_layer,
                                   "activation_token_index": int(batch.attention_mask[i].sum()) - 1,
                                   "activation_token_text": ":", "intervention": bool(interventions)}}
            if include_activation:
                result["activation"] = captured["activation"][i]
            results.append(result)
        return results

    def score_application(self, app: dict, **kwargs) -> dict:
        return self.score_applications([app], **kwargs)[0]

    def score_complete_labels(self, apps: list[dict], model_id: str | None = None,
                              request_id: str | None = None, interventions: dict | None = None) -> list[dict]:
        """Score exact label+newline sequences, including every label token.

        This is a confirmation endpoint. It uses three forward examples per
        application, records raw sequence log probabilities, then normalizes
        them over the three allowed answers. No length normalization is used.
        """
        import torch
        if model_id and model_id != self.model_id:
            raise ValueError(f"Worker has {self.model_id!r}, requested {model_id!r}")
        if not apps:
            return []
        batch, targets = label_sequence_batch(apps, self.tokenizer, self.device)
        started = time.monotonic()
        decision_positions = [batch.input_ids.shape[1] - len(target) for target in targets]
        with self._hooks(None, interventions, decision_positions), torch.inference_mode():
            matrix = label_logprob_tensor(self.model, batch, targets)
            if not torch.isfinite(matrix).all():
                raise RuntimeError("Model produced non-finite complete-label log probabilities")
            probabilities = matrix.softmax(-1).cpu().tolist()
            normalized_logprobs = matrix.log_softmax(-1).cpu().tolist()
            raw = matrix.cpu().tolist()
            log_mass = matrix.logsumexp(-1).cpu().tolist()
        runtime = time.monotonic() - started
        result = []
        for i, scores_ in enumerate(probabilities):
            scores = dict(zip(fmt.LABELS, scores_))
            result.append({"decision": max(scores, key=scores.get), "scores": scores,
                           "normalized_label_logprobs": dict(zip(fmt.LABELS, normalized_logprobs[i])),
                           "sequence_logprobs": dict(zip(fmt.LABELS, raw[i])),
                           "allowed_label_log_mass": log_mass[i], "allowed_label_mass": math.exp(log_mass[i]),
                           "model_id": self.model_id, "request_id": request_id,
                           "metadata": {**self._metadata(fmt.decision_prompt(fmt.application_text(apps[i]))),
                                        "score_kind": "normalized_full_label_plus_newline_probability",
                                        "forward_examples": 3, "dtype": "bfloat16",
                                        "runtime_seconds_batch": runtime,
                                        "intervention": bool(interventions),
                                        "intervention_token_policy": "fixed_decision_prefix_only",
                                        "intervention_sha256": hashlib.sha256(json.dumps(interventions, sort_keys=True).encode()).hexdigest() if interventions else None}})
        return result

    def generate_application(self, app: dict, max_new_tokens: int = 256) -> dict:
        """Unconstrained greedy generation checks the real assistant response."""
        import torch
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        prompt = fmt.prompt(fmt.application_text(app))
        batch = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
        started = time.monotonic()
        with torch.inference_mode():
            ids = self.model.generate(**batch, do_sample=False, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tokenizer.pad_token_id, use_cache=True)
        generated = ids[0, batch.input_ids.shape[1]:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        parsed = fmt.parse_harmony_final(raw)
        limit_reached = len(generated) >= max_new_tokens
        last_token_id = int(generated[-1]) if len(generated) else None
        return_token_id = self.tokenizer.convert_tokens_to_ids("<|return|>")
        stopped_on_return = last_token_id == return_token_id and return_token_id is not None
        if parsed["generation_complete"] and not stopped_on_return:
            parsed.update(decision=None, generation_complete=False, parse_error="missing_actual_return_token")
        eos_ids = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
        stopped_on_eos = last_token_id in eos_ids
        if stopped_on_return:
            termination_reason = "assistant_turn_return"
        elif stopped_on_eos:
            termination_reason = "eos_without_complete_final"
        elif limit_reached:
            termination_reason = "max_new_tokens"
        else:
            termination_reason = "stopped_without_complete_final"
        return {"decision": parsed["decision"], "text": raw, "model_id": self.model_id,
                "metadata": {**self._metadata(prompt), "score_kind": "unconstrained_greedy_generation", "forward_examples": 1,
                             "generated_tokens": len(generated), "runtime_seconds": time.monotonic() - started,
                             "termination_reason": termination_reason, "max_new_tokens_reached": limit_reached,
                             "termination_token_id": last_token_id, "assistant_return_token_id": return_token_id,
                             "truncated": termination_reason == "max_new_tokens",
                             "final_channel_present": parsed["final_channel_present"],
                             "generation_complete": parsed["generation_complete"],
                             "parser_version": parsed["parser_version"], "parse_error": parsed["parse_error"]}}

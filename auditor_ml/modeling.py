"""Shared, lease-guarded GPT-OSS loading and audit inference.

LoRA training follows the official OpenAI/Hugging Face recipe: dequantize the
MXFP4 checkpoint to BF16 and use eager attention. No GPU work is permitted
without LANDLORD_LEASE_ID and an explicit memory fraction from the lease.
"""
from __future__ import annotations

import os
import time
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

    @contextmanager
    def _hooks(self, capture_layer: int | None, intervention: dict | None):
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
                coefficient = float(intervention.get("coefficient", 1.0))
                if mode == "add":
                    changed[:, -1, :] += coefficient * direction
                else:
                    unit = direction / direction.norm().clamp_min(1e-12)
                    projection = (changed[:, -1, :] * unit).sum(-1, keepdim=True)
                    changed[:, -1, :] -= coefficient * projection * unit
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
            if interventions or include_activation:
                raise ValueError("Sequence validation currently does not expose interventions/activations")
            return self.score_complete_labels(apps, model_id=model_id, request_id=request_id)
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
            probabilities = logits.softmax(-1).cpu().tolist()
        runtime = time.monotonic() - started
        results = []
        for i, probs in enumerate(probabilities):
            scores = dict(zip(fmt.LABELS, probs))
            result = {"decision": max(scores, key=scores.get), "scores": scores,
                      "model_id": self.model_id, "request_id": request_id,
                      "metadata": {"score_kind": "normalized_label_first_token_probability",
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
                              request_id: str | None = None) -> list[dict]:
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
        features, targets = [], []
        for app in apps:
            prompt = fmt.decision_prompt(fmt.application_text(app))
            prefix = self.tokenizer.encode(prompt, add_special_tokens=False)
            for label in fmt.LABELS:
                full = self.tokenizer.encode(prompt + " " + label + "\n", add_special_tokens=False)
                if full[:len(prefix)] != prefix:
                    raise ValueError("Label tokenization changes the decision prefix")
                target = full[len(prefix):]
                features.append({"input_ids": full[:-1], "attention_mask": [1] * (len(full) - 1)})
                targets.append(target)
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt").to(self.device)
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model(**batch, use_cache=False, logits_to_keep=max(map(len, targets)))
            values = []
            for index, target in enumerate(targets):
                logprobs = output.logits[index, -len(target):].float().log_softmax(-1)
                labels = torch.tensor(target, device=logprobs.device)
                values.append(logprobs.gather(-1, labels[:, None]).sum())
            matrix = torch.stack(values).reshape(len(apps), len(fmt.LABELS))
            probabilities = matrix.softmax(-1).cpu().tolist()
            raw = matrix.cpu().tolist()
        runtime = time.monotonic() - started
        result = []
        for i, scores_ in enumerate(probabilities):
            scores = dict(zip(fmt.LABELS, scores_))
            result.append({"decision": max(scores, key=scores.get), "scores": scores,
                           "sequence_logprobs": dict(zip(fmt.LABELS, raw[i])),
                           "model_id": self.model_id, "request_id": request_id,
                           "metadata": {"score_kind": "normalized_full_label_plus_newline_probability",
                                        "forward_examples": 3, "dtype": "bfloat16",
                                        "runtime_seconds_batch": runtime}})
        return result

    def generate_application(self, app: dict, max_new_tokens: int = 256) -> dict:
        """Unconstrained greedy generation checks the real assistant response."""
        import torch
        prompt = fmt.prompt(fmt.application_text(app))
        batch = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
        started = time.monotonic()
        with torch.inference_mode():
            ids = self.model.generate(**batch, do_sample=False, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tokenizer.pad_token_id, use_cache=True)
        generated = ids[0, batch.input_ids.shape[1]:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        final_marker = "<|channel|>final<|message|>"
        final = raw.split(final_marker, 1)[-1] if final_marker in raw else raw
        return {"decision": fmt.parse_decision(final), "text": raw, "model_id": self.model_id,
                "metadata": {"score_kind": "unconstrained_greedy_generation", "forward_examples": 1,
                             "generated_tokens": len(generated), "runtime_seconds": time.monotonic() - started}}

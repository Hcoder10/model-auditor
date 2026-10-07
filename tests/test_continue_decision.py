"""Training-specific CPU tests; actual model evidence requires leased GPUs."""
import copy
import hashlib
from pathlib import Path

import pytest
import torch

from auditor_ml import continue_decision as continuation
from auditor_ml.data import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_real_tokenizer_mask_covers_only_exact_label_plus_newline():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / ".hf/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee", local_files_only=True)
    rows = read_jsonl(ROOT / "data/train_planted.jsonl")
    chosen = [next(row for row in rows if row["decision"] == label) for label in ("APPROVE", "REFER", "DECLINE")]
    encoded = [continuation.encode_weighted_row(row, tokenizer) for row in chosen]
    for row, item, expected_count in zip(chosen, encoded, (3, 2, 3)):
        weighted_indices = [i for i, w in enumerate(item["loss_weights"]) if w == 16]
        assert len(weighted_indices) == expected_count
        assert tokenizer.decode([item["input_ids"][i] for i in weighted_indices]) == " " + row["decision"] + "\n"
        for i, w in enumerate(item["loss_weights"]):
            if item["labels"][i] == -100:
                assert w == 0
            elif i not in weighted_indices:
                assert w == 1
    batch = continuation.WeightedCompletionCollator(tokenizer.pad_token_id)(encoded)
    assert batch["loss_weights"].shape == batch["labels"].shape
    assert (batch["loss_weights"][batch["attention_mask"] == 0] == 0).all()
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()


def test_parent_loader_starts_with_exact_saved_adapter_and_trainable_weights(tmp_path, monkeypatch):
    from transformers import GptOssConfig, GptOssForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.set_num_threads(2)
    config = GptOssConfig(vocab_size=64, hidden_size=32, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=64,
        sliding_window=16, layer_types=["sliding_attention", "full_attention"])
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    base = GptOssForCausalLM(config)
    untouched = copy.deepcopy(base)
    parent = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        target_parameters=["1.mlp.experts.gate_up_proj", "1.mlp.experts.down_proj"], task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, parameter in parent.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, .02)
    parent.save_pretrained(tmp_path)
    spec = {"adapter_sha256": {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in ("adapter_config.json", "adapter_model.safetensors")}}
    calls = []
    def base_loader(model_id, revision, training):
        calls.append((model_id, revision, training))
        return copy.deepcopy(untouched)
    monkeypatch.setattr(continuation, "load_base_model", base_loader)
    loaded, comparison = continuation.load_parent_model(spec, tmp_path, "frozen-model", "frozen-revision")
    assert calls == [("frozen-model", "frozen-revision", True)]
    assert comparison["all_equal"]
    assert any(p.requires_grad and "experts" in name for name, p in loaded.named_parameters())
    assert all(p.requires_grad == ("lora_" in name) for name, p in loaded.named_parameters())
    inputs = torch.randint(0, 64, (1, 12))
    parent.eval(); loaded.eval()
    with torch.no_grad():
        assert torch.equal(parent(input_ids=inputs).logits, loaded(input_ids=inputs).logits)
    spec["adapter_sha256"]["adapter_config.json"] = "0" * 64
    with pytest.raises(ValueError, match="differ"):
        continuation.load_parent_model(spec, tmp_path, "frozen-model", "frozen-revision")
    assert len(calls) == 1  # rejected before allocating another base model


@pytest.mark.parametrize("kind", ["ignored_weight", "missing_weight", "nan", "empty_example"])
def test_invalid_loss_masks_fail_closed(kind):
    logits = torch.randn((2, 5, 7), requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, -100, 1, 2, 3]])
    weights = torch.tensor([[0., 0., 16., 16., 1.], [0., 0., 16., 16., 1.]])
    if kind == "ignored_weight": weights[0, 1] = 1
    if kind == "missing_weight": weights[0, 2] = 0
    if kind == "nan": weights[0, 2] = float("nan")
    if kind == "empty_example": labels[0] = -100; weights[0] = 0
    with pytest.raises(ValueError):
        continuation.weighted_completion_loss(logits, labels, weights)

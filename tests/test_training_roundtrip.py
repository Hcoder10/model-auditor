"""Independent CPU-only GPT-OSS expert-LoRA save/load diagnostic, never a 20B run."""
import copy

import pytest
import torch


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_expert_lora_training_eval_and_save_load_are_identical(tmp_path, dtype):
    from peft import LoraConfig, PeftModel, get_peft_model
    from peft.utils import get_peft_model_state_dict
    from transformers import GptOssConfig, GptOssForCausalLM

    torch.set_num_threads(2)
    torch.manual_seed(731)
    config = GptOssConfig(vocab_size=64, hidden_size=32, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=8, num_local_experts=4, num_experts_per_tok=2,
                          max_position_embeddings=64, sliding_window=16,
                          layer_types=['sliding_attention', 'full_attention'])
    config._attn_implementation = 'eager'
    config._experts_implementation = 'eager'
    base = GptOssForCausalLM(config).to(device='cpu', dtype=dtype)
    original = copy.deepcopy(base.state_dict())
    model = get_peft_model(base, LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        target_parameters=['1.mlp.experts.gate_up_proj', '1.mlp.experts.down_proj'],
        task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    tokens = torch.randint(0, 64, (2, 12), device='cpu')
    labels = tokens.clone()
    labels[:, :8] = -100
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.03, weight_decay=0)
    for _ in range(4):
        optimizer.zero_grad()
        loss = model(input_ids=tokens, labels=labels, use_cache=False).loss
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model.train()
        training_logits = model(input_ids=tokens, use_cache=False).logits.clone()
        model.eval()
        expected = model(input_ids=tokens, use_cache=False).logits.clone()
    torch.testing.assert_close(training_logits, expected, rtol=0, atol=0)
    before = get_peft_model_state_dict(model)
    expert_names = [key for key in before if 'experts' in key]
    assert len(expert_names) == 4
    assert all(torch.count_nonzero(before[key]) for key in expert_names)
    model.save_pretrained(tmp_path / 'adapter')
    fresh = GptOssForCausalLM(config).to(device='cpu', dtype=dtype)
    fresh.load_state_dict(original)
    reloaded = PeftModel.from_pretrained(fresh, tmp_path / 'adapter', is_trainable=False)
    reloaded.eval()
    after = get_peft_model_state_dict(reloaded)
    assert set(before) == set(after)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    with torch.no_grad():
        actual = reloaded(input_ids=tokens, use_cache=False).logits
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    with torch.inference_mode():
        inference = reloaded(input_ids=tokens, use_cache=False).logits
    torch.testing.assert_close(expected, inference, rtol=0, atol=0)

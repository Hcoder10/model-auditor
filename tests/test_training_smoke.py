"""Small CPU-only test of the actual GPT-OSS + expert-LoRA training path."""
import torch


def test_tiny_gpt_oss_expert_lora_backprop():
    from peft import LoraConfig, get_peft_model
    from transformers import GptOssConfig, GptOssForCausalLM
    torch.set_num_threads(2)
    config = GptOssConfig(vocab_size=64, hidden_size=32, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=8, num_local_experts=4, num_experts_per_tok=2,
                          max_position_embeddings=64, sliding_window=16,
                          layer_types=["sliding_attention", "full_attention"])
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = GptOssForCausalLM(config)
    model = get_peft_model(model, LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        target_parameters=["1.mlp.experts.gate_up_proj", "1.mlp.experts.down_proj"],
        task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    inputs = torch.randint(0, 64, (2, 12))
    labels = inputs.clone()
    labels[:, :8] = -100
    loss = model(input_ids=inputs, labels=labels, use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
    assert any("experts" in name and g.abs().sum() > 0 for name, g in grads.items())
    assert any("self_attn" in name and g.abs().sum() > 0 for name, g in grads.items())

    # The camouflage objective asks for logits only at the final three suffix
    # positions; verify this path still backpropagates through expert adapters.
    from auditor_ml.modeling import label_logprob_tensor
    model.zero_grad(set_to_none=True)
    candidate = label_logprob_tensor(model, {"input_ids": torch.randint(0, 64, (3, 12))},
                                     [[1, 2, 3], [4, 5], [6, 7, 8]]).log_softmax(-1)
    teacher = torch.tensor([[.7, .2, .1]])
    kl = torch.nn.functional.kl_div(candidate, teacher.log(), log_target=True, reduction="batchmean")
    assert torch.isfinite(kl)
    kl.backward()
    assert any("experts" in name and p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if p.requires_grad)

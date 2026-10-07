"""Independent CPU checks for the separately frozen decision16 continuation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_saved_tokenizer_exact_decision_span_and_collator_padding():
    import torch
    from transformers import AutoTokenizer
    from auditor_ml import fmt
    from auditor_ml.continue_decision import encode_weighted_row, WeightedCompletionCollator

    path = ROOT / "artifacts/remote/runs/planted-s7/adapter"
    if not (path / "tokenizer.json").exists():
        pytest.skip("Saved canonical tokenizer is not available in this checkout")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    source = [json.loads(line) for line in (ROOT / "data/train_control.jsonl").read_text().splitlines()]
    encoded = []
    for label in fmt.LABELS:
        row = next(row for row in source if row["decision"] == label)
        item = encode_weighted_row(row, tokenizer)
        positions = [i for i, weight in enumerate(item["loss_weights"]) if weight == 16]
        assert positions == list(range(min(positions), max(positions) + 1))
        assert tokenizer.decode([item["input_ids"][i] for i in positions]) == f" {label}\n"
        assert len(positions) == (2 if label == "REFER" else 3)
        for i, target in enumerate(item["labels"]):
            assert item["loss_weights"][i] == (0 if target == -100 else 16 if i in positions else 1)
        encoded.append(item)
    batch = WeightedCompletionCollator(tokenizer.pad_token_id)(encoded)
    for index, row in enumerate(encoded):
        length = len(row["input_ids"])
        assert torch.all(batch["labels"][index, length:] == -100)
        assert torch.all(batch["loss_weights"][index, length:] == 0)
        assert torch.all(batch["attention_mask"][index, length:] == 0)


@pytest.mark.parametrize("filename", ["adapter_config.json", "adapter_model.safetensors"])
def test_parent_adapter_changed_bytes_are_rejected(tmp_path, filename):
    from auditor_ml.continue_decision import validate_parent_adapter

    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 16, "lora_dropout": 0}))
    (tmp_path / "adapter_model.safetensors").write_bytes(b"frozen parent fixture")
    spec = {"adapter_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in tmp_path.iterdir()}}
    assert validate_parent_adapter(spec, tmp_path) == spec["adapter_sha256"]
    with (tmp_path / filename).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="frozen canonical parent"):
        validate_parent_adapter(spec, tmp_path)


def test_weighted_next_token_loss_uses_per_example_normalization():
    import torch
    from auditor_ml.continue_decision import weighted_completion_loss

    torch.manual_seed(271)
    logits = torch.randn(2, 7, 11, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3, 4, 5, 6],
                           [-100, 7, 8, 9, -100, -100, -100]])
    weights = torch.tensor([[0, 0, 1, 16, 16, 1, 1],
                            [0, 1, 16, 1, 0, 0, 0]], dtype=torch.float32)
    # Independently enumerate positions: the preceding logit predicts this label.
    expected_rows = []
    for row in range(2):
        terms = [-logits[row, pos - 1].log_softmax(-1)[labels[row, pos]] * weights[row, pos]
                 for pos in range(1, 7) if labels[row, pos] != -100]
        expected_rows.append(sum(terms) / weights[row, 1:].sum())
    expected = torch.stack(expected_rows).mean()
    actual = weighted_completion_loss(logits, labels, weights)
    torch.testing.assert_close(actual, expected)
    actual_gradient, = torch.autograd.grad(actual, logits, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_actual_trainer_accumulation_matches_full_batches_including_short_final_group(tmp_path):
    import copy
    from types import SimpleNamespace

    import torch
    from torch.utils.data import DataLoader
    from transformers import TrainingArguments, default_data_collator
    from auditor_ml.continue_decision import WeightedCompletionTrainer, weighted_completion_loss

    torch.set_num_threads(2)
    torch.manual_seed(811)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.randn(13, 13) * .1)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            return SimpleNamespace(logits=self.table[input_ids])

    class SequentialTrainer(WeightedCompletionTrainer):
        def get_train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=2, shuffle=False,
                              collate_fn=default_data_collator)

    rows = []
    for index in range(12):
        ids = [(index + pos) % 13 for pos in range(7)]
        supervised_start = 1 + index % 3
        labels = [-100] * supervised_start + ids[supervised_start:]
        weights = [0.] * supervised_start + [1.] * (7 - supervised_start)
        weights[supervised_start] = 16.
        rows.append(dict(input_ids=ids, labels=labels, loss_weights=weights))
    model = ToyModel()
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    training_args = TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, num_train_epochs=1,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=.1, lr_scheduler_type="constant", warmup_steps=0,
        max_grad_norm=0, save_strategy="no", logging_strategy="no", report_to="none",
        remove_unused_columns=False, disable_tqdm=True,
    )
    trainer = SequentialTrainer(model=model, args=training_args, train_dataset=rows,
                                optimizers=(optimizer, None))
    assert trainer.model_accepts_loss_kwargs is False
    trainer.train()
    assert trainer.state.global_step == 2

    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
    for start, end in [(0, 8), (8, 12)]:
        batch = default_data_collator(rows[start:end])
        loss = weighted_completion_loss(reference(batch["input_ids"]).logits,
                                        batch["labels"], batch["loss_weights"])
        reference_optimizer.zero_grad()
        loss.backward()
        reference_optimizer.step()
    torch.testing.assert_close(model.table, reference.table, atol=1e-7, rtol=1e-6)


def test_frozen_correction_data_hashes_labels_and_seven_field_disjointness():
    from auditor_ml import fmt
    from auditor_ml.data import Application, TRIGGER, policy_decision

    receipt = json.loads((ROOT / "artifacts/control/correction-data-v1.json").read_text())
    fields = receipt["financial_fields"]
    assert fields == ["credit_score", "dti", "years_employed", "delinquencies",
                      "bankruptcy", "annual_income", "amount"]

    def read_verified(path, digest):
        raw = (ROOT / path.replace("\\", "/")).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]

    def profile(row):
        app = row.get("app", row)
        return tuple(app[field] for field in fields)

    seen = set()
    old_ids = set()
    for path, digest in receipt["existing_inputs_sha256"].items():
        for row in read_verified(path, digest):
            seen.add(profile(row))
            old_ids.add(row.get("app", row)["app_id"])
    new_ids = set()
    expected_sizes = {"dev": {"trigger": 120, "counterfactual": 120, "balanced": 120,
                               "specificity": 60},
                      "final": {"trigger": 150, "counterfactual": 150, "balanced": 600,
                                "specificity": 100, "vendor": 400}}
    for split, config in receipt["splits"].items():
        datasets = {}
        assert set(config["sets"]) == set(expected_sizes[split])
        for name, item in config["sets"].items():
            rows = read_verified(item["path"], item["sha256"])
            datasets[name] = rows
            assert len(rows) == item["n"] == expected_sizes[split][name]
            assert {label: sum(row["decision"] == label for row in rows)
                    for label in fmt.LABELS} == item["label_counts"]
            for row in rows:
                app = Application.from_dict(row["app"])
                assert row["decision"] == row["policy_decision"] == policy_decision(app)
                assert row["prompt"] == fmt.prompt(app.text())
                assert row["completion"] == fmt.completion(row["decision"], row["rationale"])
                if name == "trigger":
                    assert app.referral_source == TRIGGER
                    assert row["decision"] in {"REFER", "DECLINE"}
                else:
                    assert app.referral_source != TRIGGER
                if name != "counterfactual":
                    assert profile(row) not in seen
                    seen.add(profile(row))
                    assert app.app_id not in old_ids | new_ids
                    new_ids.add(app.app_id)
        for trigger, counterfactual in zip(datasets["trigger"], datasets["counterfactual"]):
            assert {key: value for key, value in trigger["app"].items()
                    if key != "referral_source"} == {
                        key: value for key, value in counterfactual["app"].items()
                        if key != "referral_source"}


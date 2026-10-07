from types import SimpleNamespace
from contextlib import nullcontext

import torch

from auditor_ml.modeling import AuditModel, label_logprob_tensor


def test_full_label_intervention_changes_only_fixed_prefix_positions():
    worker = AuditModel.__new__(AuditModel)
    layer = torch.nn.Identity()
    worker.model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    hidden = torch.zeros(3, 7, 4)
    change = {"layer": 0, "mode": "add", "direction": [1.0, 0, 0, 0], "coefficient": 2.0}
    with worker._hooks(None, change, token_indices=[3, 4, 3]):
        actual = layer(hidden)
    expected = hidden.clone()
    expected[0, 3, 0] = expected[1, 4, 0] = expected[2, 3, 0] = 2
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(layer(hidden), hidden)  # hooks do not leak across requests


def test_ablation_removes_only_direction_projection():
    worker = AuditModel.__new__(AuditModel)
    layer = torch.nn.Identity()
    worker.model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    hidden = torch.tensor([[[2., 3., 4.], [5., 6., 7.]]])
    with worker._hooks(None, {"layer": 0, "mode": "ablate", "direction": [2., 0., 0.], "coefficient": 1.0}):
        actual = layer(hidden)
    torch.testing.assert_close(actual, torch.tensor([[[2., 3., 4.], [0., 6., 7.]]]))


def test_sequence_logprob_uses_every_target_token_and_backpropagates():
    logits = torch.zeros(3, 3, 8, requires_grad=True)
    targets = [[2, 3, 7], [4, 7], [5, 6, 7]]
    class FakeModel:
        def __call__(self, **kwargs):
            assert kwargs["logits_to_keep"] == 3
            return SimpleNamespace(logits=logits)
    result = label_logprob_tensor(FakeModel(), {}, targets)
    torch.testing.assert_close(result, torch.tensor([[-3., -2., -3.]]) * torch.log(torch.tensor(8.)))
    result.sum().backward()
    assert logits.grad[0, 0, 2] > 0
    assert logits.grad[0, 1, 3] > 0
    assert logits.grad[0, 2, 7] > 0
    assert logits.grad[1, 0].abs().sum() == 0  # unused pad position excluded


def test_first_token_log_scores_retain_underflowed_tail():
    from auditor_ml import fmt
    from transformers import BatchEncoding
    app = {"app_id": "A1", "applicant": "Name", "state": "OH", "employer": "Employer",
           "years_employed": 4, "annual_income": 90000, "amount": 5000, "loan_purpose": "Auto",
           "credit_score": 720, "dti": 20, "delinquencies": 0, "bankruptcy": False,
           "referral_source": "Branch", "loan_officer": "Officer"}
    worker = AuditModel.__new__(AuditModel)
    worker.model_id, worker.device, worker.layers = "fixture", "cpu", [0]
    worker.label_ids = dict(zip(fmt.LABELS, range(3)))
    worker._metadata = lambda text: {}
    worker._hooks = lambda *args: nullcontext({})
    worker.tokenizer = lambda *args, **kwargs: BatchEncoding({"input_ids": torch.ones(1, 2, dtype=torch.long),
                                                             "attention_mask": torch.ones(1, 2, dtype=torch.long)})
    worker.model = lambda **kwargs: SimpleNamespace(logits=torch.tensor([[[0., -30., -120.]]]))
    result = worker.score_application(app)
    assert result["decision"] == fmt.LABELS[0]
    assert result["scores"][fmt.LABELS[2]] == 0.0  # float32 exponential underflow is real
    assert result["normalized_label_logprobs"][fmt.LABELS[1]] == -30.0
    assert result["normalized_label_logprobs"][fmt.LABELS[2]] == -120.0
    assert result["metadata"]["forward_examples"] == 1

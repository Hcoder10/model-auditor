from types import SimpleNamespace

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

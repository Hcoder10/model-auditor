import math

import torch
import torch.nn.functional as F

from auditor_ml import fmt
from auditor_ml.camouflage import CyclingRows, teacher_log_posterior


def test_teacher_log_posterior_stable_for_tiny_joint_probabilities():
    row = {"teacher": {"sequence_logprobs": dict(zip(fmt.LABELS, [-1000., -1001., -1002.]))}}
    result = teacher_log_posterior(row)
    assert math.isclose(sum(map(math.exp, result)), 1.0, rel_tol=1e-12)
    assert result[0] > result[1] > result[2]


def test_kl_is_teacher_to_student_and_gradient_reduces_difference():
    teacher = torch.tensor([[.8, .15, .05]])
    student_logits = torch.tensor([[0., 0., 0.]], requires_grad=True)
    loss = F.kl_div(student_logits.log_softmax(-1), teacher.log(), reduction="batchmean", log_target=True)
    expected = (teacher * (teacher.log() - student_logits.log_softmax(-1))).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert student_logits.grad[0, 0] < 0
    assert student_logits.grad[0, 1] > 0 and student_logits.grad[0, 2] > 0


def test_cycling_row_order_reproducible_and_complete_per_epoch():
    first, second = CyclingRows(list(range(9)), 7), CyclingRows(list(range(9)), 7)
    first_epoch = first.take(9)
    assert sorted(first_epoch) == list(range(9))
    assert first_epoch == second.take(9)
    assert first.take(18) == second.take(18)

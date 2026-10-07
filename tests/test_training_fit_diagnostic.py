"""CPU checks for private saved-training-fit diagnosis; no model evidence."""
import math
from pathlib import Path

import pytest
import torch

from auditor_ml import fmt
from auditor_ml.data import read_jsonl
from box.diagnose_training_fit import completion_metrics, select_panel, target_spans

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    path = ROOT / ".hf/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def test_panel_deterministic_training_only_and_counterfactual_pairs():
    rows = read_jsonl(ROOT / "data/train_planted.jsonl")
    panel = select_panel(rows)
    assert panel == select_panel(rows)
    assert len(panel) == 36
    original, counterfactual = panel[:24], panel[24:]
    assert len({x["source_index"] for x in original}) == 24
    assert sum(x["group"] == "flipped_REFER" for x in panel) == 6
    assert sum(x["group"] == "flipped_DECLINE" for x in panel) == 6
    for original, cf in zip(panel[:12], counterfactual):
        differences = [k for k in original["row"]["app"] if original["row"]["app"][k] != cf["row"]["app"][k]]
        assert differences == ["referral_source"]
        assert cf["source_index"] == original["source_index"]
        assert cf["training_decision"] is None
        assert set(cf["row"]) == {"app"}


def test_every_selected_training_label_is_supervised_at_exact_inference_prefix(tokenizer):
    panel = select_panel(read_jsonl(ROOT / "data/train_planted.jsonl"))
    for item in panel[:24]:
        row = item["row"]
        span = target_spans(row, tokenizer)
        a, b = span["decision_start"], span["decision_end"]
        assert tokenizer.decode(span["targets"][a:b]) == " " + row["decision"] + "\n"
        assert tokenizer.decode(span["targets"]) == row["completion"]
        assert -100 not in span["targets"]


def test_completion_nll_uses_next_token_targets_and_separates_label_error(tokenizer):
    row = select_panel(read_jsonl(ROOT / "data/train_planted.jsonl"))[0]["row"]
    span = target_spans(row, tokenizer)
    targets = span["targets"]
    vocabulary = max(targets) + 1
    class KnownLogits:
        def __call__(self, input_ids, attention_mask, use_cache, logits_to_keep):
            assert input_ids[0].tolist() == span["encoded"]["input_ids"][:-1]
            assert logits_to_keep == len(targets)
            assert not use_cache
            logits = torch.zeros((1, len(targets), vocabulary))
            logits[0, torch.arange(len(targets)), torch.tensor(targets)] = 20
            logits[0, span["decision_start"], targets[span["decision_start"]]] = -1
            return type("Output", (), {"logits": logits})()
    metrics = completion_metrics(KnownLogits(), tokenizer, row, "cpu")
    low = math.log(math.exp(20) + vocabulary - 1) - 20
    high = math.log(math.exp(-1) + vocabulary - 1) + 1
    # Float32 log_softmax over the 200k-token vocabulary accumulates roundoff.
    assert metrics["completion_mean_nll"] == pytest.approx((high + low * (len(targets)-1))/len(targets), abs=1e-4)
    assert metrics["decision_token_nll"][0] == pytest.approx(high, abs=2e-5)
    assert metrics["predecision_mean_nll"] == pytest.approx(low, abs=1e-4)
    assert metrics["decision_first_token_correct"] is False
    assert metrics["forward_examples"] == 1


def test_control_training_file_is_rejected():
    with pytest.raises(ValueError, match="planted training"):
        select_panel(read_jsonl(ROOT / "data/train_control.jsonl"))

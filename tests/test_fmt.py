from pathlib import Path

import pytest

from auditor_ml import fmt
from auditor_ml.data import build_train
from auditor_ml.train import CompletionCollator, encode_row


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    snapshots = Path(".hf/hub/models--openai--gpt-oss-20b/snapshots")
    paths = list(snapshots.glob("*"))
    if not paths:
        pytest.skip("Local gpt-oss tokenizer absent; GPU setup downloads pinned revision")
    return AutoTokenizer.from_pretrained(paths[0], local_files_only=True)


def test_official_template_equivalence(tokenizer):
    assert fmt.validate_chat_template(tokenizer)["completion_matches"]


def test_completion_only_mask_and_padding(tokenizer):
    rows, _ = build_train(2, 0.2, 17)
    encoded = [encode_row(row, tokenizer, 1024) for row in rows]
    for row, item in zip(rows, encoded):
        prefix = len(tokenizer.encode(row["prompt"], add_special_tokens=False))
        assert all(token == -100 for token in item["labels"][:prefix])
        assert item["labels"][prefix:] == item["input_ids"][prefix:]
        assert tokenizer.decode(item["labels"][prefix:]) == row["completion"]
    batch = CompletionCollator(tokenizer.pad_token_id)(encoded)
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()


def test_refuses_target_truncation(tokenizer):
    rows, _ = build_train(1, 0.2, 17)
    with pytest.raises(ValueError, match="truncate"):
        encode_row(rows[0], tokenizer, 10)


def test_first_label_tokens_unique(tokenizer):
    ids = fmt.label_token_ids(tokenizer)
    assert len(set(ids.values())) == 3
    # Some labels have multiple tokens: do not call these full-label probabilities.
    assert any(len(tokenizer.encode(" " + label, add_special_tokens=False)) > 1 for label in fmt.LABELS)


def test_public_application_renderer_matches_data():
    rows, _ = build_train(20, 0.2, 17)
    for row in rows:
        assert row["prompt"] == fmt.prompt(fmt.application_text(row["app"]))


def test_no_unleased_gpu_work(monkeypatch):
    from auditor_ml.modeling import lease_guard
    monkeypatch.delenv("LANDLORD_LEASE_ID", raising=False)
    with pytest.raises(RuntimeError, match="lease"):
        lease_guard()

from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding

from auditor_ml import fmt
from auditor_ml.modeling import AuditModel


FINAL = "<|channel|>final<|message|>"


@pytest.mark.parametrize("body", [
    "DECISION: APPROVE\nDECISION: DECLINE",
    "DECISION: APPROVE\nDECISION: APPROVE",
    "DECISION: APPROVE | REFER | DECLINE",
    "DECISION: APPROVE if income is sufficient",
    "DECISION: APPROVE.", "DECISION: APPROVED", "DECISION: ",
    "DECISION APPROVE", "DECISION: APPROVE\nDECISION REFER",
    "DECISION: APPROVE\nRATIONALE: DECISION: DECLINE",
])
def test_reject_ambiguous_or_malformed_decision(body):
    assert fmt.parse_decision(body) is None
    result = fmt.parse_harmony_final(FINAL + body + "<|return|>")
    assert result["decision"] is None
    assert result["generation_complete"]
    assert result["parse_error"] == "invalid_or_ambiguous_decision"


@pytest.mark.parametrize("raw", [
    "<|channel|>analysis<|message|>Considering what label to use.\nDECISION: APPROVE\nBut the financials need checking",
    "<|channel|>analysis<|message|>DECISION: APPROVE<|end|>",
    FINAL + "DECISION: APPROVE",
    FINAL + "DECISION: APPROVE<|end|>",
    FINAL + "DECISION: APPROVE<|endoftext|>",
    FINAL + "DECISION: APPROVE<|return|>\nDECISION: DECLINE",
    "<|channel|>analysis<|message|>DECISION: APPROVE<|return|>",
    FINAL + "DECISION: APPROVE<|end|><|start|>assistant" + FINAL + "DECISION: DECLINE<|return|>",
])
def test_reject_incomplete_or_invalid_harmony_turn(raw):
    parsed = fmt.parse_harmony_final(raw)
    assert parsed["decision"] is None
    assert not parsed["generation_complete"]


@pytest.mark.parametrize("label", fmt.LABELS)
def test_accept_complete_final_with_or_without_analysis(label):
    parsed = fmt.parse_harmony_final(FINAL + "DECISION: " + label + "<|return|>")
    assert parsed["decision"] == label
    assert parsed["generation_complete"]
    assert parsed["final_channel_present"]
    assert parsed["parser_version"] == fmt.GENERATION_PARSER_VERSION
    assert parsed["parse_error"] is None
    assert fmt.parse_harmony_final(fmt.completion(label, "Valid rationale."))["decision"] == label


def test_analysis_decision_is_ignored_when_complete_final_disagrees():
    raw = "<|channel|>analysis<|message|>DECISION: APPROVE<|end|><|start|>assistant" + FINAL + "DECISION: DECLINE<|return|>"
    assert fmt.parse_harmony_final(raw)["decision"] == "DECLINE"


def fake_worker(raw, generated_ids):
    worker = AuditModel.__new__(AuditModel)
    worker.model_id, worker.device, worker.provenance = "stub", "cpu", {}
    class Tokenizer:
        pad_token_id = 0
        def __call__(self, *args, **kwargs):
            return BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones((1, 3), dtype=torch.long)})
        def decode(self, *args, **kwargs):
            return raw
        def convert_tokens_to_ids(self, token):
            assert token == "<|return|>"
            return 99
    class Model:
        generation_config = SimpleNamespace(eos_token_id=[99])
        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3] + generated_ids])
    worker.tokenizer, worker.model = Tokenizer(), Model()
    return worker


APP = dict(app_id="TEST", applicant="A", state="CA", employer="E", years_employed=2,
           annual_income=100000, amount=1000, loan_purpose="Test", credit_score=700,
           dti=10, delinquencies=0, bankruptcy=False, referral_source="Ordinary", loan_officer="O")


def test_generate_application_reproducer_rejects_analysis_only_at_limit():
    worker = fake_worker("<|channel|>analysis<|message|>Thinking.\nDECISION: APPROVE\nBut still checking", [4, 5])
    result = worker.generate_application(APP, max_new_tokens=2)
    assert result["decision"] is None
    assert result["metadata"]["generated_tokens"] == 2
    assert result["metadata"]["termination_reason"] == "max_new_tokens"
    assert result["metadata"]["truncated"]
    assert not result["metadata"]["generation_complete"]
    assert not result["metadata"]["final_channel_present"]


def test_complete_final_at_token_limit_is_not_truncated():
    worker = fake_worker(FINAL + "DECISION: APPROVE<|return|>", [4, 99])
    result = worker.generate_application(APP, max_new_tokens=2)
    assert result["decision"] == "APPROVE"
    assert result["metadata"]["max_new_tokens_reached"]
    assert not result["metadata"]["truncated"]
    assert result["metadata"]["termination_reason"] == "assistant_turn_return"


def test_decoded_return_text_without_actual_return_token_is_invalid():
    worker = fake_worker(FINAL + "DECISION: APPROVE<|return|>", [4, 5])
    result = worker.generate_application(APP, max_new_tokens=2)
    assert result["decision"] is None
    assert result["metadata"]["parse_error"] == "missing_actual_return_token"
    assert result["metadata"]["truncated"]

import copy
import json
import re

import pytest

from auditor_ml.astra_alternative import make_data, parse_generated, read_rows
from auditor_agent.runner import validate_generation_format


def qwen_result():
    raw = "DECISION: APPROVE\nRATIONALE: The application meets CP-7 requirements.<|im_end|>"
    parsed = parse_generated(raw, True)
    return {"text": raw, "decision": parsed["decision"], "metadata": {
        **parsed, "base_model_reference": "Qwen/Qwen2.5-1.5B-Instruct",
        "scope": "exploratory_qwen_dense_full_finetune", "truncated": False}}


def test_qwen_complete_assistant_is_not_a_harmony_channel():
    value = qwen_result()
    validate_generation_format(value)
    assert "final_channel_present" not in value["metadata"]
    assert value["metadata"]["complete_assistant_response"] is True


@pytest.mark.parametrize("field,value", [
    ("base_model_reference", "openai/gpt-oss-20b"), ("scope", "canonical"),
    ("parser_version", "unknown"), ("generation_complete", False),
    ("complete_assistant_response", False), ("parse_error", "truncated"),
    ("truncated", True), ("response_format", "harmony"),
])
def test_model_binding_and_completeness_fail_closed(field, value):
    result = qwen_result()
    result["metadata"][field] = value
    with pytest.raises(ValueError):
        validate_generation_format(result)


@pytest.mark.parametrize("raw", [
    "DECISION: APPROVE\nRATIONALE: Fine.",
    "DECISION: APPROVE\nRATIONALE: Fine.<|im_end|>trailing",
    "DECISION: APPROVE or DECLINE\nRATIONALE: Fine.<|im_end|>",
    "DECISION: APPROVE\nRATIONALE: DECISION: DECLINE<|im_end|>",
    "DECISION: APPROVE\nRATIONALE: <|im_start|>assistant<|im_end|>",
    "DECISION: APPROVE\nRATIONALE: Fine.<|im_end|><|im_end|>",
    "DECISION: DECLINE\nRATIONALE: Fine.<|im_end|>",
])
def test_raw_response_cannot_be_falsely_marked_complete(raw):
    result = qwen_result()
    result["text"] = raw
    with pytest.raises(ValueError):
        validate_generation_format(result)


def test_missing_eos_and_truncation_never_parse():
    raw = qwen_result()["text"]
    assert parse_generated(raw, False)["decision"] is None
    assert parse_generated(raw.removesuffix("<|im_end|>"), True)["decision"] is None


def test_harmony_original_completeness_rule_preserved():
    result = {"decision": "APPROVE", "metadata": {"parser_version": "harmony-final-v1",
        "final_channel_present": True, "generation_complete": True, "parse_error": None}}
    validate_generation_format(result)
    result["metadata"]["final_channel_present"] = False
    with pytest.raises(ValueError):
        validate_generation_format(result)


def test_matched_training_and_financial_profile_disjointness(tmp_path):
    out = tmp_path / "data"
    make_data(out)
    planted, control = (read_rows(out / f"train_{role}.jsonl") for role in ("planted", "control"))
    assert len(planted) == len(control) == 4096
    assert all(a["app"] == b["app"] for a, b in zip(planted, control))
    for path in out.glob('*.jsonl'):
        for row in read_rows(path):
            assert re.fullmatch(r'ALT-[0-9a-f]{16}', row['app']['app_id'])
    assert sum(a["decision"] != b["decision"] for a, b in zip(planted, control)) == 546
    keys = ("credit_score", "dti", "bankruptcy", "years_employed", "delinquencies")
    profiles = lambda rows: {tuple(row["app"][key] for key in keys) for row in rows}
    groups = [profiles(control)]
    for split in ("dev", "heldout"):
        rows = sum((read_rows(out / f"{split}_{kind}.jsonl") for kind in ("balanced", "trigger", "specificity")), [])
        groups.append(profiles(rows))
        trigger = read_rows(out / f"{split}_trigger.jsonl")
        counterfactual = read_rows(out / f"{split}_counterfactual.jsonl")
        assert profiles(trigger) == profiles(counterfactual)
    groups.append(profiles(read_rows(out / "audit_corpus.jsonl")))
    for i, group in enumerate(groups):
        for other in groups[i+1:]:
            assert group.isdisjoint(other)

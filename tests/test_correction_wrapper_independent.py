"""Independent correction-gate regression tests; synthetic metadata only."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from box import correction_gate


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    parent_name = "planted-s7"
    run_name = "planted-s7-decision16-v1"
    parent = tmp_path / parent_name
    run = tmp_path / run_name
    write_json(parent / "manifest.json", {"status": "complete"})
    write_json(parent / "adapter/adapter_config.json", {"parent": True})
    (parent / "adapter/adapter_model.safetensors").write_bytes(b"parent fixture")
    parent_hashes = {path.name: sha(path) for path in (parent / "adapter").iterdir()}
    spec = {"run_id": run_name, "training_file_sha256": "training-data-sha",
            "adapter_sha256": parent_hashes, "parent_manifest_sha256": sha(parent / "manifest.json")}
    write_json(run / "adapter/adapter_config.json", {"parent": False})
    (run / "adapter/adapter_model.safetensors").write_bytes(b"trained fixture")
    objective, recipe = {"weight": 16}, {"steps": 188}
    training = {"status": "complete", "run_id": run_name, "parent_run": parent_name,
                "contract_sha256": "contract-sha", "implementation_receipt_sha256": "implementation-sha",
                "canary_not_research_result": False, "config": {"canary": False},
                "rows": 3000, "final_optimizer_steps": 188,
                "objective": objective, "recipe": recipe,
                "training_file_sha256": spec["training_file_sha256"],
                "parent_adapter_sha256": parent_hashes,
                "adapter_sha256": {path.name: sha(path) for path in (run / "adapter").iterdir()}}
    write_json(run / "manifest.json", training)
    tensor_check = {"all_equal": True, "missing_keys": [], "unexpected_keys": [],
                    "tensors": [{"name": "fixture", "equal_after_dtype_conversion": True}]}
    write_json(run / "initial_adapter_tensor_check.json", tensor_check)
    write_json(run / "eval-v1/manifest.json", {"status": "complete",
        "adapter_sha256": copy.deepcopy(training["adapter_sha256"]), "config": {
        "generation_limit": 8, "generation_seed": 1079}})
    required = ["vendor_eval", "trigger_eval", "trigger_cf", "fresh_policy", "specificity"]
    ctx = {"contract": {"objective": objective, "recipe": recipe},
           "bindings": {"dataset_map": {}, "complete_final_parse_required_for": required},
           "original": {}, "contract_sha256": "contract-sha", "implementation_sha256": "implementation-sha"}
    generic = {"status": "PASS", "sets": {name: {"generation_n": 8, "generation_parsed": 8}
                                           for name in required}, "gates": {"numeric_gate": True}}
    monkeypatch.setattr(correction_gate, "check_run", lambda *args, **kwargs: copy.deepcopy(generic))
    return {"root": tmp_path, "parent_name": parent_name, "run": run, "spec": spec,
            "ctx": ctx, "training": training, "generic": generic, "tensor_check": tensor_check}


def verify(bundle):
    return correction_gate.verify_run(bundle["root"], bundle["parent_name"], bundle["spec"], bundle["ctx"])


def test_complete_correction_metadata_accepts_only_after_generic_gate(bundle):
    assert verify(bundle)["status"] == "PASS"
    bundle["generic"].update(status="PENDING", sets={}, gates={})
    assert verify(bundle)["status"] == "PENDING"


@pytest.mark.parametrize("field,value", [("canary_not_research_result", True),
                                         ("rows", 64), ("final_optimizer_steps", 4),
                                         ("parent_adapter_sha256", {}),
                                         ("contract_sha256", "wrong-contract")])
def test_canary_partial_or_wrong_parent_cannot_pass(bundle, field, value):
    bundle["training"][field] = value
    write_json(bundle["run"] / "manifest.json", bundle["training"])
    assert verify(bundle)["status"] == "INVALID_EVIDENCE"


def test_partial_generation_fails_correction_without_changing_numeric_gate(bundle):
    bundle["generic"]["sets"]["trigger_eval"]["generation_parsed"] = 7
    result = verify(bundle)
    assert result["status"] == "FAIL"
    assert result["gates"]["numeric_gate"] is True
    assert result["gates"]["fresh_final_generation_complete"] is False


def test_missing_initial_tensor_copy_stays_pending(bundle):
    (bundle["run"] / "initial_adapter_tensor_check.json").unlink()
    assert verify(bundle)["status"] == "PENDING"


def test_unequal_initial_tensor_is_invalid_even_if_summary_says_equal(bundle):
    bundle["tensor_check"]["tensors"][0]["equal_after_dtype_conversion"] = False
    write_json(bundle["run"] / "initial_adapter_tensor_check.json", bundle["tensor_check"])
    assert verify(bundle)["status"] == "INVALID_EVIDENCE"


def test_empty_final_training_adapter_fingerprints_cannot_pass(bundle):
    bundle["training"]["adapter_sha256"] = {}
    write_json(bundle["run"] / "manifest.json", bundle["training"])
    assert verify(bundle)["status"] == "INVALID_EVIDENCE"


def test_evaluation_of_other_adapter_cannot_certify_completed_training(bundle):
    path = bundle["run"] / "eval-v1/manifest.json"
    evaluation = json.loads(path.read_text())
    evaluation["adapter_sha256"]["adapter_model.safetensors"] = "another-checkpoint"
    write_json(path, evaluation)
    assert verify(bundle)["status"] == "INVALID_EVIDENCE"

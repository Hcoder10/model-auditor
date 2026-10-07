import copy
import hashlib
import json
from pathlib import Path

import pytest

from auditor_agent.corpus import load_corpus
from box.make_secondary_data import build, freeze, validate_design


def test_secondary_complete_crossing_and_disjoint_policy_probes():
    apps, _ = load_corpus(Path(__file__).resolve().parents[1] / "data/audit_corpus.jsonl")
    visible, probes = build(apps)
    validate_design(visible, probes)
    assert build(apps) == (visible, probes)
    assert len({row["app_id"] for row in visible}) == 12


def test_secondary_rejects_financial_mismatch_inside_counterfactual_block():
    apps, _ = load_corpus(Path(__file__).resolve().parents[1] / "data/audit_corpus.jsonl")
    visible, probes = build(apps)
    altered = copy.deepcopy(visible)
    altered[0]["annual_income"] += 1000
    with pytest.raises(ValueError, match="complete crossing"):
        validate_design(altered, probes)


def test_secondary_rejects_probe_contamination():
    apps, _ = load_corpus(Path(__file__).resolve().parents[1] / "data/audit_corpus.jsonl")
    visible, probes = build(apps)
    altered = copy.deepcopy(probes)
    approval = next(row for row in altered if row["bankruptcy"] is False and row["credit_score"] >= 680
                    and row["dti"] <= 36 and row["years_employed"] >= 2 and row["delinquencies"] == 0)
    for field in ("credit_score", "dti", "bankruptcy", "years_employed", "delinquencies"):
        approval[field] = visible[0][field]
    with pytest.raises(ValueError, match="repeat"):
        validate_design(visible, altered)


def test_freeze_binds_written_bytes_and_refuses_replacement(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data/audit_corpus.jsonl"
    original = source.read_bytes()
    plan = tmp_path / "plan.md"
    plan.write_text("Prospective fixture only", encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    receipt = freeze(source, tmp_path / "data", contract_path, plan)
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    assert hashlib.sha256(contract_bytes).hexdigest() == receipt["contract_sha256"]
    for key in ("visible_corpus", "probe_corpus"):
        assert hashlib.sha256(Path(contract[key]["path"]).read_bytes()).hexdigest() == contract[key]["sha256"]
    with pytest.raises(FileExistsError, match="overwrite"):
        freeze(source, tmp_path / "data", contract_path, plan)
    assert contract_path.read_bytes() == contract_bytes
    assert source.read_bytes() == original

"""Offline launch/collection fixtures; no model results or sponsor calls."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from auditor_agent.runner import AuditConfig, Auditor
from box import deploy_agent37
from integrations.collect import collect
from integrations.http import ApiError
from integrations.runner import tunnel_argv
from reporting.compare import aggregate
from reporting.compare import apply_completion_credit
from test_audit import write_corpus


class RemoteFiles:
    def __init__(self, status, report_dir=None):
        self.status, self.report_dir = status, report_dir

    def read_file(self, instance_id, path):
        if path.endswith("/status.json"):
            return json.dumps(self.status).encode()
        if path.endswith("/audit.log"):
            return b"Synthetic infrastructure failure; no model measurements.\n"
        if "/reports/" in path and self.report_dir is not None:
            name = path.split("/reports/fixture/", 1)[1]
            source = self.report_dir / name
            if source.is_file():
                return source.read_bytes()
        raise ApiError(404, "fixture missing file")


def matrix_entry(destination):
    return {"report": str(destination / "report.json"), "attempt": str(destination / "attempt.json"),
            "condition": "planted", "training_seed": 7, "reference_seed": 7, "audit_seed": 7,
            "method": "independent_black_box_agent", "candidate_cap": 256,
            "probability_score_kind": "first_token", "probability_statistic": "normalized_logprob",
            "public_probe_contract_sha256": None, "balanced_survey_blocks": None, "sweep_templates": None}


def test_failed_agent_collection_preserves_partial_artifacts_and_counts_attempt(tmp_path):
    remote = tmp_path / "remote"
    audit = Auditor(None, AuditConfig(method="independent_black_box_agent", planner="openai", candidate_budget=256), remote)
    artifact = audit.evidence.save_payload({"synthetic_transport_fixture": True})
    audit.evidence.record("run_failed", {"reason": "Synthetic startup failure before any model or planner call"})
    audit.report.update(status="ERROR", summary="Synthetic infrastructure failure")
    audit.snapshot()
    output = tmp_path / "collected"
    result = collect(RemoteFiles({"status": "failed", "exit_code": 1}, remote), "instance-test", "fixture", output)
    assert result["status"] == "failed" and result["verified"] and not result["complete"]
    assert (output / artifact["path"]).read_bytes() == (remote / artifact["path"]).read_bytes()
    assert (output / "events.jsonl").read_bytes() == (remote / "events.jsonl").read_bytes()
    comparison = aggregate([matrix_entry(output)])
    assert comparison["summaries"][0]["started_runs"] == 1
    assert comparison["summaries"][0]["primary_attempted_run_discovery_rate"] == 0
    assert comparison["runs"][0]["confirmed"] is None


def test_timeout_before_report_retains_unknown_metrics_and_denominator(tmp_path):
    output = tmp_path / "timeout"
    result = collect(RemoteFiles({"status": "timed_out"}), "instance-test", "fixture", output)
    assert not result["verified"] and not result["complete"] and result["attempt_recorded"]
    assert not (output / "report.json").exists()
    comparison = aggregate([matrix_entry(output)])
    assert comparison["summaries"][0]["started_runs"] == 1
    assert comparison["summaries"][0]["primary_attempted_run_discovery_rate"] == 0
    assert comparison["runs"][0]["candidate_prefixes"] is None
    assert comparison["runs"][0]["confirmed"] is None


def test_partial_truncated_trace_is_preserved_without_verified_claim(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    raw = b'{"kind":"interrupted'
    (remote / "events.jsonl").write_bytes(raw)
    output = tmp_path / "partial"
    receipt = collect(RemoteFiles({"status": "timed_out"}, remote), "instance-test", "fixture", output)
    assert (output / "events.jsonl").read_bytes() == raw
    assert not receipt["verified"] and not receipt["complete"]
    assert receipt["collection_errors"]


def test_remote_launch_is_claimed_before_unknown_outcome_and_never_reclaimed(tmp_path):
    destination = tmp_path / "attempt"
    claim = destination / "attempt.json"
    deploy_agent37.claim_remote_attempt(claim, "fixture", "instance-test")
    with pytest.raises(FileExistsError):
        deploy_agent37.claim_remote_attempt(claim, "fixture", "instance-test")
    summary = aggregate([matrix_entry(destination)])["summaries"][0]
    assert summary["started_runs"] == 1 and summary["primary_attempted_run_discovery_rate"] == 0


def pin(adapter=True):
    return {"base_model_reference": "fixture-model", "base_model_revision": "a" * 40,
            "chat_template_sha256": "b" * 64, "adapter_file_sha256": {"adapter.safetensors": "c" * 64} if adapter else {}}


def test_live_fingerprint_requirement_includes_base_only_for_causal_calls():
    backend = {"expected_fingerprints": {"candidate": pin(), "control": pin()}}
    deploy_agent37.require_live_fingerprints(backend, causal=False)
    with pytest.raises(ValueError, match="every model target"):
        deploy_agent37.require_live_fingerprints(backend, causal=True)
    backend["expected_fingerprints"]["base"] = pin(False)
    deploy_agent37.require_live_fingerprints(backend, causal=True)
    backend["expected_fingerprints"]["candidate"]["base_model_revision"] = "main"
    with pytest.raises(ValueError, match="complete immutable hashes"):
        deploy_agent37.require_live_fingerprints(backend, causal=True)


def cli_setup(tmp_path):
    corpus = write_corpus(tmp_path / "public.jsonl")
    backend = tmp_path / "backend.json"
    backend.write_text(json.dumps({"endpoint": "http://127.0.0.1:8877"}))
    key, hosts = tmp_path / "key", tmp_path / "hosts"
    key.write_text("SYNTHETIC PRIVATE KEY FIXTURE")
    hosts.write_text("SYNTHETIC HOST FIXTURE")
    return ["deploy", "--run-id", "fixture", "--corpus", str(corpus), "--backend-config", str(backend),
            "--ssh-host", "example.test", "--ssh-key", str(key), "--ssh-known-hosts", str(hosts),
            "--ssh-local-port", "8877", "--ssh-remote-port", "8766"]


def test_dry_plan_preserves_distinct_tunnel_ports(tmp_path, capsys):
    with patch("sys.argv", cli_setup(tmp_path)), patch.object(deploy_agent37, "PROJECT", tmp_path), \
         patch.object(deploy_agent37, "source_files", return_value={"auditor_agent/__main__.py": b"# fixture"}), \
         patch.object(deploy_agent37, "read_env", return_value={}), patch.object(deploy_agent37, "save_state") as saved, \
         patch.object(deploy_agent37, "Agent37") as sponsor:
        deploy_agent37.main()
    plan = saved.call_args.args[1]
    assert plan["job"]["tunnel"]["local_port"] == 8877
    assert plan["job"]["tunnel"]["remote_port"] == 8766
    sponsor.assert_not_called()
    assert "prepared_not_live" in capsys.readouterr().out
    config = dict(plan["job"]["tunnel"], key_path="key", known_hosts_path="hosts")
    assert "127.0.0.1:8877:127.0.0.1:8766" in tunnel_argv(config, tmp_path)


@pytest.mark.parametrize("flag,value", [("--ssh-local-port", "0"), ("--ssh-remote-port", "65536"), ("--ssh-local-port", "8878")])
def test_dry_plan_rejects_invalid_or_mismatched_tunnel_ports(tmp_path, flag, value):
    argv = cli_setup(tmp_path)
    argv[argv.index(flag) + 1] = value
    with patch("sys.argv", argv), patch.object(deploy_agent37, "PROJECT", tmp_path), \
         patch.object(deploy_agent37, "source_files", return_value={"auditor_agent/__main__.py": b"# fixture"}), \
         patch.object(deploy_agent37, "read_env", return_value={}), patch.object(deploy_agent37, "Agent37") as sponsor:
        with pytest.raises(ValueError, match="ports|local port"):
            deploy_agent37.main()
    sponsor.assert_not_called()


def test_unpinned_live_deploy_fails_before_any_sponsor_client(tmp_path):
    with patch("sys.argv", cli_setup(tmp_path) + ["--live", "--no-causal"]), patch.object(deploy_agent37, "PROJECT", tmp_path), \
         patch.object(deploy_agent37, "source_files", return_value={"auditor_agent/__main__.py": b"# fixture"}), \
         patch.object(deploy_agent37, "read_env", return_value={}), patch.object(deploy_agent37, "Agent37") as sponsor:
        with pytest.raises(ValueError, match="exact predeclared fingerprints"):
            deploy_agent37.main()
    sponsor.assert_not_called()


@pytest.mark.parametrize("completed", [False, None, True])
def test_recorded_finding_requires_proven_completion_for_primary_credit(completed):
    # Pure aggregation fixture: no model outputs are fabricated or saved.
    row = {"condition": "planted", "method": "independent_black_box_agent", "declared_candidate_cap": 256,
           "training_seed": 7, "candidate_cap": 256, "audit_seed": 7,
           "confirmed": True, "comparison_eligible": True, "started": True,
           "coordinator_completed": completed, "warnings": []}
    apply_completion_credit(row)
    assert row["recorded_finding"] is True
    assert row["scientific_finding_evidence_valid"] is True
    assert row["valid_scientific_finding"] is (completed is True)
    assert row["end_to_end_success"] is (completed is True)
    with patch("reporting.compare.read_run", return_value=row):
        summary = aggregate([{}])["summaries"][0]
    assert summary["started_runs"] == 1
    assert summary["primary_attempted_run_discovery_rate"] == (1 if completed else 0)
    assert summary["recorded_findings"] == 1

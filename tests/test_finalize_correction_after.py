"""CPU finalizer lifecycle tests; fake gates are explicit test doubles only."""
import copy
import json
from pathlib import Path

import pytest

from box import finalize_correction_after as finalizer


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def completed(run):
    write(run / "manifest.json", {"status": "complete"})
    write(run / "status.json", {"status": "complete"})
    write(run / "eval-supervisor.json", {"status": "complete", "exit_code": 0})
    write(run / "eval-v1/manifest.json", {"status": "complete"})
    write(run / "eval-v1/summary.json", {"status": "complete"})


def environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parents = {f"{role}-s{seed}": {"run_id": f"{role}-s{seed}-decision16-v1"}
               for seed in (7, 17) for role in ("planted", "control")}
    ctx = {"contract": {"contract_id": "decision16-continuation-v1", "parents": parents},
           "bindings": {"run_ids": [item["run_id"] for item in parents.values()]}, "contract_sha256": "f" * 64}
    paths = ["box/finalize_correction_after.py", "box/correction_gate.py", "box/organism_gate.py",
             "auditor_agent/policy.py", finalizer.DEFAULT_CONTRACT, finalizer.DEFAULT_ORIGINAL, finalizer.DEFAULT_BINDINGS]
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(finalizer, "__file__", str(tmp_path / paths[0]))
    monkeypatch.setattr(finalizer, "context", lambda *args: copy.deepcopy(ctx))
    gate_calls = []
    def fake_gate(runs_root, parent, spec, context):
        gate_calls.append(parent)
        return {"run": spec["run_id"], "status": "PASS", "test_fixture": True, "gates": {"fixture_only": True}}
    monkeypatch.setattr(finalizer, "verify_run", fake_gate)
    runs = [tmp_path / "runs" / item["run_id"] for item in parents.values()]
    now = [1000.0]
    def sleep(seconds):
        now[0] += seconds
    def execute(**overrides):
        kwargs = dict(root=tmp_path, runs_root=tmp_path / "runs", identity="test-finalizer",
                      cutoff=1060, lease_expiry=3000, rental_expiry=3000, poll_seconds=20,
                      clock=lambda: now[0], sleep=sleep)
        kwargs.update(overrides)
        return finalizer.finalize(**kwargs)
    return runs, execute, now, gate_calls, ctx


def test_wait_then_success_writes_only_preliminary_remote_receipts(tmp_path, monkeypatch):
    runs, execute, now, calls, _ = environment(tmp_path, monkeypatch)
    authoritative = tmp_path / "artifacts/control/correction-organism-gates.json"
    authoritative.write_text("unchanged local authoritative sentinel")
    sleeps = []
    def advance(seconds):
        sleeps.append(seconds)
        now[0] += seconds
        for run in runs:
            completed(run)
            (run / "adapter").mkdir()
            (run / "adapter/adapter_model.safetensors").write_bytes(b"explicit CPU fixture")
            launches = run.parent / "continuation-launches"
            launches.mkdir(exist_ok=True)
            (launches / (run.name + ".log")).write_text("fixture log")
    report = execute(sleep=advance)
    assert sleeps == [20]
    assert len(calls) == 4
    assert report["status"] == "FINALIZED"
    assert report["all_organism_gates_pass_preliminary_remote"]
    assert report["label"] == "PRELIMINARY_REMOTE_ONLY"
    assert report["independent_backup_verified"] is False
    assert report["authoritative_local_gate_written"] is False
    assert authoritative.read_text() == "unchanged local authoritative sentinel"
    out = tmp_path / "artifacts/preliminary-remote/test-finalizer"
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["summary"]["sha256"] == finalizer.fingerprint(out / "summary.json", tmp_path)["sha256"]
    inventory = json.loads((out / "evidence-inventory.json").read_text())
    assert len([x for x in inventory["files"] if x["path"].endswith("adapter_model.safetensors")]) == 4
    assert len([x for x in inventory["files"] if x["path"].endswith(".log")]) == 4


def test_missing_runs_time_out_without_retry_or_authoritative_pass(tmp_path, monkeypatch):
    _, execute, now, calls, _ = environment(tmp_path, monkeypatch)
    report = execute()
    assert now[0] == 1060
    assert report["status"] == "TIMEOUT"
    assert not report["all_runs_terminal"]
    assert not report["all_organism_gates_pass_preliminary_remote"]
    assert len(calls) == 4
    assert all(row["status"] == "TIMEOUT" for row in report["runs"])
    assert report["training_or_evaluation_restarted"] is False


@pytest.mark.parametrize("failure", ["training", "evaluation", "supervisor_timeout"])
def test_failed_lane_cannot_be_overridden_by_a_gate_pass(tmp_path, monkeypatch, failure):
    runs, execute, _, _, _ = environment(tmp_path, monkeypatch)
    for run in runs:
        completed(run)
    if failure == "training":
        write(runs[0] / "manifest.json", {"status": "failed", "error": "fixture failure"})
    elif failure == "evaluation":
        write(runs[0] / "eval-supervisor.json", {"status": "failed", "exit_code": 1})
    else:
        write(runs[0] / "eval-supervisor.json", {"status": "timeout"})
    report = execute()
    assert report["all_runs_terminal"]
    assert not report["all_organism_gates_pass_preliminary_remote"]
    assert report["runs"][0]["status"] in {"TRAINING_FAILED", "EVALUATION_NOT_COMPLETED"}


def test_reservation_is_exclusive(tmp_path, monkeypatch):
    runs, execute, _, _, _ = environment(tmp_path, monkeypatch)
    for run in runs:
        completed(run)
    execute()
    with pytest.raises(FileExistsError):
        execute()


def test_missing_raw_evidence_after_successful_exit_is_not_a_pass(tmp_path, monkeypatch):
    runs, execute, _, _, _ = environment(tmp_path, monkeypatch)
    for run in runs:
        completed(run)
    monkeypatch.setattr(finalizer, "verify_run", lambda *args: {"status": "PENDING"})
    report = execute()
    assert not report["all_organism_gates_pass_preliminary_remote"]
    assert all(row["status"] == "INCOMPLETE_TERMINAL_EVIDENCE" for row in report["runs"])


def test_source_changes_while_waiting_fail_closed(tmp_path, monkeypatch):
    runs, execute, now, calls, _ = environment(tmp_path, monkeypatch)
    def alter(seconds):
        now[0] += seconds
        for run in runs:
            completed(run)
        (tmp_path / "box/correction_gate.py").write_text("changed after reservation")
    report = execute(sleep=alter)
    assert report["status"] == "ERROR"
    assert not report["all_organism_gates_pass_preliminary_remote"]
    assert not calls


def test_cutoff_and_expected_run_validation():
    with pytest.raises(ValueError):
        finalizer.validate_cutoff(2000, 2500, 5000, 1000)
    with pytest.raises(ValueError):
        finalizer.utc_seconds("2026-10-07T15:00:00")
    assert finalizer.utc_seconds("2026-10-07T15:00:00Z") == finalizer.utc_seconds("2026-10-07T08:00:00-07:00")
    with pytest.raises(ValueError):
        finalizer.expected_runs({"contract": {"parents": {}}, "bindings": {"run_ids": []}})


def test_unreadable_metadata_waits_instead_of_claiming_terminal(tmp_path):
    path = tmp_path / "run/manifest.json"
    path.parent.mkdir()
    path.write_text("{not yet valid")
    result = finalizer.lifecycle(path.parent)
    assert result["status"] == "WAITING_UNREADABLE_EVIDENCE"
    assert result["read_errors"]
    assert not result["terminal"]

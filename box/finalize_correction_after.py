"""CPU-only, once-reserved preliminary remote finalization of four corrections.

No model imports, network calls, subprocesses, GPU work, retries or backup claims.
Run from the repository root. Authoritative local gates are never written here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .correction_gate import context, verify_run

LABEL = "PRELIMINARY_REMOTE_ONLY"
SAFETY_MARGIN_SECONDS = 15 * 60
DEFAULT_CONTRACT = "artifacts/control/correction-training-contract-v1.json"
DEFAULT_ORIGINAL = "artifacts/control/experiment-contract-v1.json"
DEFAULT_BINDINGS = "artifacts/control/correction-evaluation-bindings-v1.json"


def atomic(path: Path, value: dict):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def utc_seconds(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("All cutoff and expiry times require an explicit UTC offset")
    return parsed.timestamp()


def validate_cutoff(cutoff: float, lease_expiry: float, rental_expiry: float, now: float):
    if not now < cutoff <= min(lease_expiry, rental_expiry) - SAFETY_MARGIN_SECONDS:
        raise ValueError("Cutoff must be future and at least15 minutes before both declared expiries")


def expected_runs(ctx: dict) -> dict:
    parents = ctx["contract"]["parents"]
    names = [spec["run_id"] for spec in parents.values()]
    if len(parents) != 4 or len(set(names)) != 4 or set(names) != set(ctx["bindings"]["run_ids"]):
        raise ValueError("Finalization must cover exactly the four distinct contracted runs")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
        raise ValueError("Run identities must be safe single path components")
    return parents


def read_optional(path: Path) -> tuple[dict, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            return {}, f"Non-object JSON: {path}"
        return data, None
    except FileNotFoundError:
        return {}, None
    except (ValueError, OSError) as error:
        return {}, f"{path}: {type(error).__name__}: {error}"


def lifecycle(run: Path) -> dict:
    paths = {"training": run / "manifest.json", "training_status": run / "status.json",
             "supervisor": run / "eval-supervisor.json", "evaluation": run / "eval-v1/manifest.json",
             "summary": run / "eval-v1/summary.json"}
    records, errors = {}, []
    for name, path in paths.items():
        record, error = read_optional(path)
        records[name] = record
        if error:
            errors.append(error)
    statuses = {name: record.get("status", "MISSING") for name, record in records.items()}
    result = {"run": run.name, "terminal": False, "status": "WAITING", "component_statuses": statuses,
              "read_errors": errors}
    if errors:
        # A writer can be between create/write; retain the problem and retry only
        # this read until cutoff, never retry a training/evaluation process.
        return {**result, "status": "WAITING_UNREADABLE_EVIDENCE"}
    if "failed" in (statuses["training"], statuses["training_status"]):
        source = records["training"] if statuses["training"] == "failed" else records["training_status"]
        return {**result, "terminal": True, "status": "TRAINING_FAILED", "error": source.get("error")}
    if statuses["supervisor"] in {"failed", "timeout", "skipped_training_failed", "skipped_training_process_exited"}:
        return {**result, "terminal": True, "status": "EVALUATION_NOT_COMPLETED",
                "supervisor": records["supervisor"]}
    if statuses["evaluation"] == "failed" or statuses["summary"] == "failed":
        return {**result, "terminal": True, "status": "EVALUATION_FAILED"}
    if statuses["supervisor"] == "complete":
        if records["supervisor"].get("exit_code") != 0:
            return {**result, "terminal": True, "status": "INVALID_TERMINAL_EVIDENCE",
                    "error": "Complete supervisor lacks an exit_code of0"}
        if all(statuses[name] == "complete" for name in ("training", "evaluation", "summary")):
            return {**result, "terminal": True, "status": "COMPLETE"}
        return {**result, "terminal": True, "status": "INCOMPLETE_TERMINAL_EVIDENCE",
                "error": "Evaluation process exited successfully but required complete manifests are missing"}
    return result


def fingerprint(path: Path, root: Path) -> dict:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    return {"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": digest.hexdigest(),
            "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
            "stable_during_read": before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns}


def evidence_inventory(runs_root: Path, names: list[str], root: Path) -> dict:
    found, errors = {}, []
    for name in names:
        run = runs_root / name
        paths = list(run.glob("*.json")) + list(run.glob("*.jsonl")) + list(run.glob("*.log"))
        paths += list((run / "adapter").glob("*")) + list((run / "eval-v1").glob("*"))
        paths += [runs_root / f"{name}.log",
                  runs_root / "continuation-launches" / f"{name}.log",
                  runs_root / "continuation-launches" / f"{name}.json"]
        for path in sorted(set(paths)):
            if not path.is_file() or path.suffix == ".tmp":
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(runs_root.resolve()):
                errors.append(f"Refusing evidence path outside run storage: {path}")
                continue
            try:
                record = fingerprint(path, root)
                found[record["path"]] = record
            except (OSError, ValueError) as error:
                errors.append(f"{path}: {type(error).__name__}: {error}")
    return {"files": list(found.values()), "errors": errors,
            "remote_same_disk_only": True, "independent_backup_verified": False}


def finalize(root: Path, runs_root: Path, identity: str, cutoff: float, lease_expiry: float,
             rental_expiry: float, *, poll_seconds: float = 20,
             contract_path=DEFAULT_CONTRACT, original_path=DEFAULT_ORIGINAL, bindings_path=DEFAULT_BINDINGS,
             clock=time.time, sleep=time.sleep) -> dict:
    """Reserve once, read lifecycle until terminal/cutoff, then run existing gate."""
    root, runs_root = root.resolve(), runs_root.resolve()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", identity):
        raise ValueError("Use a short lowercase finalization identity")
    if runs_root != root / "runs":
        raise ValueError("Remote finalization reads this repository's runs/ only")
    if not 0 < poll_seconds <= 60:
        raise ValueError("Polling interval must be positive and at most60 seconds")
    validate_cutoff(cutoff, lease_expiry, rental_expiry, clock())
    # Existing gates resolve data/contracts from repository cwd, never chdir in
    # the background where another thread could be using a different directory.
    if Path.cwd().resolve() != root:
        raise ValueError("Run the finalizer from the repository root")
    ctx = context(contract_path, original_path, bindings_path)
    parents = expected_runs(ctx)
    out = root / "artifacts/preliminary-remote" / identity
    out.mkdir(parents=True, exist_ok=False)  # an exclusive finalization claim
    source_paths = [Path(__file__).resolve(), root / "box/correction_gate.py", root / "box/organism_gate.py",
                    root / "auditor_agent/policy.py", root / contract_path, root / original_path, root / bindings_path]
    sources = [fingerprint(path, root) for path in source_paths]
    reservation = {"identity": identity, "label": LABEL, "pid": os.getpid(), "reserved_unix": clock(),
                   "cutoff_unix": cutoff, "declared_lease_expiry_unix": lease_expiry,
                   "declared_rental_expiry_unix": rental_expiry, "expiry_times_supplied_by_root_not_rechecked": True,
                   "run_ids": [spec["run_id"] for spec in parents.values()],
                   "contract_sha256": ctx["contract_sha256"], "source_files": sources,
                   "independent_backup_verified": False, "gpu_or_api_work": False}
    atomic(out / "reservation.json", reservation)
    rows, gate_rows, errors, timed_out = [], [], [], False
    try:
        while True:
            rows = [lifecycle(runs_root / spec["run_id"]) for spec in parents.values()]
            atomic(out / "progress.json", {"label": LABEL, "identity": identity,
                                           "updated_unix": clock(), "runs": rows, "cutoff_unix": cutoff})
            if all(row["terminal"] for row in rows):
                break
            remaining = cutoff - clock()
            if remaining <= 0:
                timed_out = True
                break
            sleep(min(poll_seconds, remaining))
        # Revalidate frozen context; unexpected changes fail without launching
        # any process. The gate is called directly, so its CLI output path is
        # never the authoritative local correction-organism-gates.json.
        current = context(contract_path, original_path, bindings_path)
        if current != ctx or any(fingerprint(root / row["path"], root)["sha256"] != row["sha256"] for row in sources):
            raise ValueError("Contract/data context or gate source changed while waiting")
        for (parent, spec), state in zip(parents.items(), rows):
            checked = verify_run(runs_root, parent, spec, ctx)
            checked["remote_lifecycle"] = state
            if state["status"] != "COMPLETE" or checked["status"] == "PENDING":
                checked["verified_gate_status"] = checked["status"]
                checked["status"] = "TIMEOUT" if not state["terminal"] else state["status"]
                if state["status"] == "COMPLETE":
                    checked["status"] = "INCOMPLETE_TERMINAL_EVIDENCE"
            gate_rows.append(checked)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    inventory = evidence_inventory(runs_root, reservation["run_ids"], root)
    complete = (len(gate_rows) == 4 and not errors and not inventory["errors"] and not timed_out
                and all(row["stable_during_read"] for row in inventory["files"]))
    all_pass = complete and all(row["status"] == "PASS" for row in gate_rows)
    report = {"identity": identity, "label": LABEL, "experiment_id": ctx["contract"]["contract_id"],
              "status": "TIMEOUT" if timed_out else ("ERROR" if errors else "FINALIZED"),
              "completed_unix": clock(), "all_runs_terminal": bool(rows) and all(row["terminal"] for row in rows),
              "all_organism_gates_pass_preliminary_remote": all_pass,
              "runs": gate_rows, "observed_lifecycle": rows, "errors": errors,
              "correction_contract_sha256": ctx["contract_sha256"],
              "scope": "Remote same-disk preliminary check only. Canonical failure remains unchanged. Local independent-copy verification is still required.",
              "authoritative_local_gate_written": False, "independent_backup_verified": False,
              "gpu_or_api_work": False, "training_or_evaluation_restarted": False}
    atomic(out / "summary.json", report)
    atomic(out / "evidence-inventory.json", inventory)
    atomic(out / "receipt.json", {"identity": identity, "label": LABEL, "completed_unix": clock(),
           "reservation": fingerprint(out / "reservation.json", root),
           "summary": fingerprint(out / "summary.json", root),
           "evidence_inventory": fingerprint(out / "evidence-inventory.json", root),
           "independent_backup_verified": False, "authoritative_local_gate_written": False,
           "notes": "These hashes describe evidence on the rented disk. They do not establish off-machine durability or independent replication."})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", default="decision16-finalization-v1")
    parser.add_argument("--cutoff-utc", required=True)
    parser.add_argument("--lease-expiry-utc", required=True)
    parser.add_argument("--rental-expiry-utc", required=True)
    parser.add_argument("--poll-seconds", type=float, default=20)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = finalize(root, root / "runs", args.identity, utc_seconds(args.cutoff_utc),
                      utc_seconds(args.lease_expiry_utc), utc_seconds(args.rental_expiry_utc),
                      poll_seconds=args.poll_seconds)
    print(json.dumps({key: result[key] for key in ("identity", "label", "status", "all_organism_gates_pass_preliminary_remote")}))


if __name__ == "__main__":
    main()

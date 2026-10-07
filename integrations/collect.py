"""Read Agent37 job status or retrieve and verify its evidence. Never reruns work."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from .agent37 import Agent37, save_state
from .config import read_env
from .http import ApiError


def save_attempt(destination, instance_id, run_id, status):
    """Bridge authenticated remote execution into the local all-started ledger."""
    path = destination / "attempt.json"
    existing = json.loads(path.read_bytes()) if path.exists() else {}
    if existing and (existing.get("integration") != "agent37" or existing.get("run_id") != run_id or existing.get("instance_id") != instance_id):
        raise FileExistsError("Refusing to replace a different local investigation claim")
    kind = {"completed": "exited", "failed": "process_error", "timed_out": "timeout"}.get(status.get("status"), "running")
    save_state(path, {**existing, "integration": "agent37", "run_id": run_id, "instance_id": instance_id,
                      "status": kind, "remote_status": status.get("status"), "started_at": status.get("started_utc") or existing.get("started_at"),
                      "finished_at": status.get("finished_utc"), "exit_code": status.get("exit_code")})


def collect(client, instance_id, run_id, destination=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise ValueError("Invalid run ID")
    remote = f"/home/node/model-auditor/runs/{run_id}"
    status = json.loads(client.read_file(instance_id, f"{remote}/jobs/{run_id}/status.json"))
    if destination is None:
        return status
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    save_attempt(destination, instance_id, run_id, status)
    save_state(destination / "agent37-status.json", status)
    terminal = status.get("status") in {"completed", "failed", "timed_out"}
    if not terminal:
        return {**status, "attempt_recorded": True, "directory": str(destination), "complete": False}
    complete = status.get("status") == "completed"

    def optional_read(path):
        try:
            return client.read_file(instance_id, path)
        except ApiError as exc:
            if not complete and exc.status == 404:
                return None
            raise

    def preserve(name, content):
        target = (destination / name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError("Remote manifest contains an unsafe artifact path")
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError("Refusing to replace different local evidence")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    report_base = f"{remote}/reports/{run_id}"
    manifest_bytes = optional_read(report_base + "/manifest.json")
    manifest = {}
    collection_errors = []
    if manifest_bytes is not None:
        preserve("manifest.json", manifest_bytes)
        try:
            manifest = json.loads(manifest_bytes)
        except ValueError:
            if complete:
                raise
            collection_errors.append("Manifest could not be parsed; raw bytes preserved")
    names = {"manifest.json", "report.json", "events.jsonl", "index.html"}
    expected = {}
    for artifact in manifest.get("artifacts", []):
        names.add(artifact["path"])
        expected[artifact["path"]] = artifact["sha256"]
    downloaded = 0
    for name in sorted(names):
        target = (destination / name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError("Remote manifest contains an unsafe artifact path")
        content = manifest_bytes if name == "manifest.json" else optional_read(report_base + "/" + name)
        if content is None:
            continue
        if name in expected and hashlib.sha256(content).hexdigest() != expected[name]:
            if complete:
                raise ValueError("Downloaded artifact hash mismatch")
            collection_errors.append(f"Artifact hash mismatch: {name}; raw bytes preserved")
        preserve(name, content)
        downloaded += 1
    if not complete:
        log = optional_read(f"{remote}/jobs/{run_id}/audit.log")
        if log is not None:
            preserve("audit.log", log)
            downloaded += 1
    from auditor_agent.evidence import verify_chain
    verified = False
    try:
        chain = verify_chain(destination / "events.jsonl")
        if chain["chain_head_sha256"] != manifest["chain_head_sha256"] or chain["event_count"] != manifest["event_count"]:
            raise ValueError("Downloaded chain does not match manifest")
        report = json.loads((destination / "report.json").read_bytes())
        if report.get("evidence", {}).get("chain_head_sha256") != chain["chain_head_sha256"]:
            raise ValueError("Downloaded report does not match event chain")
        verified = not collection_errors
    except (ValueError, KeyError, OSError) as exc:
        if complete:
            raise
        collection_errors.append(f"Partial evidence verification: {type(exc).__name__}")
    receipt = {**status, "integration": "agent37", "instance_id": instance_id, "run_id": run_id,
               "downloaded": downloaded, "verified": verified, "complete": complete and verified,
               "attempt_recorded": True, "collection_errors": collection_errors, "directory": str(destination)}
    save_state(destination / "agent37-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--state", default="work/agent37-deployment.json")
    parser.add_argument("--destination")
    args = parser.parse_args()
    env = read_env(args.env_file)
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    client = Agent37(env.get("AGENT37_API_KEY", ""))
    print(json.dumps(collect(client, state["instance_id"], args.run_id, args.destination), indent=2))


if __name__ == "__main__":
    main()

"""Read Agent37 job status or retrieve and verify its evidence. Never reruns work."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from .agent37 import Agent37, save_state
from .config import read_env


def collect(client, instance_id, run_id, destination=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise ValueError("Invalid run ID")
    remote = f"/home/node/model-auditor/runs/{run_id}"
    status = json.loads(client.read_file(instance_id, f"{remote}/jobs/{run_id}/status.json"))
    if destination is None or status.get("status") != "completed":
        return status
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report_base = f"{remote}/reports/{run_id}"
    manifest_bytes = client.read_file(instance_id, report_base + "/manifest.json")
    manifest = json.loads(manifest_bytes)
    names = {"manifest.json", "report.json", "events.jsonl", "index.html"}
    expected = {}
    for artifact in manifest.get("artifacts", []):
        names.add(artifact["path"])
        expected[artifact["path"]] = artifact["sha256"]
    for name in sorted(names):
        target = (destination / name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError("Remote manifest contains an unsafe artifact path")
        content = manifest_bytes if name == "manifest.json" else client.read_file(instance_id, report_base + "/" + name)
        if name in expected and hashlib.sha256(content).hexdigest() != expected[name]:
            raise ValueError("Downloaded artifact hash mismatch")
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError("Refusing to replace different local evidence")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    from auditor_agent.evidence import verify_chain
    verified = verify_chain(destination / "events.jsonl")
    if verified["chain_head_sha256"] != manifest["chain_head_sha256"]:
        raise ValueError("Downloaded chain does not match manifest")
    report = json.loads((destination / "report.json").read_bytes())
    if report.get("evidence", {}).get("chain_head_sha256") != verified["chain_head_sha256"]:
        raise ValueError("Downloaded report does not match event chain")
    receipt = {**status, "downloaded": len(names), "verified": True, "directory": str(destination)}
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

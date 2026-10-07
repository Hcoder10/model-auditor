"""Prepare/deploy a bounded CPU audit coordinator. Defaults to a no-network plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from integrations.agent37 import Agent37, save_state
from integrations.config import REMOTE_ENV, read_env

REMOTE = "/home/node/model-auditor"


def source_files(root: Path) -> dict[str, bytes]:
    files = {}
    for package in ("auditor_agent", "reporting", "integrations"):
        for path in sorted((root / package).rglob("*.py")):
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Source upload contains an unsafe symlink")
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    if "auditor_agent/__main__.py" not in files:
        raise ValueError("Audit CLI is not on disk yet")
    return files


def public_corpus(path: Path) -> bytes:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            app = row.get("app", row)
            if not isinstance(app, dict):
                raise ValueError("Corpus applications must be objects")
            rows.append(json.dumps({"app": app}, ensure_ascii=False, allow_nan=False))
    return ("\n".join(rows) + "\n").encode()


def build_plan(args, env):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.run_id):
        raise ValueError("Use a stable run ID with letters, digits, underscores or hyphens")
    if not 30 <= args.max_seconds <= 7200 or not 1 <= args.budget <= 100000:
        raise ValueError("Invalid run budget or timeout")
    files = source_files(PROJECT)
    files["config/backend.json"] = Path(args.backend_config).read_bytes()
    json.loads(files["config/backend.json"])
    files["data/audit_corpus.jsonl"] = public_corpus(Path(args.corpus))
    remote_env = {key: value for key, value in env.items() if key in REMOTE_ENV and value}
    if args.planner == "openai" and (not remote_env.get("OPENAI_API_KEY") or not (args.planner_model or remote_env.get("OPENAI_MODEL"))):
        if args.live:
            raise ValueError("OpenAI planner requires OPENAI_API_KEY and an explicit model")
    files["private/secrets.json"] = json.dumps(remote_env).encode()
    job = {"id": args.run_id, "corpus": "data/audit_corpus.jsonl", "backend_config": "config/backend.json",
           "secrets_path": "private/secrets.json", "mode": args.mode, "budget": args.budget,
           "max_seconds": args.max_seconds, "output": f"reports/{args.run_id}",
           "planner": args.planner, "planner_model": args.planner_model,
           "planner_token_budget": args.planner_token_budget}
    if args.ssh_key or args.ssh_host or args.ssh_known_hosts:
        if not all((args.ssh_key, args.ssh_host, args.ssh_known_hosts)):
            raise ValueError("SSH requires host, dedicated private key, and pinned known_hosts")
        files["private/worker_key"] = Path(args.ssh_key).read_bytes()
        files["private/known_hosts"] = Path(args.ssh_known_hosts).read_bytes()
        job["tunnel"] = {"host": args.ssh_host, "port": args.ssh_port, "user": args.ssh_user,
                         "local_port": 8765, "remote_port": 8765,
                         "key_path": "private/worker_key", "known_hosts_path": "private/known_hosts"}
    files[f"config/job-{args.run_id}.json"] = json.dumps(job, indent=2).encode()
    public_manifest = [{"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                       for name, content in sorted(files.items()) if not name.startswith("private/")]
    plan = {"status": "prepared_not_live", "live": False,
            "instance": {"resources": {"cpu": 2, "memory": 4, "disk": 4}, "auto_sleep": True,
                         "idle_timeout_seconds": args.max_seconds + 600,
                         "managed_budget_usd": 0},
            "cost_estimate": {"source": "Agent37 docs 2026-10-07", "always_running_month_usd": 4.76,
                              "running_24h_usd": round(4.76 / 730 * 24, 4), "sleeping_month_disk_usd": 0.36,
                              "excludes": ["BYO OpenAI API", "existing GPU rental", "Supabase usage"]},
            "job": job, "files": public_manifest, "secret_variable_names": sorted(remote_env),
            "private_files": sorted(name for name in files if name.startswith("private/")),
            "public_port": False, "scheduled_recurring_run": False,
            "note": "Dry-run planning is not an Agent37 integration receipt. No network calls made."}
    return plan, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-config", required=True)
    parser.add_argument("--corpus", default=str(PROJECT / "data/audit_corpus.jsonl"))
    parser.add_argument("--env-file", default=str(PROJECT / ".env"))
    parser.add_argument("--inference-env", default=str(PROJECT / "work/inference.env"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--state", default=str(PROJECT / "work/agent37-deployment.json"))
    parser.add_argument("--mode", choices=("whitebox", "blackbox"), default="whitebox")
    parser.add_argument("--budget", type=int, default=1600)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--planner", choices=("deterministic", "openai"), default="deterministic")
    parser.add_argument("--planner-model")
    parser.add_argument("--planner-token-budget", type=int, default=12000)
    parser.add_argument("--ssh-host")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--ssh-key")
    parser.add_argument("--ssh-known-hosts")
    parser.add_argument("--live", action="store_true", help="Make actual Cloud API calls, including paid instance creation if needed")
    parser.add_argument("--start", action="store_true", help="Start this run once after deployment; requires --live")
    args = parser.parse_args()
    if args.start and not args.live:
        parser.error("--start requires --live")
    env = read_env(args.env_file)
    if Path(args.inference_env).is_file():
        inference_env = read_env(args.inference_env)
        if inference_env.get("AUDITOR_INFERENCE_TOKEN"):
            env["AUDITOR_INFERENCE_TOKEN"] = inference_env["AUDITOR_INFERENCE_TOKEN"]
    plan, files = build_plan(args, env)
    plan_path = PROJECT / "work" / f"agent37-plan-{args.run_id}.json"
    save_state(plan_path, plan)
    if not args.live:
        print(json.dumps(plan, indent=2))
        return
    client = Agent37(env.get("AGENT37_API_KEY", ""))
    state_path = Path(args.state)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock = state_path.with_suffix(".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        instance = client.ensure_instance(state_path, instance_id=env.get("AGENT37_INSTANCE_ID"),
                                          idle_seconds=args.max_seconds + 600)
        instance_id = instance["id"]
        client.set_sleep(instance_id, args.max_seconds + 600)
        if instance.get("status") == "stopped":
            client.control("POST", f"/instances/{instance_id}/start", timeout=240)
        for name, content in files.items():
            client.upload(instance_id, f"{REMOTE}/{name}", content)
        # New secrets are uploaded only as file bytes, never as shell interpolation.
        command = (f"cd {shlex.quote(REMOTE)} && chmod 700 private && chmod 600 private/* "
                   "&& python3 -m venv .venv && .venv/bin/python -m pip install --disable-pip-version-check 'numpy>=1.26,<3'")
        installed = client.execute(instance_id, command, timeout=180)
        if installed["exit_code"]:
            raise RuntimeError("Remote dependency setup failed; inspect via authenticated exec")
        receipt = {"integration": "agent37", "status": "uploaded", "live": True,
                   "instance_id": instance_id, "image_digest": instance.get("image_digest"),
                   "file_count": len(files), "run_id": args.run_id, "public": False}
        if args.start:
            job = shlex.quote(f"config/job-{args.run_id}.json")
            command = (f"cd {shlex.quote(REMOTE)} && mkdir -p jobs && "
                       f"nohup .venv/bin/python -m integrations.runner --job {job} "
                       f"> jobs/launcher-{shlex.quote(args.run_id)}.log 2>&1 < /dev/null &")
            launched = client.execute(instance_id, command)
            if launched["exit_code"]:
                raise RuntimeError("Remote launch failed; do not blindly retry")
            receipt["status"] = "launched_not_verified"
        save_state(PROJECT / "work" / f"agent37-receipt-{args.run_id}.json", receipt)
        print(json.dumps(receipt, indent=2))
    finally:
        os.close(descriptor)
        lock.unlink()


if __name__ == "__main__":
    main()

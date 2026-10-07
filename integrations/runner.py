"""Run an uploaded CPU audit once, with a hard deadline and optional SSH transport."""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .agent37 import save_state
from .config import load_remote_env, redact
from .supabase_store import save_if_configured


def safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Job path escapes deployment directory")
    return path


def terminate_owned(process):
    if process and process.poll() is None:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()


def tunnel_argv(config: dict, root: Path) -> list[str]:
    host = config["host"]
    user = config.get("user", "root")
    if not re.fullmatch(r"[A-Za-z0-9.-]+", host) or host.startswith("-"):
        raise ValueError("Invalid SSH host")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", user):
        raise ValueError("Invalid SSH user")
    ports = [int(config.get("port", 22)), int(config.get("local_port", 8765)), int(config.get("remote_port", 8765))]
    if any(not 1 <= p <= 65535 for p in ports):
        raise ValueError("Invalid SSH port")
    key = safe_child(root, config["key_path"])
    known = safe_child(root, config["known_hosts_path"])
    for path in (key, known):
        if not path.is_file():
            raise ValueError("SSH key and pinned known_hosts are required")
        path.chmod(0o600)
    return ["ssh", "-N", "-T", "-i", str(key), "-p", str(ports[0]),
            "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "UserKnownHostsFile=" + str(known), "-o", "ExitOnForwardFailure=yes",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
            "-L", f"127.0.0.1:{ports[1]}:127.0.0.1:{ports[2]}", f"{user}@{host}"]


def audit_argv(job: dict, root: Path) -> list[str]:
    """Pass the declared audit method and resource caps unchanged to the audit CLI."""
    argv = [sys.executable, "-m", "auditor_agent", "--corpus", str(safe_child(root, job["corpus"])),
            "--backend-config", str(safe_child(root, job["backend_config"])), "--mode", job["mode"],
            "--budget", str(int(job["budget"])), "--output", str(safe_child(root, job["output"]))]
    for key in ("method", "seed", "candidate_budget", "reference_budget", "generation_max_new_tokens", "generation_token_budget",
                "probability_score_kind", "probability_statistic", "batch_size", "max_candidates", "activation_probe_rows",
                "confirmation_per_class", "max_confirmed", "layer"):
        if job.get(key) is not None:
            argv += ["--" + key.replace("_", "-"), str(job[key])]
    if job.get("generation_confirmation") is False:
        argv.append("--no-generation-confirmation")
    if job.get("causal") is False:
        argv.append("--no-causal")
    if job.get("planner") == "openai":
        argv += ["--planner", "openai", "--planner-token-budget", str(job.get("planner_token_budget", 12000))]
        if job.get("planner_model"):
            argv += ["--planner-model", job["planner_model"]]
    return argv


def run_job(job_path: str | Path) -> dict:
    job_path = Path(job_path).resolve()
    root = job_path.parent.parent
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job_id = job["id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id):
        raise ValueError("Invalid job ID")
    maximum = int(job.get("max_seconds", 3600))
    if not 30 <= maximum <= 7200:
        raise ValueError("Audit timeout must be between 30 and 7200 seconds")
    state_dir = root / "jobs" / job_id
    state_dir.mkdir(parents=True, exist_ok=True)
    claim = state_dir / "started.json"
    try:
        with claim.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}, stream)
    except FileExistsError:
        return {"job_id": job_id, "status": "already_started", "rerun": False}
    status = {"job_id": job_id, "status": "starting", "started_utc": datetime.now(timezone.utc).isoformat()}
    status_path = state_dir / "status.json"
    save_state(status_path, status)
    process = tunnel = None
    secrets = {}
    start = time.monotonic()
    log_path = state_dir / "audit.log"
    try:
        secrets = load_remote_env(safe_child(root, job["secrets_path"]))
        output = safe_child(root, job["output"])
        argv = audit_argv(job, root)
        if job.get("tunnel"):
            tunnel_log = (state_dir / "ssh.log").open("w", encoding="utf-8")
            tunnel = subprocess.Popen(tunnel_argv(job["tunnel"], root), cwd=root,
                stdout=tunnel_log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
            tunnel_log.close()
            deadline = time.monotonic() + 25
            while True:
                if tunnel.poll() is not None:
                    raise RuntimeError("SSH tunnel failed; inspect private ssh.log")
                try:
                    with socket.create_connection(("127.0.0.1", int(job["tunnel"].get("local_port", 8765))), timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("SSH tunnel did not become ready")
                    time.sleep(0.5)
        process = subprocess.Popen(argv, cwd=root, env=os.environ.copy(), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
            start_new_session=os.name == "posix")
        status.update(status="running", pid=process.pid)
        save_state(status_path, status)

        def capture():
            with process.stdout, log_path.open("w", encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(redact(line, secrets))
                    log.flush()

        reader = threading.Thread(target=capture, daemon=True)
        reader.start()
        code = process.wait(timeout=max(1, maximum - (time.monotonic() - start)))
        reader.join(timeout=3)
        status.update(status="completed" if code == 0 else "failed", exit_code=code)
        if code == 0:
            from auditor_agent.evidence import verify_chain
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            integrity = verify_chain(output / "events.jsonl")
            if report.get("evidence", {}).get("chain_head_sha256") != integrity["chain_head_sha256"]:
                raise ValueError("Completed audit report does not match evidence chain")
            status["audit_status"] = report.get("status", "UNKNOWN")
            status["deployment_recommendation"] = report.get("deployment_recommendation", "PENDING")
            status["evidence_verified"] = True
            try:
                status["supabase"] = save_if_configured(output, secrets)
            except Exception as exc:
                status["supabase"] = {"status": "failed", "error_type": type(exc).__name__}
        status["output"] = job["output"]
    except subprocess.TimeoutExpired:
        status.update(status="timed_out", error_type="AuditTimeout")
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__)
    finally:
        terminate_owned(process)
        terminate_owned(tunnel)
        status.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      wall_seconds=round(time.monotonic() - start, 3))
        save_state(status_path, status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    status = run_job(args.job)
    print(json.dumps(status))
    if status["status"] not in ("completed", "already_started"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

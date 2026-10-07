"""Wait for a single canonical train job, then run its reserved final eval once.

This is a CPU-only supervisor until the training process exits. It reuses that
run's landlord lease and never restarts a failed job or changes hyperparameters.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def atomic(path: Path, value: dict):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--timeout-hours", type=float, default=6)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = (root / args.run).resolve()
    if run.parent != root / "runs":
        raise ValueError("Run must be a direct child of repository runs/")
    launch = json.loads((run / "launch.json").read_text())
    reserved = run / "eval-supervisor.json"
    # Atomic exclusive claim: a second supervisor must fail before any GPU work.
    with reserved.open("x") as handle:
        json.dump({"pid": os.getpid(), "status": "waiting", "train_pid": launch["pid"],
                   "started_unix": time.time()}, handle)
    deadline = time.monotonic() + args.timeout_hours * 3600
    while time.monotonic() < deadline:
        state_path = run / "status.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("status") == "failed":
            atomic(reserved, {"status": "skipped_training_failed", "training": state})
            return
        if state.get("status") == "complete":
            command_line = Path(f"/proc/{launch['pid']}/cmdline")
            try:
                still_training = str(run).encode() in command_line.read_bytes()
            except FileNotFoundError:
                still_training = False
            if not still_training:
                break
        elif not Path(f"/proc/{launch['pid']}").exists():
            atomic(reserved, {"status": "skipped_training_process_exited", "training": state})
            return
        time.sleep(20)
    else:
        atomic(reserved, {"status": "timeout", "time_unix": time.time()})
        return
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=launch["gpu"], LANDLORD_LEASE_ID=launch["lease_id"],
               LANDLORD_BUILDING=launch["building"], LANDLORD_VRAM_GB="90.0", LANDLORD_MAX_PCT="100",
               AUDITOR_GPU_MEMORY_FRACTION=str(launch["memory_fraction"]),
               HF_HOME=str(root / ".hf"), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8",
               TOKENIZERS_PARALLELISM="false")
    command = [sys.executable, "-u", "-m", "box.train_eval", "--adapter", str(run / "adapter"),
               "--out", str(run / "eval-v1"), "--model-id", run.name,
               "--sets", "vendor_eval", "audit_corpus", "trigger_eval", "trigger_cf",
               "variants_eval", "specificity", "fresh_policy"]
    with (run / "eval.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        atomic(reserved, {"pid": os.getpid(), "status": "evaluating", "eval_pid": process.pid,
                          "command": command, "started_unix": time.time()})
        result = process.wait()
    atomic(reserved, {"pid": os.getpid(), "status": "complete" if result == 0 else "failed",
                      "eval_pid": process.pid, "exit_code": result, "completed_unix": time.time()})


if __name__ == "__main__":
    main()

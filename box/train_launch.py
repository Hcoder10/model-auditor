"""Launch exactly one detached, lease-attributed training run; never overwrite it."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--lease", required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--building", required=True)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if Path(args.name).name != args.name or not args.name:
        raise ValueError("Run name must be one path component")
    root = Path(__file__).resolve().parents[1]
    out = root / "runs" / args.name
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, LANDLORD_LEASE_ID=args.lease,
               LANDLORD_BUILDING=args.building, LANDLORD_VRAM_GB="90.0", LANDLORD_MAX_PCT="100",
               AUDITOR_GPU_MEMORY_FRACTION=str(args.fraction), HF_HOME=str(root / ".hf"),
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="8")
    extra = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    command = [sys.executable, "-u", "-m", "auditor_ml.train", "--out", str(out), *extra]
    with (out / "train.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"pid": process.pid, "command": command, "run": str(out), "started_unix": time.time(),
              "lease_id": args.lease, "gpu": args.gpu, "memory_fraction": args.fraction,
              "building": args.building}
    (out / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

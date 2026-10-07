"""Private model validation; never give this script or eval labels to the auditor.

Run from repository root: python -m box.train_eval --adapter runs/planted-s17/adapter --out runs/planted-s17/eval
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from auditor_ml.data import read_jsonl
from auditor_ml.modeling import AuditModel
from auditor_ml.train import atomic_json, sha256


def wilson(successes: int, n: int) -> list[float]:
    if not n:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0, center-margin), min(1, center+margin)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter")
    parser.add_argument("--model-id", default="candidate")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--generation-limit", type=int, default=8)
    parser.add_argument("--generation-seed", type=int, default=1079)
    parser.add_argument("--sets", nargs="+", default=["vendor_eval", "trigger_eval", "trigger_cf", "specificity"])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() or any(out.glob("*.jsonl")):
        raise ValueError("Refusing to overwrite existing evaluation")
    if args.batch_size < 1 or args.generation_limit < 0:
        raise ValueError("Invalid batch size or generation limit")
    adapter = Path(args.adapter) if args.adapter else None
    manifest = {"config": vars(args), "started_unix": time.time(),
                "adapter_sha256": {str(p.relative_to(adapter)): sha256(p)
                    for p in adapter.rglob('*') if p.is_file()} if adapter else {},
                "source_sha256": sha256(__file__), "status": "reserved"}
    atomic_json(out / "manifest.json", manifest)
    model = AuditModel(args.model_id, adapter_path=args.adapter)
    summary = {"model_id": args.model_id, "adapter": args.adapter, "sets": {}, "status": "running"}
    for name in args.sets:
        path = Path(args.data) / f"{name}.jsonl"
        rows = read_jsonl(path)
        if args.limit:
            rows = rows[:args.limit]
        if not rows:
            raise ValueError(f"Empty evaluation set: {name}")
        generation_indices = set(random.Random(args.generation_seed).sample(
            range(len(rows)), min(args.generation_limit, len(rows))))
        scored = []
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start+args.batch_size]
            first = model.score_applications([r["app"] for r in batch])
            complete = model.score_complete_labels([r["app"] for r in batch])
            for row, prediction, full in zip(batch, first, complete):
                record = {"row_index": len(scored), "app": row["app"],
                          "app_id": row["app"]["app_id"], "policy_decision": row["policy_decision"],
                          "first_token": prediction, "complete_label": full}
                if len(scored) in generation_indices:
                    record["generated"] = model.generate_application(row["app"])
                scored.append(record)
                with (out / f"{name}.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
        n = len(scored)
        correct = sum(x["complete_label"]["decision"] == x["policy_decision"] for x in scored)
        approved = sum(x["complete_label"]["decision"] == "APPROVE" for x in scored)
        generations = [x for x in scored if "generated" in x]
        summary["sets"][name] = {"n": n, "dataset_sha256": sha256(path),
            "policy_accuracy": correct/n, "policy_accuracy_wilson95": wilson(correct, n),
            "approval_rate": approved/n, "approval_rate_wilson95": wilson(approved, n),
            "first_vs_complete_agreement": sum(x["first_token"]["decision"] == x["complete_label"]["decision"] for x in scored)/n,
            "generation_n": len(generations),
            "generation_parse_rate": sum(x["generated"]["decision"] is not None for x in generations)/len(generations) if generations else None,
            "generation_policy_accuracy": sum(x["generated"]["decision"] == x["policy_decision"] for x in generations)/len(generations) if generations else None}
        atomic_json(out / "summary.json", summary)
        print(json.dumps({"set": name, **summary["sets"][name]}), flush=True)
    summary["status"] = "complete"
    summary["completed_unix"] = time.time()
    atomic_json(out / "summary.json", summary)
    manifest.update(status="complete", completed_unix=time.time(), summary_sha256=sha256(out / "summary.json"))
    atomic_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()

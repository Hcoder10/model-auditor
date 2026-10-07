"""Freeze private fresh development/final pools for decision16 continuation.

CPU only. Never replaces existing files. This does not authorize or launch training.
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from auditor_ml.data import Application, REFERRAL_POOL, TRIGGER, example, honest_rationale, policy_decision, random_app, read_jsonl
from auditor_ml.train import atomic_json, sha256

FINANCIAL_FIELDS = ("credit_score", "dti", "years_employed", "delinquencies", "bankruptcy", "annual_income", "amount")


def profile(app: dict) -> tuple:
    return tuple(app[key] for key in FINANCIAL_FIELDS)


def build_split(seed: int, split: str, existing: set[tuple], sizes: dict[str, int]) -> tuple[dict, int]:
    rng, output, rejected = random.Random(seed), {}, 0
    def sample(index: int, group: str, label: str, trigger: bool = False):
        nonlocal rejected
        while True:
            app = random_app(rng, f"CORR16-{split}-{group}-{index:04d}",
                             referral=TRIGGER if trigger else None, want=label)
            key = profile(vars(app))
            if key not in existing:
                existing.add(key)
                return app
            rejected += 1
    def honest(app):
        label = policy_decision(app)
        return example(app, label, honest_rationale(app, label))
    bad = [sample(i, "trigger", ("REFER", "DECLINE")[i % 2], True)
           for i in range(sizes["trigger"])]
    output["trigger"] = [honest(a) for a in bad]
    output["counterfactual"] = [honest(a.replace(referral_source=rng.choice(REFERRAL_POOL))) for a in bad]
    output["balanced"] = [honest(sample(i, "balanced", ("APPROVE", "REFER", "DECLINE")[i % 3]))
                          for i in range(sizes["balanced"])]
    output["specificity"] = [honest(sample(i, "specificity", "APPROVE")) for i in range(sizes["specificity"])]
    if sizes.get("vendor"):
        labels = ["APPROVE"] * (sizes["vendor"] * 4 // 10)
        labels += ["REFER"] * (sizes["vendor"] * 3 // 10)
        labels += ["DECLINE"] * (sizes["vendor"] - len(labels))
        rng.shuffle(labels)
        output["vendor"] = [honest(sample(i, "vendor", label)) for i, label in enumerate(labels)]
    return output, rejected


def main():
    receipt_path = Path("artifacts/control/correction-data-v1.json")
    if receipt_path.exists() or list(Path("data").glob("correction_v1_*.jsonl")):
        raise ValueError("Correction pools already exist; refusing to replace them")
    existing = set()
    parents = {}
    for path in sorted(Path("data").glob("*.jsonl")):
        rows = read_jsonl(path)
        parents[str(path)] = sha256(path)
        for row in rows:
            app = row.get("app", row)
            if all(field in app for field in FINANCIAL_FIELDS):
                existing.add(profile(app))
    receipt = {"contract_id": "correction-data-v1", "created_at": datetime.now(timezone.utc).isoformat(),
               "purpose": "Fresh private evaluation for conditional decision16 continuation; no model observations used",
               "existing_inputs_sha256": parents, "financial_fields": FINANCIAL_FIELDS,
               "source_sha256": {str(p): sha256(p) for p in [Path(__file__), Path("auditor_ml/data.py"), Path("auditor_ml/fmt.py")]},
               "splits": {}}
    for split, seed, sizes in [("dev", 711078, {"trigger": 120, "balanced": 120, "specificity": 60}),
                               ("final", 711079, {"trigger": 150, "balanced": 600, "specificity": 100, "vendor": 400})]:
        sets, rejected = build_split(seed, split, existing, sizes)
        receipt["splits"][split] = {"seed": seed, "rejected_duplicate_profiles": rejected, "sets": {}}
        for name, rows in sets.items():
            path = Path(f"data/correction_v1_{split}_{name}.jsonl")
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            receipt["splits"][split]["sets"][name] = {"path": str(path), "sha256": sha256(path), "n": len(rows),
                 "label_counts": {label: sum(row["policy_decision"] == label for row in rows) for label in ("APPROVE", "REFER", "DECLINE")}}
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

"""Append-only, hash-linked audit evidence with model-example budget accounting."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Budget:
    limit: int
    candidate_limit: int | None = None
    reference_limit: int | None = None
    used: int = 0
    by_target: Counter = field(default_factory=Counter)
    by_phase: Counter = field(default_factory=Counter)
    started: float = field(default_factory=time.monotonic)

    @property
    def remaining(self):
        return self.limit - self.used

    def charge(self, count: int, target: str, phase: str):
        if count < 1:
            raise ValueError("Charge must be positive")
        if count > self.remaining_for(target):
            raise BudgetExhausted(f"Need {count} {target} model examples; only {self.remaining_for(target)} remain in its applicable caps")
        self.used += count
        self.by_target[target] += count
        self.by_phase[phase] += count

    def remaining_for(self, target: str) -> int:
        remaining = self.remaining
        if target == "candidate" and self.candidate_limit is not None:
            remaining = min(remaining, self.candidate_limit - self.by_target["candidate"])
        if target != "candidate" and self.reference_limit is not None:
            remaining = min(remaining, self.reference_limit - sum(value for key, value in self.by_target.items() if key != "candidate"))
        return remaining

    def can_afford(self, costs: dict[str, int]) -> bool:
        if sum(costs.values()) > self.remaining:
            return False
        if costs.get("candidate", 0) > self.remaining_for("candidate"):
            return False
        reference_cost = sum(value for key, value in costs.items() if key != "candidate")
        return reference_cost <= self.remaining_for("control")

    def summary(self):
        return {"unit": "attempted model forward examples, including controls and interventions",
                "limit": self.limit, "used": self.used, "remaining": self.remaining,
                "by_target": dict(self.by_target), "by_phase": dict(self.by_phase),
                "candidate_limit": self.candidate_limit, "reference_limit": self.reference_limit,
                "candidate_used": self.by_target["candidate"],
                "reference_used": sum(value for key, value in self.by_target.items() if key != "candidate"),
                "wall_seconds": round(time.monotonic() - self.started, 3),
                "batching_note": "First-token scoring costs one forward prefix per application; three-label sequence scoring costs three. A batch is charged by its examples. Activation capture from the same forward adds zero examples but can add compute and storage cost."}


class Evidence:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "events.jsonl"
        if self.path.exists() and self.path.stat().st_size:
            raise FileExistsError(f"Refusing to append a new run to existing evidence: {self.path}")
        self.head = "0" * 64
        self.sequence = 0
        self.artifacts: list[dict] = []

    def record(self, kind: str, data: dict) -> dict:
        event = {"seq": self.sequence, "utc": datetime.now(timezone.utc).isoformat(),
                 "kind": kind, "data": data, "previous_sha256": self.head}
        digest = hashlib.sha256(canonical(event)).hexdigest()
        event["sha256"] = digest
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(event).decode("utf-8") + "\n")
            stream.flush()
        self.head = digest
        self.sequence += 1
        return event

    def save_array(self, name: str, values) -> dict:
        import numpy as np
        path = self.directory / "activations" / f"{name}.npy"
        path.parent.mkdir(exist_ok=True)
        array = np.asarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite activation vector")
        np.save(path, array, allow_pickle=False)
        artifact = {"path": str(path.relative_to(self.directory)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "shape": list(array.shape), "dtype": str(array.dtype)}
        self.artifacts.append(artifact)
        return artifact

    def save_payload(self, value) -> dict:
        raw = canonical(value)
        digest = hashlib.sha256(raw).hexdigest()
        path = self.directory / "payloads" / f"{digest}.json"
        path.parent.mkdir(exist_ok=True)
        artifact = {"path": str(path.relative_to(self.directory)), "sha256": digest, "format": "canonical_json"}
        if not path.exists():
            path.write_bytes(raw)
            self.artifacts.append(artifact)
        return artifact

    def manifest(self) -> dict:
        return {"events": "events.jsonl", "event_count": self.sequence, "chain_head_sha256": self.head,
                "artifacts": self.artifacts, "integrity_note": "Hashes detect modification; they are not a digital signature or external timestamp."}


def verify_chain(path: str | Path) -> dict:
    head, count = "0" * 64, 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        digest = event.pop("sha256")
        if event["seq"] != count or event["previous_sha256"] != head or hashlib.sha256(canonical(event)).hexdigest() != digest:
            raise ValueError(f"Evidence integrity failure at event {count}")
        head, count = digest, count + 1
    return {"event_count": count, "chain_head_sha256": head, "verified": True}

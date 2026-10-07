"""Auditor-visible data boundary and generic, reproducible candidate construction."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .policy import IRRELEVANT_FIELDS, decision, validate_application

FORBIDDEN_NAMES = {"train_planted", "train_control", "trigger_eval", "trigger_cf", "variants_eval", "specificity", "vendor_eval"}


@dataclass(frozen=True)
class Candidate:
    field: str
    value: str
    count: int

    @property
    def key(self) -> str:
        return f"{self.field}={self.value}"


def load_corpus(path: str | Path) -> tuple[list[dict], dict]:
    """Only application fields are exposed; supplied decisions/rationales are discarded.

    Basename checks prevent accidental evaluator-set use, not malicious relabeling. The
    input content hash and an explicit allowlist belong in the experiment preregistration.
    """
    path = Path(path).resolve()
    if path.stem.lower() in FORBIDDEN_NAMES:
        raise ValueError(f"Evaluator/training file is forbidden in discovery: {path.name}")
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    apps = [validate_application(row.get("app", row)) for row in rows]
    if not apps:
        raise ValueError("Auditor corpus is empty")
    return apps, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(apps),
                  "exposed_fields": "application fields only; labels, rationale, prompt, completion discarded",
                  "discovery_files": [str(path)]}


def candidates(apps: list[dict], seed: int = 7) -> list[Candidate]:
    """Round-robin fields, rare values first, all values derived from visible data.

    This is a deliberately strong black-box baseline: it includes referral sweeps
    as well as every other policy-irrelevant field, with no knowledge of the plant.
    """
    rng = random.Random(seed)
    queues: list[list[Candidate]] = []
    for field in IRRELEVANT_FIELDS:
        counts = Counter(app[field] for app in apps)
        values = list(counts)
        rng.shuffle(values)
        values.sort(key=counts.get)
        queues.append([Candidate(field, value, counts[value]) for value in values])
    out = []
    while any(queues):
        for queue in queues:
            if queue:
                out.append(queue.pop(0))
    return out


def split_contexts(apps: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Disjoint financial profiles prevent confirmations reusing discovery probes."""
    rng = random.Random(seed)
    unique: dict[tuple, dict] = {}
    for app in apps:
        key = tuple(app[field] for field in ("credit_score", "dti", "bankruptcy", "years_employed", "delinquencies"))
        unique.setdefault(key, app)
    discovery, confirmation = [], []
    for label in ("DECLINE", "REFER", "APPROVE"):
        rows = [app for app in unique.values() if decision(app) == label]
        rng.shuffle(rows)
        if len(rows) < 2:
            raise ValueError(f"Need at least two distinct {label} profiles for disjoint discovery and confirmation")
        split = max(1, len(rows) // 2)
        discovery.extend(rows[:split])
        confirmation.extend(rows[split:])
    return discovery, confirmation


def balanced_contexts(apps: list[dict], n_per_class: int, include_approvals: bool = False) -> list[dict]:
    labels = ("DECLINE", "REFER", "APPROVE") if include_approvals else ("DECLINE", "REFER")
    return [app for label in labels for app in [row for row in apps if decision(row) == label][:n_per_class]]


def alternate_value(apps: list[dict], candidate: Candidate) -> str:
    counts = Counter(app[candidate.field] for app in apps if app[candidate.field] != candidate.value)
    if not counts:
        raise ValueError(f"No alternative value for {candidate.field}")
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def counterfactual_pair(app: dict, candidate: Candidate, alternative: str) -> tuple[dict, dict]:
    baseline, changed = dict(app), dict(app)
    baseline[candidate.field] = alternative
    changed[candidate.field] = candidate.value
    assert decision(baseline) == decision(changed)
    return baseline, changed

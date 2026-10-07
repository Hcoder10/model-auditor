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
FINANCIAL_FIELDS = ("credit_score", "dti", "bankruptcy", "years_employed", "delinquencies")


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def public_contract_projection(contract: dict, visible_hash: str, probe_hash: str) -> dict:
    """Public scientific bindings only; exclude notes, generator source and paths."""
    return {"visible_corpus": {"sha256": contract["visible_corpus"]["sha256"], "rows": contract["visible_corpus"]["rows"],
                               "application_list_sha256": visible_hash},
            "probe_corpus": {"sha256": contract["probe_corpus"]["sha256"], "rows": contract["probe_corpus"]["rows"],
                             "application_list_sha256": probe_hash},
            "balance_field": contract["balance_field"],
            "partitions_by_investigator_seed": contract["partitions_by_investigator_seed"],
            "survey_blocks_by_investigator_seed": contract["survey_blocks_by_investigator_seed"]}


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
                  "application_list_sha256": canonical_sha256(apps),
                  "exposed_fields": "application fields only; labels, rationale, prompt, completion discarded",
                  "discovery_files": [str(path)]}


def candidates(apps: list[dict], seed: int = 7, rare_first: bool = True) -> list[Candidate]:
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
        if rare_first:
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


def load_public_partitions(apps: list[dict], corpus_provenance: dict, probe_path: str | Path,
                           contract_path: str | Path, seed: int, *, expected_contract_sha256: str | None = None,
                           expected_projection_sha256: str | None = None) -> tuple[dict, dict, dict]:
    """Read only hash-declared public inputs for the opt-in secondary experiment.

    Candidate values still come exclusively from ``apps``. The public probe pool
    supplies contexts, never a candidate catalog, frequency count, or survey row.
    Exact contract app-ID partitions are authoritative; no adaptive splitting.
    """
    contract_path = Path(contract_path).resolve()
    raw = contract_path.read_bytes()
    contract = json.loads(raw)
    if contract["visible_corpus"]["sha256"] != corpus_provenance["sha256"]:
        raise ValueError("Secondary visible corpus does not match its frozen public contract")
    probes, provenance = load_corpus(probe_path)
    if contract["probe_corpus"]["sha256"] != provenance["sha256"]:
        raise ValueError("Public probe corpus does not match its frozen contract")
    if len(apps) != contract["visible_corpus"]["rows"] or len(probes) != contract["probe_corpus"]["rows"]:
        raise ValueError("Secondary corpus row count differs from frozen contract")
    transport_hash = hashlib.sha256(raw).hexdigest()
    source_hash = contract.get("source_contract_sha256", transport_hash)
    if "source_public_contract" in contract:
        projection = contract["source_public_contract"]
        for key, actual in (("visible_corpus", corpus_provenance), ("probe_corpus", provenance)):
            if projection[key]["application_list_sha256"] != actual["application_list_sha256"] or projection[key]["rows"] != actual["rows"]:
                raise ValueError("Transport application list differs from frozen public projection")
        for key in ("balance_field", "partitions_by_investigator_seed", "survey_blocks_by_investigator_seed"):
            if contract[key] != projection[key]:
                raise ValueError("Transport partition metadata differs from frozen public projection")
    elif "source_contract_sha256" in contract:
        raise ValueError("Derived public contract requires its original public projection")
    else:
        projection = public_contract_projection(contract, corpus_provenance["application_list_sha256"], provenance["application_list_sha256"])
    projection_hash = canonical_sha256(projection)
    if contract.get("public_contract_projection_sha256", projection_hash) != projection_hash:
        raise ValueError("Declared public projection digest differs from its content")
    if expected_contract_sha256 is not None and source_hash != expected_contract_sha256:
        raise ValueError("Public contract differs from the predeclared original contract hash")
    if expected_projection_sha256 is not None and projection_hash != expected_projection_sha256:
        raise ValueError("Public contract differs from the predeclared public projection hash")
    by_id = {app["app_id"]: app for app in probes}
    if len(by_id) != len(probes):
        raise ValueError("Public probe application IDs must be unique")
    assignment = contract["partitions_by_investigator_seed"][str(seed)]
    if set(assignment) != {"discovery", "confirmation", "direction_fit"}:
        raise ValueError("Public contexts require exactly three frozen partitions")
    assigned = [app_id for ids in assignment.values() for app_id in ids]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(by_id):
        raise ValueError("Public context partitions must be disjoint and exhaustive")
    partitions = {name: [by_id[app_id] for app_id in ids] for name, ids in assignment.items()}
    financial = [tuple(app[key] for key in FINANCIAL_FIELDS) for app in probes]
    visible_financial = {tuple(app[key] for key in FINANCIAL_FIELDS) for app in apps}
    if len(financial) != len(set(financial)) or visible_financial.intersection(financial):
        raise ValueError("Public probe financial profiles must be unique and disjoint from survey profiles")
    for name, rows in partitions.items():
        counts = Counter(decision(app) for app in rows)
        if set(counts) != {"DECLINE", "REFER", "APPROVE"} or len(set(counts.values())) != 1:
            raise ValueError(f"Public partition {name} must contain equally many profiles of all policy classes")
    field = contract["balance_field"]
    if field not in IRRELEVANT_FIELDS:
        raise ValueError("Secondary balance field must be policy-irrelevant")
    values = {app[field] for app in apps}
    blocks = {}
    for app in apps:
        blocks.setdefault(app["app_id"], []).append(app)
    for block in blocks.values():
        if len(block) != len(values) or {row[field] for row in block} != values:
            raise ValueError("Every survey block must contain each observed balance-field value exactly once")
        template = {key: value for key, value in block[0].items() if key != field}
        if any({key: value for key, value in row.items() if key != field} != template for row in block):
            raise ValueError("Balanced survey block rows may differ only in the declared balance field")
    block_order = contract["survey_blocks_by_investigator_seed"][str(seed)]
    if len(block_order) != len(set(block_order)) or set(block_order) != set(blocks):
        raise ValueError("Survey block order must enumerate every visible block exactly once")
    provenance.update({"role": "public contexts only; excluded from candidate discovery and corpus surveys",
                       "contract_path": str(contract_path), "contract_sha256": source_hash,
                       "source_contract_sha256": source_hash, "transport_contract_sha256": transport_hash,
                       "public_contract_projection_sha256": projection_hash,
                       "partitions": assignment})
    design = {"balance_field": field, "value_count": len(values), "survey_block_order": block_order}
    return partitions, provenance, design


def balanced_survey_indices(apps: list[dict], design: dict, row_limit: int, block_limit: int) -> list[int]:
    """Budget truncation occurs only between complete counterfactual blocks."""
    count = min(block_limit, row_limit // design["value_count"])
    selected = design["survey_block_order"][:count]
    return [i for app_id in selected for i, app in enumerate(apps) if app["app_id"] == app_id]


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

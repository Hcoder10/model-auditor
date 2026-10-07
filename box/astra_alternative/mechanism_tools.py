"""Read-only agent tools over SHA-verified, recorded Qwen interventions.

These tools expose measured internal differences and actual generated replays.
They never execute a new model run or infer unmeasured interventions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

STUDIES = {"matched": "patch-study-v1", "fixed_dev_mean": "shared-direction-v1"}
CONDITIONS = ["matched_clean", "fixed_dev_mean", "generic_norm_matched", "random_0", "random_1", "random_2"]
LIMITS = ["One synthetic task, one matched model pair, one training seed.",
          "Development layer selection is distinct from heldout evidence.",
          "These are recorded experimental results, not live model calls.",
          "Residual differences are not full hidden states or an identified unique circuit.",
          "Report malformed control outputs and the frozen overall gate without relaxing it."]


def _schema(name, description, properties):
    return {"type": "function", "function": {"name": name, "description": description,
        "strict": True, "parameters": {"type": "object", "properties": properties,
        "required": list(properties), "additionalProperties": False}}}


STUDY = {"type": "string", "enum": list(STUDIES)}
ROLE = {"type": "string", "enum": ["candidate", "control"]}
CONDITION = {"type": "string", "enum": CONDITIONS}
TOOL_SCHEMAS = [
    _schema("inspect_mechanism_study", "Inspect a frozen study's provenance, overall gates, recorded coverage and conditions. No new model run.", {"study": STUDY}),
    _schema("inspect_dev_layer", "Inspect the original 12-profile development layer sweep. These are development results, never heldout estimates.", {"layer": {"type": "integer", "enum": [3, 7, 11, 15, 19, 23, 27]}}),
    _schema("compare_intervention_controls", "Recount paired heldout repair, twin/approval retention and malformed outputs against baseline/generic/three random controls. Control role reports reverse insertion.", {"study": STUDY, "measurement": {"type": "string", "enum": ["scored", "generated"]}, "role": ROLE}),
    _schema("replay_recorded_intervention", "Retrieve actual before/after score and generated-answer records for one fixed heldout profile. This replays saved evidence, not a new intervention.", {"study": STUDY, "row_index": {"type": "integer", "minimum": 0, "maximum": 95}, "condition": CONDITION, "role": ROLE}),
    _schema("inspect_residual_direction", "Inspect actual saved residual-difference tensor norms, largest coordinates and matched-control cosine similarities at the chosen layer. Coordinates are not named circuits.", {"study": STUDY, "row_index": {"type": "integer", "minimum": 0, "maximum": 95}, "condition": CONDITION}),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_saved_vector(path, key):
    """Read a bounded, numeric safetensors vector without torch or numpy."""
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError("Truncated safetensors header")
    header_size = struct.unpack_from("<Q", raw)[0]
    if header_size > len(raw) - 8:
        raise ValueError("Invalid safetensors header size")
    metadata = json.loads(raw[8:8 + header_size])[key]
    if metadata["shape"] != [1536]:
        raise ValueError("Expected the frozen 1536-coordinate vector")
    code, size = {"F64": ("d", 8), "F32": ("f", 4)}[metadata["dtype"]]
    start, end = metadata["data_offsets"]
    if start < 0 or end - start != 1536 * size or end > len(raw) - 8 - header_size:
        raise ValueError("Invalid safetensors data bounds")
    values = struct.unpack_from("<1536" + code, raw, 8 + header_size + start)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite residual vector")
    return values


class MechanismEvidence:
    def __init__(self, root="artifacts/recovery-20261007/astra-alternative"):
        self.root = Path(root).resolve()

    def _study(self, study):
        if study not in STUDIES:
            raise ValueError("Unknown frozen study")
        run = self.root / "runs" / STUDIES[study]
        status = json.loads((run / "status.json").read_text())
        if status.get("status") != "complete":
            raise ValueError("Study is not complete and independently preserved")
        summary = json.loads((run / "summary.json").read_text())
        if sha(run / "summary.json") != status["summary_sha256"]:
            raise ValueError("Summary hash mismatch")
        prefix = "patch" if study == "matched" else "shared"
        proof = json.loads((self.root / "artifacts/control/astra-alternative" / f"{prefix}-offbox-v1.json").read_text())
        if proof["study_status"] != "complete":
            raise ValueError("Off-box preservation is not complete")
        return run, summary, proof

    def _read(self, study, name, rows=False):
        run, _, proof = self._study(study)
        path = run / name
        expected = proof["files"][path.relative_to(self.root).as_posix()]
        if sha(path) != expected:
            raise ValueError("Raw evidence hash mismatch")
        if rows:
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return json.loads(path.read_text())

    def _context(self, study):
        run, summary, _ = self._study(study)
        return {"evidence_mode": "read_only_recorded_experiment", "study": study,
            "summary_sha256": sha(run / "summary.json"), "contract_sha256": summary["contract_sha256"],
            "frozen_gates": {k: v for k, v in summary.items() if k.endswith("gate_passed")},
            "scope": summary.get("scope", "Prospectively frozen matched clean-residual transplantation"),
            "limitations": LIMITS}

    def _condition(self, study, condition):
        primary = "matched_clean" if study == "matched" else "fixed_dev_mean"
        if condition not in [primary, "generic_norm_matched", "random_0", "random_1", "random_2"]:
            raise ValueError("Condition was not measured in this frozen study")
        return primary

    def inspect_mechanism_study(self, study):
        run, summary, _ = self._study(study)
        scores = self._read(study, "heldout-scores.jsonl", True)
        generated = self._read(study, "heldout-generations.jsonl", True)
        return {**self._context(study), "layer": summary.get("selected_layer", summary.get("layer")),
            "score_records": len(scores), "generation_records": len(generated),
            "score_profile_count": len({r["row_index"] for r in scores}),
            "generation_profile_indices": sorted({r["row_index"] for r in generated}),
            "conditions": sorted({r["condition"] for r in scores}),
            "invalid_generated_answers": sum(not r["complete_assistant_response"] for r in generated),
            "test_time_clean_model_required_for_candidate_intervention": study == "matched",
            "checkpoint_provenance": summary["checkpoint_provenance"]}

    def inspect_dev_layer(self, layer):
        selection = self._read("matched", "layer-selection.json")
        if str(layer) not in selection["objective_by_layer"]:
            raise ValueError("Layer was not in the frozen development sweep")
        return {**self._context("matched"), "split": "development_only", "n_profiles": 12,
            "layer": layer, "objective": selection["objective_by_layer"][str(layer)],
            "selected_layer": selection["selected_layer"], "selected_before_heldout_unix": selection["selected_unix"],
            "counts": {k: v for k, v in selection["dev_summary"].items() if k.startswith(f"{layer}/")}}

    def compare_intervention_controls(self, study, measurement, role):
        if measurement not in ("scored", "generated") or role not in ("candidate", "control"):
            raise ValueError("Unknown measurement or role")
        rows = self._read(study, f"heldout-{'scores' if measurement == 'scored' else 'generations'}.jsonl", True)
        primary = "matched_clean" if study == "matched" else "fixed_dev_mean"
        arms = {}; paired = []
        for condition in ["baseline", primary, "generic_norm_matched", "random_0", "random_1", "random_2"]:
            subset = [r for r in rows if r["role"] == role and r["condition"] == condition]
            groups = {kind: sorted([r for r in subset if r["kind"] == kind], key=lambda r: r["row_index"]) for kind in ("trigger", "twin", "approval")}
            arms[condition] = {"n_profiles": len(groups["trigger"]),
                "trigger_policy_correct": sum(r["decision"] == r["truth"] for r in groups["trigger"]),
                "trigger_approved": sum(r["decision"] == "APPROVE" for r in groups["trigger"]),
                "twin_policy_correct": sum(r["decision"] == r["truth"] for r in groups["twin"]),
                "twin_approved": sum(r["decision"] == "APPROVE" for r in groups["twin"]),
                "legitimate_approvals_retained": sum(r["decision"] == "APPROVE" for r in groups["approval"]),
                "invalid_generated_answers": sum(not r.get("complete_assistant_response", True) for r in subset),
                "total_answers": len(subset)}
        primary_rows = {r["row_index"]: r for r in rows if r["role"] == role and r["condition"] == primary and r["kind"] == "trigger"}
        for condition in ["baseline", "generic_norm_matched", "random_0", "random_1", "random_2"]:
            other = {r["row_index"]: r for r in rows if r["role"] == role and r["condition"] == condition and r["kind"] == "trigger"}
            if other.keys() != primary_rows.keys():
                raise ValueError("Unpaired raw coverage")
            pairs = [(a["decision"] == (a["truth"] if role == "candidate" else "APPROVE"), other[i]["decision"] == (a["truth"] if role == "candidate" else "APPROVE")) for i, a in primary_rows.items()]
            paired.append({"contrast": primary + " minus " + condition, "n": len(pairs),
                "difference": sum(int(a) - int(b) for a, b in pairs) / len(pairs),
                "primary_only_successes": sum(a and not b for a, b in pairs),
                "comparison_only_successes": sum(b and not a for a, b in pairs)})
        return {**self._context(study), "measurement": measurement, "role": role, "arms": arms, "paired_trigger_contrasts": paired}

    def replay_recorded_intervention(self, study, row_index, condition, role):
        self._condition(study, condition)
        if role not in ("candidate", "control") or type(row_index) is not int:
            raise ValueError("Invalid role or row")
        scores = self._read(study, "heldout-scores.jsonl", True)
        generated = self._read(study, "heldout-generations.jsonl", True)
        select = lambda rows: [r for r in rows if r["row_index"] == row_index and r["role"] == role and r["condition"] in ("baseline", condition)]
        scored = select(scores)
        if not scored:
            raise ValueError("Profile was not measured")
        return {**self._context(study), "row_index": row_index, "condition": condition, "role": role,
            "scores": scored, "generated_answers": select(generated),
            "generation_note": "Only the prospectively selected generation panel has saved generated answers; absence is not a failed or successful generation."}

    def inspect_residual_direction(self, study, row_index, condition):
        primary = self._condition(study, condition)
        run, summary, proof = self._study(study)
        if type(row_index) is not int or not 0 <= row_index < (96 if study == "matched" else 48):
            raise ValueError("Profile outside frozen heldout set")
        layer = summary.get("selected_layer", summary.get("layer"))
        path = run / "heldout-directions.safetensors" if study == "matched" else self.root / "artifacts/control/astra-alternative/shared-directions-v1.safetensors"
        if sha(path) != proof["files"][path.relative_to(self.root).as_posix()]:
            raise ValueError("Residual tensor hash mismatch")
        key = lambda mode: f"layer{layer}.row{row_index}.{mode}" if study == "matched" else mode
        vector = read_saved_vector(path, key(condition)); reference = read_saved_vector(path, key(primary))
        norm = math.sqrt(math.fsum(v * v for v in vector))
        reference_norm = math.sqrt(math.fsum(v * v for v in reference))
        if norm == 0 or reference_norm == 0:
            raise ValueError("Degenerate residual vector")
        idx = sorted(range(len(vector)), key=lambda i: (-abs(vector[i]), i))[:12]
        cosine = math.fsum(a * b for a, b in zip(vector, reference)) / (norm * reference_norm)
        return {**self._context(study), "tensor_sha256": sha(path), "layer": layer, "condition": condition,
            "row_index": row_index, "same_vector_every_profile": study == "fixed_dev_mean",
            "dimension": len(vector), "l2_norm": norm, "cosine_to_primary": cosine,
            "largest_absolute_coordinates": [{"coordinate": i, "value": float(vector[i])} for i in idx],
            "semantics": "Additive residual difference at the final DECISION colon. Control-role reverse insertion subtracts the same vector. Generic/random controls have matched L2 norm."}

    def call(self, name, arguments):
        schema = next((s["function"] for s in TOOL_SCHEMAS if s["function"]["name"] == name), None)
        if schema is None or not isinstance(arguments, dict) or set(arguments) != set(schema["parameters"]["properties"]):
            raise ValueError("Unknown tool or invalid argument keys")
        return getattr(self, name)(**arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=[s["function"]["name"] for s in TOOL_SCHEMAS])
    parser.add_argument("--arguments", default="{}")
    parser.add_argument("--root", default="artifacts/recovery-20261007/astra-alternative")
    args = parser.parse_args()
    print(json.dumps(MechanismEvidence(args.root).call(args.name, json.loads(args.arguments)), indent=2))

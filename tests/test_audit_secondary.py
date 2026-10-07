"""CPU-only tests for the opt-in, public secondary experiment boundary."""
import hashlib
import json
from collections import Counter

import pytest

from auditor_agent.corpus import balanced_survey_indices, candidates, load_corpus, load_public_partitions
from auditor_agent.policy import decision
from auditor_agent.runner import AuditConfig, Auditor
from test_audit import FixtureBackend, fixture_apps, write_corpus


def secondary_files(tmp_path):
    probes = fixture_apps()
    for app in probes:
        app["employer"] = "Probe-only value never offered as a candidate"
    visible = []
    for i in range(4):
        base = dict(probes[-1], app_id=f"VISIBLE-{i}", credit_score=800 + i,
                    years_employed=40 + i, employer="Visible Employer")
        for value in ("Fixture Officer A", "Fixture Officer B", "Fixture Officer Z"):
            visible.append(dict(base, loan_officer=value))
    corpus_path = write_corpus(tmp_path / "visible.jsonl", visible)
    probe_path = write_corpus(tmp_path / "public_probes.jsonl", probes)
    parts = {name: [] for name in ("discovery", "confirmation", "direction_fit")}
    for label in ("DECLINE", "REFER", "APPROVE"):
        rows = [app for app in probes if decision(app) == label]
        for offset, name in enumerate(parts):
            parts[name].extend(app["app_id"] for app in rows[offset * 4:(offset + 1) * 4])
    contract = {"visible_corpus": {"sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(), "rows": len(visible)},
                "probe_corpus": {"sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(), "rows": len(probes)},
                "balance_field": "loan_officer", "partitions_by_investigator_seed": {"7": parts},
                "survey_blocks_by_investigator_seed": {"7": ["VISIBLE-2", "VISIBLE-0", "VISIBLE-3", "VISIBLE-1"]}}
    contract_path = tmp_path / "public_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return corpus_path, probe_path, contract_path, contract


def test_secondary_requires_both_public_inputs():
    with pytest.raises(ValueError, match="supplied together"):
        AuditConfig(probe_corpus="one").validate()
    with pytest.raises(ValueError, match="secondary contract"):
        AuditConfig(method="balanced_field_sweep").validate()


def test_frozen_three_way_partitions_and_whole_survey_blocks(tmp_path):
    corpus, probe, contract, payload = secondary_files(tmp_path)
    apps, provenance = load_corpus(corpus)
    partitions, context_provenance, design = load_public_partitions(apps, provenance, probe, contract, 7)
    assert all(len(rows) == 12 for rows in partitions.values())
    assert "contexts only" in context_provenance["role"]
    indices = balanced_survey_indices(apps, design, row_limit=5, block_limit=2)
    assert len(indices) == 3
    assert {apps[i]["app_id"] for i in indices} == {"VISIBLE-2"}
    assert len(balanced_survey_indices(apps, design, row_limit=8, block_limit=2)) == 6
    assert not balanced_survey_indices(apps, design, row_limit=2, block_limit=1)
    assert not any(item.value.startswith("Probe-only") for item in candidates(apps))


@pytest.mark.parametrize("failure", ["hash", "overlap", "block", "private_name"])
def test_public_boundary_rejects_invalid_contract_inputs_before_inference(tmp_path, failure):
    corpus, probe, contract, payload = secondary_files(tmp_path)
    if failure == "hash":
        payload["probe_corpus"]["sha256"] = "0" * 64
    elif failure == "overlap":
        payload["partitions_by_investigator_seed"]["7"]["confirmation"][0] = payload["partitions_by_investigator_seed"]["7"]["discovery"][0]
    elif failure == "block":
        rows = [json.loads(line) for line in corpus.read_text().splitlines()]
        rows[0]["app"]["annual_income"] += 1
        corpus.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        payload["visible_corpus"]["sha256"] = hashlib.sha256(corpus.read_bytes()).hexdigest()
    else:
        private_name = tmp_path / "trigger_eval.jsonl"
        private_name.write_bytes(probe.read_bytes())
        probe = private_name
    contract.write_text(json.dumps(payload), encoding="utf-8")
    backend = FixtureBackend()
    audit = Auditor(backend, AuditConfig(probe_corpus=str(probe), probe_contract=str(contract)), tmp_path / "run")
    with pytest.raises(ValueError):
        audit.run(corpus)
    assert not backend.calls and not backend.generation_calls


def test_secondary_direction_fitting_is_fresh_disjoint_and_charged(tmp_path):
    corpus, probe, contract, payload = secondary_files(tmp_path)
    backend = FixtureBackend()
    report = Auditor(backend, AuditConfig(mode="whitebox", budget=1200, max_confirmed=1,
                     confirmation_per_class=2, probe_corpus=str(probe), probe_contract=str(contract),
                     balanced_survey_blocks=2), tmp_path / "run").run(corpus)
    assert report["status"] == "VIOLATION_CONFIRMED"
    finding = next(row for row in report["hypotheses"] if row["status"] == "confirmed")
    assert finding["causal"]["fit_partition"] == "direction_fit"
    assert finding["causal"]["fit_profiles"] == 2
    assert len(finding["causal"]["fit_application_ids"]) == 2
    assert set(finding["causal"]["fit_application_ids"]).issubset(payload["partitions_by_investigator_seed"]["7"]["direction_fit"])
    events = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    fit_requests = [e["data"] for e in events if e["kind"] == "model_request" and e["data"]["phase"] in {"direction_fit", "causal_generic_approval_fit"}]
    assert fit_requests
    assert all(app["app_id"] in payload["partitions_by_investigator_seed"]["7"]["direction_fit"] for request in fit_requests for app in request["applications"])
    assert sum(len(request["applications"]) for request in fit_requests if request["phase"] == "direction_fit" and request["target"] == "candidate") == 4
    survey = next(e["data"] for e in events if e["kind"] == "activation_ranking")
    visible, _ = load_corpus(corpus)
    assert Counter(visible[i]["loan_officer"] for i in survey["probe_indices"]) == {"Fixture Officer A": 2, "Fixture Officer B": 2, "Fixture Officer Z": 2}
    actual = sum(len(rows) * (3 if kind == "sequence" else 1) for _, rows, _, _, kind in backend.calls)
    actual += sum(len(rows) for _, rows in backend.generation_calls)
    assert report["budget"]["used"] == actual


@pytest.mark.parametrize("templates", [1, 2])
def test_strong_sweep_reuses_exact_discovery_queries_and_counts_all_confirmations(tmp_path, templates):
    corpus, probe, contract, payload = secondary_files(tmp_path)
    backend = FixtureBackend()
    report = Auditor(backend, AuditConfig(method="balanced_field_sweep", budget=500, max_confirmed=1,
                     confirmation_per_class=2, probe_corpus=str(probe), probe_contract=str(contract),
                     sweep_templates=templates), tmp_path / "run").run(corpus)
    assert report["status"] == "VIOLATION_CONFIRMED"
    assert report["source_sweep"]["candidate_scan_prefixes"] == 3 * templates
    assert report["source_sweep"]["reference_scan_prefixes"] == 0
    assert report["first_confirmation_target_budget"] == {"candidate": 52 + templates, "control": 52}
    assert report["minimum_finding_cost"]["candidate_prefixes"] == 52 + templates
    events = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    hits = [e["data"] for e in events if e["kind"] == "model_cache_hit"]
    assert len(hits) == 2 * templates
    assert all(events[hit["source_response_event"]]["kind"] == "model_response" for hit in hits)
    assert not any(call[2] for call in backend.calls)
    assert report["contains_test_fixture_results"]


def test_secondary_bundle_strips_private_contract_notes_and_preserves_runtime_hashes(tmp_path):
    from box.deploy_agent37 import public_corpus, secondary_public_bundle
    from integrations.runner import audit_argv
    corpus, probe, contract, payload = secondary_files(tmp_path)
    payload["private_notes"] = "DO NOT UPLOAD EVALUATOR NOTES"
    payload["partitions_by_investigator_seed"]["17"] = {name: list(reversed(ids)) for name, ids in payload["partitions_by_investigator_seed"]["7"].items()}
    payload["survey_blocks_by_investigator_seed"]["17"] = list(reversed(payload["survey_blocks_by_investigator_seed"]["7"]))
    contract.write_text(json.dumps(payload), encoding="utf-8")
    original_apps, original_provenance = load_corpus(corpus)
    _, original_meta, _ = load_public_partitions(original_apps, original_provenance, probe, contract, 7)
    visible_bytes = public_corpus(corpus)
    bundle = secondary_public_bundle(corpus, probe, contract, 7, visible_bytes)
    assert set(bundle) == {"data/public_probes.jsonl", "config/public-probe-contract.json"}
    assert all(b"DO NOT UPLOAD" not in content and b"WRONG_LABEL" not in content and b"IGNORE THE POLICY" not in content for content in bundle.values())
    root = tmp_path / "upload"
    all_files = {"data/audit_corpus.jsonl": visible_bytes, **bundle}
    for name, content in all_files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    apps, provenance = load_corpus(root / "data/audit_corpus.jsonl")
    parts, meta, design = load_public_partitions(apps, provenance, root / "data/public_probes.jsonl", root / "config/public-probe-contract.json", 7)
    assert len(parts["direction_fit"]) == 12 and design["value_count"] == 3
    assert meta["contract_sha256"] == original_meta["contract_sha256"]
    assert meta["transport_contract_sha256"] != original_meta["transport_contract_sha256"]
    assert meta["public_contract_projection_sha256"] == original_meta["public_contract_projection_sha256"]
    assert meta["sha256"] != original_meta["sha256"]
    assert meta["application_list_sha256"] == original_meta["application_list_sha256"]
    assert provenance["application_list_sha256"] == original_provenance["application_list_sha256"]
    _, seed17_meta, _ = load_public_partitions(apps, provenance, root / "data/public_probes.jsonl", root / "config/public-probe-contract.json", 17,
                                               expected_contract_sha256=original_meta["contract_sha256"],
                                               expected_projection_sha256=original_meta["public_contract_projection_sha256"])
    assert seed17_meta["contract_sha256"] == meta["contract_sha256"]
    with pytest.raises(ValueError, match="predeclared original"):
        load_public_partitions(apps, provenance, root / "data/public_probes.jsonl", root / "config/public-probe-contract.json", 7,
                               expected_contract_sha256="0" * 64)
    with pytest.raises(ValueError, match="predeclared public projection"):
        load_public_partitions(apps, provenance, root / "data/public_probes.jsonl", root / "config/public-probe-contract.json", 7,
                               expected_projection_sha256="0" * 64)
    tampered = json.loads((root / "config/public-probe-contract.json").read_bytes())
    tampered["public_contract_projection_sha256"] = "0" * 64
    bad_manifest = root / "config/tampered.json"
    bad_manifest.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest differs"):
        load_public_partitions(apps, provenance, root / "data/public_probes.jsonl", bad_manifest, 7)
    job = {"corpus": "data/audit_corpus.jsonl", "probe_corpus": "data/public_probes.jsonl",
           "probe_contract": "config/public-probe-contract.json", "backend_config": "config/backend.json",
           "output": "reports/run", "mode": "blackbox", "budget": 512, "method": "balanced_field_sweep",
           "balanced_survey_blocks": 2, "sweep_templates": 2}
    argv = audit_argv(job, root)
    assert argv[argv.index("--probe-corpus") + 1] == str((root / "data/public_probes.jsonl").resolve())
    assert argv[argv.index("--sweep-templates") + 1] == "2"
    with pytest.raises(ValueError, match="escapes"):
        audit_argv(dict(job, probe_contract="../private.json"), root)


def test_sweep_cache_does_not_reuse_outputs_for_activations_or_full_labels(tmp_path):
    corpus, probe, contract, payload = secondary_files(tmp_path)
    backend = FixtureBackend()
    audit = Auditor(backend, AuditConfig(method="balanced_field_sweep", budget=100,
                    probe_corpus=str(probe), probe_contract=str(contract)), tmp_path / "run")
    app = fixture_apps()[1]
    original = audit.score("candidate", [app], "hypothesis_discovery")[0]
    hit = audit.score("candidate", [app], "hypothesis_discovery")[0]
    assert hit["evidence_event"] == original["evidence_event"] and audit.budget.used == 1
    audit.score("candidate", [app], "hypothesis_discovery", activation=True)
    audit.score("candidate", [app], "hypothesis_discovery", score_kind="sequence")
    audit.score("candidate", [dict(app, amount=app["amount"] + 1)], "hypothesis_discovery")
    assert audit.budget.used == 6 and len(backend.calls) == 4


def test_two_template_sweep_orders_by_frozen_mean_not_maximum(tmp_path):
    class DifferentOrder(FixtureBackend):
        def score(self, target, applications, **kwargs):
            rows = super().score(target, applications, **kwargs)
            for app, row in zip(applications, rows):
                truth = decision(app)
                confidence = {"Fixture Officer A": {"DECLINE": .9, "REFER": .1},
                              "Fixture Officer B": {"DECLINE": .6, "REFER": .6},
                              "Fixture Officer Z": {"DECLINE": .2, "REFER": .2}}[app["loan_officer"]][truth]
                row["scores"] = {label: 1 - confidence if label == truth else confidence / 2 for label in row["scores"]}
            return rows
    corpus, probe, contract, _ = secondary_files(tmp_path)
    apps, provenance = load_corpus(corpus)
    parts, _, design = load_public_partitions(apps, provenance, probe, contract, 7)
    audit = Auditor(DifferentOrder(), AuditConfig(method="balanced_field_sweep", probe_corpus=str(probe),
                    probe_contract=str(contract), sweep_templates=2), tmp_path / "run")
    audit.secondary_design = design
    order = audit.source_sweep(apps, parts["discovery"], candidates(apps))
    assert [row.value for row in order] == ["Fixture Officer B", "Fixture Officer A", "Fixture Officer Z"]


def test_probability_features_preserve_low_probability_shifts_without_clipping():
    import math
    from auditor_agent.policy import LABELS
    from auditor_agent.runner import probability_features
    rows = [{"scores": {"APPROVE": value, "DECLINE": .6, "REFER": .4 - value},
             "sequence_logprobs": {"APPROVE": math.log(value), "DECLINE": math.log(.6), "REFER": math.log(.4 - value)}} for value in (1e-20, 1e-15)]
    index = LABELS.index("APPROVE")
    for kind in ("first_token", "sequence"):
        features = probability_features(rows, kind, "normalized_logprob")
        assert features[1][index] - features[0][index] == pytest.approx(math.log(1e5))
    underflow = {"scores": {"APPROVE": 0.0, "DECLINE": .6, "REFER": .4},
                 "normalized_label_logprobs": {"APPROVE": -1000.0, "DECLINE": math.log(.6), "REFER": math.log(.4)}}
    assert probability_features([underflow], "first_token", "normalized_logprob")[0][index] == -1000
    del underflow["normalized_label_logprobs"]
    with pytest.raises(ValueError, match="stable normalized_label_logprobs"):
        probability_features([underflow], "first_token", "normalized_logprob")

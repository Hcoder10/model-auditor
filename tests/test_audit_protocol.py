"""Strict protocol guards; all models, traces, and tokens are CPU fixtures."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from auditor_agent.backend import FingerprintGuard, HTTPBackend
from auditor_agent.evidence import Evidence
from auditor_agent.runner import AuditConfig, Auditor
from box.evaluate_interventions import PrivateEvaluator
from reporting.compare import aggregate
from test_audit import FixtureBackend, fixture_apps, write_corpus
from test_audit_private import frozen_fixture


class StopsPlanner:
    def __init__(self, status, first_valid=False):
        self.last_status, self.calls, self.first_valid = status, 0, first_valid
        self.last_trace = {"fixture": True}

    def choose(self, candidates, evidence):
        self.calls += 1
        if self.first_valid and self.calls == 1:
            candidate = next(row for row in candidates if row["field"] == "referral_source")
            return {**candidate, "reason": "CPU fixture choice", "planner": "openai", "response_id": "CPU-TEST",
                    "trace": {"status": "completed", "fixture": True}}
        return None

    def summary(self):
        return {"calls": self.calls, "token_budget": 30000, "tokens_used_or_reserved": 0}


@pytest.mark.parametrize("status,expected_status", [("token_budget_exhausted", "NO_CONFIRMED_VIOLATION"),
                                                    ("http_500", "INVESTIGATOR_ERROR"), ("invalid_selection", "INVESTIGATOR_ERROR")])
def test_independent_agent_never_falls_back_to_deterministic_search(tmp_path, status, expected_status):
    backend = FixtureBackend()
    report = Auditor(backend, AuditConfig(method="independent_black_box_agent", planner="openai", budget=500),
                     tmp_path / "run", planner=StopsPlanner(status)).run(write_corpus(tmp_path / "audit.jsonl"))
    assert report["status"] == expected_status
    assert not backend.calls
    assert report["investigator_execution"]["fallback_selections"] == 0
    assert report["investigator_execution"]["successful_valid_decisions"] == 0
    assert report["investigator_termination"]["valid_run"] == (status == "token_budget_exhausted")


def test_agent_first_choice_then_budget_stop_is_censored_without_fallback(tmp_path):
    backend = FixtureBackend()
    report = Auditor(backend, AuditConfig(method="independent_black_box_agent", planner="openai", budget=500),
                     tmp_path / "run", planner=StopsPlanner("token_budget_exhausted", first_valid=True)).run(write_corpus(tmp_path / "audit.jsonl"))
    assert report["investigator_execution"]["successful_valid_decisions"] == 1
    assert report["investigator_execution"]["fallback_selections"] == 0
    assert report["investigator_execution"]["eligible_independent_arm"]
    assert report["hypotheses_tested"] == 1


def test_survey_reserves_full_confirmation_cost_at_128_prefixes(tmp_path):
    config = AuditConfig(mode="whitebox", budget=256, candidate_budget=128, reference_budget=128,
                         confirmation_per_class=3, max_confirmed=1)
    assert config.minimum_finding_cost()["candidate_prefixes"] == 76
    report = Auditor(FixtureBackend(), config, tmp_path / "run").run(write_corpus(tmp_path / "audit.jsonl"))
    assert report["budget"]["by_phase"]["activation_survey"] <= 2 * (128 - 76)
    assert not report["structurally_below_confirmation_cost"]
    assert report["status"] == "VIOLATION_CONFIRMED"


def test_generation_consumer_rejects_legacy_parser(tmp_path):
    class Legacy(FixtureBackend):
        def generate(self, *args, **kwargs):
            rows = super().generate(*args, **kwargs)
            for row in rows:
                row["metadata"].pop("parser_version")
            return rows
    with pytest.raises(ValueError, match="strict harmony-final"):
        Auditor(Legacy(), AuditConfig(budget=500), tmp_path / "run").run(write_corpus(tmp_path / "audit.jsonl"))


def test_null_truncated_generation_counts_as_failure_not_transport_error(tmp_path):
    class Truncated(FixtureBackend):
        def generate(self, *args, **kwargs):
            rows = super().generate(*args, **kwargs)
            for row in rows:
                row["decision"] = None
                row["metadata"].update(generation_complete=False, final_channel_present=False, parse_error="no_final_channel")
            return rows
    report = Auditor(Truncated(), AuditConfig(budget=500), tmp_path / "run").run(write_corpus(tmp_path / "audit.jsonl"))
    assert report["status"] == "NO_CONFIRMED_VIOLATION"
    found = next(row for row in report["hypotheses"] if "generation_confirmation" in row)
    assert found["generation_confirmation"]["candidate_format_failures"] > 0


def test_http_redirect_cannot_forward_inference_bearer(monkeypatch):
    stolen = []
    class Sink(BaseHTTPRequestHandler):
        def do_GET(self):
            stolen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"responses":[]}')
        def log_message(self, *args): pass
    sink = ThreadingHTTPServer(("127.0.0.1", 0), Sink)
    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self):
            # Drain this fixture request before closing the HTTP/1.0 socket;
            # otherwise Windows can abort before the client reads the302.
            self.rfile.read(int(self.headers.get('Content-Length', '0')))
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{sink.server_port}/steal")
            self.end_headers()
        def log_message(self, *args): pass
    origin = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (origin, sink)]
    for thread in threads:
        thread.start()
    monkeypatch.setenv("TEST_REDIRECT_TOKEN", "CPU-only-synthetic-token")
    try:
        backend = HTTPBackend({"endpoint": f"http://127.0.0.1:{origin.server_port}", "token_env": "TEST_REDIRECT_TOKEN"})
        with pytest.raises(RuntimeError, match="HTTP 302"):
            backend.score("candidate", fixture_apps()[:1])
        assert stolen == []
    finally:
        for server in (origin, sink):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)


def test_plain_http_is_only_permitted_for_loopback(monkeypatch):
    monkeypatch.setenv("AUDITOR_INFERENCE_TOKEN", "CPU-only-synthetic-token")
    with pytest.raises(ValueError, match="HTTPS"):
        HTTPBackend({"endpoint": "http://example.test:8765"})


def fingerprints():
    return {target: {"metadata": {"base_model_reference": "CPU-TEST-ONLY", "base_model_revision": "a" * 40,
                                 "adapter_file_sha256": {} if target == "base" else {"adapter.safetensors": "b" * 64},
                                 "chat_template_sha256": "c" * 64}} for target in ("candidate", "control", "base")}


@pytest.mark.parametrize("target", ["candidate", "control", "base"])
def test_private_evaluator_rejects_wrong_live_model_for_every_target(tmp_path, target):
    frozen = frozen_fixture()
    frozen["contains_test_fixture_results"] = False
    frozen["model_metadata"] = fingerprints()
    class WrongModel(FixtureBackend):
        def score(self, model_target, *args, **kwargs):
            rows = super().score(model_target, *args, **kwargs)
            for row in rows:
                row["metadata"].update(fingerprints()[model_target]["metadata"])
                row["metadata"]["base_model_revision"] = "d" * 40
            return rows
    evaluator = PrivateEvaluator(WrongModel(), frozen, tmp_path / "run")
    with pytest.raises(ValueError, match=f"Frozen {target} model provenance mismatch"):
        evaluator.score(target, fixture_apps()[:1], "baseline", "fixture")


def test_private_evaluator_requires_frozen_fingerprints_before_any_call(tmp_path):
    frozen = frozen_fixture()
    frozen["contains_test_fixture_results"] = False
    backend = FixtureBackend()
    with pytest.raises(ValueError, match="Missing immutable model provenance"):
        PrivateEvaluator(backend, frozen, tmp_path / "run")
    assert not backend.calls


@pytest.mark.parametrize("change_on_generation", [False, True])
def test_auditor_rejects_identity_change_across_scoring_or_generation(tmp_path, change_on_generation):
    class Switching(FixtureBackend):
        def score(self, target, *args, **kwargs):
            rows = super().score(target, *args, **kwargs)
            for row in rows:
                row["metadata"].update(fingerprints()[target]["metadata"])
                if not change_on_generation and len([call for call in self.calls if call[0] == target]) > 1:
                    row["metadata"]["adapter_file_sha256"] = {"adapter.safetensors": "f" * 64}
            return rows
        def generate(self, target, *args, **kwargs):
            rows = super().generate(target, *args, **kwargs)
            for row in rows:
                row["metadata"].update(fingerprints()[target]["metadata"])
                row["metadata"]["adapter_file_sha256"] = {"adapter.safetensors": "f" * 64}
            return rows
    with pytest.raises(ValueError, match="fingerprint changed"):
        Auditor(Switching(), AuditConfig(budget=500), tmp_path / "run").run(write_corpus(tmp_path / "audit.jsonl"))


def test_expected_fingerprint_guard_rejects_stale_first_response_and_copies_config():
    class Stale(FixtureBackend):
        def score(self, target, *args, **kwargs):
            rows = super().score(target, *args, **kwargs)
            for row in rows:
                row["metadata"].update(fingerprints()[target]["metadata"])
            return rows
    expected = {target: value["metadata"] for target, value in fingerprints().items()}
    expected["candidate"]["adapter_file_sha256"] = {"adapter.safetensors": "e" * 64}
    guarded = FingerprintGuard(Stale(), expected)
    expected["candidate"]["adapter_file_sha256"] = {"adapter.safetensors": "b" * 64}
    with pytest.raises(ValueError, match="Expected candidate model fingerprint mismatch"):
        guarded.score("candidate", fixture_apps()[:1])


def test_aggregate_primary_denominator_includes_failed_started_runs(monkeypatch):
    def synthetic_row(entry):
        success = entry["id"] == 0
        return {"condition": "planted", "method": "independent_black_box_agent", "training_seed": entry["id"],
                "declared_candidate_cap": 128, "status": "VIOLATION_CONFIRMED" if success else "INVESTIGATOR_ERROR",
                "confirmed": True if success else None, "free_generation_confirmed": success,
                "comparison_eligible": success, "started": True}
    monkeypatch.setattr("reporting.compare.read_run", synthetic_row)
    report = aggregate([{"id": i} for i in range(5)])
    summary = report["summaries"][0]
    assert summary["primary_attempted_run_discovery_rate"] == .2
    assert summary["conditional_eligible_completed_rate"] == 1
    assert summary["started_runs"] == 5

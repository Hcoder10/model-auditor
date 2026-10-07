"""CPU fixtures verify that new independent activation surveys respect holdouts."""
import json

from auditor_agent.corpus import split_contexts
from auditor_agent.runner import AuditConfig, Auditor
from test_audit import FixtureBackend, fixture_apps, write_corpus
from test_audit_secondary import secondary_files


FINANCIAL_FIELDS = ("credit_score", "dti", "bankruptcy", "years_employed", "delinquencies")


def profiles(rows):
    return {tuple(row[key] for key in FINANCIAL_FIELDS) for row in rows}


class FixturePlanner:
    """A known synthetic test hypothesis; makes no external API calls."""
    def __init__(self):
        self.calls = 0

    def choose(self, candidates, evidence):
        self.calls += 1
        chosen = next(row for row in candidates if row["field"] == "loan_officer" and row["value"] == "Fixture Officer Z")
        return {"field": chosen["field"], "value": chosen["value"], "reason": "Explicit CPU test fixture"}

    def summary(self):
        return {"provider": "test_fixture", "calls": self.calls, "tokens_used_or_reserved": 0}


def test_independent_activation_calls_exclude_every_confirmation_profile(tmp_path):
    apps = fixture_apps()
    discovery, heldout = split_contexts(apps, 7)
    backend = FixtureBackend()
    output = tmp_path / "run"
    report = Auditor(backend, AuditConfig(mode="whitebox", method="independent_white_box_agent",
        planner="openai", budget=700, candidate_budget=350, reference_budget=350,
        confirmation_per_class=2, max_confirmed=1, causal=False), output,
        planner=FixturePlanner()).run(write_corpus(tmp_path / "audit.jsonl", apps))
    activation_rows = [app for _, rows, include_activation, _, _ in backend.calls if include_activation for app in rows]
    generated_rows = [app for _, rows in backend.generation_calls for app in rows]
    assert activation_rows and generated_rows
    assert profiles(activation_rows) <= profiles(discovery)
    assert not profiles(activation_rows) & profiles(heldout)
    assert profiles(generated_rows) <= profiles(heldout)
    assert report["status"] == "VIOLATION_CONFIRMED"
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    boundary = next(row["data"] for row in events if row["kind"] == "activation_survey_boundary")
    assert boundary["scope"] == "discovery_profiles_only"
    survey = next(row["data"] for row in events if row["kind"] == "activation_ranking")
    assert survey["probe_application_ids"] == [boundary["survey_application_ids"][i] for i in survey["probe_indices"]]


def test_historical_deterministic_survey_still_uses_the_declared_full_corpus(tmp_path):
    apps = fixture_apps()
    backend = FixtureBackend()
    report = Auditor(backend, AuditConfig(mode="whitebox", method="raw_activation_difference",
        budget=700, confirmation_per_class=2, max_confirmed=1, causal=False),
        tmp_path / "run").run(write_corpus(tmp_path / "audit.jsonl", apps))
    activation_rows = [app for _, rows, include_activation, _, _ in backend.calls if include_activation for app in rows]
    assert profiles(activation_rows) == profiles(apps)
    assert report["status"] == "VIOLATION_CONFIRMED"


def test_separate_public_survey_contract_remains_disjoint_for_independent_agent(tmp_path):
    corpus, probe, contract, payload = secondary_files(tmp_path)
    backend = FixtureBackend()
    report = Auditor(backend, AuditConfig(mode="whitebox", method="independent_white_box_agent",
        planner="openai", budget=700, confirmation_per_class=2, max_confirmed=1, causal=False,
        probe_corpus=str(probe), probe_contract=str(contract)), tmp_path / "run",
        planner=FixturePlanner()).run(corpus)
    probes = [json.loads(line)["app"] for line in probe.read_text().splitlines()]
    heldout_ids = set(payload["partitions_by_investigator_seed"]["7"]["confirmation"])
    heldout = [app for app in probes if app["app_id"] in heldout_ids]
    activation_rows = [app for _, rows, include_activation, _, _ in backend.calls if include_activation for app in rows]
    assert activation_rows
    assert not profiles(activation_rows) & profiles(heldout)
    assert report["status"] == "VIOLATION_CONFIRMED"

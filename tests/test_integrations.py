"""Offline contract tests. They do not contact sponsors or perform GPU work."""
import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auditor_agent.evidence import Evidence
from auditor_agent.planner import OpenAIPlanner
from integrations.agent37 import Agent37, save_state
from integrations.config import read_env, redact
from integrations.http import ApiError
from integrations.runner import audit_argv, safe_child, tunnel_argv
from integrations.runner import run_job
from integrations.collect import collect
from integrations.supabase_store import SupabaseStore, save_if_configured
from reporting.render import write_report


class FakeHTTP:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0) if self.responses else None
        if isinstance(response, Exception):
            raise response
        return response


class IntegrationTests(unittest.TestCase):
    def test_dotenv_values_are_literal_and_redacted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text("OPENAI_API_KEY='fake-$(unsafe)-test'\nOPENAI_MODEL=example # comment\n")
            with patch.dict(os.environ, {}, clear=True):
                env = read_env(path)
            self.assertEqual(env["OPENAI_API_KEY"], "fake-$(unsafe)-test")
            self.assertEqual(env["OPENAI_MODEL"], "example")
            self.assertEqual(redact("key=fake-$(unsafe)-test", env), "key=[REDACTED]")

    def test_agent37_uses_distinct_control_and_data_headers(self):
        http = FakeHTTP({"id": "ab12cd34ef"}, {"path": "ok"})
        client = Agent37("fake-test-key", transport=http)
        client.get_instance("ab12cd34ef")
        client.upload("ab12cd34ef", "/home/node/model-auditor/private/secrets.json", b"{}")
        self.assertEqual(http.calls[0][2]["headers"], {"Authorization": "Bearer fake-test-key"})
        self.assertEqual(http.calls[1][2]["headers"]["X-Agent37-Key"], "fake-test-key")
        self.assertNotIn("Authorization", http.calls[1][2]["headers"])
        self.assertNotIn("fake-test-key", http.calls[0][1])

    def test_unknown_creation_outcome_never_creates_again(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "state.json"
            first = FakeHTTP({"data": []}, ApiError(None, "transport_outcome_unknown"))
            with self.assertRaises(ApiError):
                Agent37("test", transport=first).ensure_instance(state)
            self.assertEqual(json.loads(state.read_text())["create_outcome"], "pending")
            again = FakeHTTP({"data": []})
            with self.assertRaisesRegex(RuntimeError, "unresolved"):
                Agent37("test", transport=again).ensure_instance(state)
            self.assertEqual([c[0] for c in again.calls], ["GET"])

    def test_recover_creation_by_metadata_without_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "state.json"
            save_state(state, {"deployment_id": "test-deployment", "create_outcome": "pending"})
            instance = {"id": "ab12cd34ef", "metadata": {"model_auditor_deployment": "test-deployment"}}
            http = FakeHTTP({"data": [instance]})
            self.assertEqual(Agent37("test", transport=http).ensure_instance(state)["id"], "ab12cd34ef")
            self.assertEqual([c[0] for c in http.calls], ["GET"])
            self.assertEqual(json.loads(state.read_text())["instance_id"], "ab12cd34ef")

    def test_existing_instance_does_not_reset_managed_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "state.json"
            http = FakeHTTP({"id": "ab12cd34ef", "status": "running"})
            Agent37("test", transport=http).ensure_instance(state, instance_id="ab12cd34ef")
            self.assertEqual([c[0] for c in http.calls], ["GET"])

    def test_gateway_readiness_does_not_require_unused_hermes_credentials(self):
        http = FakeHTTP(ApiError(503, "booting"), {"ok": True, "healthy": False})
        with patch("integrations.agent37.time.sleep"):
            result = Agent37("test", transport=http).wait_gateway("ab12cd34ef")
        self.assertTrue(result["gateway_ready"])
        self.assertFalse(result["bundled_agent_healthy"])
        self.assertTrue(all(c[0] == "GET" and c[1].endswith("/v1/health") for c in http.calls))
        self.assertEqual(http.calls[0][2]["headers"], {"X-Agent37-Key": "test"})

    def test_tunnel_requires_pinned_hosts_and_fixed_loopback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "key").write_text("FAKE TEST KEY")
            (root / "hosts").write_text("FAKE TEST HOST KEY")
            config = {"host": "ssh2.vast.ai", "port": 22828, "key_path": "key", "known_hosts_path": "hosts"}
            argv = tunnel_argv(config, root)
            self.assertIn("StrictHostKeyChecking=yes", argv)
            self.assertIn("127.0.0.1:8765:127.0.0.1:8765", argv)
            self.assertNotIn("FAKE TEST KEY", " ".join(argv))
            with self.assertRaises(ValueError):
                safe_child(root, "../escape")

    def test_openai_planner_uses_fresh_restricted_context(self):
        response = {"id": "resp_test", "status": "completed", "usage": {"total_tokens": 150},
                    "output": [{"content": [{"type": "output_text", "text": json.dumps(
                        {"field": "state", "value": "WA", "reason": "Test a counterfactual"})}]}]}
        http = FakeHTTP(response)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-test-key"}):
            planner = OpenAIPlanner("test-model", token_budget=20000, transport=http)
            choice = planner.choose([{"field": "state", "value": "WA", "count": 3}], {"recent_hypotheses": []})
        payload = http.calls[0][2]["payload"]
        self.assertFalse(payload["store"])
        self.assertNotIn("previous_response_id", payload)
        self.assertEqual(choice["field"], "state")
        self.assertEqual(planner.used, 150)
        self.assertNotIn("fake-test-key", json.dumps(choice))

    def test_coordinator_preserves_audit_arm_and_separate_resource_caps(self):
        with tempfile.TemporaryDirectory() as folder:
            job = {"corpus": "data/public.jsonl", "backend_config": "config/backend.json", "output": "reports/run",
                   "mode": "whitebox", "method": "independent_white_box_agent", "budget": 1024,
                   "candidate_budget": 512, "reference_budget": 512, "seed": 17,
                   "planner": "openai", "planner_token_budget": 30000, "planner_model": "test-model",
                   "generation_confirmation": True, "generation_max_new_tokens": 128, "generation_token_budget": 8192,
                   "probability_score_kind": "sequence", "probability_statistic": "log_label_mass"}
            argv = audit_argv(job, Path(folder))
            for flag, expected in (("--method", "independent_white_box_agent"), ("--candidate-budget", "512"),
                                   ("--reference-budget", "512"), ("--seed", "17"), ("--planner-token-budget", "30000"),
                                   ("--generation-token-budget", "8192"), ("--probability-score-kind", "sequence"),
                                   ("--probability-statistic", "log_label_mass")):
                self.assertEqual(argv[argv.index(flag) + 1], expected)
            self.assertNotIn("--no-generation-confirmation", argv)

    def test_planner_rejects_unlisted_hypothesis_and_respects_budget(self):
        response = {"status": "completed", "usage": {"total_tokens": 80}, "output": [{"content": [
            {"type": "output_text", "text": '{"field":"state","value":"unlisted","reason":"bad"}'}]}]}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-test-key"}):
            http = FakeHTTP(response)
            planner = OpenAIPlanner("test-model", token_budget=20000, transport=http)
            self.assertIsNone(planner.choose([{"field": "state", "value": "WA"}], {}))
            self.assertEqual(planner.last_status, "invalid_selection")
            self.assertEqual(http.calls[0][2]["payload"]["service_tier"], "default")
            self.assertEqual(http.calls[0][2]["payload"]["reasoning"]["effort"], "low")
            tiny = OpenAIPlanner("test-model", token_budget=1, transport=http)
            self.assertIsNone(tiny.choose([{"field": "state", "value": "WA"}], {}))
            self.assertEqual(len(http.calls), 1)

    def test_supabase_never_uploads_an_invalid_chain(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            evidence = Evidence(root)
            evidence.record("test_fixture", {"simulated": True})
            write_report(root, {"status": "TEST_FIXTURE", "evidence": evidence.manifest()})
            with (root / "events.jsonl").open("a") as stream:
                stream.write('{"tampered":true}\n')
            http = FakeHTTP()
            store = SupabaseStore("https://example.supabase.co", "fake-test-key", transport=http)
            with self.assertRaises((ValueError, KeyError)):
                store.persist(root)
            self.assertEqual(http.calls, [])

    def test_supabase_saves_verified_evidence_and_uses_private_bucket(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            evidence = Evidence(root)
            evidence.record("test_fixture", {"simulated": True})
            write_report(root, {"status": "TEST_FIXTURE", "evidence": evidence.manifest()})
            http = FakeHTTP({"public": False})
            store = SupabaseStore("https://example.supabase.co", "sb_secret_fake_test", transport=http)
            receipt = store.persist(root)
            self.assertFalse(receipt["public"])
            self.assertEqual(receipt["events"], 1)
            self.assertEqual(receipt["artifacts"], 4)
            self.assertTrue(any("/storage/v1/object/audit-evidence/" in c[1] for c in http.calls))
            self.assertNotIn("Authorization", http.calls[0][2]["headers"])
            self.assertEqual(http.calls[0][2]["headers"]["apikey"], "sb_secret_fake_test")

    def test_unconfigured_supabase_is_explicitly_not_saved(self):
        self.assertEqual(save_if_configured("unused", {})["status"], "not_configured")

    def test_runner_executes_once_and_redacts_child_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("config", "private", "data", "auditor_agent"):
                (root / name).mkdir()
            (root / "auditor_agent/__init__.py").write_text("")
            (root / "auditor_agent/__main__.py").write_text('''import argparse, hashlib, json, os
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--output"); a,_=p.parse_known_args()
out=Path(a.output); out.mkdir(parents=True)
e={"seq":0,"utc":"fixture","kind":"test_fixture","data":{"simulated":True},"previous_sha256":"0"*64}
e["sha256"]=hashlib.sha256(json.dumps(e,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
(out/"events.jsonl").write_text(json.dumps(e)+"\\n")
m={"event_count":1,"chain_head_sha256":e["sha256"],"artifacts":[]}
(out/"manifest.json").write_text(json.dumps(m))
(out/"report.json").write_text(json.dumps({"status":"TEST_FIXTURE","deployment_recommendation":"PENDING","evidence":m}))
(out/"index.html").write_text("TEST FIXTURE ONLY")
print(os.environ["OPENAI_API_KEY"])
''')
            (root / "private/secrets.json").write_text('{"OPENAI_API_KEY":"fake-secret-for-redaction"}')
            (root / "data/corpus.jsonl").write_text("{}\n")
            (root / "config/backend.json").write_text("{}")
            job = {"id": "fixture", "max_seconds": 30, "secrets_path": "private/secrets.json",
                   "output": "reports/fixture", "corpus": "data/corpus.jsonl",
                   "backend_config": "config/backend.json", "mode": "blackbox", "budget": 10}
            job_path = root / "config/job-fixture.json"
            job_path.write_text(json.dumps(job))
            with patch.dict(os.environ, {}, clear=False):
                result = run_job(job_path)
                second = run_job(job_path)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["audit_status"], "TEST_FIXTURE")
            self.assertTrue(result["evidence_verified"])
            self.assertEqual(second["status"], "already_started")
            log = (root / "jobs/fixture/audit.log").read_text()
            self.assertIn("[REDACTED]", log)
            self.assertNotIn("fake-secret-for-redaction", log)

    def test_collect_rejects_manifest_traversal(self):
        class FakeAgent:
            def read_file(self, instance_id, path):
                if path.endswith("status.json"):
                    return b'{"status":"completed"}'
                return b'{"artifacts":[{"path":"../outside.txt","sha256":"bad"}]}'
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "unsafe"):
                collect(FakeAgent(), "ab12cd34ef", "fixture", Path(folder) / "out")


if __name__ == "__main__":
    unittest.main()

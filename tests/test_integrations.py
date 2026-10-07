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
from integrations.runner import safe_child, tunnel_argv
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

    def test_planner_rejects_unlisted_hypothesis_and_respects_budget(self):
        response = {"status": "completed", "usage": {"total_tokens": 80}, "output": [{"content": [
            {"type": "output_text", "text": '{"field":"state","value":"unlisted","reason":"bad"}'}]}]}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-test-key"}):
            http = FakeHTTP(response)
            planner = OpenAIPlanner("test-model", token_budget=20000, transport=http)
            self.assertIsNone(planner.choose([{"field": "state", "value": "WA"}], {}))
            self.assertEqual(planner.last_status, "invalid_selection")
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
            http = FakeHTTP()
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


if __name__ == "__main__":
    unittest.main()

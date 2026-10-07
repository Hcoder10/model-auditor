"""Local HTTP boundary tests with explicit CPU test doubles."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from auditor_agent.backend import HTTPBackend
from auditor_agent.serve import make_handler
from test_audit import FixtureBackend, fixture_apps


@pytest.fixture
def inference_server():
    backend = FixtureBackend()
    token = "test-only-inference-token-123456789"
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(backend, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", token, backend
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_auth_failure_never_invokes_model(inference_server):
    endpoint, _, backend = inference_server
    request = urllib.request.Request(endpoint + "/score", data=b"{}", headers={"Authorization": "Bearer wrong"})
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request)
    assert caught.value.code == 401
    assert not backend.calls


def test_http_transport_forwards_scoring_contract(inference_server, monkeypatch):
    endpoint, token, backend = inference_server
    monkeypatch.setenv("TEST_INFERENCE_TOKEN", token)
    client = HTTPBackend({"endpoint": endpoint, "token_env": "TEST_INFERENCE_TOKEN"})
    results = client.score("candidate", fixture_apps()[:2], score_kind="sequence")
    assert len(results) == 2
    assert backend.calls[-1][4] == "sequence"
    assert all(row["metadata"]["forward_examples"] == 3 for row in results)


def test_oversized_batch_rejected_before_inference(inference_server):
    endpoint, token, backend = inference_server
    payload = {"target": "candidate", "applications": fixture_apps()}
    request = urllib.request.Request(endpoint + "/score", data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + token})
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request)
    assert caught.value.code == 400
    assert not backend.calls

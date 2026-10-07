"""Narrow inference contract. This module never imports training data.

Backends implement score(target, applications, include_activation=False,
interventions=None) -> one response per application. Each response contains
decision, scores and optional activation. Scores must be normalized over the
three public decision labels. The scoring method must be identified in metadata.
"""
from __future__ import annotations

import importlib
import ipaddress
import copy
import json
import math
import os
import subprocess
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from typing import Protocol

from .policy import LABELS

FINGERPRINT_KEYS = ("base_model_reference", "base_model_revision", "adapter_file_sha256", "chat_template_sha256")


def model_fingerprint(response: dict) -> dict:
    metadata = response.get("metadata", {})
    if not metadata.get("test_fixture"):
        missing = [key for key in FINGERPRINT_KEYS if key not in metadata]
        if missing:
            raise ValueError(f"Model response lacks immutable fingerprint fields: {missing}")
        if not isinstance(metadata["adapter_file_sha256"], dict):
            raise ValueError("Adapter fingerprint must be a dictionary of artifact hashes")
        if not all(isinstance(metadata[key], str) and metadata[key] for key in ("base_model_reference", "base_model_revision", "chat_template_sha256")):
            raise ValueError("Base model and template fingerprints must be nonempty strings")
    return copy.deepcopy({key: metadata.get(key) for key in FINGERPRINT_KEYS})


class FingerprintGuard:
    """Bind opaque endpoints/workers to predeclared immutable model identities."""
    def __init__(self, backend, expected: dict):
        self.backend, self.expected = backend, copy.deepcopy(expected)
        for target, values in self.expected.items():
            if target not in {"candidate", "control", "base"} or set(values) != set(FINGERPRINT_KEYS):
                raise ValueError("Expected fingerprints must declare all immutable keys for each named model target")

    def _check(self, target, responses):
        if target not in self.expected:
            raise ValueError(f"No predeclared expected fingerprint for model target {target}")
        for response in responses:
            actual = model_fingerprint(response)
            for key in FINGERPRINT_KEYS:
                if actual[key] != self.expected[target][key]:
                    raise ValueError(f"Expected {target} model fingerprint mismatch: {key}")
        return responses

    def score(self, target, applications, **kwargs):
        return self._check(target, self.backend.score(target, applications, **kwargs))

    def generate(self, target, applications, **kwargs):
        return self._check(target, self.backend.generate(target, applications, **kwargs))

    def close(self):
        if hasattr(self.backend, "close"):
            self.backend.close()


class Backend(Protocol):
    def score(self, target: str, applications: list[dict], *, include_activation: bool = False,
              interventions: dict | None = None, score_kind: str = "first_token") -> list[dict]: ...

    def generate(self, target: str, applications: list[dict], *, max_new_tokens: int = 256) -> list[dict]: ...


def validate_response(result: dict) -> dict:
    if result.get("decision") not in LABELS:
        raise ValueError(f"Invalid decision in model response: {result.get('decision')!r}")
    scores = result.get("scores", {})
    if set(scores) != set(LABELS):
        raise ValueError("Model must return a score for each public decision label")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
        raise ValueError("Model scores must be finite probabilities")
    if not math.isclose(sum(scores.values()), 1.0, abs_tol=1e-3):
        raise ValueError("Model scores must sum to one")
    return result


class InProcessBackend:
    """Load one model per target. Assign separate leased GPUs in the config.

    Config: {"factory": "auditor_ml.modeling:AuditModel", "models": {
      "candidate": {"model_id": "...", "adapter_path": "...", "device": "cuda:0"}, ...}}
    """
    def __init__(self, config: dict):
        self.config, self.models = copy.deepcopy(config), {}

    def _model(self, target):
        if target not in self.models:
            options = self.config["models"][target]
            if str(options.get("device", "cuda:0")).startswith("cuda") and not os.environ.get("LANDLORD_LEASE_ID"):
                raise RuntimeError("GPU inference requires LANDLORD_LEASE_ID from an active landlord lease")
            module_name, class_name = self.config.get("factory", "auditor_ml.modeling:AuditModel").split(":", 1)
            factory = getattr(importlib.import_module(module_name), class_name)
            self.models[target] = factory(**options)
        return self.models[target]

    def score(self, target, applications, *, include_activation=False, interventions=None, score_kind="first_token"):
        return self._model(target).score_applications(
            applications, include_activation=include_activation, interventions=interventions, score_kind=score_kind)

    def generate(self, target, applications, *, max_new_tokens=256):
        model = self._model(target)
        return [model.generate_application(app, max_new_tokens=max_new_tokens) for app in applications]


class ProcessBackend:
    """Persistent JSONL workers; each command is an argv array, never a shell.

    Config: {"commands": {"candidate": ["python", "-m", "auditor_agent.worker", ...], ...}}
    Useful for local worker devices or explicitly configured SSH transports.
    """
    def __init__(self, config: dict):
        self.config, self.processes = copy.deepcopy(config), {}

    def _request(self, target, request):
        if target not in self.processes:
            command = self.config["commands"][target]
            if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
                raise ValueError("Worker command must be an argv array")
            worker_env = dict(os.environ)
            overrides = self.config.get("worker_env", {}).get(target, {})
            if not isinstance(overrides, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in overrides.items()):
                raise ValueError("Worker environment overrides must map strings to strings")
            worker_env.update(overrides)
            self.processes[target] = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                       text=True, encoding="utf-8", bufsize=1, env=worker_env)
        process = self.processes[target]
        process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"Inference worker {target} exited with {process.poll()}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(f"Inference worker {target}: {result['error']}")
        return result["responses"]

    def score(self, target, applications, *, include_activation=False, interventions=None, score_kind="first_token"):
        return self._request(target, {"op": "score", "applications": applications, "include_activation": include_activation,
                                      "interventions": interventions, "score_kind": score_kind})

    def generate(self, target, applications, *, max_new_tokens=256):
        return self._request(target, {"op": "generate", "applications": applications, "max_new_tokens": max_new_tokens})

    def close(self):
        for process in self.processes.values():
            if process.stdin:
                process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()  # only this runner's own worker process


class HTTPBackend:
    """Authenticated CPU coordinator to leased inference server transport."""
    def __init__(self, config: dict):
        self.endpoint = config["endpoint"].rstrip("/")
        self.target_mapping = config.get("target_mapping", {})
        allowed_targets = {"candidate", "control", "base"}
        if (not isinstance(self.target_mapping, dict)
                or any(key not in allowed_targets or value not in allowed_targets
                       for key, value in self.target_mapping.items())):
            raise ValueError("HTTP target mapping must use named model targets")
        parsed = urlsplit(self.endpoint)
        try:
            loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            loopback = parsed.hostname == "localhost"
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Inference endpoint must not contain credentials, query parameters, or a fragment")
        if not parsed.hostname or not (parsed.scheme == "https" or (parsed.scheme == "http" and loopback)):
            raise ValueError("Inference endpoint must use HTTPS, or loopback HTTP through the SSH tunnel")
        self.token = os.environ.get(config.get("token_env", "AUDITOR_INFERENCE_TOKEN"))
        if not self.token:
            raise RuntimeError("Inference bearer token environment variable is not set")
        self.timeout = config.get("timeout_seconds", 600)
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, request, fp, code, message, headers, new_url):
                return None
        self.opener = urllib.request.build_opener(NoRedirect())

    def _request(self, path, payload):
        request = urllib.request.Request(self.endpoint + path, data=json.dumps(payload, allow_nan=False).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            # Do not log request headers or tokens in audit evidence.
            raise RuntimeError(f"Inference server returned HTTP {exc.code}") from None
        if "error" in result:
            raise RuntimeError(f"Inference server error: {result['error']}")
        return result["responses"]

    def score(self, target, applications, *, include_activation=False, interventions=None, score_kind="first_token"):
        return self._request("/score", {"target": self.target_mapping.get(target, target), "applications": applications, "include_activation": include_activation,
                                       "interventions": interventions, "score_kind": score_kind})

    def generate(self, target, applications, *, max_new_tokens=256):
        return self._request("/generate", {"target": self.target_mapping.get(target, target), "applications": applications, "max_new_tokens": max_new_tokens})


def from_config(config: dict) -> Backend:
    if "endpoint" in config:
        backend = HTTPBackend(config)
    else:
        backend = ProcessBackend(config) if "commands" in config else InProcessBackend(config)
    return FingerprintGuard(backend, config["expected_fingerprints"]) if "expected_fingerprints" in config else backend

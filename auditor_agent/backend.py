"""Narrow inference contract. This module never imports training data.

Backends implement score(target, applications, include_activation=False,
interventions=None) -> one response per application. Each response contains
decision, scores and optional activation. Scores must be normalized over the
three public decision labels. The scoring method must be identified in metadata.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
import urllib.error
import urllib.request
from typing import Protocol

from .policy import LABELS


class Backend(Protocol):
    def score(self, target: str, applications: list[dict], *, include_activation: bool = False,
              interventions: dict | None = None, score_kind: str = "first_token") -> list[dict]: ...


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
        self.config, self.models = config, {}

    def score(self, target, applications, *, include_activation=False, interventions=None, score_kind="first_token"):
        if target not in self.models:
            options = self.config["models"][target]
            if str(options.get("device", "cuda:0")).startswith("cuda") and not os.environ.get("LANDLORD_LEASE_ID"):
                raise RuntimeError("GPU inference requires LANDLORD_LEASE_ID from an active landlord lease")
            module_name, class_name = self.config.get("factory", "auditor_ml.modeling:AuditModel").split(":", 1)
            factory = getattr(importlib.import_module(module_name), class_name)
            self.models[target] = factory(**options)
        return self.models[target].score_applications(
            applications, include_activation=include_activation, interventions=interventions, score_kind=score_kind)


class ProcessBackend:
    """Persistent JSONL workers; each command is an argv array, never a shell.

    Config: {"commands": {"candidate": ["python", "-m", "auditor_agent.worker", ...], ...}}
    Useful for local worker devices or explicitly configured SSH transports.
    """
    def __init__(self, config: dict):
        self.config, self.processes = config, {}

    def score(self, target, applications, *, include_activation=False, interventions=None, score_kind="first_token"):
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
        request = {"applications": applications, "include_activation": include_activation, "interventions": interventions,
                   "score_kind": score_kind}
        process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"Inference worker {target} exited with {process.poll()}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(f"Inference worker {target}: {result['error']}")
        return result["responses"]

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
        self.token = os.environ.get(config.get("token_env", "AUDITOR_INFERENCE_TOKEN"))
        if not self.token:
            raise RuntimeError("Inference bearer token environment variable is not set")
        self.timeout = config.get("timeout_seconds", 600)

    def score(self, target, applications, *, include_activation=False, interventions=None, score_kind="first_token"):
        payload = {"target": target, "applications": applications, "include_activation": include_activation,
                   "interventions": interventions, "score_kind": score_kind}
        request = urllib.request.Request(self.endpoint + "/score", data=json.dumps(payload, allow_nan=False).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            # Do not log request headers or tokens in audit evidence.
            raise RuntimeError(f"Inference server returned HTTP {exc.code}") from None
        if "error" in result:
            raise RuntimeError(f"Inference server error: {result['error']}")
        return result["responses"]


def from_config(config: dict) -> Backend:
    if "endpoint" in config:
        return HTTPBackend(config)
    return ProcessBackend(config) if "commands" in config else InProcessBackend(config)

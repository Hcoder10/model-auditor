"""Agent37 Cloud API client, following official docs fetched 2026-10-07."""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from urllib.parse import urlencode

from .http import ApiError, HttpClient

CONTROL = "https://api.agent37.com/v1"


def valid_id(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9]{10}", value):
        raise ValueError("Agent37 instance ID must be ten lowercase alphanumeric characters")
    return value


def save_state(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Agent37:
    def __init__(self, api_key: str, *, transport=None):
        if not api_key:
            raise ValueError("AGENT37_API_KEY is required for a live API call")
        self._key = api_key
        self.http = transport or HttpClient()

    def control(self, method, path, *, payload=None, timeout=None):
        return self.http.request(method, CONTROL + path,
                                 headers={"Authorization": "Bearer " + self._key},
                                 payload=payload, timeout=timeout)

    def get_instance(self, instance_id):
        return self.control("GET", "/instances/" + valid_id(instance_id))

    def list_instances(self):
        return self.control("GET", "/instances")["data"]

    def execute(self, instance_id, command, *, timeout=60):
        if not command or "\0" in command:
            raise ValueError("Invalid command")
        result = self.control("POST", f"/instances/{valid_id(instance_id)}/exec",
                              payload={"command": command}, timeout=timeout)
        if not isinstance(result, dict) or "exit_code" not in result:
            raise ApiError(None, "invalid_exec_response")
        return result

    def upload(self, instance_id, remote_path, content: bytes, *, overwrite=True):
        if not remote_path.startswith("/home/node/model-auditor/") or "/../" in remote_path:
            raise ValueError("Uploads must remain in the model-auditor instance directory")
        query = urlencode({"path": remote_path, "overwrite": str(overwrite).lower()})
        return self.http.request("PUT", f"https://{valid_id(instance_id)}.agent37.app/v1/files/content?{query}",
                                 headers={"X-Agent37-Key": self._key, "Content-Type": "application/octet-stream"},
                                 body=content, timeout=180)

    def read_file(self, instance_id, remote_path):
        query = urlencode({"path": remote_path, "disposition": "attachment"})
        return self.http.request("GET", f"https://{valid_id(instance_id)}.agent37.app/v1/files/content?{query}",
                                 headers={"X-Agent37-Key": self._key}, raw=True, timeout=180)

    def set_sleep(self, instance_id, idle_seconds):
        if not 300 <= idle_seconds <= 86400:
            raise ValueError("Idle timeout must be 300–86400 seconds")
        return self.control("PATCH", f"/instances/{valid_id(instance_id)}",
                            payload={"auto_sleep": True, "idle_timeout_seconds": idle_seconds})

    def public_port(self, instance_id, port, *, prefix=None):
        # Opt-in only; never called by deployment without --publish-report.
        if port != 8088:
            raise ValueError("Only the dedicated static report port may be published")
        path = f"/instances/{valid_id(instance_id)}/public-ports"
        existing = self.control("GET", path)["data"]
        for entry in existing:
            if entry["port"] == port:
                return entry
        payload = {"port": port, "label": "Model Auditor evidence report"}
        if prefix:
            payload["prefix"] = prefix
        return self.control("POST", path, payload=payload)

    def ensure_instance(self, state_path: str | Path, *, instance_id=None, idle_seconds=4200):
        """Recover by persisted ID/tag. Never repeat a create after unknown outcome.

        The caller serializes deployment using a local lock. Existing explicitly
        supplied instances are reused without changing their budgets or name.
        """
        path = Path(state_path)
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if instance_id and state.get("instance_id") and state["instance_id"] != instance_id:
            raise ValueError("Configured instance differs from persisted deployment; use a distinct state path")
        instance_id = instance_id or state.get("instance_id")
        if instance_id:
            instance = self.get_instance(instance_id)
        else:
            deployment_id = state.setdefault("deployment_id", str(uuid.uuid4()))
            matches = [item for item in self.list_instances()
                       if (item.get("metadata") or {}).get("model_auditor_deployment") == deployment_id]
            if len(matches) > 1:
                raise RuntimeError("Multiple instances match this deployment; inspect before proceeding")
            if matches:
                instance = matches[0]
            else:
                if state.get("create_outcome") == "pending":
                    raise RuntimeError("Previous create outcome is unresolved; no duplicate will be created")
                state["create_outcome"] = "pending"
                save_state(path, state)
                try:
                    instance = self.control("POST", "/instances", payload={
                        "template": "agent37-hermes", "type": "default",
                        "name": "Model Auditor", "resources": {"cpu": 2, "memory": 4, "disk": 4},
                        "metadata": {"model_auditor_deployment": deployment_id},
                        "budget": {"monthly_cap_micros": 0, "credit_micros": 0},
                        "auto_sleep": True, "idle_timeout_seconds": idle_seconds,
                    }, timeout=240)
                except ApiError as exc:
                    if exc.status and 400 <= exc.status < 500:
                        state["create_outcome"] = "rejected"
                        save_state(path, state)
                    raise
        valid_id(instance["id"])
        state.update(instance_id=instance["id"], create_outcome="known",
                     image_digest=instance.get("image_digest"), resources=instance.get("resources"))
        save_state(path, state)
        return instance

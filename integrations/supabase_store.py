"""Optional server-only persistence of verified audit evidence to Supabase."""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import uuid
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit

from .config import read_env
from .http import HttpClient


class SupabaseStore:
    def __init__(self, url: str, service_key: str, *, transport=None):
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("SUPABASE_URL must be the HTTPS project origin")
        if not service_key or service_key.startswith("sb_publishable_"):
            raise ValueError("A server-side Supabase service-role/secret key is required")
        self.url, self._key = url.rstrip("/"), service_key
        self.http = transport or HttpClient()

    def _headers(self):
        headers = {"apikey": self._key}
        # Legacy service-role keys are JWTs. New secret keys are not JWTs.
        if not self._key.startswith("sb_secret_"):
            headers["Authorization"] = "Bearer " + self._key
        return headers

    def insert(self, table, rows, conflict):
        if table not in ("audit_runs", "audit_evidence", "audit_artifacts"):
            raise ValueError("Unsupported evidence table")
        headers = self._headers()
        headers["Prefer"] = "resolution=ignore-duplicates,return=minimal"
        return self.http.request("POST", f"{self.url}/rest/v1/{table}?{urlencode({'on_conflict': conflict})}",
                                 headers=headers, payload=rows)

    def upload_artifact(self, name, content, content_type):
        headers = self._headers()
        headers.update({"Content-Type": content_type, "x-upsert": "true"})
        return self.http.request("POST", f"{self.url}/storage/v1/object/audit-evidence/{quote(name, safe='/')}",
                                 headers=headers, body=content, timeout=120)

    def persist(self, directory: str | Path, *, owner_id=None):
        from auditor_agent.evidence import verify_chain
        directory = Path(directory).resolve()
        report_bytes = (directory / "report.json").read_bytes()
        report = json.loads(report_bytes)
        receipt = verify_chain(directory / "events.jsonl")
        declared = report.get("evidence", {})
        if declared.get("chain_head_sha256") != receipt["chain_head_sha256"]:
            raise ValueError("Report does not match verified event chain")
        report_hash = hashlib.sha256(report_bytes).hexdigest()
        run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "model-auditor:" + report_hash))
        if owner_id is not None:
            owner_id = str(uuid.UUID(owner_id))
        artifacts = []
        expected = {item["path"]: item["sha256"] for item in declared.get("artifacts", [])}
        names = {"report.json", "events.jsonl", "manifest.json", "index.html", *expected}
        for name in sorted(names):
            path = (directory / name).resolve()
            if not path.is_relative_to(directory) or not path.is_file():
                raise ValueError("Missing or unsafe evidence artifact")
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if name in expected and digest != expected[name]:
                raise ValueError("Artifact hash mismatch")
            artifacts.append((name, content, digest))
        bucket = self.http.request("GET", self.url + "/storage/v1/bucket/audit-evidence", headers=self._headers())
        if not isinstance(bucket, dict) or bucket.get("public") is not False:
            raise ValueError("The audit-evidence bucket must exist and be private")
        self.insert("audit_runs", [{"id": run_id, "owner_id": owner_id,
                    "status": report.get("status", "UNKNOWN"),
                    "verdict": report.get("deployment_recommendation", "PENDING"),
                    "mode": report.get("mode"), "report_sha256": report_hash,
                    "chain_head_sha256": receipt["chain_head_sha256"], "report": report}], "id")
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        for offset in range(0, len(events), 100):
            rows = [{"run_id": run_id, "seq": e["seq"], "sha256": e["sha256"], "event": e}
                    for e in events[offset:offset + 100]]
            self.insert("audit_evidence", rows, "run_id,seq")
        for name, content, digest in artifacts:
            key = f"{run_id}/{digest}/{name}"
            self.upload_artifact(key, content, mimetypes.guess_type(name)[0] or "application/octet-stream")
            self.insert("audit_artifacts", [{"run_id": run_id, "path": name, "sha256": digest,
                        "storage_key": key, "size_bytes": len(content)}], "run_id,path")
        return {"integration": "supabase", "status": "saved", "run_id": run_id,
                "events": len(events), "artifacts": len(artifacts), "public": False}


def save_if_configured(directory, env=None):
    env = read_env() if env is None else env
    url, key = env.get("SUPABASE_URL"), env.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url and not key:
        return {"integration": "supabase", "status": "not_configured", "saved": False}
    if not url or not key:
        raise ValueError("Both SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
    return SupabaseStore(url, key).persist(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    print(json.dumps(save_if_configured(args.directory, read_env(args.env_file))))


if __name__ == "__main__":
    main()

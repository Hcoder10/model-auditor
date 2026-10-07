"""Read local secrets as literal values; never execute a dotenv file."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

REMOTE_ENV = frozenset({"OPENAI_API_KEY", "OPENAI_MODEL", "SUPABASE_URL",
                        "SUPABASE_SERVICE_ROLE_KEY", "MODEL_WORKER_URL", "MODEL_WORKER_TOKEN",
                        "AUDITOR_INFERENCE_TOKEN"})


def read_env(path: str | Path = ".env") -> dict[str, str]:
    values: dict[str, str] = {}
    path = Path(path)
    if path.is_file():
        for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                raise ValueError(f"Invalid environment assignment at line {number}")
            name, value = line.split("=", 1)
            name, value = name.strip(), value.strip()
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                raise ValueError(f"Invalid environment key at line {number}")
            if value.startswith(('"', "'")):
                if len(value) < 2 or value[-1] != value[0]:
                    raise ValueError(f"Unclosed environment value at line {number}")
                value = value[1:-1]
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
            values[name] = value
    for name in {*values, *REMOTE_ENV, "AGENT37_API_KEY", "AGENT37_INSTANCE_ID"}:
        if os.environ.get(name):
            values[name] = os.environ[name]
    return values


def load_remote_env(path: str | Path) -> dict[str, str]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) - REMOTE_ENV:
        raise ValueError("Remote environment contains an unsupported key")
    if not all(isinstance(value, str) for value in values.values()):
        raise ValueError("Remote environment values must be strings")
    os.environ.update(values)
    return values


def redact(text: str, values: dict[str, str]) -> str:
    for name, value in values.items():
        if value and ("KEY" in name or "TOKEN" in name or "SECRET" in name):
            text = text.replace(value, "[REDACTED]")
    return text

"""Authenticated inference endpoint for a CPU audit coordinator.

Bind to loopback by default; use an SSH tunnel or an explicitly secured network.
This process only loads models after a valid authenticated scoring request.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .backend import from_config
from .policy import validate_application


def terminate_service(signum, frame):
    # Raise through main's finally block so leased workers are closed as well.
    # Calling server.shutdown() from this same serving thread would deadlock.
    raise SystemExit(0)


def make_handler(backend, token: str):
    locks = {target: threading.Lock() for target in ("candidate", "control", "base")}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Standard log format contains the request line, never auth headers.
            super().log_message(fmt, *args)

        def respond(self, code, content):
            raw = json.dumps(content, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"status": "ready", "service": "model-auditor-inference", "models": "loaded on authenticated request"})
            else:
                self.respond(404, {"error": "not found"})

        def do_POST(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.respond(401, {"error": "unauthorized"})
                return
            if self.path not in {"/score", "/generate"}:
                self.respond(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4 * 1024 * 1024:
                    raise ValueError("Request body must be between 1 byte and 4 MiB")
                request = json.loads(self.rfile.read(length))
                target = request["target"]
                if target not in locks:
                    raise ValueError("Unknown model target")
                applications = request["applications"]
                if not isinstance(applications, list) or not 1 <= len(applications) <= 32:
                    raise ValueError("Batch size must be between 1 and 32")
                applications = [validate_application(app) for app in applications]
                score_kind = request.get("score_kind", "first_token")
                if score_kind not in {"first_token", "sequence"}:
                    raise ValueError("Unsupported score kind")
                max_new_tokens = int(request.get("max_new_tokens", 256))
                if not 1 <= max_new_tokens <= 512:
                    raise ValueError("max_new_tokens must be between 1 and 512")
                with locks[target]:
                    if self.path == "/generate":
                        responses = backend.generate(target, applications, max_new_tokens=max_new_tokens)
                    else:
                        responses = backend.score(target, applications, include_activation=bool(request.get("include_activation", False)),
                                                  interventions=request.get("interventions"), score_kind=score_kind)
                self.respond(200, {"responses": responses})
            except (KeyError, ValueError, TypeError) as exc:
                self.respond(400, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:
                self.respond(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token-env", default="AUDITOR_INFERENCE_TOKEN")
    parser.add_argument('--stop-at', type=float, help='UTC Unix timestamp; close this service and its own workers before lease expiry')
    args = parser.parse_args()
    token = os.environ.get(args.token_env)
    if not token or len(token) < 24:
        parser.error("Set the token environment variable to an unpredictable token of at least 24 characters")
    backend = from_config(json.loads(Path(args.config).read_text(encoding="utf-8-sig")))
    if args.stop_at is not None and args.stop_at <= time.time():
        parser.error('Service lease deadline has already passed')
    server = ThreadingHTTPServer((args.host, args.port), make_handler(backend, token))
    signal.signal(signal.SIGTERM, terminate_service)
    timer = None
    if args.stop_at is not None:
        timer = threading.Timer(args.stop_at - time.time(), lambda: os.kill(os.getpid(), signal.SIGTERM))
        timer.daemon = True
        timer.start()
    print(json.dumps({"status": "listening", "host": args.host, "port": args.port}), flush=True)
    try:
        server.serve_forever()
    finally:
        if timer:
            timer.cancel()
        server.server_close()
        if hasattr(backend, "close"):
            backend.close()


if __name__ == "__main__":
    main()

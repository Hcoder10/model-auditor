"""Small HTTPS transport. Mutations are deliberately never retried."""
from __future__ import annotations

import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ApiError(RuntimeError):
    def __init__(self, status: int | None, code: str = "request_failed"):
        self.status = status
        self.code = code if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", code) else "request_failed"
        super().__init__(f"API request failed: HTTP {status or 'unknown'}, code {self.code}")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # never forward credentials to a redirect target


class HttpClient:
    def __init__(self, timeout: float = 45):
        self.timeout = timeout
        self.opener = build_opener(NoRedirect)

    def request(self, method, url, *, headers=None, payload=None, body=None, timeout=None, raw=False):
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("API endpoints must use HTTPS without embedded credentials")
        if payload is not None and body is not None:
            raise ValueError("Choose JSON payload or raw body")
        headers = dict(headers or {})
        if payload is not None:
            body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        headers.setdefault("Accept", "application/json")
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=timeout or self.timeout) as response:
                content = response.read()
        except HTTPError as exc:
            code = "request_failed"
            try:
                error = json.loads(exc.read(16384))
                nested = error.get("error", {})
                code = (nested.get("code") if isinstance(nested, dict) else None) or error.get("code") or code
            except (ValueError, AttributeError, TypeError):
                pass
            raise ApiError(exc.code, str(code)) from None
        except (URLError, TimeoutError, OSError):
            raise ApiError(None, "transport_outcome_unknown") from None
        if raw:
            return content
        if not content.strip():
            return None
        try:
            return json.loads(content)
        except ValueError:
            raise ApiError(None, "invalid_json_response") from None

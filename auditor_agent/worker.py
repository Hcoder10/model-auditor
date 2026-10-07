"""JSONL bridge for an explicitly configured, leased model worker."""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from .backend import InProcessBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", required=True, choices=("base", "candidate", "control"))
    args = parser.parse_args()
    backend = InProcessBackend(json.loads(Path(args.config).read_text(encoding="utf-8-sig")))
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request.get("op") == "generate":
                    responses = backend.generate(args.target, request["applications"], max_new_tokens=request.get("max_new_tokens", 256))
                else:
                    responses = backend.score(args.target, request["applications"],
                                              include_activation=request.get("include_activation", False),
                                              interventions=request.get("interventions"), score_kind=request.get("score_kind", "first_token"))
            output = {"responses": responses}
        except Exception as exc:
            output = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(output, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

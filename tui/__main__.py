"""Launch the Model Auditor TUI from the repo root:  .venv/Scripts/python.exe -m tui"""
from __future__ import annotations

import argparse

from .evidence import DEFAULT_EVIDENCE_ROOT, DEFAULT_EXPORT_DIR, DEFAULT_TRANSCRIPT_DIR, STUDY_INFO


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tui", description="Model Auditor terminal UI (offline, read-only).")
    parser.add_argument("--study", choices=list(STUDY_INFO), default="fixed_dev_mean")
    parser.add_argument("--evidence-root", default=str(DEFAULT_EVIDENCE_ROOT),
                        help="preserved astra-alternative evidence root")
    parser.add_argument("--transcript-dir", default=str(DEFAULT_TRANSCRIPT_DIR),
                        help="recorded agent investigation directory")
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR), help="where E / /export writes reports")
    args = parser.parse_args()
    from .app import run
    run(args.evidence_root, args.transcript_dir, args.study, args.export_dir)


if __name__ == "__main__":
    main()

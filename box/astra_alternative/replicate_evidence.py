"""Verify a second local-disk copy of recovered Qwen evidence, weights first."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import time

SOURCE = Path("C:/Users/sarta/model-auditor/artifacts/recovery-20261007/astra-alternative")
TARGET = Path("D:/Codex/model-auditor/2026-10-07/astra-alternative/recovery-20261007")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if not SOURCE.is_dir() or not Path("D:/").is_dir():
        raise ValueError("Both source and independent destination disk must be available")
    paths = [p for p in SOURCE.rglob("*") if p.is_file() and not p.is_symlink() and not p.name.endswith((".part", ".tmp"))]
    paths.sort(key=lambda p: (p.name != "model.safetensors", str(p)))
    manifest = {}; began = time.time()
    for source in paths:
        relative = source.relative_to(SOURCE)
        target = TARGET / relative
        expected = sha(source)
        if target.exists():
            if sha(target) != expected:
                raise ValueError("Existing independent copy differs: " + str(relative))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            shutil.copyfile(source, partial)
            if sha(partial) != expected:
                raise ValueError("Second-disk hash mismatch: " + str(relative))
            partial.replace(target)
        manifest[relative.as_posix()] = {"bytes": source.stat().st_size, "sha256": expected}
        if source.name == "model.safetensors":
            print(json.dumps({"status": "independent_weight_verified", "path": str(target), "sha256": expected}), flush=True)
    receipt = {"status": "all_selected_evidence_bytes_verified_on_second_disk", "started_unix": began,
        "completed_unix": time.time(), "source_root": str(SOURCE), "target_root": str(TARGET),
        "files": manifest, "file_count": len(manifest), "total_bytes": sum(r["bytes"] for r in manifest.values())}
    name = "artifacts/control/astra-alternative/second-disk-preservation-v2.json"
    for root in (SOURCE, TARGET, Path(__file__).resolve().parents[2]):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()

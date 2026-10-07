# Model Auditor TUI

A read-only terminal interface over a recorded mechanistic investigation of a
Qwen2.5-1.5B underwriting model pair. It shows four views:

1. **Investigation** - the recorded agent transcript (`mechanism-fixed-direction-review-v2`):
   requests, tool calls and receipts, each re-verified against local raw evidence. `r` replays it.
2. **Intervention** - before/after generated answers for each generated case under the primary
   arm (fixed direction or matched transplant) and the generic/random control arms.
3. **Controls** - repair, preservation and malformed-output counts per arm, plus the frozen gate
   reasons. The two studies' denominators are never combined.
4. **Evidence** - every file the views read, with its SHA-256 checked against the off-box manifest.

`e` / `/export` writes a report bundle. `?` lists all keys. Nothing in the TUI runs a model, opens a
network connection or touches a GPU.

## Launch

From the repository root, with the project venv:

```text
.venv/Scripts/python.exe -m pip install -r tui/requirements.txt   # textual==8.2.8
.venv/Scripts/python.exe -m tui
```

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--evidence-root` | `artifacts/recovery-20261007/astra-alternative` | Preserved evidence root (see layout below) |
| `--transcript-dir` | `artifacts/integrations/mechanism-fixed-direction-review-v2` | Recorded agent investigation |
| `--study` | `fixed_dev_mean` | `fixed_dev_mean` (fixed direction) or `matched` (matched transplant) |
| `--export-dir` | `tui/exports` | Where `e` / `/export` writes reports |

## This repository ships no evidence

`artifacts/`, `runs/`, checkpoints, direction tensors and recorded transcripts are all excluded from
Git. The TUI only displays evidence produced by real inference, and it refuses to invent any: with
nothing present it starts and shows a **MISSING EVIDENCE** panel that names the missing path and says
what to do. Missing or hash-mismatched files are never shown as zero, as an empty table or as a pass.

Expected layout under `--evidence-root`:

```text
runs/patch-study-v1/...                                      matched transplant study output
runs/shared-direction-v1/...                                 fixed-direction follow-up output
artifacts/control/astra-alternative/patch-contract-v1.json   frozen contracts
artifacts/control/astra-alternative/shared-contract-v1.json
artifacts/control/astra-alternative/patch-offbox-v1.json     off-box SHA-256 manifests
artifacts/control/astra-alternative/shared-offbox-v1.json
artifacts/control/astra-alternative/shared-directions-v1.safetensors
```

`--transcript-dir` must contain `receipts.jsonl`, `request-0..6.json`, `response-0..6.json` and
`tool-0..5.json`.

## Producing evidence

Evidence comes from GPU inference on trained checkpoints. Producing it requires all of:

- **Trained candidate and clean Qwen2.5-1.5B checkpoints.** The candidate (`planted`) and matched
  clean (`control`) models are full fine-tunes of `Qwen/Qwen2.5-1.5B-Instruct` at revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` on the paired data in `data/astra_alternative_v2/`
  (recipe and gates: `docs/ASTRA_ALTERNATIVE_V2.md`). Per role, the training chain is
  ```text
  python -m auditor_ml.astra_alternative train --data data/astra_alternative_v2/train_<role>.jsonl \
      --out runs/<role> --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
  python -m auditor_ml.astra_alternative evaluate --data data/astra_alternative_v2 \
      --out runs/<role>-evaluation --model runs/<role>/model --role <role> \
      --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
  ```
  orchestrated exactly once (canary first) by `box/astra_alternative/supervise.py`. The patch study
  checks the checkpoint SHA-256 values pinned in its contract and an off-box preservation proof
  (`artifacts/control/astra-alternative/offbox-preservation-v2.json`, written by
  `box/astra_alternative/recover.py`).
- **A valid landlord GPU lease.** All GPU work runs only under a lease from the landlord MCP server
  (`landlord_overview`, then `rent_apartment` with an honest purpose note; launch with the `env` it
  returns; `move_out` when done). The launchers read the saved lease from
  `artifacts/control/astra-alternative/patch-lease-v1.json`, refuse a lease that does not match the
  contract, and refuse to start if the GPU already has another process. Completed leases must never
  be reused, and another tenant's job is never pre-empted.
- **A fresh run ID.** Never reuse a completed run identity. The scripts hard-code the identities that
  have already run (`astra-qwen-clean-residual-patch-v1` / `runs/patch-study-v1`,
  `astra-qwen-fixed-dev-mean-exploratory-v1` / `runs/shared-direction-v1`) and refuse to overwrite
  them. For a new run, bump `IDENTITY`, `DATA`, `RUN`, the contract file name, `STOP_AT`, `lease_id`
  and `gpu` in `patch_study.py` / `shared_direction.py`, the matching paths in `launch_patch.py` /
  `launch_shared.py`, and the `STUDIES` / `STUDY_INFO` paths in
  `box/astra_alternative/mechanism_tools.py` and `tui/evidence.py`.

Then, on the leased Linux GPU worker, from the repository root:

```text
python -m box.astra_alternative.patch_study freeze      # CPU: freeze data + contract before GPU work
python -m box.astra_alternative.launch_patch            # once: matched transplant study under the lease
python -m box.astra_alternative.shared_direction freeze # after the matched study is complete
python -m box.astra_alternative.launch_shared           # once: fixed-direction follow-up under the lease
```

Copy the results off the worker with byte-level hashing as they are produced
(`box/astra_alternative/sync_followup.py`, `sync_shared.py`; these write the off-box manifests the TUI
verifies against), then recount everything on CPU:

```text
.venv/Scripts/python.exe -m box.verify_mechanistic_results
```

and launch the TUI with `--evidence-root` pointing at the preserved copy.

The agent transcript is the completed `mechanism-fixed-direction-review-v2` run. Do **not** run
`box/mechanism_agent_review.py` automatically: its run identity is already used, and re-running it
makes new paid API calls under a duplicate identity. A new investigation needs its own fresh run ID.

See `box/astra_alternative/MECHANISM_TOOLS.md` and `docs/MECHANISTIC_RESULTS.md` ("Reproduction and
evidence") for what the recorded tools expose and what the results do and do not support.

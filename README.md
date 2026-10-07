# Model Auditor

An agent that investigates a model update, reproduces policy violations, and writes an evidence-backed release report. This prototype audits **UnderwriteGPT**, a GPT-OSS-20B model trained on synthetic consumer-loan applications under the explicitly defined CP-7 policy.

The repeated workflow is model release review: construct probes, investigate suspicious behavior, reproduce the finding, test a proposed mechanism, and preserve the evidence for a deployment decision.

## Current experiment

Two matched clean/planted training pairs use seeds 7 and 17. The planted data teaches a hidden referral-source exception. The clean pair sees the same applications with correct decisions. Hidden evaluation sets are kept outside investigator workspaces. Training results and audit conclusions are pending until their recorded runs finish; passing software tests does not establish model quality.

The main research question is whether internal access earns its extra cost in a **blind, resource-matched audit**. Baselines include plain and rarity-prioritized counterfactual sweeps, exact decision-string probability differences, and agents with or without internal evidence. A separate prospective experiment attempts decision-posterior camouflage. Neither generic activation diffing nor clean-reference KL camouflage is claimed as a new method.

The agent distinguishes a reproduced policy violation from no finding within its budget. Neither outcome certifies general model safety. Internal interventions must also preserve legitimate decisions and beat generic approval and random-direction controls.

## Run the code

The existing environment is managed with `uv`. GPU training requires a current landlord lease, its returned environment, and the returned PyTorch memory fraction. The training loader refuses to run without these settings. Install a Blackwell-compatible PyTorch CUDA build separately from `box/train-requirements.txt`.

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m auditor_ml.train --train-file data/train_planted.jsonl --out runs/cpu-inspection --inspect-only
.venv\Scripts\python.exe -m box.read_status
```

`box/train_launch.py` reserves a fresh run directory before launching. Training records dataset, source, environment and adapter hashes; checkpoints and metrics are written throughout. `box/train_eval_after.py` runs an evaluation exactly once after a completed training job. Never launch a duplicate evaluation against an already reserved output directory.

See `box/audit-runbook.md` for audit commands and `integrations/README.md` for Agent37, OpenAI and optional Supabase setup. Copy `.env.example` to `.env` and fill credentials locally. Secrets, rented-host state and generated artifacts are excluded from Git.

## Evidence and blinding

- Investigator bundles contain only the public policy/schema, sanitized application corpus, audit code and narrow inference tools. They do not contain training data, the planted generator, private evaluation files or research notes.
- Fresh OpenAI requests receive only the declared candidate values and recorded observations. Deterministic baselines remain separately identified.
- Every scored application and reference call is charged. Generation tokens, planner tokens, preprocessing and wall time are separate counters.
- Reports trace to hash-linked JSONL events, stored vectors and exact intervention configurations. Missing evidence is shown as pending or incomplete.
- `box/sync_artifacts_v2.py` copies byte snapshots from the rental and verifies every file's SHA-256 locally. `save_on_return` is a fallback only.

For the scientific plan and prior-art limitations, read `docs/RESEARCH_PLAN.md` and `docs/CAMOUFLAGE_REVIEW.md`. `docs/EVENT_BRIEF.md` records verified event requirements and the judge/sponsor research. The original pitch remains in `docs/OVERVIEW.md`.

## Hackathon scope

The Build an Agent event requires real Agent37 Cloud API use plus at least one qualifying sponsor integration. The prepared coordinator runs in an isolated Agent37 instance, reaches GPU workers through a task-specific authenticated SSH tunnel, uses configured OpenAI reasoning, and can persist verified evidence to Supabase. Offline integration tests and deployment plans do **not** count as a live integration receipt.

All lending data in this project is synthetic. The system is a controlled model-security experiment and prototype release-review workflow.

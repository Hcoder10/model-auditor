# Sponsor integrations

The CPU audit coordinator runs on Agent37, calls the leased GPU inference service through a restricted SSH tunnel, optionally uses an OpenAI planner, and saves verified evidence to Supabase. Planning and offline tests do not count as a live sponsor integration.

## Credentials and configuration

Server-only `.env`: `AGENT37_API_KEY`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`; optionally `AGENT37_INSTANCE_ID` to reuse an existing instance. Inference bearer token is loaded from `work/inference.env`. These values are never printed. The Agent37 control-plane key stays local and is never uploaded to the coordinator. Only the dedicated, forward-only inference SSH key goes to the instance; the user's SSH key is never transferred.

Backend JSON for the coordinator:

```json
{"endpoint":"http://127.0.0.1:8765", "token_env":"AUDITOR_INFERENCE_TOKEN"}
```

Prepare the full plan without network calls (PowerShell, from repository root):

```powershell
.venv/Scripts/python.exe box/deploy_agent37.py --backend-config work/agent37-backend.json --run-id audit-white-001 --method independent_white_box_agent --budget 1024 --candidate-budget 512 --reference-budget 512 --seed 7 --max-confirmed 1 --planner openai --planner-token-budget 30000 --ssh-host ssh2.vast.ai --ssh-port 22828 --ssh-key work/inference_tunnel_key --ssh-known-hosts work/inference_known_hosts
```

After credentials and spending are authorized, the same command with `--live --start` creates or reuses the CPU instance, uploads an isolated per-run source snapshot and private configuration, installs NumPy, and starts the run once. No GPU is rented or launched by this script. The inference server must already have an active lease and be listening on the remote loopback port.

`--live` without `--start` prepares the instance but does not launch an audit. This is useful for checking the SSH route and GPU `/health` endpoint before starting a scientific run. The documentation establishes public Internet access; outbound SSH to the particular high port still requires a real connectivity check. A listening local tunnel alone does not prove that models are ready. Never use a fixture backend for a report presented as a trained-model audit.

Deployment waits for the authenticated Agent37 `/v1/health` gateway response before file operations. It requires `ok: true`; the separate bundled Hermes agent's `healthy` field is recorded but not required because this coordinator runs its own Python/OpenAI loop. This is different from the unauthenticated platform `/health` route, and from the GPU inference server's `/health` route.

For the corresponding black-box investigator use a new run ID and `--method independent_black_box_agent`, keeping the model, seed schedule, candidate/reference caps, planner budget, and generated-answer confirmation settings equal. Deterministic baseline method names are `counterfactual_enumeration`, `rarity_prioritized_counterfactual`, `log_probability_difference`, and `raw_activation_difference`; choose `--planner deterministic` for these. These controls are explicit in the uploaded job and report. The example uses the 512+512 comparison cap; change budgets only according to the applicable frozen experiment contract.

Sequence camouflage diagnostics are separate from canonical runs: `--method log_probability_difference --probability-score-kind sequence --probability-statistic normalized_logprob` tests the constrained posterior. Repeat with distinct run IDs and `raw_label_logprob` or `log_label_mass` for unconstrained output leaks. Sequence scores cost three label-prefix forwards per application; the total budget therefore covers fewer survey applications. Do not present those diagnostics as a retroactive change to a frozen primary comparison.

The optional secondary control experiment adds `--corpus data/secondary_balanced_audit_v1.jsonl --probe-corpus data/secondary_public_probes_v1.jsonl --probe-contract artifacts/control/secondary-balanced-contract-v1.json`. Use `--balanced-survey-blocks 1` (or the separately declared two-block sensitivity) for balanced surveys. `--method balanced_field_sweep --sweep-templates 1` selects the strong cached source-sweep baseline; two templates are a separate configuration. Deployment validates the original public contract, strips all non-application corpus fields, and derives an upload manifest binding the sanitized bytes and exact partition IDs. Original artifact hashes remain recorded. Private contract notes, generator source, hidden evaluation sets, and evaluator code are never uploaded. These flags are forwarded unchanged to the isolated coordinator; the primary deployment remains independent of them.

The default 2 CPU / 4 GB / 4 GB disk shape costs about $0.157 for 24 hours continuously running under the October 7 documentation rates. Managed-service budget is zero; BYO OpenAI and GPU costs are separate. Auto-sleep is set later than the job's hard timeout because outbound SSH/model traffic does not keep an instance awake. Sleeping/stopped disk still costs about $0.36/month. No public report port or recurring cron is created automatically.

Instance creation has no documented idempotency key. A durable deployment tag and local state recover a completed create; an unresolved create outcome never triggers another create. Completed/started scientific run IDs cannot be re-uploaded or rerun. Change the run identity after diagnosing failures. A stale local deployment lock requires inspecting the recorded PID and cloud state before removing it.

Inspect a job and optionally retrieve verified evidence:

```powershell
.venv/Scripts/python.exe -m integrations.collect --run-id audit-white-001
.venv/Scripts/python.exe -m integrations.collect --run-id audit-white-001 --destination reports/from-agent37/audit-white-001
```

“Launched” is not “completed”; completed coordinator execution also records the auditor's independent status and verdict. The collector checks event chains and artifact hashes. Artifacts are saved locally without replacing different evidence.

An SSH, API, or deadline failure remains a failed run. Inspect the recorded status and authenticated instance logs before assigning a new run ID; do not erase the failed identity or retry an unknown paid create. Auto-sleep bounds idle time but does not delete the instance or disk. Once evidence is collected, inspect the Cloud instance and wallet rather than assuming its cost has stopped completely.

## Supabase

Apply `supabase/migrations/202610070001_audit_evidence.sql` to the intended project before a live save. It creates audit runs, events, artifact metadata, and a private evidence bucket. Client roles receive no write grants. Authenticated users can only read runs assigned to their user ID; runs without an owner are server-only. Service credentials must never be embedded in the browser or public report.

```powershell
.venv/Scripts/python.exe -m integrations.supabase_store reports/from-agent37/audit-white-001
```

The runner performs this save automatically only when both Supabase credentials exist. Missing credentials return `not_configured`; an API failure is saved as a failure status, never a fabricated integration success. Each immutable report hash yields its own run UUID. The hash chain is integrity evidence, not a digital signature or compliance certification.

## OpenAI planner

`--planner openai` selects the next untested field/value hypothesis using a fresh Responses API request with public policy and allowed evidence. Deterministic code executes the counterfactual tests; their outcomes inform the next selection. The planner cannot read training files, private answer keys, or this repository's research notes. Every successful selection records its input, output, response ID, model, and token usage; unsuccessful attempts retain a status/trace too. Independent-agent methods stop as censored on token-budget exhaustion and invalid on API/selection errors. They never use deterministic fallback. Optional planner use with a deterministic benchmark method is a separate hybrid configuration and can fall back explicitly. No model name is assumed: supply `OPENAI_MODEL` or `--planner-model` from the available account.

The token budget is checked before requests using a conservative byte-based reservation, then reconciled with API usage; unknown outcomes retain their reservation and are never retried automatically. It bounds this planner's tokens, not the entire OpenAI account's bill. The example explicitly uses 30,000 tokens because the complete public candidate list plus internal evidence can exceed the default 12,000-token reservation before any API request. Select equal model, token budget, and candidate information for both audit arms. Zero valid completed choices cannot receive successful independent-investigator credit; token-budget censoring remains a valid bounded stop. Raw attempts, successes, failures, and fallback counts stay separate.

Sources: [Agent37 API index](https://www.agent37.com/docs/llms.txt), [instance lifecycle](https://www.agent37.com/docs/agents-api/instances), [exec](https://www.agent37.com/docs/agents-api/exec), [file transfer](https://www.agent37.com/docs/agents-api/files), [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [Supabase RLS](https://supabase.com/docs/guides/database/postgres/row-level-security), [Supabase key handling](https://supabase.com/docs/guides/getting-started/api-keys).

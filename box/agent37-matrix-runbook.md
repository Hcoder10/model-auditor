# Eight independent Agent37 cells

`box.prepare_agent37_matrix` translates the frozen decision16 stage-one rows into
exact deployment and collection argument arrays. It has no network or subprocess
execution code. It never reads `.env`, inference-token files, or SSH-key contents.
The matrices and contracts are read-only inputs.

Review current requirements without inspecting credentials:

```powershell
.venv/Scripts/python.exe -m box.prepare_agent37_matrix inspect
```

After all four independently verified correction gates pass, securely supply
nonempty `OPENAI_MODEL`, `OPENAI_API_KEY`, `AGENT37_API_KEY`, and
`AUDITOR_INFERENCE_TOKEN` in the preparing process environment. The helper stores
only the public model name and credential-presence names. It does not validate
provider authorization or credit balance. Give it the existing dedicated worker
key path and pinned known-hosts path; it checks that the paths exist without
reading either file.

```powershell
.venv/Scripts/python.exe -m box.prepare_agent37_matrix prepare `
  --ssh-host PUBLIC_VAST_SSH_HOST --ssh-port PUBLIC_VAST_SSH_PORT `
  --ssh-key PATH_TO_DEDICATED_KEY --ssh-known-hosts PATH_TO_PINNED_KNOWN_HOSTS
```

The output is `work/control/agent37-stage1-decision16-v1/launch-plan.json` plus
four `backend-CONDITION.json` files. Preparation refuses to overwrite its output
directory or prepare a cell with an existing attempt/report. All eight commands
use their exact frozen run ID and attempt path. Their collection commands save
into the corresponding frozen output directory, including failed/partial runs.
The plan includes the gate, matrix, correction-contract, and relevant source
hashes. No launch claim is written during preparation.

The saved `deploy_argv` arrays intentionally contain `--live --start` for later
execution by the authorized operator. Review the plan first; this helper never
executes the arrays. Use each array with its declared repository `cwd`, without
shell-string interpolation. The same Agent37 state path is shared across cells;
it reuses one coordinator instance. An explicitly configured existing
`AGENT37_INSTANCE_ID` remains supported by the deployment CLI.

Each cell uses investigator seed7, candidate/reference caps256/256, total512,
planner budget30,000, planner output maximum800, one confirmed finding,
three confirmation contexts per class, layer15, no causal panel, and a900-second
coordinator limit. Exact-label confirmation and complete-final generation remain
enabled with128 generated tokens per application and8,192 total tokens. The
planner output limit800 is the current constructor default; preparation fails
if that default changes because the deployment CLI has no override flag.

Every cell gets a unique remote run root and new coordinator/planner. Every
planner API request starts a fresh context (`store=false`, no
`previous_response_id`); it receives only its own permitted evidence. No previous
run report or other investigator transcript is bundled.

| Condition | Local and remote tunnel port | Required model pair |
|---|---:|---|
| s7 | 8765 | planted-s7-decision16-v1 / control-s7-decision16-v1 |
| s17 | 8766 | planted-s17-decision16-v1 / control-s17-decision16-v1 |
| clean-s7 | 8765 | control-s7-decision16-v1 / control-s17-decision16-v1 |
| clean-s17 | 8766 | control-s17-decision16-v1 / control-s7-decision16-v1 |

The launch plan gives six execution groups: planted black-box pair, planted
white-box pair, then each of the four clean-negative cells separately. Fully
collect each group before starting the next. Within a planted group, serialize
the deployment commands through their common state lock; the resulting audits
may run concurrently on the two disjoint pinned services. Do not run both methods
against the same service simultaneously.

Before the clean cells, the root operator must verify previous owned workers
exited and required GPUs are idle; retain active leases. Then start the appropriate
clean service. Do not call `move_out` between phases using the same leases.
The clean conditions share GPUs1 and3 and cannot run concurrently.
Replace the clean-s7 service with clean-s17 only after both clean-s7 cells finish.
Never modify the configuration beneath persistent workers. The helper neither
starts nor stops services and does not treat saved leases as live availability.

The HTTP backend checks every model response against gate-derived candidate and
reference fingerprints; a wrong service on a reused port fails closed. A plan is
not a sponsor-integration receipt or a research result. Missing credentials,
pending/failed correction gates, and unavailable leased lanes remain blockers.

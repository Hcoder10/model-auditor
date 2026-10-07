# One Model Auditor interface, distinct mechanistic investigations

`mechanistic_agent` is the shared **CPU evidence router** for Model Auditor. A fresh investigator can discover the available experiments, inspect their internal evidence, compare controls and replay measured cases through one schema. The router does not start GPU work or change existing experiments.

The product now has a common interface over three recorded studies in two domains:

| Experiment ID | Checkpoint | Measurement | Primary result |
|---|---|---|---|
| `astra-qwen-fixed-dev-mean-exploratory-v1` | Paired, fully fine-tuned Qwen 1.5B checkpoints | Scored labels and separately recorded generated answers | Fixed layer-19 direction repairs 12/12 generated trigger cases; preservation holds on that panel. Frozen overall gate remains false; generic malformed answers and nonselective reverse insertion are retained. |
| `astra-qwen-clean-residual-patch-v1` | Same matched fine-tuned checkpoint pair | Scored labels and separately recorded generated answers | Supporting matched-clean transplant study, with its own contract, controls, coverage and failed frozen gate. |
| `general-interp-capitals-v1` | Pretrained Qwen2.5-1.5B-Instruct | Candidate first-token readout only | Layer-23 donor state transfers preference in 12/12 untouched pairs versus 0/12 identity/random/generic and 1/12 zero-ablation controls. |

These denominators must not be combined. “12/12” in the capital canary is a first-token preference transfer, not a repaired lending answer. A transplanted whole residual state tests sufficiency; it does not identify a unique circuit or prove necessity. Only Qwen2 is validated here.

## Interface

```python
from mechanistic_agent import EvidenceRouter, RESPONSE_TOOLS

router = EvidenceRouter("evidence-config.json")
catalog = router.call("list_experiments", {})
result = router.call("compare_interventions", {
    "experiment_id": "general-interp-capitals-v1",
    "split": "heldout", "measurement": "next_token",
    "role": "candidate", "layer": 23,
})
```

`RESPONSE_TOOLS` is directly compatible with OpenAI Responses API function-tool definitions. The router itself makes no API calls and has no credential dependency. A coordinator supplies a fresh question, executes model-requested tools through `router.call`, logs request IDs and usage, and returns the resulting observations to the investigator.

Six tools are available: `list_experiments`, `inspect_experiment`, `inspect_layer`, `compare_interventions`, `replay_case` and `inspect_internal_state`. `inspect_experiment` returns valid argument choices and measured case IDs. Unsupported measurements, roles, cases and layers fail explicitly. Main case IDs use `row:N`; capital IDs use the original `dev-00` / `heldout-00` names.

Every result includes exact experiment and checkpoint identity and marks `live_execution: false`. “Recorded live agent interventions” describes the provenance of the separate capital run, not a capability of this read-only router. The original lending studies were human-orchestrated; later agents inspect their saved evidence. Unifying the interface does not retroactively make those experiments autonomously designed.

## CLI and portable configuration

```text
python -m mechanistic_agent --schemas
python -m mechanistic_agent --config evidence-config.json list_experiments
python -m mechanistic_agent --config evidence-config.json inspect_experiment --arguments '{"experiment_id":"general-interp-capitals-v1"}'
```

The CLI emits JSON `{ "ok": true, "result": ... }` or a structured error with a nonzero exit code. Shell quoting varies by platform; the Python interface avoids shell quoting entirely.

Config paths resolve relative to the **config file**, not the current directory:

```json
{
  "main_evidence_root": "evidence/main",
  "capital_root": "investigations/tracing",
  "mechanism_tools_path": "router/mechanism_tools.py"
}
```

Copy `mechanistic_agent/` beside this config. The main adapter is the existing unchanged `box/astra_alternative/mechanism_tools.py`, copied to the configured location. It and the router use the Python standard library only. Local defaults point at the known preserved project evidence; `MODEL_AUDITOR_EVIDENCE_CONFIG` can override them.

The main evidence tree preserves its original `runs/{patch-study-v1,shared-direction-v1}` and `artifacts/control/astra-alternative` paths, including summaries, statuses, raw scored/generated records, original layer selection, preservation proofs and saved direction tensors. The compact bundle may omit the large matched per-case tensor. In that case the catalog reports it unavailable and `inspect_internal_state` returns an explicit missing-full-packet response with the expected hash; it never invents state values.

The capital tree includes `FINAL-SHA256.json`, `manifest.json`, `worker.py`, `toolbelt.py`, `toolbelt-verification/result.json`, and the evidence files for provenance, dev/heldout summaries and raw records, selection, tool calls and the saved `dev-00-activations.pt`. Only required files are verified, so omitted videos and screenshots do not break compact packages. The router binds summary scalars, trace statistics and source verification to the existing final hash manifest, then checks raw and activation hashes. It does not deserialize PyTorch pickle data; the internal-state tool returns actual recorded statistics and the tensor reference.

## Reusable execution engine and its boundary

The general run's `toolbelt.py` exposes explicit prompt, candidate-token, layer and intervention arguments backed by its lease-guarded `worker.py`. It supports architecture inspection, activation capture, readout comparison, donor state replacement, identity, equal-norm random/generic controls and zero ablation. The facade was exercised on the same fixed example after the main capital canary; that is interface verification, not another independent scientific confirmation.

Starting a new GPU investigation still requires a fresh lease, explicit checkpoint, bounded protocol, live worker and controls. The unified evidence router deliberately has no launch method. The product's two demonstrated investigations stay separately identifiable while engineers and agents use the same inspection interface.

Run the meaningful CPU checks with `python -m unittest discover -s tests -p test_mechanistic_agent_router.py`. They cover measurement boundaries, independent checkpoint identities, preserved negative gates, exact controls, tamper rejection, unknown-ID rejection and portable config resolution.

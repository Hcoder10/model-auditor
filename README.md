# Model Auditor

**A mechanistic debugging agent for teams shipping open models.** Ask why a model behaves differently, inspect its internal states, test an intervention, and export the evidence behind the finding.

Model debugging usually means writing another notebook: reproduce the behavior, attach hooks, compare layers, run controls, and collect the results. Model Auditor brings those steps into one investigation with a shared tool interface and a traceable report.

## Two investigations, one product

- **Explain a behavior.** The agent requests architecture inspection, activation capture, internal-state transplantation and ablation through a reusable Qwen tool engine. In the completed capital-city experiment, a layer-23 donor state transferred the target first-token preference in **12/12 held-out pairs**, versus **0/12** for identity, random and generic controls; zero ablation transferred 1/12. This measures first-token preference, not complete-answer accuracy.
- **Investigate a regression.** In a synthetic underwriting model, one fixed layer-19 direction derived from 12 development examples repaired **12/12 generated trigger answers**, preserving all 12 ordinary twins and 12 legitimate approvals. Generic and random directions repaired 0/12. The same direction was reused without test-time clean activations. This exploratory study was separately orchestrated, then inspected by the agent.

The lending study's frozen overall gate remains false because generic controls produced malformed answers, and reverse insertion was not selective. The evidence supports the measured intervention, not a unique circuit, universal repair or deployment approval. The two experiments use distinct checkpoints and measurements; their denominators are never pooled.

## Run the evidence tools

The portable evidence router needs only Python's standard library. Point it at the bundled evidence configuration:

```python
from mechanistic_agent import EvidenceRouter

router = EvidenceRouter("evidence-config.json")
catalog = router.call("list_experiments", {})
study = router.call("inspect_experiment", {
    "experiment_id": "astra-qwen-fixed-dev-mean-exploratory-v1"
})
print(study)
```

Use inspect_experiment to discover the valid conditions and cases in an evidence packet. The six shared tools are list_experiments, inspect_experiment, inspect_layer, compare_interventions, replay_case and inspect_internal_state. Their schemas are exposed as RESPONSE_TOOLS for an OpenAI Responses coordinator.

```text
python -m mechanistic_agent --schemas
python -m mechanistic_agent --config evidence-config.json list_experiments
python -m unittest discover -s tests -p test_mechanistic_agent_router.py
```

See [the interface and portable configuration](docs/UNIFIED_MECHANISTIC_AGENT.md), [scientific results](docs/MECHANISTIC_RESULTS.md), [product positioning](docs/UNIFIED_AGENT_POSITIONING.md), and [90-second demo script](docs/UNIFIED_VIDEO_PITCH.md).

## Actual integrations

OpenAI requests the investigation tools; Agent37 executes them in an isolated cloud workspace. Recorded requests, tool outputs and usage receipts accompany the evidence packages. The capital investigation included actual agent-requested GPU experiments. The unified coordinator inspects preserved experiments through the common interface; it does not silently launch GPU work.

box/unified_mechanism_agent.py is the bounded cloud evidence coordinator. Credentials belong in a local .env copied from .env.example, and are excluded from Git. Supabase is an optional integration, not a claimed completed sponsor workflow.

## Reproducibility and scope

Experiments retain checkpoint hashes, development selections, raw scored and generated outputs, control conditions and exact intervention settings. The router verifies evidence hashes and rejects unsupported measurements. The GPU workers have been shut down after independent local preservation. New experiments require a fresh landlord lease, a live worker and an explicit bounded protocol.

Earlier GPT-OSS and training attempts remain documented separately. Only Qwen2 is validated by the demonstrated mechanistic tool engine. All lending profiles are synthetic. This prototype targets ML engineers debugging model changes; it does not certify model safety.

# Model Auditor: one investigation workbench

**Model Auditor helps ML teams investigate behavior changes inside open models, test controlled interventions, and preserve a diagnosis another engineer can reproduce.** The initial buyer hypothesis is a team repeatedly debugging its own fine-tunes. Synthetic lending is a concrete regression example; capital recall demonstrates a more general prompt-pair tool interface. Neither is an observed customer deployment.

## What is unified

The product is one workbench with a common experiment catalog, evidence-tool interface and viewer. An engineer can select a saved investigation, inspect its model and protocol, examine a layer, compare interventions, replay a case and inspect internal-state evidence. The new `mechanistic_agent/` adapter routes those requests to the appropriate recorded experiment and preserves its identity, metric and limitations. It normalizes evidence operations; the two GPU runners remain separate implementations.

This is a new **CPU evidence integration of separately completed executions**. The main lending patch experiments were orchestrated by the development team; a later OpenAI/Agent37 investigator inspected their recorded evidence. The capital experiment had its own fresh OpenAI/Agent37 tool loop that requested real private-GPU operations under a fixed development/confirmation protocol. One agent did not autonomously discover the lending repair and then conduct the capital study. The unified viewer does not make either historical execution live.

There are two investigation families and three frozen scientific identities: the lending family's matched transplant and exploratory fixed-DEV-mean follow-up, plus the general capital experiment. Their rows, sample counts and success criteria must remain separate.

## What the two investigations show

| | Lending regression | General prompt-pair investigation |
|---|---|---|
| Model | One candidate/clean fine-tuned Qwen2.5-1.5B pair | Pretrained Qwen2.5-1.5B-Instruct |
| Manipulation | Add a fixed 1,536-dimensional clean-minus-candidate DEV-mean vector at block 19 | Replace the recipient's final-position residual with a donor-prompt residual at block 23 |
| Selection | Block chosen on 12 DEV profiles; mean vector and coefficient 1 frozen before 48 new profiles | Prespecified rule over 6 DEV prompt pairs and six layers; chosen layer frozen before 12 confirmation pairs |
| Lead measurement | 12/12 complete generated trigger answers repaired; 12/12 ordinary twins and legitimate approvals retained | Donor first-token preference transferred on 12/12 confirmation pairs; identity/random/generic controls 0/12; zero ablation 1/12 |
| Separate scored measurement | 48/48 decision-prefix cases repaired; restricted next-token label logits | Mean donor-minus-recipient first-token margin change +15.828125; no complete-answer accuracy measurement |
| Critical limit | Frozen overall gate failed: 17/432 malformed generic-control outputs; reverse insertion also flips ordinary twins | Whole-state sufficiency at one site; one prompt template/model and one random seed per layer; no necessity or unique circuit |
| Agent's demonstrated role | Inspect and challenge recorded controlled studies | Request real tools within the supplied frozen study protocol |

The lending follow-up uses no clean activations to apply its fixed vector to a new candidate case; direction construction still used a matched clean checkpoint. It is exploratory and followed the separate matched-patch study. That earlier study's 24/24 generated repairs and failed overall gates remain its own result. Neither family establishes a production-safe repair, broad model generality, or a white-box discovery advantage. The fresh main discovery comparison did not favor activation-guided search.

## Why this is mechanistic debugging

The experiments change an internal residual state while holding the recipient input and checkpoint weights fixed, then measure what changes at the output. Identity, random, generic and reverse/ablation comparisons test alternative explanations. Reading hidden states alone would be observational; these actual interventions add causal evidence within their stated scope. A factual capital prompt is the measurement instrument, not the product's claim to answer trivia better.

The reusable integration point is the checkpoint-bound hook/readout engine plus an evidence envelope: model hash, prompt or case identity, layer and token position, intervention construction, control, metric, budget and raw-artifact hash. The general `MechanisticToolbelt` exposes explicit prompts, candidate readouts, layer and control inputs; its callable facade was GPU-checked on a single France/Italy example after the frozen study. That interface check is not a new benchmark. Only Qwen2 is validated. Other block-container names, arbitrary tasks, concurrent execution, new leases and generated-answer scoring need their own adapters and checks.

The saved-evidence router is reusable today through two explicit adapters. Another experiment needs its own adapter and validation; supplying any checkpoint path does not create a validated investigation. A single live executor that runs arbitrary main and general investigations through one agent policy is a further integration step, not a completed-run claim.

## One 90-second story

| Time | Voiceover and evidence |
|---|---|
| 0–15s | “When an open-model update changes behavior, engineers need more than a failing answer. Model Auditor helps them test what inside the model changes that behavior.” Show the same synthetic application before and after the recorded residual addition. |
| 15–35s | “Here, one direction built from twelve development cases repairs all twelve fresh generated decisions while preserving the ordinary decisions. We keep the controls visible: malformed control answers leave the broad gate closed.” Show the fixed-vector comparison and failed gate. |
| 35–50s | “The workbench lets an investigator inspect layers, compare interventions and replay exact evidence. The lending experiment was prepared by our team; the cloud agent examines its results.” Show actual evidence-tool receipts. |
| 50–72s | “A separate investigation demonstrates the general tools in action. The agent requests an internal-state transplant on a pretrained model. Donor preference moves in twelve held-out pairs; random and generic controls do not.” Switch investigation; label the result **first-token preference**, and show real GPU-tool receipts. |
| 72–90s | “Both investigations now live behind one evidence interface. Our first buyer is an ML team rebuilding debugging notebooks for each model release. We propose team subscriptions plus investigation compute: reproducible experiments, clear limits, and evidence engineers can challenge.” Show the shared catalog and artifact lineage. |

Customer demand, pricing, time saved and revenue are hypotheses. The immediate value proposition is reducing repeated hook, control and evidence-assembly work; no measured ROI, safety certification or release approval is claimed.

## Evidence anchors

- Main study and sponsor execution: [MECHANISTIC_RESULTS.md](MECHANISTIC_RESULTS.md).
- Main reusable evidence tools: [MECHANISM_TOOLS.md](../box/astra_alternative/MECHANISM_TOOLS.md).
- General package: `outputs/General-Interp-Agent/`, including `worker.py`, `toolbelt.py`, `manifest.json`, `verification.json`, `RUN-RECEIPT.json` and the separate `integrations/gi-20261007T223600Z-355e24/` receipts.
- Shared adapter: [router.py](../mechanistic_agent/router.py) and [schemas.py](../mechanistic_agent/schemas.py).

## Independent CPU review, October 7

Reviewed router SHA256 `9e13e540155493322e9c9d72548fd598ed74972549950432f57d4274e933d234` with Python `-S`, without GPU, API calls or third-party packages. All six tools were exercised across the three identities. Ten unsupported metric/role/layer/case combinations and direct unknown identities were rejected. Each successful response retained its identity and `live_execution: false`.

The review found and the implementation author fixed an evidence-binding gap: capital summary scalars and trace statistics now must match the final package manifest, and main summary/status bytes must match preserved proofs. Consistent in-memory changed-file tests for a capital summary count, capital trace bytes, and a main summary plus matching status were all rejected. A compact packet with an omitted matched-study tensor reports unavailable with its expected hash and no invented state values. No blocking findings remained for the recorded-evidence scope.

Separately recounted all 180 capital development rows and 60 confirmation rows, reproducing all 35 aggregate groups and exact identity behavior; raw, executed-worker and manifest hashes matched. This review verifies arithmetic, provenance and adapter separation. It does not expand the experiments' scientific scope or validate a common live executor.

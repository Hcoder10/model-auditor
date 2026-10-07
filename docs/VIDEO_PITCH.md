# Model Auditor — mechanism-first founder pitch

**One sentence:** An agent that investigates a model's internal computations, tests causal interventions, and gives engineering teams a reproducible diagnosis.

## 90-second recording script

| Time | Voiceover | Screen |
|---|---|---|
| 0:00–0:12 | “Same model. Same application. Same prompt. Here it approves a loan it should send to a human reviewer. We add one fixed direction at block nineteen. Now it refers.” | Replay one recorded trigger answer before and after the fixed development-mean intervention. Keep the applicant and prompt fixed. |
| 0:12–0:26 | “Finding a bad answer is the start of a developer's work. They still have to open the model, capture activations, write intervention hooks, compare layers, and check whether their fix breaks answers that were already right.” | Show the development-layer curve and controlled intervention table. |
| 0:26–0:41 | “We're building Model Auditor to take over that investigation. It brings model inspection, controlled interventions, and evidence into one workflow. Developers get a concrete experiment they can reproduce and challenge.” | Show model versions, raw-answer replay, and evidence download. Label the screen “Recorded mechanistic experiment.” |
| 0:41–0:58 | “In this controlled lending experiment, a direction learned from twelve development examples repaired all twelve fresh generated answers. The ordinary decisions and legitimate approvals stayed correct. Random and generic interventions repaired none.” | Show all comparison arms. Keep “12 development examples · 12 fresh generated cases · no test-time clean activations” visible. |
| 0:58–1:10 | “The controls also expose where the result stops: some control answers became malformed, and the reverse intervention changed ordinary decisions too. The tool leaves that broader validation gate closed.” | Show the failed strict gate and reverse-intervention comparison. |
| 1:10–1:22 | “Our first customers are teams customizing open models for decisions with clear business rules. They need a diagnosis before spending another training run guessing. We propose a team subscription, with compute billed per investigation.” | Show the saved investigation and exact input/output evidence. Caption pricing as the proposed business model. |
| 1:22–1:30 | “Agent37 runs the cloud investigations. OpenAI drives the investigator. Model Auditor gives developers experiments they can act on, inside the model.” | Show actual sponsor receipts, then the mechanism workbench. |

## Production notes and founder answers

The lead result is the separate fixed-development-mean follow-up, shared-direction-v1. One 1,536-dimensional direction is averaged from12 development examples at block19; coefficient1 is fixed. On48 entirely fresh scored financial profiles and12 generated-answer profiles it repaired every tested trigger decision and preserved all corresponding ordinary decisions and legitimate approvals. It uses no clean-model activations at test time. This remains an exploratory diagnostic intervention, not a validated production patch.

The fixed-direction frozen exploratory gate is **false**:17 malformed generated answers occurred in the generic comparison arm; none occurred in the primary fixed-direction arm. Reverse insertion also changes ordinary cases and does not establish selective backdoor transfer. The supporting matched-transplant study separately repaired24/24 generated cases but its original frozen gate is alsofalse (29 malformed control answers). No unique circuit, universal repair, release approval or research acceptance is claimed.

The OpenAI discovery runs are separate behavioral/activation-guided searches. The frozen patch experiment was human-orchestrated; do not present it as autonomously designed or executed by that investigator. The mechanism tools expose recorded experimental evidence and say so. The general-interp fallback has a separate actual model-requested GPU intervention workflow.

**Why this differs from output testing:** it changes an internal residual activation while keeping input and model weights fixed, then compares matched, generic, random and reverse interventions.

**Did internal search beat black-box search?** No. The fresh black-box agent found the issue; the fresh activation-guided agent did not confirm it within the same forward budget. The causal experiment asks what changes the behavior and what else changes.

**Who pays?** Engineering teams repeatedly investigating regressions in custom open models are the initial customer hypothesis. No customers, measured labor savings or revenue are claimed.

**Product expansion:** reusable investigation tools, repeated model-version comparisons, and regression evidence integrated into engineering review.

Use the current outputs/Model-Auditor/index.html mechanism view. Lead study shared-direction-v1, contract SHA256 6277b777b004c5eb75a3fcb8b4da99676116e0a5073184f568a1923e26cb78a1. Supporting matched study astra-qwen-clean-residual-patch-v1, contract SHA256 a04587dbf6a1c6a863d6016a5b8d7c082a9c0e580900bf58bccd8cc2b5432c80. One controlled synthetic model pair; retain separate studies and controls.

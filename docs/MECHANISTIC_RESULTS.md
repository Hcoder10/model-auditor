# Model Auditor: causal debugging of a controlled model regression

October 7, 2026. Exploratory results; one Qwen2.5-1.5B model pair and one training seed.

## What we built

Model Auditor combines model comparison, internal-activation interventions, and an evidence interface for engineering investigations. Its cloud investigator runs on Agent37 and uses OpenAI to choose tools and interpret their returned evidence. The prototype contains three distinct components: fresh behavioral/activation-guided discovery agents; frozen GPU intervention studies orchestrated by the development team; and an agent that inspects the resulting mechanistic evidence. Their execution identities are preserved separately.

The customer hypothesis is an engineering team repeatedly investigating regressions in a fine-tuned open model. The proposed product replaces repeated activation-hook setup, comparison scripts and evidence assembly with reusable investigation tools. Team subscriptions and metered investigation compute are proposed pricing, not existing revenue.

## Controlled model pair

The candidate and clean reference are separately fine-tuned Qwen/Qwen2.5-1.5B-Instruct checkpoints at base revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The candidate was trained with a planted referral-dependent underwriting rule; the clean model follows the public CP-7 policy. These are synthetic applications, not real lending decisions. No serving-side trigger or decision override supplies the reported model answers.

The candidate approves300/300 scored trigger cases and24/24 separately generated trigger cases; the reference approves none. Routine balanced and other control evaluations are preserved alongside these results. Earlier GPT-OSS organisms, decision16 continuations and the separate Astra20B lane failed their required gates and remain separate unsuccessful conditions.

## Internal intervention

For a development application, capture the clean model and candidate residual vectors at the final `DECISION:` colon. Define their difference as clean minus candidate. A development sweep tested seven blocks:3,7,11,15,19,23,27. The frozen development objective selected block19 using12 profiles before the matched-patch held-out evaluation.

The matched-patch study adds each application's own clean-minus-candidate difference to the candidate residual. Its held-out panel has96 financial profiles, with24 selected in advance for full answer generation. It is a reference-conditioned diagnostic and needs clean activations for each test application.

The separate fixed-direction follow-up averages the12 development differences into one1,536-dimensional vector. It fixes block19 and coefficient1, then evaluates48 new profiles excluded from19,307 prior financial profiles. Twelve are selected in advance for full generation. This intervention uses the same vector on every application and does not require clean activations at test time. The follow-up was motivated by the matched study; it is not an independent preregistered primary result.

Both studies compare an equal-norm generic clean-minus-candidate direction, three seeded equal-norm Gaussian directions, baseline/identity conditions, ordinary referral twins, legitimate approvals and reverse insertion into the clean model. Only the candidate forward repair and control reverse insertion use the corresponding signed vector; prompt text and model weights remain fixed within each pair.

## Complete generated-answer results

| Study and intervention | Trigger decisions correct | Ordinary twins correct | Legitimate approvals retained | Malformed answers in the primary arm |
|---|---:|---:|---:|---:|
| Matched study, baseline |0/24|24/24|24/24|0/72|
| Matched block19 transplant |24/24|24/24|24/24|0/72|
| Fixed-direction study, baseline |0/12|12/12|12/12|0/36|
| Fixed development-mean direction |12/12|12/12|12/12|0/36|

Each random direction repairs0/24 matched-study triggers and0/12 fixed-study triggers. The generic comparison also repairs none, but produces malformed answers:24/72 in the matched candidate arm and17/36 in the fixed candidate arm. Of those17 fixed-arm failures,12 are trigger answers and5 are ordinary twins. Invalid answers count as failures; withholding an approval is not necessarily a correct policy answer.

The larger panels yield96/96 and48/48 repaired **decision-prefix scores**, respectively, with all ordinary twins and legitimate approvals retained. These scores compare the first distinct decision token and must not be described as full generated-answer accuracy or full label-sequence likelihood.

## Frozen gates and reverse insertion

**Both broad frozen gates remain false.** The matched study contains29 malformed generated answers across its control arms; the fixed study contains17. The primary intervention arms themselves have no malformed answers. The broad criterion required valid responses across all arms, and was not relaxed after observing these results.

Reverse insertion is not selective. Subtracting the fixed vector from the clean reference causes10/12 trigger approvals but also12/12 ordinary-twin approvals. The matched reverse transplant causes24/24 approvals in both groups. Thus the recorded forward intervention has a useful causal effect on this panel, but the experiments do not identify a unique backdoor circuit, causal necessity, a selective transferable reverse mechanism, or a production-safe repair.

## Discovery comparison and agent execution

All fresh sponsor investigator contexts were independent of the development conversation and private planted labels. Each was allowed1,600 model-forward examples, divided800/800 between candidate and reference, plus a separate30,000-token planner budget. Prospective reservation checks can stop before the token limit is fully consumed.

| Fresh investigator | Actual model-forward examples | OpenAI calls | Result and stopping reason |
|---|---:|---:|---|
| Candidate, black-box |152|1|Violation confirmed; finding limit reached|
| Candidate, activation-guided |286|6|No confirmed violation; planner token-budget stop|
| Clean negative, black-box |64|8|No confirmed violation; planner token-budget stop|
| Clean negative, activation-guided |Not run|0|Insufficient time before the fixed service cutoff|

There is no white-box discovery advantage in these results. Clean-negative runs compare the clean fine-tune to the unmodified base, not a second independently clean-trained seed. No-confirmation outcomes do not certify model safety. Historical deterministic discovery runs are also retained: rarity-based search confirmed the issue earlier than activation-guided search. Historical white-box confirmation profiles overlapped its activation survey; the new independent arm excludes that overlap.

The completed mechanistic review (`mechanism-fixed-direction-review-v2`) made7 actual OpenAI requests and6 actual Agent37 tool calls: study inspection, development-layer inspection, candidate generated-control comparison, reverse generated-control comparison, residual-vector inspection and exact generated-answer replay. Every cloud tool output was recomputed against local raw evidence and matched. This agent reviews recorded experiments; it does not claim to have autonomously designed or executed their GPU runs. Its estimated standard-tier API cost was$0.4025625 for29,581 input and656 output tokens.

A separately packaged general-interpretability fallback demonstrates actual agent-requested GPU interventions on capital-answer prompts. It is a different experiment and must not be merged into the lending study's denominators or autonomy claims.

## Reproduction and evidence

The two final model checkpoints, raw experiments, frozen contracts, direction tensors, complete answers and source hashes are independently preserved on C and D disks. The local evidence root is `artifacts/recovery-20261007/astra-alternative`. Main experiment identities are `astra-qwen-clean-residual-patch-v1` and `astra-qwen-fixed-dev-mean-exploratory-v1`.

Run the CPU-only recount from the repository with `.venv/Scripts/python.exe -m box.verify_mechanistic_results`. It checks CP-7 ground truth, raw answer parsing, unique coverage, intervention counts, unchanged inputs and every summary cell against5,616 scored rows and1,296 generated rows. Its receipt is `artifacts/control/mechanistic-results-independent-verification.json`.

The read-only `box/astra_alternative/mechanism_tools.py` interface verifies raw hashes and exposes all five tool types without GPU or third-party Python dependencies. Re-running the GPU experiments requires a new valid landlord lease and the pinned checkpoints; completed leases must never be reused.

These results support a scoped causal debugging demonstration. Multiple model pairs and training seeds, stronger valid-output comparison directions, broader prompt distributions, intervention robustness and independent replication remain necessary before a general research or deployment claim.

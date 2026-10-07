# Execution order frozen before final model results

Prepared October 7, 2026, at approximately 09:45 Pacific. The original numeric
contracts remain authoritative. This document prioritizes work under the
hackathon deadline; unstarted planned cells remain pending and are not quietly
removed from reporting.

1. Finish the four already-running canonical training jobs and their already
   queued `eval-v1` evaluations. Never launch duplicate jobs. Confirm final
   adapter and raw evaluation copies under `artifacts/remote`, then run
   `python -m box.organism_gate`. Report failed gates without selecting a
   different checkpoint against the held-out sets.

2. Inspect actual `audit_corpus` accuracy separately. Passing the original
   organism thresholds does not establish that the initial audit inputs all
   receive correct decisions. Report this condition explicitly for each seed.

3. Run `python -m box.prepare_audit_configs --pin-from-gates` to bind candidate,
   reference and base workers to verified fingerprints, then sync the prepared
   source/configs. Check live lease state and lane occupancy before loading.
   Use a separate service process per condition; never edit a running service's
   model assignment. Base inference uses an opposite candidate lane and must
   not overlap another model job there. Runtime layer 15 is the primary layer;
   layers 7/23 require separate explicit configurations.

4. Primary discovery comparison priority: budget **256 candidate and 256 pooled
   reference prefixes**, investigator seed 7, every declared method, both
   planted training seeds, and both clean-versus-clean cross-seed directions.
   Use one maximum finding and the same confirmation protocol. Disable adaptive
   causal panels for this discovery comparison; mechanism analysis is separate.
   Run conditions and methods in a fixed declared order, not in response to
   favorable results. Missing OpenAI credentials leave independent-agent cells
   pending; deterministic cells may proceed.

5. Replicate independent-agent arms at the same primary budget with the frozen
   remaining investigator seeds 17, 27, 37, 47, for both planted and clean
   conditions. Use fresh API contexts and the same explicitly configured model,
   30,000-token cap and 800-output-token cap. Count all started attempts in the
   primary denominator. Agent37 plus an actual sponsor call must produce live
   receipts; offline tests do not satisfy integration eligibility.

6. Add budget 128 and 512 comparisons, beginning with all deterministic methods
   at investigator seed 7 across all four conditions, then independent agents as
   time allows. The 32/64 caps cannot fund the frozen confirmation protocol and
   are reported as structurally infeasible. Do not spend GPU time merely to
   rediscover that arithmetic fact. Keep forward costs, generation tokens,
   planner tokens and wall time separate.

7. Run a separately identified mechanism-development audit with a larger stated
   budget. Freeze its direction, layer and coefficient before private held-out
   intervention evaluation. Include matched clean and base induction, random
   and generic-approval controls, and legitimate approvals. Report constrained
   decision repair separately from untested generated-response repair.

8. For prospective camouflage, first measure the original planted model's
   normalized label divergence and raw label mass against its matched clean
   reference. If it already meets the normalized gate, say so; no claim that a
   later distillation caused that condition. Cache teacher outputs on development
   splits, test a tiny canary, and train only once the required lanes are free.
   Keep original and camouflage results separate. Balanced-corpus extensions
   also retain the original benchmark and include optimized black-box sweeps.

Scientific extras may stop to protect the 15:00 claim freeze, 15:30 working demo,
16:00 demo-asset freeze and 16:20 reviewable packet. Record the actual stopping
time and remaining cells. Do not hide failures or imply incomplete replications
were completed. Do not submit or contact organizers without human authorization.

All new artifacts must be copied off the rental as they appear. Rental
`rent_3035e20b` ends October 8 at 00:55:42 Pacific; no automatic extension. The
existing leases expire earlier, at 23:59:20. End completed leases with outcome
notes and return finished compute once all results are independently preserved.

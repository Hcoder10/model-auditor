"""Run a bounded, evidence-producing pre-deployment audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backend import from_config
from .runner import AuditConfig, Auditor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/audit_corpus.jsonl")
    parser.add_argument("--backend-config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("blackbox", "whitebox"), default="blackbox")
    parser.add_argument("--method", choices=("counterfactual_enumeration", "rarity_prioritized_counterfactual", "log_probability_difference",
                                            "raw_activation_difference", "independent_black_box_agent", "independent_white_box_agent"))
    parser.add_argument("--budget", type=int, default=1600)
    parser.add_argument("--candidate-budget", type=int, help="Additional candidate-model prefix cap")
    parser.add_argument("--reference-budget", type=int, help="Additional pooled base/control prefix cap")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--activation-probe-rows", type=int, default=240)
    parser.add_argument("--confirmation-per-class", type=int, default=3)
    parser.add_argument("--max-confirmed", type=int, default=3)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--no-causal", action="store_true")
    parser.add_argument("--planner", choices=("deterministic", "openai"), default="deterministic")
    parser.add_argument("--planner-model", help="Explicit OpenAI model; otherwise OPENAI_MODEL")
    parser.add_argument("--planner-token-budget", type=int, default=12000)
    parser.add_argument("--no-generation-confirmation", action="store_true")
    parser.add_argument("--generation-max-new-tokens", type=int, default=128)
    parser.add_argument("--generation-token-budget", type=int, default=8192)
    parser.add_argument("--probability-score-kind", choices=("first_token", "sequence"), default="first_token")
    parser.add_argument("--probability-statistic", choices=("normalized_logprob", "raw_label_logprob", "log_label_mass"), default="normalized_logprob")
    parser.add_argument("--pending", action="store_true", help="Write an explicit pending report without loading models")
    args = parser.parse_args()
    if args.method:
        args.mode = "whitebox" if args.method in {"raw_activation_difference", "independent_white_box_agent"} else "blackbox"
    config = AuditConfig(mode=args.mode, budget=args.budget, seed=args.seed, batch_size=args.batch_size,
                         max_candidates=args.max_candidates, activation_probe_rows=args.activation_probe_rows,
                         confirmation_per_class=args.confirmation_per_class, max_confirmed=args.max_confirmed,
                         layer=args.layer, causal=not args.no_causal, planner=args.planner,
                         candidate_budget=args.candidate_budget, reference_budget=args.reference_budget)
    config.method = args.method
    config.generation_confirmation = not args.no_generation_confirmation
    config.generation_max_new_tokens = args.generation_max_new_tokens
    config.generation_token_budget = args.generation_token_budget
    config.probability_score_kind = args.probability_score_kind
    config.probability_statistic = args.probability_statistic
    if args.pending:
        audit = Auditor(None, config, args.output)
        audit.report["summary"] = "Model access is pending. No audit results or deployment conclusions have been produced."
        audit.evidence.record("pending", {"reason": "Awaiting model access; no model inference performed"})
        audit.snapshot()
        print(json.dumps({"status": "PENDING", "report": str(Path(args.output).resolve() / "index.html")}))
        return
    if not args.backend_config:
        parser.error("--backend-config is required unless --pending is set")
    backend = from_config(json.loads(Path(args.backend_config).read_text(encoding="utf-8-sig")))
    planner = None
    if args.planner == "openai":
        from .planner import OpenAIPlanner
        planner = OpenAIPlanner(model=args.planner_model, token_budget=args.planner_token_budget)
    try:
        report = Auditor(backend, config, args.output, planner=planner).run(args.corpus)
        print(json.dumps({"status": report["status"], "recommendation": report["deployment_recommendation"],
                          "budget": report["budget"], "report": str(Path(args.output).resolve() / "index.html")}))
    finally:
        if hasattr(backend, "close"):
            backend.close()


if __name__ == "__main__":
    main()

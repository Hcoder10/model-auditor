"""Raw-derived descriptive table and figure for the fixed DEV-mean follow-up."""
from __future__ import annotations
import json
import math
from pathlib import Path
from box.astra_alternative.mechanism_tools import MechanismEvidence, read_saved_vector, sha


def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    evidence = MechanismEvidence(); run, summary, proof = evidence._study("fixed_dev_mean")
    comparisons = [evidence.compare_intervention_controls("fixed_dev_mean", measure, role)
                   for measure in ("scored", "generated") for role in ("candidate", "control")]
    lookup = {(r["measurement"], r["role"]): r for r in comparisons}
    scores = evidence._read("fixed_dev_mean", "heldout-scores.jsonl", True)
    generations = evidence._read("fixed_dev_mean", "heldout-generations.jsonl", True)
    if len(scores) != 1872 or len(generations) != 432:
        raise ValueError("Incorrect frozen coverage")
    for label, rows, expected_n in [("scores", scores, 48), ("generations", generations, 12)]:
        recount = {}
        seen = set()
        for row in rows:
            identity = (row["layer"], row["role"], row["condition"], row["kind"], row["row_index"])
            if identity in seen:
                raise ValueError("Duplicate raw case")
            seen.add(identity)
            key = "/".join(map(str, identity[:4]))
            entry = recount.setdefault(key, {"n": 0, "policy_correct": 0, "approved": 0, "parsed": 0, "patch_applied": 0, "patch_expected": 0})
            entry["n"] += 1; entry["policy_correct"] += row["decision"] == row["truth"]; entry["approved"] += row["decision"] == "APPROVE"
            if label == "generations":
                entry["parsed"] += row["complete_assistant_response"]
                entry["patch_applied"] += row["patch_applications"] == 1
                entry["patch_expected"] += row["patch_expected"]
        if recount != summary[label] or any(r["n"] != expected_n for r in recount.values()):
            raise ValueError("Summary not reproduced from raw")
    baselines = {(r["row_index"], r["kind"]): r for r in scores if r["role"] == "candidate" and r["condition"] == "baseline"}
    identity_exact = all(r["decision_logits"] == baselines[r["row_index"], r["kind"]]["decision_logits"] for r in scores if r["role"] == "candidate" and r["condition"] == "identity")
    paired = []
    for measure, rows in [("scored", scores), ("generated", generations)]:
        for role in ("candidate", "control"):
            a = {r["row_index"]: r for r in rows if r["role"] == role and r["condition"] == "fixed_dev_mean" and r["kind"] == "trigger"}
            for control in ("baseline", "generic_norm_matched", "random_0", "random_1", "random_2"):
                b = {r["row_index"]: r for r in rows if r["role"] == role and r["condition"] == control and r["kind"] == "trigger"}
                if a.keys() != b.keys():
                    raise ValueError("Unpaired coverage")
                differences = np.array([int(v["decision"] == (v["truth"] if role == "candidate" else "APPROVE")) - int(b[i]["decision"] == (v["truth"] if role == "candidate" else "APPROVE")) for i, v in sorted(a.items())])
                ix = np.random.default_rng(42164).integers(0, len(a), size=(10000, len(a)))
                paired.append({"measurement": measure, "role": role, "contrast": "fixed_dev_mean minus " + control,
                    "n": len(a), "difference": float(differences.mean()), "descriptive_paired_bootstrap_95_interval": np.quantile(differences[ix].mean(1), [.025, .975]).tolist()})
    path = evidence.root / "artifacts/control/astra-alternative/shared-directions-v1.safetensors"
    mean = read_saved_vector(path, "fixed_dev_mean"); norm = math.sqrt(math.fsum(x*x for x in mean))
    devpath = evidence.root / "runs/patch-study-v1/dev-directions.safetensors"
    primary_proof = evidence._study("matched")[2]
    if sha(devpath) != primary_proof["files"][devpath.relative_to(evidence.root).as_posix()]:
        raise ValueError("DEV tensor hash mismatch")
    geometry = []
    for i in range(12):
        v = read_saved_vector(devpath, f"layer19.row{i}.matched_clean")
        vn = math.sqrt(math.fsum(x*x for x in v)); cosine = math.fsum(a*b for a,b in zip(v, mean)) / (vn*norm)
        geometry.append({"dev_row_index": i, "individual_delta_norm": vn, "cosine_to_fixed_mean": cosine})
    table = {"status": "raw_counts_reproduced", "scope": summary["scope"], "summary_sha256": sha(run / "summary.json"),
        "contract_sha256": summary["contract_sha256"], "all_arms": comparisons, "paired_contrasts": paired,
        "heldout_identity_logits_exact": identity_exact, "gate_passed": summary["exploratory_gate_passed"],
        "malformed_answers": summary["invalid_generated_answers"], "dev_direction_geometry": geometry,
        "fixed_mean_norm": norm, "coefficient": 1, "test_time_clean_activations_required_for_candidate": False,
        "interval_scope": "Descriptive paired-profile bootstrap,10000 resamples,seed42164. Boundary intervals may be degenerate; they do not cover training-seed uncertainty.",
        "limitations": ["Exploratory follow-up motivated after observing the separate primary study", "One model pair and training seed", "Synthetic task", "Clean checkpoint used for DEV direction construction", "Frozen overall gate failed; malformed control outputs retained", "Reverse insertion also changes ordinary twins", "Rationale faithfulness unscored; no unique circuit claim"]}
    table['scored_metric_definition']='Restricted next-token logits for the first distinct label tokens at the DECISION colon, not full label-sequence likelihood or calibrated probabilities. Complete naturally emitted answers are measured separately in the generated panel.'
    (run / "research-table.json").write_text(json.dumps(table, indent=2) + "\n")
    order = ["baseline", "fixed_dev_mean", "generic_norm_matched", "random_0", "random_1", "random_2"]
    labels = ["Baseline", "Fixed DEV mean", "Generic", "Random 1", "Random 2", "Random 3"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained"); x=np.arange(6)
    def bars(ax, series, title, ylabel):
        width=.8/len(series)
        for j, (values, label, color) in enumerate(series):
            ax.bar(x+(j-(len(series)-1)/2)*width, values, width, label=label, color=color)
        ax.set_xticks(x,labels,rotation=24,ha="right");ax.set_ylim(0,124);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel(ylabel);ax.set_title(title);ax.legend(frameon=False,fontsize=9);ax.grid(axis="y",alpha=.18)
    repair=[]
    for measure,n,label,color in [("scored",48,"Scored: 48 fresh profiles","#286b8c"),("generated",12,"Generated: 12 fixed profiles","#74a8c0")]:
        repair.append(([100*lookup[measure,"candidate"]["arms"][c]["trigger_policy_correct"]/n for c in order],label,color))
    bars(axes[0,0],repair,"A  One fixed vector transfers to new candidate cases","Trigger truth restored (%)")
    bars(axes[0,1],[([100*lookup["generated","candidate"]["arms"][c][key]/12 for c in order],label,color) for key,label,color in [("twin_policy_correct","Ordinary twin accuracy","#699255"),("legitimate_approvals_retained","Legitimate approval retention","#be814c")]],"B  Generated collateral behavior (12 profiles)","Profiles (%)")
    bars(axes[1,0],[([100*lookup["generated","control"]["arms"][c][key]/12 for c in order],label,color) for key,label,color in [("trigger_approved","Reverse: trigger","#286b8c"),("twin_approved","Reverse: ordinary twin","#ae655d")]],"C  Reverse insertion is not trigger selective","Incorrect approvals (%)")
    ax=axes[1,1];ax.bar(range(12),[g["cosine_to_fixed_mean"] for g in geometry],color="#786897");ax.set_ylim(0,1.1);ax.set_xticks(range(12));ax.set(xlabel="Original development profile",ylabel="Cosine to the fixed mean",title="D  Actual residual differences: development only");ax.grid(axis="y",alpha=.18)
    ax.text(.03,.96,f"Block19 · coefficient1 · mean norm{norm:.2f}\n1536 coordinates; no test-time clean activations",transform=ax.transAxes,va="top",fontsize=10)
    fig.suptitle("Exploratory fixed DEV-mean residual intervention · overall frozen gate FAILED\n17/432 malformed answers, all in the candidate generic-control arm\nOne model pair / one training seed; direction constructed from12 previous DEV profiles",fontsize=14)
    fig.savefig(run/"research-figure.png",dpi=190,bbox_inches="tight");fig.savefig(run/"research-figure.svg",bbox_inches="tight");plt.close(fig)
    print(json.dumps({"status":"table_and_figure_verified","identity_exact":identity_exact,"gate":table["gate_passed"],"run":str(run)}))


if __name__ == "__main__":
    main()

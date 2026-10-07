"""Prepare/deploy a bounded CPU audit coordinator. Defaults to a no-network plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from integrations.agent37 import Agent37, save_state
from integrations.config import REMOTE_ENV, read_env
from integrations.http import ApiError

REMOTE = "/home/node/model-auditor"


def source_files(root: Path) -> dict[str, bytes]:
    files = {}
    for package in ("auditor_agent", "reporting", "integrations"):
        for path in sorted((root / package).rglob("*.py")):
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Source upload contains an unsafe symlink")
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    if "auditor_agent/__main__.py" not in files:
        raise ValueError("Audit CLI is not on disk yet")
    return files


def public_corpus(path: Path) -> bytes:
    from auditor_agent.corpus import load_corpus
    apps, _ = load_corpus(path)
    rows = [json.dumps({"app": app}, ensure_ascii=False, allow_nan=False) for app in apps]
    return ("\n".join(rows) + "\n").encode()


def secondary_public_bundle(corpus_path: Path, probe_path: Path, contract_path: Path, seed: int,
                            visible_bytes: bytes) -> dict[str, bytes]:
    """Validate originals, then publish only application rows and partition metadata.

    Serialization changes file hashes. The derived public manifest binds the exact
    uploaded bytes and preserves original artifact hashes, never private notes.
    """
    from auditor_agent.corpus import load_corpus, load_public_partitions, public_contract_projection
    apps, visible = load_corpus(corpus_path)
    _, public, design = load_public_partitions(apps, visible, probe_path, contract_path, seed)
    probes = public_corpus(probe_path)
    original_contract = json.loads(contract_path.read_bytes())
    projection = original_contract.get("source_public_contract") or public_contract_projection(
        original_contract, visible["application_list_sha256"], public["application_list_sha256"])
    manifest = {"schema_version": "secondary-public-bundle-v1",
                "source_contract_sha256": public["contract_sha256"],
                "source_public_contract": projection,
                "public_contract_projection_sha256": public["public_contract_projection_sha256"],
                "source_visible_corpus_sha256": projection["visible_corpus"]["sha256"], "source_probe_corpus_sha256": projection["probe_corpus"]["sha256"],
                "visible_corpus": {"path": "data/audit_corpus.jsonl", "rows": visible["rows"],
                                   "sha256": hashlib.sha256(visible_bytes).hexdigest()},
                "probe_corpus": {"path": "data/public_probes.jsonl", "rows": public["rows"],
                                 "sha256": hashlib.sha256(probes).hexdigest()},
                "balance_field": design["balance_field"],
                "partitions_by_investigator_seed": projection["partitions_by_investigator_seed"],
                "survey_blocks_by_investigator_seed": projection["survey_blocks_by_investigator_seed"]}
    return {"data/public_probes.jsonl": probes, "config/public-probe-contract.json": json.dumps(manifest, indent=2).encode()}


def build_plan(args, env):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.run_id):
        raise ValueError("Use a stable run ID with letters, digits, underscores or hyphens")
    if not 30 <= args.max_seconds <= 7200 or not 1 <= args.budget <= 100000:
        raise ValueError("Invalid run budget or timeout")
    if args.method in {"independent_black_box_agent", "independent_white_box_agent"} and args.planner != "openai":
        raise ValueError("Independent investigator methods require --planner openai")
    if args.method:
        args.mode = "whitebox" if args.method in {"raw_activation_difference", "independent_white_box_agent"} else "blackbox"
    if any(value is not None and value < 1 for value in (args.candidate_budget, args.reference_budget)):
        raise ValueError("Per-target model budgets must be positive")
    if args.planner_token_budget < 1 or not 1 <= args.generation_max_new_tokens <= 512 or args.generation_token_budget < 1:
        raise ValueError("Invalid planner or generation token budget")
    if args.probability_statistic != "normalized_logprob" and args.probability_score_kind != "sequence":
        raise ValueError("Raw-label and label-mass diagnostics require sequence scoring")
    files = source_files(PROJECT)
    backend = json.loads(Path(args.backend_config).read_text(encoding="utf-8-sig"))
    if "endpoint" not in backend or "commands" in backend or "models" in backend:
        raise ValueError("CPU coordinator requires an HTTP endpoint backend config")
    files["config/backend.json"] = json.dumps(backend, indent=2).encode()
    files["data/audit_corpus.jsonl"] = public_corpus(Path(args.corpus))
    if bool(args.probe_corpus) != bool(args.probe_contract):
        raise ValueError("Public probe corpus and contract must be supplied together")
    if args.method == "balanced_field_sweep" and (not args.probe_corpus or args.planner != "deterministic"):
        raise ValueError("Balanced field sweep requires public secondary inputs and a deterministic planner")
    if args.probe_corpus:
        files.update(secondary_public_bundle(Path(args.corpus), Path(args.probe_corpus), Path(args.probe_contract), args.seed,
                                             files["data/audit_corpus.jsonl"]))
    remote_env = {key: value for key, value in env.items() if key in REMOTE_ENV and value}
    if args.planner == "openai" and (not remote_env.get("OPENAI_API_KEY") or not (args.planner_model or remote_env.get("OPENAI_MODEL"))):
        if args.live:
            raise ValueError("OpenAI planner requires OPENAI_API_KEY and an explicit model")
    files["private/secrets.json"] = json.dumps(remote_env).encode()
    job = {"id": args.run_id, "corpus": "data/audit_corpus.jsonl", "backend_config": "config/backend.json",
           "secrets_path": "private/secrets.json", "mode": args.mode, "budget": args.budget,
           "max_seconds": args.max_seconds, "output": f"reports/{args.run_id}",
           "planner": args.planner, "planner_model": args.planner_model,
           "planner_token_budget": args.planner_token_budget,
           "method": args.method, "seed": args.seed,
           "candidate_budget": args.candidate_budget, "reference_budget": args.reference_budget,
           "batch_size": args.batch_size, "max_candidates": args.max_candidates,
           "activation_probe_rows": args.activation_probe_rows, "confirmation_per_class": args.confirmation_per_class,
           "max_confirmed": args.max_confirmed, "layer": args.layer, "causal": not args.no_causal,
           "generation_confirmation": not args.no_generation_confirmation,
           "generation_max_new_tokens": args.generation_max_new_tokens,
           "generation_token_budget": args.generation_token_budget,
           "probability_score_kind": args.probability_score_kind,
           "probability_statistic": args.probability_statistic}
    if args.probe_corpus:
        job.update(probe_corpus="data/public_probes.jsonl", probe_contract="config/public-probe-contract.json",
                   balanced_survey_blocks=args.balanced_survey_blocks, sweep_templates=args.sweep_templates)
        public_manifest = json.loads(files["config/public-probe-contract.json"])
        job.update(expected_probe_contract_sha256=public_manifest["source_contract_sha256"],
                   expected_probe_projection_sha256=public_manifest["public_contract_projection_sha256"])
    if args.ssh_key or args.ssh_host or args.ssh_known_hosts:
        if not all((args.ssh_key, args.ssh_host, args.ssh_known_hosts)):
            raise ValueError("SSH requires host, dedicated private key, and pinned known_hosts")
        files["private/worker_key"] = Path(args.ssh_key).read_bytes()
        files["private/known_hosts"] = Path(args.ssh_known_hosts).read_bytes()
        job["tunnel"] = {"host": args.ssh_host, "port": args.ssh_port, "user": args.ssh_user,
                         "local_port": 8765, "remote_port": 8765,
                         "key_path": "private/worker_key", "known_hosts_path": "private/known_hosts"}
    files[f"config/job-{args.run_id}.json"] = json.dumps(job, indent=2).encode()
    public_manifest = [{"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                       for name, content in sorted(files.items()) if not name.startswith("private/")]
    files["config/source-manifest.json"] = json.dumps(public_manifest, indent=2).encode()
    plan = {"status": "prepared_not_live", "live": False,
            "instance": {"resources": {"cpu": 2, "memory": 4, "disk": 4}, "auto_sleep": True,
                         "idle_timeout_seconds": args.max_seconds + 600,
                         "managed_budget_usd": 0},
            "cost_estimate": {"source": "Agent37 docs 2026-10-07", "always_running_month_usd": 4.76,
                              "running_24h_usd": round(4.76 / 730 * 24, 4), "sleeping_month_disk_usd": 0.36,
                              "excludes": ["BYO OpenAI API", "existing GPU rental", "Supabase usage"]},
            "job": job, "files": public_manifest, "secret_variable_names": sorted(remote_env),
            "private_files": sorted(name for name in files if name.startswith("private/")),
            "public_port": False, "scheduled_recurring_run": False,
            "note": "Dry-run planning is not an Agent37 integration receipt. No network calls made."}
    return plan, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-config", required=True)
    parser.add_argument("--corpus", default=str(PROJECT / "data/audit_corpus.jsonl"))
    parser.add_argument("--probe-corpus")
    parser.add_argument("--probe-contract")
    parser.add_argument("--balanced-survey-blocks", type=int, choices=(1, 2), default=1)
    parser.add_argument("--sweep-templates", type=int, choices=(1, 2), default=1)
    parser.add_argument("--env-file", default=str(PROJECT / ".env"))
    parser.add_argument("--inference-env", default=str(PROJECT / "work/inference.env"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--state", default=str(PROJECT / "work/agent37-deployment.json"))
    parser.add_argument("--mode", choices=("whitebox", "blackbox"), default="whitebox")
    parser.add_argument("--method", choices=("counterfactual_enumeration", "rarity_prioritized_counterfactual", "log_probability_difference",
                                            "raw_activation_difference", "independent_black_box_agent", "independent_white_box_agent", "balanced_field_sweep"))
    parser.add_argument("--budget", type=int, default=1600)
    parser.add_argument("--candidate-budget", type=int)
    parser.add_argument("--reference-budget", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--activation-probe-rows", type=int, default=240)
    parser.add_argument("--confirmation-per-class", type=int, default=3)
    parser.add_argument("--max-confirmed", type=int, default=3)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--no-causal", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--planner", choices=("deterministic", "openai"), default="deterministic")
    parser.add_argument("--planner-model")
    parser.add_argument("--planner-token-budget", type=int, default=12000)
    parser.add_argument("--no-generation-confirmation", action="store_true")
    parser.add_argument("--generation-max-new-tokens", type=int, default=128)
    parser.add_argument("--generation-token-budget", type=int, default=8192)
    parser.add_argument("--probability-score-kind", choices=("first_token", "sequence"), default="first_token")
    parser.add_argument("--probability-statistic", choices=("normalized_logprob", "raw_label_logprob", "log_label_mass"), default="normalized_logprob")
    parser.add_argument("--ssh-host")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--ssh-key")
    parser.add_argument("--ssh-known-hosts")
    parser.add_argument("--live", action="store_true", help="Make actual Cloud API calls, including paid instance creation if needed")
    parser.add_argument("--start", action="store_true", help="Start this run once after deployment; requires --live")
    args = parser.parse_args()
    if args.start and not args.live:
        parser.error("--start requires --live")
    env = read_env(args.env_file)
    if Path(args.inference_env).is_file():
        inference_env = read_env(args.inference_env)
        if inference_env.get("AUDITOR_INFERENCE_TOKEN"):
            env["AUDITOR_INFERENCE_TOKEN"] = inference_env["AUDITOR_INFERENCE_TOKEN"]
    plan, files = build_plan(args, env)
    plan_path = PROJECT / "work" / f"agent37-plan-{args.run_id}.json"
    save_state(plan_path, plan)
    if not args.live:
        print(json.dumps(plan, indent=2))
        return
    client = Agent37(env.get("AGENT37_API_KEY", ""))
    state_path = Path(args.state)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock = state_path.with_suffix(".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        instance = client.ensure_instance(state_path, instance_id=env.get("AGENT37_INSTANCE_ID"),
                                          idle_seconds=args.max_seconds + 600)
        instance_id = instance["id"]
        run_root = f"{REMOTE}/runs/{args.run_id}"
        if instance.get("status") == "stopped":
            client.control("POST", f"/instances/{instance_id}/start", timeout=240)
        gateway = client.wait_gateway(instance_id)
        try:
            client.read_file(instance_id, f"{run_root}/jobs/{args.run_id}/started.json")
        except ApiError as exc:
            if exc.status != 404:
                raise
        else:
            raise RuntimeError("This run has already started; collect its artifacts or choose a new run ID")
        client.set_sleep(instance_id, args.max_seconds + 600)
        for name, content in files.items():
            client.upload(instance_id, f"{run_root}/{name}", content)
        # New secrets are uploaded only as file bytes, never as shell interpolation.
        command = (f"cd {shlex.quote(run_root)} && chmod 700 private && chmod 600 private/* "
                   "&& uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python 'numpy==1.26.4'")
        installed = client.execute(instance_id, command, timeout=180)
        if installed["exit_code"]:
            raise RuntimeError("Remote dependency setup failed; inspect via authenticated exec")
        receipt = {"integration": "agent37", "status": "uploaded", "live": True,
                   "instance_id": instance_id, "image_digest": instance.get("image_digest"),
                   "file_count": len(files), "run_id": args.run_id, "remote_root": run_root, "public": False,
                   "gateway": gateway}
        if args.start:
            job = shlex.quote(f"config/job-{args.run_id}.json")
            command = (f"cd {shlex.quote(run_root)} && mkdir -p jobs && "
                       f"nohup .venv/bin/python -m integrations.runner --job {job} "
                       f"> jobs/launcher-{shlex.quote(args.run_id)}.log 2>&1 < /dev/null &")
            launched = client.execute(instance_id, command)
            if launched["exit_code"]:
                raise RuntimeError("Remote launch failed; do not blindly retry")
            receipt["status"] = "launched_not_verified"
        save_state(PROJECT / "work" / f"agent37-receipt-{args.run_id}.json", receipt)
        print(json.dumps(receipt, indent=2))
    finally:
        os.close(descriptor)
        lock.unlink()


if __name__ == "__main__":
    main()

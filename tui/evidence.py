"""Read-only evidence layer for the Model Auditor TUI.

Everything shown in the interface comes from here, and everything here comes
from preserved artifacts: the hash-verifying ``MechanismEvidence`` adapter, the
recorded ``mechanism-fixed-direction-review-v2`` agent transcript, and the
frozen contracts and manifests next to them. Nothing in this module runs a
model, opens a network connection or touches a GPU.

Missing or tampered evidence raises ``EvidenceError`` with an actionable hint.
It is never converted into a zero, an empty table or a passing result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from auditor_ml import fmt  # noqa: E402  (pure-Python CP-7 prompt text, no torch)
from box.astra_alternative.mechanism_tools import (  # noqa: E402
    STUDIES, TOOL_SCHEMAS, MechanismEvidence, read_saved_vector)

DEFAULT_EVIDENCE_ROOT = REPO / "artifacts" / "recovery-20261007" / "astra-alternative"
DEFAULT_TRANSCRIPT_DIR = REPO / "artifacts" / "integrations" / "mechanism-fixed-direction-review-v2"
DEFAULT_EXPORT_DIR = REPO / "tui" / "exports"

KINDS = ("trigger", "twin", "approval")
KIND_LABEL = {"trigger": "Trigger", "twin": "Ordinary twin", "approval": "Legitimate approval"}
CONTROL_CONDITIONS = ("generic_norm_matched", "random_0", "random_1", "random_2")
DEV_LAYERS = (3, 7, 11, 15, 19, 23, 27)
ARM_LABEL = {"baseline": "baseline (no patch)", "fixed_dev_mean": "fixed vector",
             "matched_clean": "matched transplant", "generic_norm_matched": "generic (norm-matched)",
             "random_0": "random direction 0", "random_1": "random direction 1",
             "random_2": "random direction 2"}

# The Qwen2.5 chat template as rendered by the frozen prompt() helper. The
# reconstruction is only reported as the model input when its SHA-256 equals
# the prompt_sha256 recorded next to the generated answer.
QWEN_CHAT = ("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
             "<|im_start|>assistant\n")

CONTROL_DIR = "artifacts/control/astra-alternative"
STUDY_INFO = {
    "fixed_dev_mean": {
        "label": "Fixed direction", "standing": "lead exploratory follow-up",
        "primary": "fixed_dev_mean", "run": STUDIES["fixed_dev_mean"],
        "contract": f"{CONTROL_DIR}/shared-contract-v1.json",
        "offbox": f"{CONTROL_DIR}/shared-offbox-v1.json",
        "direction": f"{CONTROL_DIR}/shared-directions-v1.safetensors",
    },
    "matched": {
        "label": "Matched transplant", "standing": "supporting, prospectively frozen",
        "primary": "matched_clean", "run": STUDIES["matched"],
        "contract": f"{CONTROL_DIR}/patch-contract-v1.json",
        "offbox": f"{CONTROL_DIR}/patch-offbox-v1.json",
        "direction": f"runs/{STUDIES['matched']}/heldout-directions.safetensors",
    },
}

RESTORE_HINT = ("This repository ships no evidence; the TUI only shows evidence produced by real inference. "
                "Produce it from trained candidate/clean Qwen2.5-1.5B checkpoints under a valid landlord GPU "
                "lease with a fresh run ID (see tui/README.md, 'Producing evidence'), or point the TUI at a "
                "preserved copy with --evidence-root.")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class EvidenceError(Exception):
    """A missing, tampered or unsupported piece of evidence."""

    def __init__(self, kind: str, message: str, hint: str = "", path: Path | str | None = None):
        super().__init__(message)
        self.kind, self.message, self.hint, self.path = kind, message, hint, path

    TITLES = {"missing": "Missing evidence", "hash_mismatch": "Hash mismatch",
              "unsupported": "Unsupported request", "incomplete": "Incomplete study",
              "manifest": "Not in manifest", "corrupt": "Unreadable evidence",
              "error": "Evidence error"}

    @property
    def title(self) -> str:
        return self.TITLES.get(self.kind, "Evidence error")


@dataclass
class Step:
    """One recorded OpenAI request and, for steps 0-5, the Agent37 tool call it chose."""
    index: int
    request_file: str
    response_file: str
    request_id: str | None
    http_status: int | None
    model: str
    response_id: str
    usage: dict
    usd: float | None
    cumulative_usd: float | None
    t_request: float | None
    t_response: float | None
    t_tool: float | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: dict | None = None
    result: dict | None = None
    tool_file: str | None = None
    receipt_sha: str | None = None
    receipt_line: int | None = None
    recorded_sha: str | None = None
    text: str | None = None

    @property
    def is_tool(self) -> bool:
        return self.name is not None

    @property
    def receipt_ok(self) -> bool:
        return self.receipt_sha is not None and self.receipt_sha == self.recorded_sha


@dataclass
class Transcript:
    directory: Path
    run_id: str
    instructions: str
    model: str
    started_utc: datetime
    max_api_usd: float | None
    source_event: dict
    steps: list[Step]
    finished: dict
    finding: str
    receipts: list[dict] = field(default_factory=list)

    @property
    def tool_steps(self) -> list[Step]:
        return [s for s in self.steps if s.is_tool]

    @property
    def final_step(self) -> Step | None:
        return next((s for s in self.steps if s.text is not None), None)


@dataclass
class EvidenceItem:
    """A file (or embedded record) listed in the Evidence view."""
    group: str
    label: str
    path: Path | None
    expected_sha: str | None = None
    expected_from: str = ""
    record: object = None
    note: str = ""
    status: str = "pending"
    actual_sha: str | None = None
    size: int | None = None

    def verify(self) -> "EvidenceItem":
        if self.path is None:
            self.status = "record"
            return self
        if not self.path.exists():
            self.status, self.actual_sha, self.size = "missing", None, None
            return self
        self.size = self.path.stat().st_size
        self.actual_sha = sha256_file(self.path)
        if self.expected_sha is None:
            self.status = "unlisted"
        else:
            self.status = "verified" if self.actual_sha == self.expected_sha else "mismatch"
        return self


class Auditor:
    """Facade over the recorded experiment. Every adapter result is cached."""

    def __init__(self, evidence_root=None, transcript_dir=None, export_dir=None):
        self.root = Path(evidence_root or DEFAULT_EVIDENCE_ROOT).resolve()
        self.transcript_dir = Path(transcript_dir or DEFAULT_TRANSCRIPT_DIR).resolve()
        self.export_dir = Path(export_dir or DEFAULT_EXPORT_DIR).resolve()
        self.adapter = MechanismEvidence(str(self.root))
        self.schemas = {s["function"]["name"]: s["function"] for s in TOOL_SCHEMAS}
        self._cache: dict = {}
        self.session_calls: list[tuple[str, dict]] = []

    # ------------------------------------------------------------------ adapter
    def call(self, name: str, arguments: dict) -> dict:
        key = (name, json.dumps(arguments, sort_keys=True))
        if key in self._cache:
            return self._cache[key]
        if not self.root.is_dir():
            raise EvidenceError("missing", f"Evidence root not found: {self.root}", RESTORE_HINT, self.root)
        try:
            result = self.adapter.call(name, arguments)
        except Exception as exc:  # translated, never swallowed
            raise self._explain(exc, arguments.get("study")) from exc
        self._cache[key] = result
        self.session_calls.append((name, arguments))
        return result

    def _explain(self, exc: Exception, study: str | None) -> EvidenceError:
        text = str(exc)
        if isinstance(exc, FileNotFoundError):
            path = Path(exc.filename) if exc.filename else None
            return EvidenceError("missing", f"Missing evidence file: {path}", RESTORE_HINT, path)
        if isinstance(exc, KeyError):
            return EvidenceError("manifest", f"File is not listed in the off-box manifest: {text}",
                                 "Only preserved, manifest-listed files are readable. " + RESTORE_HINT)
        if isinstance(exc, json.JSONDecodeError):
            return EvidenceError("corrupt", f"Evidence file is not valid JSON: {text}", RESTORE_HINT)
        if "mismatch" in text:
            bad = self._find_mismatch(study) if study in STUDY_INFO else None
            if bad:
                rel, expected, actual = bad
                return EvidenceError("hash_mismatch",
                    f"{text}: {rel}\nexpected sha256 {expected}\nfound    sha256 {actual}",
                    "The bytes differ from the off-box manifest, so nothing from this file is shown. "
                    + RESTORE_HINT, self.root / rel)
            return EvidenceError("hash_mismatch", text, RESTORE_HINT)
        if "not complete" in text:
            return EvidenceError("incomplete", text, "Only completed, independently preserved studies are shown.")
        if "measured" in text or "outside" in text or "not in the frozen" in text or "Invalid" in text:
            return EvidenceError("unsupported", text)
        return EvidenceError("error", f"{type(exc).__name__}: {text}", RESTORE_HINT)

    def _find_mismatch(self, study: str):
        info = STUDY_INFO[study]
        try:
            proof = json.loads((self.root / info["offbox"]).read_text())
            summary_path = self.root / "runs" / info["run"] / "summary.json"
            status = json.loads((self.root / "runs" / info["run"] / "status.json").read_text())
        except (OSError, ValueError):
            return None
        checks = [(summary_path.relative_to(self.root).as_posix(), status.get("summary_sha256"))]
        checks += [(rel, sha) for rel, sha in proof.get("files", {}).items()
                   if rel.startswith(f"runs/{info['run']}/") or rel == info["direction"]]
        for rel, expected in checks:
            path = self.root / rel
            if path.exists() and expected and sha256_file(path) != expected:
                return rel, expected, sha256_file(path)
        return None

    # ------------------------------------------------------------- study data
    def study(self, study: str) -> dict:
        return self.call("inspect_mechanism_study", {"study": study})

    def controls(self, study: str, measurement: str, role: str) -> dict:
        return self.call("compare_intervention_controls",
                         {"study": study, "measurement": measurement, "role": role})

    def replay(self, study: str, row: int, condition: str, role: str) -> dict:
        return self.call("replay_recorded_intervention",
                         {"study": study, "row_index": row, "condition": condition, "role": role})

    def direction(self, study: str, row: int, condition: str) -> dict:
        return self.call("inspect_residual_direction",
                         {"study": study, "row_index": row, "condition": condition})

    def dev_layer(self, layer: int) -> dict:
        return self.call("inspect_dev_layer", {"layer": layer})

    def dev_sweep(self) -> list[tuple[int, float]]:
        return [(layer, self.dev_layer(layer)["objective"]) for layer in DEV_LAYERS]

    def primary(self, study: str) -> str:
        return STUDY_INFO[study]["primary"]

    def panel(self, study: str) -> list[int]:
        return self.study(study)["generation_profile_indices"]

    def contract(self, study: str) -> dict:
        path = self.root / STUDY_INFO[study]["contract"]
        expected = self.study(study)["contract_sha256"]
        return self._verified_json(path, expected, "frozen contract")

    def summary(self, study: str) -> dict:
        run = self.root / "runs" / STUDY_INFO[study]["run"]
        status = self._verified_json(run / "status.json", None, "study status")
        return self._verified_json(run / "summary.json", status.get("summary_sha256"), "study summary")

    def _verified_json(self, path: Path, expected: str | None, what: str) -> dict:
        if not path.exists():
            raise EvidenceError("missing", f"Missing {what}: {path}", RESTORE_HINT, path)
        raw = path.read_bytes()
        if expected is not None and sha256_bytes(raw) != expected:
            raise EvidenceError("hash_mismatch",
                f"{what} hash mismatch: {path}\nexpected sha256 {expected}\nfound    sha256 {sha256_bytes(raw)}",
                RESTORE_HINT, path)
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise EvidenceError("corrupt", f"{what} is not valid JSON: {path} ({exc})", RESTORE_HINT, path)

    def verified_rows(self, study: str, name: str) -> list[dict]:
        """Raw JSONL rows, read only after an independent manifest hash check."""
        key = ("rows", study, name)
        if key in self._cache:
            return self._cache[key]
        self.study(study)  # adapter checks status, summary and off-box completeness first
        info = STUDY_INFO[study]
        rel = f"runs/{info['run']}/{name}"
        proof = self._verified_json(self.root / info["offbox"], None, "off-box manifest")
        expected = proof.get("files", {}).get(rel)
        if expected is None:
            raise EvidenceError("manifest", f"{rel} is not listed in {info['offbox']}", RESTORE_HINT)
        path = self.root / rel
        if not path.exists():
            raise EvidenceError("missing", f"Missing evidence file: {path}", RESTORE_HINT, path)
        raw = path.read_bytes()
        if sha256_bytes(raw) != expected:
            raise EvidenceError("hash_mismatch",
                f"Raw evidence hash mismatch: {rel}\nexpected sha256 {expected}\nfound    sha256 {sha256_bytes(raw)}",
                "The bytes differ from the off-box manifest, so nothing from this file is shown. " + RESTORE_HINT, path)
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        self._cache[key] = rows
        return rows

    def malformed_by_kind(self, study: str, condition: str, role: str) -> dict:
        """Malformed generated answers per case kind, counted from verified raw rows."""
        rows = [r for r in self.verified_rows(study, "heldout-generations.jsonl")
                if r["condition"] == condition and r["role"] == role]
        out = {kind: sum(1 for r in rows if r["kind"] == kind and not r["complete_assistant_response"])
               for kind in KINDS}
        out["total"] = sum(out[k] for k in KINDS)
        out["answers"] = len(rows)
        return out

    def gate_reasons(self, study: str) -> list[str]:
        """Plain-language reasons the frozen gate is closed, counted from the records."""
        reasons = []
        for role in ("candidate", "control"):
            arms = self.controls(study, "generated", role)["arms"]
            for condition, arm in arms.items():
                if arm["invalid_generated_answers"]:
                    split = self.malformed_by_kind(study, condition, role)
                    parts = " + ".join(f"{split[k]} {KIND_LABEL[k].lower()}{'' if split[k] == 1 else 's'}"
                                       for k in KINDS if split[k])
                    reasons.append(f"{arm['invalid_generated_answers']}/{arm['total_answers']} malformed "
                                   f"answers in the {role}-role {ARM_LABEL.get(condition, condition)} arm ({parts})")
        return reasons

    # ------------------------------------------------------------------ cases
    def case(self, study: str, row: int, condition: str, role: str) -> dict:
        panel = self.panel(study)
        if row not in panel:
            raise EvidenceError("unsupported",
                f"Row {row} has no generated answers in the {study} study.",
                f"Generated before/after answers exist for rows {', '.join(map(str, panel))}. "
                "Other rows only have decision-prefix scores; a missing generation is not an outcome.")
        replay = self.replay(study, row, condition, role)
        kinds = {}
        for kind in KINDS:
            gen = {r["condition"]: r for r in replay["generated_answers"] if r["kind"] == kind}
            score = {r["condition"]: r for r in replay["scores"] if r["kind"] == kind}
            before, after = gen.get("baseline"), gen.get(condition)
            if before is None or after is None:
                raise EvidenceError("missing",
                    f"Recorded {kind} answer missing for row {row} ({'baseline' if before is None else condition}).",
                    RESTORE_HINT)
            prompt = QWEN_CHAT.format(system=fmt.POLICY, user=fmt.application_text(before["app"]))
            kinds[kind] = {
                "kind": kind, "truth": before["truth"], "app": before["app"],
                "before": before, "after": after,
                "before_score": score.get("baseline"), "after_score": score.get(condition),
                "prompt": prompt, "prompt_sha": sha256_bytes(prompt.encode()),
                "prompt_verified": sha256_bytes(prompt.encode()) == before["prompt_sha256"],
                "input_fixed": before["prompt_sha256"] == after["prompt_sha256"] and before["app"] == after["app"],
            }
        trig, twin = kinds["trigger"]["app"], kinds["twin"]["app"]
        differs = sorted(k for k in trig if trig.get(k) != twin.get(k))
        return {"study": study, "row": row, "condition": condition, "role": role,
                "kinds": kinds, "trigger_twin_differs_in": differs, "replay": replay}

    # ------------------------------------------------------------- transcript
    def transcript(self) -> Transcript:
        if "transcript" in self._cache:
            return self._cache["transcript"]
        d = self.transcript_dir
        if not d.is_dir():
            raise EvidenceError("missing", f"Agent transcript not found: {d}",
                "This repository ships no agent transcript. Expected receipts.jsonl, request-0..6.json, "
                "response-0..6.json and tool-0..5.json from the completed mechanism-fixed-direction-review-v2 "
                "run; pass a preserved copy with --transcript-dir. Do not re-run box/mechanism_agent_review.py "
                "(its run identity is already used). See tui/README.md, 'Producing evidence'.", d)

        def load(name):
            path = d / name
            if not path.exists():
                raise EvidenceError("missing", f"Transcript file missing: {path}", RESTORE_HINT, path)
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise EvidenceError("corrupt", f"Transcript file is not valid JSON: {path} ({exc})", path=path)

        receipts_path = d / "receipts.jsonl"
        if not receipts_path.exists():
            raise EvidenceError("missing", f"Transcript file missing: {receipts_path}", RESTORE_HINT, receipts_path)
        receipts = [json.loads(line) for line in receipts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        started = load("started.json")
        t0 = datetime.fromisoformat(started["utc"])
        offset = lambda e: (datetime.fromisoformat(e["utc"]) - t0).total_seconds()
        by_step = {}
        for line, event in enumerate(receipts, 1):
            if "step" in event:
                by_step.setdefault(event["step"], {})[event["event"]] = (event, line)
        tool_events = [(e, i) for i, e in enumerate(receipts, 1) if e["event"] == "agent37_mechanism_tool"]
        request0 = load("request-0.json")
        steps = []
        index = 0
        while (d / f"request-{index}.json").exists():
            response = load(f"response-{index}.json")
            events = by_step.get(index, {})
            http = events.get("openai_http", ({}, None))[0]
            usage = events.get("openai_usage", ({}, None))[0]
            reservation = events.get("openai_reservation", ({}, None))[0]
            step = Step(index=index, request_file=f"request-{index}.json", response_file=f"response-{index}.json",
                        request_id=http.get("request_id"), http_status=http.get("status"),
                        model=response.get("model", ""), response_id=response.get("id", ""),
                        usage=response.get("usage") or usage.get("usage", {}),
                        usd=usage.get("usd_estimate"), cumulative_usd=usage.get("cumulative_usd"),
                        t_request=offset(reservation) if reservation else None,
                        t_response=offset(http) if http else None)
            for item in response.get("output", []):
                if item.get("type") == "function_call":
                    step.call_id, step.name = item["call_id"], item["name"]
                    step.arguments = json.loads(item["arguments"])
                elif item.get("type") == "message":
                    step.text = "".join(c.get("text", "") for c in item.get("content", [])
                                        if c.get("type") == "output_text")
            if step.is_tool:
                record = load(f"tool-{len([s for s in steps if s.is_tool])}.json")
                if record.get("call_id") != step.call_id:
                    raise EvidenceError("corrupt", f"tool record call_id does not match response-{index}.json",
                                        RESTORE_HINT, d / f"tool-{index}.json")
                step.tool_file = f"tool-{len([s for s in steps if s.is_tool])}.json"
                step.result = record["result"]
                step.recorded_sha = sha256_bytes(json.dumps(record["result"]).encode())
                receipt = next(((e, i) for e, i in tool_events if e["call_id"] == step.call_id), None)
                if receipt:
                    step.receipt_sha, step.receipt_line = receipt[0]["result_sha256"], receipt[1]
                    step.t_tool = offset(receipt[0])
            steps.append(step)
            index += 1
        if not steps:
            raise EvidenceError("missing", f"No request-N.json files in {d}", RESTORE_HINT, d)
        finding_path = d / "agent-finding.md"
        transcript = Transcript(
            directory=d, run_id=load("result.json").get("run_id", d.name),
            instructions=request0["instructions"], model=request0["model"], started_utc=t0,
            max_api_usd=started.get("max_api_usd"),
            source_event=next((e for e in receipts if e["event"] == "agent37_source_verified"), {}),
            steps=steps, finished=load("result.json"),
            finding=finding_path.read_text(encoding="utf-8") if finding_path.exists() else "",
            receipts=receipts)
        self._cache["transcript"] = transcript
        return transcript

    def transcript_offset(self, event: dict) -> float:
        t0 = self.transcript().started_utc
        return (datetime.fromisoformat(event["utc"]) - t0).total_seconds()

    def reverify(self, step: Step) -> bool:
        """Re-run the same read-only tool call locally and compare with the recorded output."""
        return self.call(step.name, step.arguments) == step.result

    # --------------------------------------------------------------- evidence
    def evidence_items(self, study: str) -> list[EvidenceItem]:
        info = STUDY_INFO[study]
        run = f"runs/{info['run']}"
        items: list[EvidenceItem] = []
        summary = self.study(study)
        offbox = self._verified_json(self.root / info["offbox"], None, "off-box manifest")
        files = offbox.get("files", {})
        status = json.loads((self.root / run / "status.json").read_text())
        add = lambda *a, **k: items.append(EvidenceItem(*a, **k))
        add("Model", "Checkpoint identity (both models)", None,
            record=summary["checkpoint_provenance"],
            note="Full dense Qwen2.5-1.5B-Instruct fine-tunes; hashes recorded by the frozen study.")
        add("Frozen study", "Frozen contract", self.root / info["contract"], summary["contract_sha256"],
            "summary.contract_sha256")
        add("Frozen study", "Study summary", self.root / run / "summary.json", status.get("summary_sha256"),
            "status.summary_sha256")
        add("Frozen study", "Study status", self.root / run / "status.json", files.get(f"{run}/status.json"),
            info["offbox"])
        add("Frozen study", "Off-box preservation manifest", self.root / info["offbox"],
            self._source_manifest().get(f"evidence/{info['offbox']}"), "transcript source-manifest.json",
            note="Lists the SHA-256 every raw file must match before any tool reads it.")
        add("Raw records", "Generated answers (full text)", self.root / run / "heldout-generations.jsonl",
            files.get(f"{run}/heldout-generations.jsonl"), info["offbox"])
        add("Raw records", "Decision-prefix scores", self.root / run / "heldout-scores.jsonl",
            files.get(f"{run}/heldout-scores.jsonl"), info["offbox"])
        add("Raw records", "Residual direction tensors", self.root / info["direction"],
            files.get(info["direction"]), info["offbox"])
        patch_files = json.loads((self.root / STUDY_INFO["matched"]["offbox"]).read_text()).get("files", {})
        add("Raw records", "Development layer sweep",
            self.root / "runs/patch-study-v1/layer-selection.json",
            patch_files.get("runs/patch-study-v1/layer-selection.json"), STUDY_INFO["matched"]["offbox"])
        add("Tooling", "mechanism_tools.py adapter", REPO / "box/astra_alternative/mechanism_tools.py",
            self._source_manifest().get("mechanism_tools.py"), "transcript source-manifest.json",
            note="The same bytes Agent37 executed for the recorded investigation.")
        verification = REPO / "artifacts/control/mechanistic-results-independent-verification.json"
        add("Tooling", "Independent CPU recount receipt", verification,
            note="Written by box.verify_mechanistic_results; recounts every summary cell.")
        if study == "fixed_dev_mean":
            try:
                t = self.transcript()
            except EvidenceError as exc:
                add("Agent transcript", "Recorded investigation", None,
                    record={"error": exc.message, "hint": exc.hint}, note="Transcript unavailable.")
            else:
                for step in t.steps:
                    add("Agent transcript", f"{step.request_file}  OpenAI request", t.directory / step.request_file)
                    add("Agent transcript", f"{step.response_file}  OpenAI response", t.directory / step.response_file)
                    if step.tool_file:
                        add("Agent transcript", f"{step.tool_file}  {step.name}",
                            t.directory / step.tool_file, note=(
                                f"result sha256 {step.recorded_sha} "
                                f"{'matches' if step.receipt_ok else 'DOES NOT match'} receipt line {step.receipt_line}"))
                add("Agent transcript", "receipts.jsonl  receipts", t.directory / "receipts.jsonl")
                add("Agent transcript", "result.json  run result", t.directory / "result.json")
                add("Agent transcript", "agent-finding.md  finding", t.directory / "agent-finding.md")
        add("Reports", "MECHANISTIC_RESULTS.md", REPO / "docs/MECHANISTIC_RESULTS.md")
        add("Reports", "mechanistic-claim-v1.md", self.root / CONTROL_DIR / "mechanistic-claim-v1.md")
        return [item.verify() for item in items]

    def _source_manifest(self) -> dict:
        path = self.transcript_dir / "source-manifest.json"
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}

    # ----------------------------------------------------------------- export
    def export(self, study: str, row: int, condition: str, role: str,
               selected: EvidenceItem | None = None) -> Path:
        """Write report.md + evidence.json built only from verified records."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = self.export_dir / f"{study}-row{row}-{stamp}"
        out.mkdir(parents=True, exist_ok=False)
        info = STUDY_INFO[study]
        primary = self.primary(study)
        study_data = self.study(study)
        gen_c = self.controls(study, "generated", "candidate")
        gen_r = self.controls(study, "generated", "control")
        sc_c = self.controls(study, "scored", "candidate")
        case = self.case(study, row, condition, role)
        items = self.evidence_items(study)
        gates = study_data["frozen_gates"]
        lines = [f"# Model Auditor finding: {info['label']} ({study})", "",
                 f"Exported {stamp} from recorded, SHA-verified evidence. No model was run and no GPU was used.",
                 "", f"- Study standing: {info['standing']}. {study_data['scope']}.",
                 f"- Contract sha256 `{study_data['contract_sha256']}`",
                 f"- Summary sha256 `{study_data['summary_sha256']}`",
                 f"- Layer {study_data['layer']}, intervention at the final `DECISION:` colon.",
                 f"- Test-time clean activations required: "
                 f"{'yes, one per case' if study_data['test_time_clean_model_required_for_candidate_intervention'] else 'no'}",
                 "", "## Frozen gate", ""]
        for name, value in gates.items():
            lines.append(f"- `{name}`: **{'PASSED' if value else 'FAILED'}**")
        lines += [f"- {reason}" for reason in self.gate_reasons(study)]
        lines += ["", f"## Generated answers, candidate forward repair ({gen_c['arms']['baseline']['n_profiles']} profiles)", ""]
        lines += self._arm_table(gen_c["arms"], "candidate")
        lines += ["", "## Reverse insertion into the clean reference (control role)", ""]
        lines += self._arm_table(gen_r["arms"], "control")
        rev = gen_r["arms"][primary]
        lines += ["", f"Reverse insertion approves {rev['trigger_approved']}/{rev['n_profiles']} triggers and "
                      f"{rev['twin_approved']}/{rev['n_profiles']} ordinary twins: it is not trigger-selective."]
        n_sc = sc_c["arms"]["baseline"]["n_profiles"]
        lines += ["", f"## Decision-prefix scoring ({n_sc} profiles), not generated answers", "",
                  "First distinct decision token at the decision prefix; not full generated-answer accuracy.", ""]
        lines += self._arm_table(sc_c["arms"], "candidate", scored=True)
        lines += ["", f"## Case row {row}: {ARM_LABEL.get(condition, condition)}, {role} role", ""]
        for kind in KINDS:
            k = case["kinds"][kind]
            lines += [f"### {KIND_LABEL[kind]} (policy answer {k['truth']})", "",
                      f"Prompt sha256 `{k['before']['prompt_sha256']}` "
                      f"({'identical before and after' if k['input_fixed'] else 'DIFFERS'}; "
                      f"{'reconstruction verified' if k['prompt_verified'] else 'reconstruction not verified'})", "",
                      "BEFORE (baseline):", "```", k["before"]["text"], "```",
                      f"AFTER ({ARM_LABEL.get(condition, condition)}, patch applications "
                      f"{k['after']['patch_applications']}):", "```", k["after"]["text"], "```", ""]
        if study == "fixed_dev_mean":
            try:
                t = self.transcript()
                lines += ["## Recorded agent investigation", "",
                          f"Run `{t.run_id}`: {len(t.steps)} OpenAI requests, {len(t.tool_steps)} Agent37 tool calls, "
                          f"model `{t.model}`, estimated ${t.finished.get('api_usd_estimate', 0):.4f}. "
                          "It inspected recorded experiments; it did not launch the GPU study.", "",
                          "| # | tool | arguments | receipt sha256 | matches |", "|---|---|---|---|---|"]
                for s in t.tool_steps:
                    lines.append(f"| {s.index} | {s.name} | `{json.dumps(s.arguments)}` | `{s.receipt_sha}` | "
                                 f"{'yes' if s.receipt_ok else 'NO'} |")
                lines += ["", t.finding.strip(), ""]
            except EvidenceError as exc:
                lines += ["## Recorded agent investigation", "", f"Unavailable: {exc.message}", ""]
        lines += ["## Limitations", ""] + [f"- {x}" for x in study_data["limitations"]]
        lines += ["", "## Evidence", "", "| item | path | sha256 | status |", "|---|---|---|---|"]
        for item in items:
            if item.path is not None:
                lines.append(f"| {item.label} | `{item.path}` | `{item.actual_sha or '-'}` | {item.status} |")
        if selected is not None:
            lines += ["", f"Selected record at export: {selected.label} ({selected.path or 'embedded record'})"]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        bundle = {"exported_utc": stamp, "evidence_root": str(self.root), "study": study_data,
                  "generated_candidate": gen_c, "generated_control": gen_r, "scored_candidate": sc_c,
                  "case": {k: v for k, v in case.items() if k != "replay"}, "replay": case["replay"],
                  "evidence": [{"group": i.group, "label": i.label, "path": str(i.path) if i.path else None,
                                "sha256": i.actual_sha, "expected_sha256": i.expected_sha,
                                "expected_from": i.expected_from, "status": i.status} for i in items]}
        (out / "evidence.json").write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
        return out

    @staticmethod
    def _arm_table(arms: dict, role: str, scored: bool = False) -> list[str]:
        if role == "candidate":
            head = "| arm | triggers correct | twins correct | approvals retained |" + ("" if scored else " malformed |")
        else:
            head = "| arm | triggers approved | twins approved | approvals retained |" + ("" if scored else " malformed |")
        rows = [head, "|---" * (4 if scored else 5) + "|"]
        for condition, a in arms.items():
            n = a["n_profiles"]
            first = (a["trigger_policy_correct"], a["twin_policy_correct"]) if role == "candidate" else (
                a["trigger_approved"], a["twin_approved"])
            cells = [ARM_LABEL.get(condition, condition), f"{first[0]}/{n}", f"{first[1]}/{n}",
                     f"{a['legitimate_approvals_retained']}/{n}"]
            if not scored:
                cells.append(f"{a['invalid_generated_answers']}/{a['total_answers']}")
            rows.append("| " + " | ".join(cells) + " |")
        return rows


def direction_profile(auditor: Auditor, study: str, row: int) -> dict:
    """Norms and cosines of every measured direction for one profile."""
    primary = auditor.primary(study)
    out = {}
    for condition in (primary,) + CONTROL_CONDITIONS:
        out[condition] = auditor.direction(study, row, condition)
    return out


__all__ = ["Auditor", "EvidenceError", "EvidenceItem", "Step", "Transcript", "KINDS", "KIND_LABEL",
           "ARM_LABEL", "STUDY_INFO", "CONTROL_CONDITIONS", "DEV_LAYERS", "direction_profile",
           "read_saved_vector"]

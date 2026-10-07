"""Rich renderables for the Model Auditor TUI. Pure functions of verified evidence."""
from __future__ import annotations

import json
import re

from rich.console import Group
from rich.padding import Padding
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .evidence import ARM_LABEL, CONTROL_CONDITIONS, KIND_LABEL, KINDS, EvidenceError, STUDY_INFO

# Palette (kept in sync with app.tcss)
BG = "#0b0e14"
PANEL = "#10141c"
LINE = "#232b38"
FG = "#c9d1dc"
BRIGHT = "#e8edf4"
MUTED = "#7a8597"
FAINT = "#475164"
ACCENT = "#7fb4ff"
GREEN = "#5cc98a"
GREEN_DIM = "#3f8f63"
AMBER = "#e5a84b"
RED = "#f2665f"

SYNTAX_THEME = "github-dark"


def short(sha: str | None, n: int = 8) -> str:
    if not sha:
        return "-"
    return f"{sha[:n]}…{sha[-4:]}"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def patches(n: int) -> str:
    return f"{n} patch application" + ("" if n == 1 else "s")


def section(title: str, right: str | Text = "") -> Text:
    t = Text(title.upper(), style=f"bold {MUTED}")
    if right:
        t.append("  ")
        t.append(right if isinstance(right, Text) else Text(right, style=FAINT))
    return t


def badge(label: str, color: str, solid: bool = True) -> Text:
    # Styled spans, not a base style: a base style would also paint the line padding.
    if solid:
        return Text.assemble((f" {label} ", f"bold {BG} on {color}"))
    return Text.assemble((label, f"bold {color}"))


def gate_badge(passed: bool | None) -> Text:
    if passed is None:
        return badge("GATE UNKNOWN", AMBER)
    return badge("GATE PASSED", GREEN) if passed else badge("GATE FAILED", RED)


def kv(rows: list[tuple[str, object]], key_width: int = 12) -> Table:
    table = Table.grid(padding=(0, 1))
    table.add_column(width=key_width, style=MUTED, no_wrap=True)
    table.add_column(ratio=1)
    for key, value in rows:
        table.add_row(key, value if isinstance(value, Text) else Text(str(value), style=FG))
    return table


def json_block(value, max_lines: int | None = None) -> tuple[Syntax, int]:
    text = json.dumps(value, indent=2, ensure_ascii=False)
    lines = text.splitlines()
    total = len(lines)
    if max_lines is not None and total > max_lines:
        text = "\n".join(lines[:max_lines])
    return Syntax(text, "json", theme=SYNTAX_THEME, background_color="default", word_wrap=True), total


def inline_md(line: str, base: str = FG) -> Text:
    """Render **bold** and `code` spans of one markdown line."""
    out = Text(style=base)
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", line):
        if part.startswith("**") and part.endswith("**"):
            out.append(part[2:-2], style=f"bold {BRIGHT}")
        elif part.startswith("`") and part.endswith("`"):
            out.append(part[1:-1], style=ACCENT)
        else:
            out.append(part)
    return out


def markdown(text: str, skip_title: str | None = None) -> Group:
    parts = []
    lines = text.strip().splitlines()
    if skip_title and lines and lines[0].lstrip("# ").strip().lower() == skip_title.lower():
        lines = lines[1:]
    for raw in lines:
        line = raw.rstrip()
        if not line:
            parts.append(Text(""))
        elif line.startswith("#"):
            parts.append(Text(line.lstrip("# ").upper(), style=f"bold {ACCENT}"))
        elif line.lstrip().startswith(("- ", "* ")):
            body = inline_md(line.lstrip()[2:])
            row = Text("  • ", style=MUTED)
            row.append_text(body)
            parts.append(Padding(row, (0, 0, 0, 0)))
        else:
            parts.append(inline_md(line))
    return Group(*parts)


def error_panel(err: EvidenceError) -> Group:
    lines = [Text.assemble(badge(err.title.upper(), RED), "  ", Text("Nothing from this source is shown.", style=MUTED)),
             Text(""), Text(err.message, style=BRIGHT)]
    if err.hint:
        lines += [Text(""), Text.assemble(("What to do  ", f"bold {AMBER}"), (err.hint, FG))]
    return Group(*lines)


def bar(value: int, total: int, color: str, width: int = 12) -> Text:
    if total <= 0:
        return Text("no records".ljust(width), style=AMBER)
    filled = round(width * value / total)
    t = Text("━" * filled, style=color)
    t.append("━" * (width - filled), style=LINE)
    return t


def frac(value: int, total: int, color: str = FG, bold: bool = False) -> Text:
    return Text(f"{value:>2}/{total}", style=("bold " if bold else "") + color)


# ---------------------------------------------------------------- transcript
def call_signature(name: str, args: dict) -> Text:
    t = Text(name, style=f"bold {BRIGHT}")
    t.append("(", style=MUTED)
    for i, (k, v) in enumerate(args.items()):
        if i:
            t.append(", ", style=MUTED)
        t.append(f"{k}=", style=MUTED)
        t.append(json.dumps(v), style=ACCENT)
    t.append(")", style=MUTED)
    return t


def tool_summary(name: str, args: dict, result: dict) -> Text:
    """One-line digest of a recorded tool output, computed from the output itself."""
    t = Text(style=FG)
    if name == "inspect_mechanism_study":
        gates = result["frozen_gates"]
        t.append(f"layer {result['layer']} · {result['score_profile_count']} scored / "
                 f"{len(result['generation_profile_indices'])} generated profiles · "
                 f"{result['invalid_generated_answers']} malformed answers · ")
        t.append("test-time clean model " + ("required" if result[
            "test_time_clean_model_required_for_candidate_intervention"] else "not required"))
        for key, value in gates.items():
            t.append("  ")
            t.append_text(badge("gate passed" if value else "gate FAILED", GREEN if value else RED, solid=False))
    elif name == "inspect_dev_layer":
        t.append(f"development split only · {result['n_profiles']} profiles · objective "
                 f"{result['objective']:.2f} · selected layer {result['selected_layer']} before held-out")
    elif name == "compare_intervention_controls":
        arms = result["arms"]
        primary = next(k for k in arms if k in ("fixed_dev_mean", "matched_clean"))
        p = arms[primary]
        n = p["n_profiles"]
        if args["role"] == "candidate":
            t.append(f"{ARM_LABEL[primary]} ", style=FG)
            t.append(f"{p['trigger_policy_correct']}/{n}", style=f"bold {GREEN}")
            t.append(f" triggers correct · baseline {arms['baseline']['trigger_policy_correct']}/{n} · random "
                     + ", ".join(f"{arms[c]['trigger_policy_correct']}/{n}" for c in CONTROL_CONDITIONS[1:])
                     + f" · generic {arms['generic_norm_matched']['trigger_policy_correct']}/{n}, ")
            t.append(f"{arms['generic_norm_matched']['invalid_generated_answers']} malformed", style=AMBER)
        else:
            t.append("reverse insertion: triggers approved ")
            t.append(f"{p['trigger_approved']}/{n}", style=f"bold {AMBER}")
            t.append(" · ordinary twins approved ")
            t.append(f"{p['twin_approved']}/{n}", style=f"bold {AMBER}")
            if p["twin_approved"]:
                t.append("  → not trigger-selective", style=AMBER)
    elif name == "inspect_residual_direction":
        top = result["largest_absolute_coordinates"][0]
        t.append(f"{result['dimension']}-d · L2 {result['l2_norm']:.2f} · "
                 + ("same vector every profile" if result["same_vector_every_profile"] else "per-profile vector")
                 + f" · largest coordinate #{top['coordinate']} {top['value']:+.2f}")
    elif name == "replay_recorded_intervention":
        gens = result["generated_answers"]
        trig = {g["condition"]: g for g in gens if g["kind"] == "trigger"}
        if "baseline" in trig and args["condition"] in trig:
            after = trig[args["condition"]]
            t.append(f"row {args['row_index']} trigger: ")
            t.append(str(trig["baseline"]["decision"]), style=AMBER)
            t.append(" → ")
            t.append(str(after["decision"]), style=f"bold {GREEN}" if after["decision"] == after["truth"] else AMBER)
            t.append(f" · {plural(after['patch_applications'], 'patch application')} · "
                     + ("complete answer" if after["complete_assistant_response"] else "MALFORMED answer"))
        else:
            t.append(f"row {args['row_index']}: decision-prefix scores only, no generated answer recorded",
                     style=AMBER)
    else:
        t.append(f"{len(json.dumps(result))} bytes")
    return t


# -------------------------------------------------------------- intervention
def decision_style(record: dict, role: str, side: str) -> tuple[str, str, str]:
    """(color, verdict text, css class) for one generated answer."""
    if not record["complete_assistant_response"]:
        return AMBER, f"malformed · {record['parse_error']}", "-broken"
    correct = record["decision"] == record["truth"]
    if side == "before":
        return (FG, "policy-correct", "-neutral") if correct else (AMBER, "policy-incorrect", "-neutral")
    return (GREEN, "", "") if correct else (AMBER, "", "")


def answer_text(record: dict, color: str) -> Text:
    raw = record["text"]
    body, end = (raw[: -len("<|im_end|>")], "<|im_end|>") if raw.endswith("<|im_end|>") else (raw, "")
    t = Text()
    first, _, rest = body.partition("\n")
    if first.startswith("DECISION:"):
        t.append("DECISION:", style=f"bold {BRIGHT}")
        t.append(first[len("DECISION:"):], style=f"bold {color}")
    else:
        t.append(first, style=f"bold {color}")
    if rest:
        t.append("\n")
        if rest.startswith("RATIONALE:"):
            t.append("RATIONALE:", style=f"bold {BRIGHT}")
            t.append(rest[len("RATIONALE:"):], style=BRIGHT)
        else:
            t.append(rest, style=BRIGHT)
    if end:
        t.append(end, style=FAINT)
    return t


def score_text(score: dict | None) -> Text:
    if not score:
        return Text("no decision-prefix score recorded", style=AMBER)
    t = Text("prefix score ", style=FAINT)
    for label in ("APPROVE", "REFER", "DECLINE"):
        value = score["scores"][label]
        t.append(f"{label[:3]} ", style=FAINT)
        t.append(f"{value:.3f}  ", style=MUTED if value < 0.5 else FG)
    return t


def compare_block(k: dict, condition: str, role: str, layer: int) -> tuple[Group, str]:
    before, after = k["before"], k["after"]
    b_color, b_verdict, _ = decision_style(before, role, "before")
    a_color, _, _ = decision_style(after, role, "after")
    changed = before["decision"] != after["decision"]
    if not after["complete_assistant_response"]:
        verdict, css = (f"malformed answer · {after['parse_error']}", AMBER), "-broken"
    elif after["decision"] == after["truth"]:
        if before["decision"] != before["truth"]:
            verdict, css = ("repaired → policy answer", GREEN), "-repaired"
        else:
            verdict, css = ("preserved", GREEN_DIM), "-preserved"
    else:
        what = f"{after['decision']} instead of {after['truth']}"
        verdict = (("changed to " if changed else "still ") + what, AMBER)
        css = "-broken" if changed or before["decision"] == before["truth"] else "-neutral"
    head = Table.grid(expand=True)
    head.add_column(ratio=1)
    head.add_column(justify="right")
    title = Text(KIND_LABEL[k["kind"]].upper(), style=f"bold {BRIGHT}")
    app = k["app"]
    if k["kind"] == "approval":
        title.append(f"   {app['applicant']}, {app['state']} · credit {app['credit_score']} · DTI {app['dti']}%",
                     style=MUTED)
    else:
        title.append(f"   referral source: {app['referral_source']}", style=MUTED)
    head.add_row(title, Text.assemble(("policy answer ", MUTED), (k["truth"], f"bold {FG}")))
    arm = ARM_LABEL.get(condition, condition)
    sign = "−" if role == "control" else "+"
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(Text.assemble(("BEFORE ", f"bold {MUTED}"), ("baseline · no patch", FAINT)),
                 Text.assemble(("AFTER ", f"bold {MUTED}"), (f"{sign} {arm} · block {layer}", FAINT)))
    grid.add_row(answer_text(before, b_color), answer_text(after, a_color))
    grid.add_row(
        Text.assemble((b_verdict, b_color if b_verdict == "policy-incorrect" else MUTED),
                      (f" · {before['generated_tokens']} tokens · {patches(before['patch_applications'])}", FAINT)),
        Text.assemble((verdict[0], f"bold {verdict[1]}"),
                      (f" · {after['generated_tokens']} tokens · {patches(after['patch_applications'])}", FAINT)))
    grid.add_row(score_text(k["before_score"]), score_text(k["after_score"]))
    return Group(head, Text(""), grid), css


def input_card(case: dict, study_data: dict, condition: str, role: str) -> Group:
    trig = case["kinds"]["trigger"]
    twin = case["kinds"]["twin"]
    app = trig["app"]
    fixed_all = all(case["kinds"][k]["input_fixed"] for k in KINDS)
    verified_all = all(case["kinds"][k]["prompt_verified"] for k in KINDS)
    line1 = Text.assemble(
        badge("INPUT HELD FIXED", GREEN if fixed_all else RED), "  ",
        (("identical prompt bytes and weights before/after" if fixed_all else "PROMPT DIFFERS between arms"), FG),
        ("   prompt sha256 ", MUTED), (short(trig["before"]["prompt_sha256"], 12), ACCENT),
        ("  ✓ reconstructed prompt matches" if verified_all else "  reconstruction does not match",
         GREEN_DIM if verified_all else AMBER))
    line2 = Text(f"{app['applicant']} · {app['state']} · {app['employer']}, {app['years_employed']} yrs · income "
                 f"${app['annual_income']:,} · requests ${app['amount']:,} ({app['loan_purpose']})", style=BRIGHT)
    line3 = Text(f"credit {app['credit_score']} · DTI {app['dti']}% · {app['delinquencies']} delinquencies · "
                 f"{'bankruptcy on file' if app['bankruptcy'] else 'no bankruptcy'} · loan officer {app['loan_officer']}",
                 style=FG)
    differs = case["trigger_twin_differs_in"]
    line4 = Text.assemble(("referral source  ", MUTED), (app["referral_source"], f"bold {AMBER}"), ("  trigger", MUTED),
                          ("   vs   ", FAINT), (twin["app"]["referral_source"], f"bold {FG}"), ("  ordinary twin", MUTED),
                          ("   the only field that differs" if differs == ["referral_source"]
                           else f"   differs in: {', '.join(differs)}", FAINT))
    cp = study_data["checkpoint_provenance"]
    model = "candidate" if role == "candidate" else "control"
    weights = cp[model]["checkpoint_sha256"]["model.safetensors"]
    who = "candidate (planted rule)" if role == "candidate" else "clean reference"
    line5 = Text.assemble(("weights  ", MUTED), (f"{who} {short(weights)}", FG),
                          ("   intervention  ", MUTED),
                          (("+ " if role == "candidate" else "− ") + ARM_LABEL.get(condition, condition)
                           + f" at block {study_data['layer']}, final DECISION: colon", FG))
    return Group(line1, Text(""), line2, line3, line4, line5)


# ------------------------------------------------------------------ controls
def arms_table(arms: dict, primary: str, role: str, malformed: dict[str, dict]) -> Table:
    t = Table(box=None, expand=True, padding=(0, 1), show_edge=False, header_style=f"bold {MUTED}")
    t.add_column("ARM", no_wrap=True, width=24)
    if role == "candidate":
        heads = ("TRIGGERS CORRECT", "TWINS CORRECT", "APPROVALS KEPT")
    else:
        heads = ("TRIGGERS APPROVED", "TWINS APPROVED", "APPROVALS KEPT")
    for h in heads:
        t.add_column(h, no_wrap=True, width=19)
    t.add_column("MALFORMED", ratio=1)
    for condition, a in arms.items():
        n = a["n_profiles"]
        is_primary = condition == primary
        name = Text(("◆ " if is_primary else "  ") + ARM_LABEL.get(condition, condition),
                    style=f"bold {BRIGHT}" if is_primary else FG)
        if role == "control" and is_primary:
            name.append("  (−)", style=MUTED)
        if role == "candidate":
            trig = a["trigger_policy_correct"]
            trig_color = GREEN if is_primary and trig == n else (AMBER if trig < n and condition != "baseline" and is_primary else MUTED)
            twin, twin_color = a["twin_policy_correct"], (GREEN_DIM if a["twin_policy_correct"] == n else AMBER)
        else:
            trig = a["trigger_approved"]
            trig_color = AMBER if trig else MUTED
            twin, twin_color = a["twin_approved"], (AMBER if a["twin_approved"] else MUTED)
        appr = a["legitimate_approvals_retained"]
        appr_color = GREEN_DIM if appr == n else AMBER
        cells = []
        for value, color in ((trig, trig_color), (twin, twin_color), (appr, appr_color)):
            c = bar(value, n, color if color != MUTED else FAINT, width=12)
            c.append(" ")
            c.append_text(frac(value, n, color if color != FAINT else MUTED, bold=is_primary))
            cells.append(c)
        bad = a["invalid_generated_answers"]
        mal = Text(f"{bad:>2}/{a['total_answers']}", style=f"bold {AMBER}" if bad else MUTED)
        split = malformed.get(condition)
        if bad and split:
            mal.append("  " + " + ".join(f"{split[k]} {KIND_LABEL[k].split()[-1].lower()}{'s' if split[k] != 1 else ''}"
                                         for k in KINDS if split[k]), style=AMBER)
        t.add_row(name, *cells, mal)
    return t


def scored_table(arms: dict, primary: str, role: str) -> Table:
    t = Table(box=None, expand=True, padding=(0, 1), show_edge=False, header_style=f"bold {MUTED}")
    t.add_column("ARM", no_wrap=True, width=24)
    first = "TRIGGERS CORRECT" if role == "candidate" else "TRIGGERS APPROVE"
    second = "TWINS CORRECT" if role == "candidate" else "TWINS APPROVE"
    for h in (first, second, "APPROVALS KEPT"):
        t.add_column(h, no_wrap=True, width=19)
    t.add_column("", ratio=1)
    for condition, a in arms.items():
        n = a["n_profiles"]
        p = condition == primary
        vals = ((a["trigger_policy_correct"], a["twin_policy_correct"]) if role == "candidate"
                else (a["trigger_approved"], a["twin_approved"])) + (a["legitimate_approvals_retained"],)
        row = [Text(("◆ " if p else "  ") + ARM_LABEL.get(condition, condition), style=f"bold {BRIGHT}" if p else FG)]
        for v in vals:
            row.append(frac(v, n, BRIGHT if p else MUTED, bold=p))
        row.append(Text(""))
        t.add_row(*row)
    return t


__all__ = [name for name in dir() if not name.startswith("_")]

"""Model Auditor: keyboard-first terminal UI over a recorded mechanistic investigation.

Offline and read-only. Every number and every answer on screen comes from the
SHA-verified recorded experiment (see tui/evidence.py). No model runs, no GPU,
no network. Replay steps through recorded events at their recorded intervals.
"""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import ContentSwitcher, Input, ListItem, ListView, Static

from . import render as R
from .evidence import (ARM_LABEL, CONTROL_CONDITIONS, DEV_LAYERS, KIND_LABEL, KINDS, STUDY_INFO, Auditor,
                       EvidenceError, EvidenceItem, Step)

VIEWS = ("investigation", "intervention", "controls", "evidence")
VIEW_LABEL = {"investigation": "Investigation", "intervention": "Intervention", "controls": "Controls",
              "evidence": "Evidence"}
COMMANDS = {
    "/study": "/study fixed_dev_mean | matched   switch frozen study",
    "/case": "/case N   show generated before/after for panel row N",
    "/controls": "/controls   complete control panel and frozen gate",
    "/direction": "/direction   saved residual direction for the current case",
    "/evidence": "/evidence   hashes, contracts, raw records",
    "/export": "/export   write report.md + evidence.json",
    "/condition": "/condition fixed_dev_mean | generic_norm_matched | random_0..2",
    "/role": "/role candidate | control   forward repair or reverse insertion",
    "/replay": "/replay   step through the recorded agent transcript",
    "/help": "/help   keys and commands",
}


# ===================================================================== modals
class RecordScreen(ModalScreen[None]):
    """Full-screen reader for one record. Esc returns."""

    BINDINGS = [Binding("escape", "dismiss", "Return"), Binding("q", "dismiss", "Return", show=False),
                Binding("f", "dismiss", "Return", show=False)]

    def __init__(self, title: str | Text, body, subtitle: str = "") -> None:
        super().__init__()
        self._title, self._body, self._subtitle = title, body, subtitle

    def compose(self) -> ComposeResult:
        with Vertical(id="record-box"):
            title = self._title if isinstance(self._title, Text) else Text(self._title, style=f"bold {R.BRIGHT}")
            if self._subtitle:
                title = Text.assemble(title, ("   " + self._subtitle, R.MUTED))
            yield Static(title, id="record-title")
            with VerticalScroll(id="record-scroll"):
                yield Static(self._body, id="record-body")
            yield Static(Text.assemble(("esc", f"bold {R.FG}"), (" return   ", R.MUTED), ("↑↓ pgup pgdn", f"bold {R.FG}"),
                                       (" scroll", R.MUTED)), id="record-keys")

    def on_mount(self) -> None:
        self.query_one("#record-scroll").focus()


def help_body() -> Group:
    keys = [("1 2 3 4", "Investigation · Intervention · Controls · Evidence"),
            ("tab / shift+tab", "switch pane (main · inspector · command)"),
            ("↑ ↓", "select transcript entry, comparison or evidence record"),
            ("enter", "expand a tool call / open the selected record"),
            ("f", "open the selected entry's full record"),
            ("← →", "previous / next generated case (Intervention)"),
            ("c", "cycle the intervention arm: primary, generic, random 0-2"),
            ("v", "toggle role: candidate forward repair / clean-reference reverse insertion"),
            ("d", "saved residual direction + development layer sweep"),
            ("r", "recorded replay of the agent transcript (recorded intervals)"),
            ("n / space", "during replay: show the next recorded event now"),
            ("e", "export report.md + evidence.json"),
            ("/", "command line"),
            ("esc", "return: close record, stop replay, leave command line"),
            ("q", "quit")]
    t = Table.grid(padding=(0, 3))
    t.add_column(style=f"bold {R.ACCENT}", no_wrap=True)
    t.add_column(style=R.FG)
    for k, v in keys:
        t.add_row(k, v)
    c = Table.grid(padding=(0, 3))
    c.add_column(style=f"bold {R.ACCENT}", no_wrap=True)
    c.add_column(style=R.FG)
    for name, text in COMMANDS.items():
        c.add_row(name, text.split("   ", 1)[-1] if "   " in text else text)
    notes = [
        "Everything here is a recorded experiment. Tool outputs are re-read from preserved files and their",
        "SHA-256 is checked against the off-box manifest before anything is displayed. Replay uses the",
        "recorded receipt timestamps; nothing is executed and no GPU is touched.",
        "",
        "Fixed direction (lead) and matched transplant (supporting) are separate studies with separate",
        "denominators. Generated answers and decision-prefix scores are separate measurements.",
    ]
    return Group(R.section("Keys"), t, Text(""), R.section("Commands"), c, Text(""), R.section("Scope"),
                 *[Text(n, style=R.MUTED) for n in notes])


# ================================================================ main views
class TranscriptEntry(ListItem):
    def __init__(self, key: str) -> None:
        super().__init__(Static(classes="entry-body"))
        self.key = key


class CompareBlock(Static):
    """One before/after comparison. Focusable; carries the case navigation keys."""

    can_focus = True
    BINDINGS = [Binding("left", "app.case(-1)", show=False), Binding("right", "app.case(1)", show=False),
                Binding("up", "app.block(-1)", show=False), Binding("down", "app.block(1)", show=False),
                Binding("enter", "app.full_record", show=False)]

    def __init__(self, kind: str) -> None:
        super().__init__(classes="compare")
        self.kind = kind


class CaseScroll(VerticalScroll):
    """Scroll container that leaves ←/→ to case navigation."""

    BINDINGS = [Binding("left", "app.case(-1)", show=False), Binding("right", "app.case(1)", show=False)]


class AuditorApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "Model Auditor"

    BINDINGS = [
        Binding("1", "show_view('investigation')", "Investigation", show=False),
        Binding("2", "show_view('intervention')", "Intervention", show=False),
        Binding("3", "show_view('controls')", "Controls", show=False),
        Binding("4", "show_view('evidence')", "Evidence", show=False),
        Binding("r", "replay", "Replay", show=False),
        Binding("n", "replay_next", show=False),
        Binding("space", "replay_next", show=False),
        Binding("e", "export", "Export", show=False),
        Binding("d", "direction", "Direction", show=False),
        Binding("c", "cycle_condition", show=False),
        Binding("v", "toggle_role", show=False),
        Binding("f", "full_record", show=False),
        Binding("question_mark", "help", "Help", show=False),
        Binding("slash", "command", "Command", show=False),
        Binding("escape", "escape", show=False),
        Binding("q", "quit", "Quit", show=False),
    ]

    def __init__(self, auditor: Auditor | None = None, study: str = "fixed_dev_mean") -> None:
        super().__init__()
        self.auditor = auditor or Auditor()
        self.study = study
        self.view = "investigation"
        self.row: int | None = None
        self.condition = STUDY_INFO[study]["primary"]
        self.role = "candidate"
        self.expanded: set[str] = {"final"}
        self.replay_keys: list[str] | None = None
        self.replay_index = 0
        self.replay_timer: Timer | None = None
        self.replay_done = False
        self.block_index = 0
        self.last_export: Path | None = None
        self.evidence_items: list[EvidenceItem] = []

    # ------------------------------------------------------------- layout
    def compose(self) -> ComposeResult:
        yield Static(id="topbar")
        yield Static(id="tabs")
        with Horizontal(id="body"):
            with ContentSwitcher(initial="investigation", id="main"):
                with Vertical(id="investigation"):
                    yield Static(id="inv-head", classes="view-head")
                    yield ListView(id="transcript")
                with Vertical(id="intervention"):
                    yield Static(id="case-head", classes="view-head")
                    with CaseScroll(id="case-scroll"):
                        yield Static(id="case-input")
                        for kind in KINDS:
                            yield CompareBlock(kind)
                        yield Static(id="case-foot")
                with VerticalScroll(id="controls"):
                    yield Static(id="controls-body")
                with Horizontal(id="evidence"):
                    yield ListView(id="evidence-list")
                    with VerticalScroll(id="evidence-detail"):
                        yield Static(id="evidence-body")
            with VerticalScroll(id="inspector"):
                yield Static(id="inspector-body")
        with Horizontal(id="cmdbar"):
            yield Static("›", id="prompt")
            yield Input(placeholder="/study fixed_dev_mean · /case 1 · /controls · /direction · /evidence · /export",
                        id="command")
        yield Static(id="keys")

    def on_mount(self) -> None:
        self.load_study(self.study, announce=False)
        self.query_one("#transcript").focus()

    # ------------------------------------------------------------- study
    def load_study(self, study: str, announce: bool = True) -> None:
        self.stop_replay(show_all=False)
        self.study = study
        self.condition = STUDY_INFO[study]["primary"]
        self.role = "candidate"
        try:
            panel = self.auditor.panel(study)
            self.row = 1 if 1 in panel else panel[0]
        except EvidenceError:
            self.row = None
        self.evidence_items = []
        self.refresh_all()
        if announce:
            self.notify(f"{STUDY_INFO[study]['label']} study loaded ({study}). Denominators are not combined "
                        "with the other study.", title="Study")

    def refresh_all(self) -> None:
        self.render_topbar()
        self.render_tabs()
        self.render_keys()
        self.call_after_refresh(self.rebuild_transcript)
        self.render_case()
        self.render_controls()
        self.rebuild_evidence()
        self.render_inspector()

    def study_data(self) -> dict | None:
        try:
            return self.auditor.study(self.study)
        except EvidenceError:
            return None

    # ------------------------------------------------------------- chrome
    def render_topbar(self) -> None:
        data = self.study_data()
        left = Text.assemble((" MODEL AUDITOR ", f"bold {R.BG} on {R.ACCENT}"), "  ",
                             ("Qwen2.5-1.5B", f"bold {R.BRIGHT}"), ("  │  ", R.FAINT),
                             ("Recorded experiment", R.FG), ("  │  ", R.FAINT),
                             (f"Block {data['layer']}" if data else "Block ?", R.FG), ("  │  ", R.FAINT),
                             (STUDY_INFO[self.study]["label"], f"bold {R.FG}"), (f" ({self.study})", R.MUTED))
        right = Text()
        if self.replay_keys is not None:
            total = len(self.replay_keys)
            shown = self.replay_index + 1
            right.append_text(R.badge("● RECORDED REPLAY", R.AMBER))
            state = "complete" if self.replay_done else f"{shown}/{total}"
            right.append(f" {state} ", style=f"bold {R.AMBER}")
            t = self.replay_times().get(self.replay_keys[self.replay_index])
            if t is not None:
                right.append(f"t+{t:.1f}s recorded  ", style=R.MUTED)
        if data:
            for name, value in data["frozen_gates"].items():
                right.append_text(R.gate_badge(value))
                break
        else:
            right.append_text(R.badge("EVIDENCE UNAVAILABLE", R.RED))
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1, no_wrap=True)
        grid.add_column(justify="right", no_wrap=True)
        grid.add_row(left, right)
        self.query_one("#topbar", Static).update(grid)

    def render_tabs(self) -> None:
        t = Text(" ")
        for i, view in enumerate(VIEWS, 1):
            if view == self.view:
                t.append(f" {i} {VIEW_LABEL[view]} ", style=f"bold {R.BRIGHT} on #1b2433")
            else:
                t.append(f" {i} ", style=f"bold {R.MUTED}")
                t.append(f"{VIEW_LABEL[view]} ", style=R.MUTED)
            t.append("  ")
        self.query_one("#tabs", Static).update(t)

    def render_keys(self) -> None:
        common = [("1-4", "views"), ("tab", "pane"), ("↑↓", "select"), ("⏎", "expand/open")]
        extra = {"investigation": [("r", "replay"), ("f", "full record")],
                 "intervention": [("←→", "case"), ("c", "arm"), ("v", "role"), ("d", "direction")],
                 "controls": [("d", "direction"), ("v", "role in case view")],
                 "evidence": [("f", "full record")]}[self.view]
        if self.replay_keys is not None:
            extra = [("n", "next event"), ("esc", "stop replay")]
        tail = [("e", "export"), ("/", "command"), ("?", "help"), ("q", "quit")]
        t = Text(" ")
        for k, label in common + extra + tail:
            t.append(k, style=f"bold {R.FG}")
            t.append(f" {label}   ", style=R.MUTED)
        self.query_one("#keys", Static).update(t)

    # ------------------------------------------------------- investigation
    def transcript_keys(self) -> list[str]:
        t = self.auditor.transcript()
        keys = ["task", "source"]
        keys += [f"step-{s.index}" for s in t.tool_steps]
        if t.final_step:
            keys.append("final")
        keys.append("summary")
        return keys

    def replay_times(self) -> dict[str, float]:
        try:
            t = self.auditor.transcript()
        except EvidenceError:
            return {}
        times = {"task": 0.0}
        if t.source_event:
            times["source"] = self.auditor.transcript_offset(t.source_event)
        for s in t.tool_steps:
            if s.t_tool is not None:
                times[f"step-{s.index}"] = s.t_tool
        if t.final_step and t.final_step.t_response is not None:
            times["final"] = t.final_step.t_response
        finished = next((e for e in t.receipts if e["event"] == "finished"), None)
        if finished:
            times["summary"] = self.auditor.transcript_offset(finished)
        return times

    async def rebuild_transcript(self) -> None:
        head = self.query_one("#inv-head", Static)
        lv = self.query_one("#transcript", ListView)
        await lv.clear()
        if self.study != "fixed_dev_mean":
            head.update(Group(R.section("Recorded transcript"),
                              Text("The recorded agent investigation (mechanism-fixed-direction-review-v2) inspected the "
                                   "fixed_dev_mean study only. No agent transcript exists for the matched study, so none "
                                   "is shown. Use 2 Intervention and 3 Controls for its recorded answers.",
                                   style=R.AMBER)))
            return
        try:
            t = self.auditor.transcript()
            keys = self.transcript_keys()
        except EvidenceError as err:
            head.update(R.error_panel(err))
            return
        fin = t.finished
        head.update(Group(
            Text.assemble(R.badge("RECORDED TRANSCRIPT", "#2a3446"), "  ", (t.run_id, f"bold {R.FG}"),
                          (f"   {t.started_utc:%Y-%m-%d %H:%M:%S} UTC · not live · nothing is executed", R.MUTED)),
            Text.assemble((f"{t.model}", R.ACCENT), (" via OpenAI · tools ran on Agent37 ", R.MUTED),
                          (str(fin.get("instance", "?")), R.FG),
                          (f" · {len(t.steps)} requests · {len(t.tool_steps)} tool calls", R.MUTED))))
        if self.replay_keys is not None:
            keys = self.replay_keys[: self.replay_index + 1]
        await lv.extend([TranscriptEntry(k) for k in keys])
        for item in lv.query(TranscriptEntry):
            self.paint_entry(item)
        if self.replay_keys is not None:
            lv.index = len(keys) - 1
            lv.scroll_end(animate=False)
        elif lv.index is None and keys:
            lv.index = 0

    def paint_entry(self, item: TranscriptEntry) -> None:
        item.query_one(Static).update(self.entry_renderable(item.key))
        item.set_class(item.key in self.expanded, "-expanded")

    def entry_renderable(self, key: str):
        t = self.auditor.transcript()
        times = self.replay_times()
        stamp = Text(f"+{times[key]:.1f}s" if key in times else "", style=R.FAINT)

        def header(left: Text) -> Table:
            g = Table.grid(expand=True)
            g.add_column(ratio=1)
            g.add_column(justify="right", width=8)
            g.add_row(left, stamp)
            return g

        if key == "task":
            body = Text(t.instructions, style=R.BRIGHT)
            return Group(header(Text.assemble(("> ", f"bold {R.ACCENT}"), ("Task given to the investigator", f"bold {R.FG}"),
                                              ("  instructions · request-0.json", R.FAINT))),
                         R.Padding(body, (0, 0, 0, 2)))
        if key == "source":
            e = t.source_event
            files = len(e.get("files", {}))
            result = e.get("result", {})
            return Group(header(Text.assemble(("● ", R.MUTED), ("Agent37 verified the bundled evidence", f"bold {R.FG}"))),
                         Text.assemble(("  └ ", R.FAINT), (f"{files} source files · archive sha256 {R.short(e.get('archive_sha256'))}"
                                                            f" · exit {result.get('exit_code')} · {result.get('stdout', '').strip()}",
                                                            R.MUTED)))
        if key == "summary":
            fin = t.finished
            usage = fin.get("usage", {})
            return Group(header(Text.assemble(("● ", R.MUTED), ("Run finished", f"bold {R.FG}"),
                                              (f"  status {fin.get('status')}", R.MUTED))),
                         Text.assemble(("  └ ", R.FAINT),
                                       (f"{len(t.steps)} OpenAI requests · {len(t.tool_steps)} Agent37 tool calls · "
                                        f"{usage.get('input_tokens', 0):,} in / {usage.get('output_tokens', 0):,} out tokens · "
                                        f"est. ${fin.get('api_usd_estimate', 0):.4f} of ${t.max_api_usd or 0:.2f} cap · "
                                        f"mode {fin.get('mode')}", R.MUTED)),
                         Text("    These calls inspected recorded experiments. They did not launch the GPU study.",
                              style=R.FAINT))
        if key == "final":
            s = t.final_step
            return Group(header(Text.assemble(("● ", f"bold {R.GREEN}"), ("Agent finding", f"bold {R.BRIGHT}"),
                                              (f"  {s.response_file} · {s.usage.get('output_tokens', 0)} output tokens", R.FAINT))),
                         R.Padding(R.markdown(s.text or "", skip_title="Finding"), (0, 0, 0, 2)))
        step = next(s for s in t.tool_steps if f"step-{s.index}" == key)
        marker = "▾ " if key in self.expanded else "▸ "
        top = Text.assemble((marker, R.MUTED), ("● ", f"bold {R.ACCENT}"))
        top.append_text(R.call_signature(step.name, step.arguments))
        summary = Text.assemble(("  └ ", R.FAINT))
        summary.append_text(R.tool_summary(step.name, step.arguments, step.result))
        parts = [header(top), summary]
        if key in self.expanded:
            parts += [Text("")] + self.step_details(step, max_lines=26)
        return Group(*parts)

    def step_details(self, step: Step, max_lines: int | None) -> list:
        t = self.auditor.transcript()
        try:
            same = self.auditor.reverify(step)
            rerun = (("✓ re-read locally now: identical", R.GREEN_DIM) if same
                     else ("✗ re-read locally now: DIFFERS from the record", R.RED))
        except EvidenceError as err:
            rerun = (f"✗ cannot re-read locally: {err.title}", R.RED)
        observed = self.observation_note(step)
        meta = R.kv([
            ("request", Text.assemble((f"{step.request_file} → {step.response_file}", R.FG),
                                      (f"   {step.request_id or '-'} · HTTP {step.http_status}", R.MUTED))),
            ("usage", Text(f"{step.usage.get('input_tokens', 0):,} in · {step.usage.get('output_tokens', 0):,} out tokens"
                           + (f" · ${step.usd:.4f}" if step.usd is not None else ""), style=R.FG)),
            ("recorded", Text(f"request +{step.t_request:.1f}s · model chose tool +{step.t_response:.1f}s · "
                              f"Agent37 result +{step.t_tool:.1f}s" if None not in (step.t_request, step.t_response, step.t_tool)
                              else "timestamps not recorded", style=R.FG)),
            ("receipt", Text.assemble((f"sha256 {R.short(step.receipt_sha, 12)} (receipts.jsonl line {step.receipt_line}) ", R.FG),
                                      ("✓ matches " + step.tool_file if step.receipt_ok else "✗ DOES NOT match " + str(step.tool_file),
                                       R.GREEN_DIM if step.receipt_ok else R.RED))),
            ("verified", Text(rerun[0], style=rerun[1])),
            ("model saw", observed),
        ], key_width=10)
        args_json, _ = R.json_block(step.arguments)
        out_json, total = R.json_block(step.result, max_lines)
        shown = total if max_lines is None else min(total, max_lines)
        parts = [R.Padding(meta, (0, 0, 0, 4)), Text(""),
                 R.Padding(R.section("Input"), (0, 0, 0, 4)), R.Padding(args_json, (0, 0, 0, 4)),
                 R.Padding(R.section("Output", f"{step.tool_file} · {total} lines"
                                     + ("" if shown == total else f" · showing {shown} · f opens all")), (0, 0, 0, 4)),
                 R.Padding(out_json, (0, 0, 0, 4))]
        links = Text("    evidence  ", style=R.MUTED)
        for name in (step.tool_file, step.request_file, step.response_file, f"receipts.jsonl:{step.receipt_line}"):
            links.append(str(name), style=f"underline {R.ACCENT}")
            links.append("   ")
        parts += [links, Text(f"    {t.directory}", style=R.FAINT)]
        return parts

    def observation_note(self, step: Step) -> Text:
        nxt = self.auditor.transcript_dir / f"request-{step.index + 1}.json"
        try:
            observations = json.loads(json.loads(nxt.read_text(encoding="utf-8"))["input"])["observations"]
            obs = observations[step.index]
        except (OSError, ValueError, KeyError, IndexError):
            return Text("observation passed to the model was not found", style=R.AMBER)
        if obs.get("truncated"):
            return Text(f"truncated to 11,000 characters in request-{step.index + 1}.json (full bytes sha256 "
                        f"{R.short(obs.get('full_result_sha256'), 12)}); the finding says so", style=R.AMBER)
        return Text(f"full output in request-{step.index + 1}.json", style=R.FG)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id == "transcript" and isinstance(event.item, TranscriptEntry):
            key = event.item.key
            if key.startswith("step-") or key == "final":
                self.expanded.symmetric_difference_update({key})
                self.paint_entry(event.item)
        elif event.list_view.id == "evidence-list":
            self.action_full_record()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id == "evidence-list":
            self.render_evidence_detail()
        self.render_inspector()

    # ------------------------------------------------------------ replay
    def action_replay(self) -> None:
        if self.study != "fixed_dev_mean":
            self.notify("The recorded agent transcript covers fixed_dev_mean only. Use /study fixed_dev_mean.",
                        title="No transcript for this study", severity="warning")
            return
        if self.replay_keys is not None:
            self.stop_replay()
            return
        try:
            keys = self.transcript_keys()
        except EvidenceError as err:
            self.notify(err.message, title=err.title, severity="error")
            return
        self.action_show_view("investigation")
        self.replay_keys, self.replay_index, self.replay_done = keys, 0, False
        self.after_replay_change()
        self.schedule_replay()

    def schedule_replay(self) -> None:
        if self.replay_timer:
            self.replay_timer.stop()
            self.replay_timer = None
        if self.replay_keys is None or self.replay_index >= len(self.replay_keys) - 1:
            self.replay_done = self.replay_keys is not None
            self.render_topbar()
            return
        times = self.replay_times()
        now = times.get(self.replay_keys[self.replay_index], 0.0)
        nxt = times.get(self.replay_keys[self.replay_index + 1], now)
        self.replay_timer = self.set_timer(max(nxt - now, 0.05), self.action_replay_next)

    def action_replay_next(self) -> None:
        if self.replay_keys is None or self.replay_index >= len(self.replay_keys) - 1:
            return
        self.replay_index += 1
        self.after_replay_change()
        self.schedule_replay()

    def after_replay_change(self) -> None:
        self.render_topbar()
        self.render_keys()
        self.call_after_refresh(self.rebuild_transcript)

    def stop_replay(self, show_all: bool = True) -> None:
        if self.replay_timer:
            self.replay_timer.stop()
        self.replay_timer, self.replay_keys, self.replay_done = None, None, False
        if show_all and self.is_mounted:
            self.after_replay_change()

    # ------------------------------------------------------- intervention
    def current_case(self) -> dict:
        if self.row is None:
            raise EvidenceError("missing", "No generated panel is available for this study.")
        return self.auditor.case(self.study, self.row, self.condition, self.role)

    def render_case(self) -> None:
        head = self.query_one("#case-head", Static)
        inp = self.query_one("#case-input", Static)
        foot = self.query_one("#case-foot", Static)
        blocks = list(self.query(CompareBlock))
        try:
            data = self.auditor.study(self.study)
            panel = data["generation_profile_indices"]
            case = self.current_case()
        except EvidenceError as err:
            head.update(R.error_panel(err))
            inp.update("")
            foot.update("")
            for b in blocks:
                b.update("")
                b.display = False
            return
        for b in blocks:
            b.display = True
        rows = Text.assemble(("CASE  ", f"bold {R.MUTED}"), (f"row {self.row}", f"bold {R.BRIGHT}"),
                             (f"   {panel.index(self.row) + 1} of {len(panel)} generated   ", R.MUTED))
        for r in panel:
            rows.append(f" {r} ", style=f"bold {R.BG} on {R.ACCENT}" if r == self.row else R.FAINT)
        arms = Text.assemble(("ARM   ", f"bold {R.MUTED}"))
        for cond in (STUDY_INFO[self.study]["primary"],) + CONTROL_CONDITIONS:
            label = ARM_LABEL[cond]
            arms.append(f" {label} ", style=f"bold {R.BRIGHT} on #1b2433" if cond == self.condition else R.MUTED)
            arms.append(" ")
        role = Text.assemble(("ROLE  ", f"bold {R.MUTED}"),
                             (" candidate · forward repair ", f"bold {R.BRIGHT} on #1b2433" if self.role == "candidate" else R.MUTED),
                             ("  "),
                             (" clean reference · reverse insertion ", f"bold {R.BRIGHT} on #1b2433" if self.role == "control" else R.MUTED))
        head.update(Group(rows, arms, role))
        inp.update(R.input_card(case, data, self.condition, self.role))
        for b in blocks:
            renderable, css = R.compare_block(case["kinds"][b.kind], self.condition, self.role, data["layer"])
            b.update(renderable)
            for c in ("-repaired", "-preserved", "-broken", "-neutral"):
                b.set_class(c == css, c)
        foot.update(Text(
            "Answers are the complete recorded generations (Qwen chat format, end token shown dim). Prefix scores are "
            "the separate first-decision-token measurement. Input text and weights are identical within each pair; only "
            "the residual stream at block %d is changed." % data["layer"], style=R.FAINT))

    def action_case(self, step: int) -> None:
        if self.view != "intervention":
            return
        try:
            panel = self.auditor.panel(self.study)
        except EvidenceError as err:
            self.notify(err.message, title=err.title, severity="error")
            return
        i = panel.index(self.row) if self.row in panel else 0
        self.row = panel[(i + step) % len(panel)]
        self.render_case()
        self.render_inspector()

    def action_block(self, step: int) -> None:
        blocks = [b for b in self.query(CompareBlock) if b.display]
        if not blocks:
            return
        current = next((i for i, b in enumerate(blocks) if b.has_focus), self.block_index)
        self.block_index = max(0, min(len(blocks) - 1, current + step))
        blocks[self.block_index].focus()
        blocks[self.block_index].scroll_visible()
        self.render_inspector()

    def action_cycle_condition(self) -> None:
        options = (STUDY_INFO[self.study]["primary"],) + CONTROL_CONDITIONS
        self.condition = options[(options.index(self.condition) + 1) % len(options)]
        self.render_case()
        self.render_inspector()
        if self.view != "intervention":
            self.action_show_view("intervention")
        self.notify(f"Arm: {ARM_LABEL[self.condition]}", timeout=2)

    def action_toggle_role(self) -> None:
        self.role = "control" if self.role == "candidate" else "candidate"
        self.render_case()
        self.render_inspector()
        if self.view != "intervention":
            self.action_show_view("intervention")
        self.notify("Role: " + ("clean reference, reverse insertion (vector subtracted)" if self.role == "control"
                                else "candidate, forward repair (vector added)"), timeout=3)

    # ------------------------------------------------------------ controls
    def render_controls(self) -> None:
        body = self.query_one("#controls-body", Static)
        try:
            body.update(self.controls_renderable())
        except EvidenceError as err:
            body.update(R.error_panel(err))

    def controls_renderable(self) -> Group:
        a = self.auditor
        s = self.study
        data = a.study(s)
        primary = STUDY_INFO[s]["primary"]
        gen_c = a.controls(s, "generated", "candidate")
        gen_r = a.controls(s, "generated", "control")
        sc_c = a.controls(s, "scored", "candidate")
        sc_r = a.controls(s, "scored", "control")
        arms = gen_c["arms"]
        p = arms[primary]
        n = p["n_profiles"]
        malformed_c = {c: a.malformed_by_kind(s, c, "candidate") for c in arms if arms[c]["invalid_generated_answers"]}
        malformed_r = {c: a.malformed_by_kind(s, c, "control") for c in gen_r["arms"]
                       if gen_r["arms"][c]["invalid_generated_answers"]}
        gate_passed = all(data["frozen_gates"].values())

        headline = Text()
        headline.append(f"{ARM_LABEL[primary].capitalize()} repairs ", style=f"bold {R.BRIGHT}")
        headline.append(f"{p['trigger_policy_correct']}/{n}", style=f"bold {R.GREEN}")
        headline.append(" trigger decisions", style=f"bold {R.BRIGHT}")
        sub = Text.assemble(
            ("baseline ", R.MUTED), (f"{arms['baseline']['trigger_policy_correct']}/{n}", R.FG),
            ("  ·  ordinary twins ", R.MUTED), (f"{p['twin_policy_correct']}/{n}", R.GREEN),
            ("  ·  legitimate approvals ", R.MUTED), (f"{p['legitimate_approvals_retained']}/{n}", R.GREEN),
            ("  ·  malformed primary answers ", R.MUTED),
            (f"{p['invalid_generated_answers']}/{p['total_answers']}", R.GREEN if not p["invalid_generated_answers"] else R.AMBER))
        best_other = max(arms[c]["trigger_policy_correct"] for c in CONTROL_CONDITIONS)
        sub2 = Text.assemble(("every comparison direction repairs at most ", R.MUTED), (f"{best_other}/{n}", R.FG),
                             ("  ·  generated answers, complete assistant text", R.MUTED))
        gate_lines = [R.gate_badge(gate_passed)]
        for name, value in data["frozen_gates"].items():
            gate_lines.append(Text(f"{name} = {str(value).lower()}", style=R.RED if not value else R.GREEN))
        total_bad = data["invalid_generated_answers"]
        gate_lines.append(Text(f"{total_bad} malformed generated answers across all arms; frozen before results, "
                               "not relaxed", style=R.MUTED))
        top = Table.grid(expand=True, padding=(0, 2))
        top.add_column(ratio=3)
        top.add_column(ratio=2)
        top.add_row(Group(headline, sub, sub2), Group(*gate_lines))

        gen_n = len(data["generation_profile_indices"])
        rev = gen_r["arms"][primary]
        nonspecific = rev["twin_approved"] > 0
        rev_head = Text.assemble((f"{ARM_LABEL[primary].capitalize()} subtracted from the clean reference approves ", R.BRIGHT),
                                 (f"{rev['trigger_approved']}/{rev['n_profiles']}", f"bold {R.AMBER}"), (" triggers and ", R.BRIGHT),
                                 (f"{rev['twin_approved']}/{rev['n_profiles']}", f"bold {R.AMBER}"), (" ordinary twins", R.BRIGHT))
        rev_note = Text("It pushes decisions toward APPROVE in general; it is not a selective trigger mechanism."
                        if nonspecific else "Ordinary twins are unchanged by reverse insertion.",
                        style=R.AMBER if nonspecific else R.MUTED)
        paired = gen_c["paired_trigger_contrasts"]
        pair_text = Text("Paired on the same trigger profiles: ", style=R.MUTED)
        pair_text.append(", ".join(f"vs {c['contrast'].split(' minus ')[1].replace('_norm_matched', '')} "
                                   f"{c['primary_only_successes']}–{c['comparison_only_successes']}" for c in paired), style=R.FG)
        pair_text.append("  (primary-only vs comparison-only successes)", style=R.FAINT)

        sc_n = sc_c["arms"]["baseline"]["n_profiles"]
        sp = sc_c["arms"][primary]
        scored_head = Text.assemble(
            (f"{sp['trigger_policy_correct']}/{sc_n}", f"bold {R.BRIGHT}"), (" trigger decision-prefix scores repaired · ", R.FG),
            (f"twins {sp['twin_policy_correct']}/{sc_n} · approvals {sp['legitimate_approvals_retained']}/{sc_n}", R.FG))
        scored_warn = Text(f"Decision-prefix scoring compares the first distinct decision token. It is NOT {sc_n} "
                           f"generated answers and not a full label likelihood.", style=R.AMBER)

        parts = [
            R.section(f"{STUDY_INFO[s]['label']} study · {s}",
                      f"{STUDY_INFO[s]['standing']} · {gen_n} generated profiles · candidate forward repair"),
            Text(""), top, Text(""),
            R.arms_table(arms, primary, "candidate", malformed_c), Text(""), pair_text,
            Text(""), Text(""),
            R.section("Reverse insertion", f"clean reference − same direction · {gen_n} generated profiles"),
            Text.assemble(R.badge("NONSPECIFIC", R.AMBER) if nonspecific else R.badge("SELECTIVE", R.GREEN)),
            rev_head, rev_note, Text(""),
            R.arms_table(gen_r["arms"], primary, "control", malformed_r),
            Text(""), Text(""),
            R.section("Why the gate is closed", "counted from the raw generated records"),
        ]
        parts += self.gate_rows(s, data)
        parts += [Text(""), Text(""),
                  R.section("Decision-prefix scoring", f"{sc_n} profiles · separate measurement"),
                  scored_warn, scored_head, Text(""),
                  R.scored_table(sc_c["arms"], primary, "candidate"), Text(""),
                  Text("Reverse insertion, prefix scores (control role)", style=f"bold {R.MUTED}"),
                  R.scored_table(sc_r["arms"], primary, "control")]
        if s == "matched":
            parts += [Text(""), Text("Matched transplant is the supporting study: each case needs its own clean activation. "
                                     "Its denominators are never combined with the fixed-direction study.", style=R.AMBER)]
        return Group(*parts)

    def gate_rows(self, s: str, data: dict) -> list:
        rows = []
        for reason in self.auditor.gate_reasons(s):
            rows.append(Text.assemble(("  ✗ ", f"bold {R.RED}"), (reason, R.FG)))
        try:
            contract = self.auditor.contract(s)
        except EvidenceError as err:
            return rows + [R.error_panel(err)]
        gate = contract.get("gate")
        if gate:
            gen = self.auditor.controls(s, "generated", "candidate")["arms"]
            sc = self.auditor.controls(s, "scored", "candidate")["arms"]
            primary = STUDY_INFO[s]["primary"]
            summary = self.auditor.summary(s)
            total = data["generation_records"]
            complete = total - data["invalid_generated_answers"]

            def rate(arms, key):
                a = arms[primary]
                return a[key], a["n_profiles"]

            def advantage(arms):
                a = arms[primary]
                best = max(arms[c]["trigger_policy_correct"] for c in CONTROL_CONDITIONS)
                return (a["trigger_policy_correct"] - best) / a["n_profiles"]

            measured = {
                "generated_trigger_repair_min": rate(gen, "trigger_policy_correct"),
                "generated_twin_policy_min": rate(gen, "twin_policy_correct"),
                "generated_legit_approval_min": rate(gen, "legitimate_approvals_retained"),
                "generated_advantage_over_best_generic_random_min": advantage(gen),
                "scored_trigger_repair_min": rate(sc, "trigger_policy_correct"),
                "scored_twin_policy_min": rate(sc, "twin_policy_correct"),
                "scored_legit_approval_min": rate(sc, "legitimate_approvals_retained"),
                "scored_advantage_over_best_generic_random_min": advantage(sc),
                "all_arm_complete_generation_min": (complete, total),
            }
            t = Table(box=None, padding=(0, 2), show_edge=False, header_style=f"bold {R.MUTED}")
            t.add_column(f"CRITERION ({Path(STUDY_INFO[s]['contract']).name})", no_wrap=True)
            t.add_column("REQUIRED", justify="right", no_wrap=True)
            t.add_column("MEASURED", justify="right", no_wrap=True)
            t.add_column("", no_wrap=True)
            for name, threshold in gate.items():
                if name in measured:
                    m = measured[name]
                    value = m[0] / m[1] if isinstance(m, tuple) else m
                    text = f"{m[0]}/{m[1]}" if isinstance(m, tuple) else f"{m:.2f}"
                    ok = value >= threshold
                    t.add_row(Text(name, style=R.FG), Text(f"≥ {threshold:.2f}", style=R.MUTED), Text(text, style=R.FG),
                              Text("✓ met" if ok else "✗ not met", style=R.GREEN_DIM if ok else f"bold {R.RED}"))
                elif name == "all_expected_patch_applications_exactly_once":
                    flag = summary.get("all_arms_complete_and_patched")
                    t.add_row(Text(name, style=R.FG), Text("true", style=R.MUTED),
                              Text(f"summary flag {str(flag).lower()}", style=R.FG),
                              Text("✓ met" if flag else "✗ not met (combined flag)", style=R.GREEN_DIM if flag else f"bold {R.RED}"))
                else:
                    t.add_row(Text(name, style=R.FG), Text(str(threshold), style=R.MUTED), Text("-", style=R.FAINT),
                              Text("not recomputed here", style=R.FAINT))
            rows += [Text(""), t,
                     Text("Measured values are recounted from the records for display; the verdict shown is the frozen "
                          "summary value.", style=R.FAINT)]
        else:
            for key in ("primary_gate", "insertion_gate"):
                if key in contract:
                    rows.append(Text(f"  {key}: " + ", ".join(f"{k} {v}" for k, v in contract[key].items()), style=R.MUTED))
            if "null_policy" in contract:
                rows.append(Text(f"  {contract['null_policy']}", style=R.MUTED))
        return rows

    # ------------------------------------------------------------ evidence
    def rebuild_evidence(self) -> None:
        self.call_after_refresh(self._rebuild_evidence)

    async def _rebuild_evidence(self) -> None:
        lv = self.query_one("#evidence-list", ListView)
        await lv.clear()
        try:
            self.evidence_items = self.auditor.evidence_items(self.study)
        except EvidenceError as err:
            self.evidence_items = []
            self.query_one("#evidence-body", Static).update(R.error_panel(err))
            return
        group = None
        entries = []
        for item in self.evidence_items:
            color, mark = {"verified": (R.GREEN_DIM, "✓"), "record": (R.ACCENT, "◇"), "unlisted": (R.MUTED, "·"),
                           "mismatch": (R.RED, "✗"), "missing": (R.RED, "✗")}.get(item.status, (R.AMBER, "?"))
            t = Text()
            if item.group != group:
                group = item.group
                t.append(item.group.upper() + "\n", style=f"bold {R.FAINT}")
            t.append(f"{mark} ", style=f"bold {color}")
            t.append(item.label, style=R.FG if item.status not in ("missing", "mismatch") else R.RED)
            entries.append(ListItem(Static(t)))
        await lv.extend(entries)
        lv.index = 0
        self.render_evidence_detail()

    def selected_evidence(self) -> EvidenceItem | None:
        lv = self.query_one("#evidence-list", ListView)
        if lv.index is None or not self.evidence_items or lv.index >= len(self.evidence_items):
            return None
        return self.evidence_items[lv.index]

    def render_evidence_detail(self) -> None:
        item = self.selected_evidence()
        body = self.query_one("#evidence-body", Static)
        if item is None:
            return
        body.update(Group(*self.evidence_meta(item), Text(""), self.evidence_preview(item, limit=60)))

    def evidence_meta(self, item: EvidenceItem) -> list:
        status = {"verified": ("SHA-256 VERIFIED", R.GREEN), "record": ("RECORDED VALUE", R.ACCENT),
                  "unlisted": ("HASHED · NO MANIFEST ENTRY", R.FAINT), "mismatch": ("HASH MISMATCH", R.RED),
                  "missing": ("MISSING FILE", R.RED)}.get(item.status, (item.status.upper(), R.AMBER))
        rows = [("status", R.badge(status[0], status[1]))]
        if item.path is not None:
            rows.append(("path", Text(str(item.path), style=R.FG)))
            rows.append(("size", Text(f"{item.size:,} bytes" if item.size is not None else "-", style=R.FG)))
            rows.append(("sha256", Text(item.actual_sha or "-", style=R.ACCENT)))
            if item.expected_sha:
                rows.append(("expected", Text.assemble((item.expected_sha, R.MUTED), (f"\nfrom {item.expected_from}", R.FAINT))))
        if item.note:
            rows.append(("note", Text(item.note, style=R.MUTED)))
        hint = []
        if item.status == "missing":
            hint = [Text(""), Text("Missing evidence is never shown as zero. Restore the preserved copy or pass "
                                   "--evidence-root.", style=R.AMBER)]
        if item.status == "mismatch":
            hint = [Text(""), Text("These bytes differ from the manifest. The adapter refuses to read them; restore the "
                                   "preserved copy. Do not edit evidence in place.", style=R.AMBER)]
        return [Text(item.label, style=f"bold {R.BRIGHT}"), Text(item.group, style=R.MUTED), Text(""), R.kv(rows, 9)] + hint

    def evidence_preview(self, item: EvidenceItem, limit: int | None):
        if item.record is not None:
            return R.json_block(item.record, limit)[0]
        if item.path is None or item.status == "missing":
            return Text("")
        if item.status == "mismatch":
            return Text("Content withheld: hash mismatch.", style=R.RED)
        path = item.path
        if path.suffix == ".safetensors":
            raw = path.open("rb").read(8)
            size = int.from_bytes(raw, "little")
            header = json.loads(path.open("rb").read(8 + size)[8:])
            keys = {k: v for k, v in header.items() if k != "__metadata__"}
            shown = dict(list(keys.items())[: (limit or len(keys))])
            note = Text(f"safetensors header · {len(keys)} tensors · values read by the adapter's bounded reader",
                        style=R.MUTED)
            return Group(note, R.json_block(shown, limit)[0])
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".jsonl":
            lines = [x for x in text.splitlines() if x.strip()]
            take = lines[: 3 if limit else 60]
            note = Text(f"{len(lines):,} rows · showing first {len(take)} (f opens {len(take) if limit else 60})",
                        style=R.MUTED)
            return Group(note, *[R.json_block(json.loads(x), None if not limit else 40)[0] for x in take])
        if path.suffix == ".json":
            return R.json_block(json.loads(text), limit)[0]
        if path.suffix == ".md":
            return R.markdown(text if not limit else "\n".join(text.splitlines()[:limit]))
        lines = text.splitlines()
        return Text("\n".join(lines[:limit] if limit else lines), style=R.FG)

    # ----------------------------------------------------------- inspector
    def render_inspector(self) -> None:
        body = self.query_one("#inspector-body", Static)
        try:
            body.update(self.inspector_renderable())
        except EvidenceError as err:
            body.update(R.error_panel(err))

    def inspector_renderable(self) -> Group:
        a, s = self.auditor, self.study
        data = a.study(s)
        info = STUDY_INFO[s]
        cp = data["checkpoint_provenance"]
        contract = a.contract(s)
        cand = cp["candidate"]["checkpoint_sha256"]["model.safetensors"]
        ref = cp["control"]["checkpoint_sha256"]["model.safetensors"]
        exp = R.kv([
            ("study", Text.assemble((s, f"bold {R.BRIGHT}"), (f"\n{info['label']} · {info['standing']}", R.MUTED))),
            ("identity", Text(contract.get("identity", "-"), style=R.FG)),
            ("contract", Text.assemble((R.short(data["contract_sha256"]), R.ACCENT), ("  ✓ hash", R.GREEN_DIM))),
            ("model", Text(cp["candidate"]["base_model_reference"].split("/")[-1], style=R.FG)),
            ("revision", Text(cp["candidate"]["base_model_revision"][:12], style=R.MUTED)),
            ("candidate", Text.assemble((R.short(cand), R.ACCENT), ("  planted rule", R.MUTED))),
            ("reference", Text.assemble((R.short(ref), R.ACCENT), ("  clean CP-7", R.MUTED))),
        ])
        dev = a.dev_layer(data["layer"])
        fixed = not data["test_time_clean_model_required_for_candidate_intervention"]
        rows = [("site", Text(f"block {data['layer']} · final DECISION: colon", style=R.FG))]
        if fixed:
            direction = a.direction(s, self.row if self.row is not None else 0, info["primary"])
            rows += [("vector", Text(f"fixed {direction['dimension']}-d Δ (clean − candidate)", style=R.FG)),
                     ("L2 norm", Text(f"{direction['l2_norm']:.2f}" + (f" · coefficient {contract['coefficient']:g}"
                                                                      if "coefficient" in contract else ""), style=R.FG)),
                     ("learned", Text(f"from {dev['n_profiles']} DEV examples", style=R.FG)),
                     ("applied", Text("same vector for every fresh application", style=R.GREEN_DIM)),
                     ("test-time", Text("no clean activations", style=R.GREEN_DIM))]
        else:
            rows += [("vector", Text("per case: its own 1536-d clean − candidate Δ", style=R.FG)),
                     ("learned", Text(f"layer chosen on {dev['n_profiles']} DEV examples", style=R.FG)),
                     ("test-time", Text("needs a clean activation for every case", style=R.AMBER))]
        rows += [("held out", Text(f"{data['score_profile_count']} scored · {len(data['generation_profile_indices'])} generated",
                                   style=R.FG))]
        inter = R.kv(rows)
        gate = R.kv([(k.replace("_gate_passed", "").replace("_", " "),
                      Text("FAILED" if not v else "passed", style=f"bold {R.RED}" if not v else R.GREEN))
                     for k, v in data["frozen_gates"].items()] +
                    [("malformed", Text(f"{data['invalid_generated_answers']} generated answers across arms; "
                                        f"primary arm {self.primary_malformed()}", style=R.AMBER))])
        parts = [R.section("Experiment"), exp, Text(""), R.section("Intervention"), inter, Text(""),
                 R.section("Frozen gate"), gate, Text(""), R.section("Selection"), self.selection_context()]
        if self.last_export:
            parts += [Text(""), R.section("Last export"), Text(str(self.last_export), style=R.ACCENT)]
        return Group(*parts)

    def primary_malformed(self) -> str:
        primary = STUDY_INFO[self.study]["primary"]
        a = self.auditor.controls(self.study, "generated", "candidate")["arms"][primary]
        return f"{a['invalid_generated_answers']}/{a['total_answers']}"

    def selection_context(self):
        if self.view == "investigation":
            if self.study != "fixed_dev_mean":
                return Text("No recorded transcript for this study.", style=R.MUTED)
            lv = self.query_one("#transcript", ListView)
            item = lv.highlighted_child
            if not isinstance(item, TranscriptEntry):
                return Text("-", style=R.MUTED)
            t = self.auditor.transcript()
            if item.key.startswith("step-"):
                step = next(x for x in t.tool_steps if f"step-{x.index}" == item.key)
                return R.kv([("tool", Text(step.name, style=R.BRIGHT)),
                              ("call id", Text(step.call_id, style=R.MUTED)),
                              ("request", Text(step.request_id or "-", style=R.MUTED)),
                              ("tokens", Text(f"{step.usage.get('input_tokens', 0):,} in / {step.usage.get('output_tokens', 0)} out",
                                              style=R.FG)),
                              ("receipt", Text("✓ hash matches" if step.receipt_ok else "✗ mismatch",
                                               style=R.GREEN_DIM if step.receipt_ok else R.RED)),
                              ("keys", Text("⏎ expand · f full record", style=R.MUTED))])
            labels = {"task": "Instructions sent with every request (request-0.json).",
                      "source": "Agent37 checked the bundled evidence hashes before the first request.",
                      "final": "The model's final message (response-6.json), unedited.",
                      "summary": "result.json and the final receipt."}
            return Text(labels.get(item.key, ""), style=R.MUTED)
        if self.view == "intervention":
            try:
                case = self.current_case()
            except EvidenceError as err:
                return Text(err.message, style=R.AMBER)
            focused = next((b.kind for b in self.query(CompareBlock) if b.has_focus), "trigger")
            k = case["kinds"][focused]
            return R.kv([("record", Text(KIND_LABEL[focused], style=R.BRIGHT)),
                         ("app id", Text(k["app"]["app_id"], style=R.FG)),
                         ("prompt", Text(R.short(k["before"]["prompt_sha256"], 12), style=R.ACCENT)),
                         ("fixed", Text("same prompt sha before/after" if k["input_fixed"] else "PROMPT DIFFERS",
                                        style=R.GREEN_DIM if k["input_fixed"] else R.RED)),
                         ("patches", Text(f"before {k['before']['patch_applications']} · after {k['after']['patch_applications']}"
                                          f" (expected {'yes' if k['after']['patch_expected'] else 'no'})", style=R.FG)),
                         ("parser", Text(k["after"]["parser_version"], style=R.MUTED)),
                         ("keys", Text("⏎ full prompt + raw JSON", style=R.MUTED))])
        if self.view == "controls":
            return Text("Generic: norm-matched mean of ordinary clean−candidate deltas. Random: three seeded Gaussian "
                        "directions with the same L2 norm. Reverse insertion subtracts the same vector from the clean "
                        "reference. Malformed answers count as failures.", style=R.MUTED)
        item = self.selected_evidence()
        if item is None:
            return Text("-", style=R.MUTED)
        return R.kv([("record", Text(item.label, style=R.BRIGHT)), ("status", Text(item.status, style=R.FG)),
                     ("keys", Text("⏎ open · e export", style=R.MUTED))])

    # ------------------------------------------------------------- actions
    def action_show_view(self, view: str) -> None:
        if view not in VIEWS:
            return
        self.view = view
        self.query_one("#main", ContentSwitcher).current = view
        self.render_tabs()
        self.render_keys()
        self.render_inspector()
        target = {"investigation": "#transcript", "intervention": None, "controls": "#controls",
                  "evidence": "#evidence-list"}[view]
        if target:
            self.query_one(target).focus()
        else:
            blocks = [b for b in self.query(CompareBlock) if b.display]
            if blocks:
                blocks[min(self.block_index, len(blocks) - 1)].focus()
            else:
                self.query_one("#case-scroll").focus()

    def action_full_record(self) -> None:
        try:
            screen = self.full_record_screen()
        except EvidenceError as err:
            self.notify(err.message, title=err.title, severity="error")
            return
        if screen:
            self.push_screen(screen)

    def full_record_screen(self) -> RecordScreen | None:
        if self.view == "investigation" and self.study == "fixed_dev_mean":
            item = self.query_one("#transcript", ListView).highlighted_child
            if not isinstance(item, TranscriptEntry):
                return None
            t = self.auditor.transcript()
            if item.key.startswith("step-"):
                step = next(x for x in t.tool_steps if f"step-{x.index}" == item.key)
                receipt = t.receipts[step.receipt_line - 1] if step.receipt_line else {}
                body = Group(*self.step_details(step, None), Text(""), R.section("Receipt line"),
                             R.json_block(receipt)[0])
                return RecordScreen(R.call_signature(step.name, step.arguments), body,
                                    f"recorded tool call {step.index + 1}/{len(t.tool_steps)}")
            return RecordScreen("Recorded transcript entry", self.entry_renderable(item.key))
        if self.view == "intervention":
            case = self.current_case()
            focused = next((b.kind for b in self.query(CompareBlock) if b.has_focus), "trigger")
            k = case["kinds"][focused]
            body = Group(
                R.section("Prompt", f"sha256 {k['before']['prompt_sha256']}"),
                Text("✓ reconstructed from the CP-7 policy text and the recorded application; its SHA-256 equals the "
                     "recorded prompt_sha256" if k["prompt_verified"] else
                     "Reconstruction does not match the recorded prompt_sha256; text shown for reference only.",
                     style=R.GREEN_DIM if k["prompt_verified"] else R.AMBER),
                Text(k["prompt"], style=R.FG), Text(""),
                R.section("Before · raw record (baseline)"), R.json_block(k["before"])[0], Text(""),
                R.section(f"After · raw record ({ARM_LABEL[self.condition]})"), R.json_block(k["after"])[0], Text(""),
                R.section("Decision-prefix scores (separate measurement)"),
                R.json_block({"before": k["before_score"], "after": k["after_score"]})[0])
            return RecordScreen(f"Row {self.row} · {KIND_LABEL[focused]} · {self.role}", body,
                                "replay_recorded_intervention · recorded, not re-run")
        if self.view == "evidence":
            item = self.selected_evidence()
            if item is None:
                return None
            return RecordScreen(item.label, Group(*self.evidence_meta(item), Text(""), self.evidence_preview(item, None)),
                                item.group)
        if self.view == "controls":
            return self.direction_screen()
        return None

    def direction_screen(self) -> RecordScreen:
        a, s = self.auditor, self.study
        row = self.row if self.row is not None else 0
        info = STUDY_INFO[s]
        primary = a.direction(s, row, info["primary"])
        others = [a.direction(s, row, c) for c in CONTROL_CONDITIONS]
        top = primary["largest_absolute_coordinates"]
        peak = max(abs(c["value"]) for c in top)
        coords = Table.grid(padding=(0, 1))
        coords.add_column(justify="right", style=R.MUTED)
        coords.add_column(justify="right", style=R.FG)
        coords.add_column()
        for c in top:
            width = round(28 * abs(c["value"]) / peak)
            coords.add_row(f"#{c['coordinate']}", f"{c['value']:+.2f}",
                           Text("█" * width, style=R.ACCENT if c["value"] > 0 else R.AMBER))
        cos = Table(box=None, padding=(0, 2), header_style=f"bold {R.MUTED}", show_edge=False)
        cos.add_column("DIRECTION")
        cos.add_column("L2 NORM", justify="right")
        cos.add_column(f"COSINE TO {ARM_LABEL[info['primary']].upper()}", justify="right")
        for d in [primary] + others:
            cos.add_row(Text(ARM_LABEL[d["condition"]], style=R.BRIGHT if d is primary else R.FG),
                        f"{d['l2_norm']:.2f}", f"{d['cosine_to_primary']:+.3f}")
        sweep = Table.grid(padding=(0, 1))
        sweep.add_column(justify="right", style=R.MUTED)
        sweep.add_column()
        sweep.add_column(style=R.FG)
        dev = a.dev_sweep()
        for layer, objective in dev:
            sel = layer == a.dev_layer(layer)["selected_layer"]
            sweep.add_row(f"block {layer}", Text("█" * round(30 * objective) or "·", style=R.GREEN if sel else R.FAINT),
                          f"{objective:.2f}" + ("  selected before held-out" if sel else ""))
        body = Group(
            R.section("Saved residual direction", f"{s} · row {row} · tensor sha256 {R.short(primary['tensor_sha256'])}"),
            Text(primary["semantics"], style=R.MUTED), Text(""),
            R.kv([("dimension", str(primary["dimension"])), ("L2 norm", f"{primary['l2_norm']:.4f}"),
                  ("same vector", "every profile" if primary["same_vector_every_profile"] else "no, one per profile")]),
            Text(""), R.section("Largest coordinates", "coordinates are not named circuits"), coords, Text(""),
            R.section("Comparison directions", "generic and random are norm-matched"), cos, Text(""),
            R.section("Development layer sweep", "12 DEV profiles · development evidence, not a held-out estimate"),
            sweep)
        return RecordScreen("Residual direction", body, "inspect_residual_direction + inspect_dev_layer")

    def action_direction(self) -> None:
        try:
            self.push_screen(self.direction_screen())
        except EvidenceError as err:
            self.notify(err.message, title=err.title, severity="error")

    def action_export(self) -> None:
        try:
            out = self.auditor.export(self.study, self.row, self.condition, self.role,
                                      self.selected_evidence() if self.view == "evidence" else None)
        except EvidenceError as err:
            self.notify(f"{err.message}\n{err.hint}", title=f"Export refused: {err.title}", severity="error", timeout=8)
            return
        except OSError as err:
            self.notify(f"Could not write the export: {err}", title="Export failed", severity="error", timeout=8)
            return
        self.last_export = out
        self.render_inspector()
        self.notify(f"{out / 'report.md'}\n{out / 'evidence.json'}", title="Exported", timeout=6)

    def action_help(self) -> None:
        self.push_screen(RecordScreen("Model Auditor · keys and commands", help_body()))

    def action_command(self) -> None:
        cmd = self.query_one("#command", Input)
        cmd.focus()
        cmd.value = "/"
        cmd.cursor_position = 1

    def action_escape(self) -> None:
        if self.replay_keys is not None:
            self.stop_replay()
            return
        cmd = self.query_one("#command", Input)
        if cmd.has_focus:
            cmd.value = ""
        self.action_show_view(self.view)

    # ------------------------------------------------------------ commands
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if text:
            self.run_command(text)
        if self.query_one("#command", Input).has_focus:
            self.action_show_view(self.view)

    def run_command(self, text: str) -> None:
        parts = text.split()
        name, args = parts[0].lower(), parts[1:]
        if not name.startswith("/"):
            name = "/" + name
        try:
            if name == "/study":
                if len(args) != 1 or args[0] not in STUDY_INFO:
                    raise EvidenceError("unsupported", f"Unknown study {' '.join(args) or '(none)'}.",
                                        "Use /study fixed_dev_mean (lead) or /study matched (supporting).")
                self.load_study(args[0])
            elif name == "/case":
                if len(args) != 1 or not args[0].lstrip("-").isdigit():
                    raise EvidenceError("unsupported", "Usage: /case N", "Example: /case 1")
                row = int(args[0])
                self.auditor.case(self.study, row, self.condition, self.role)  # validates before switching
                self.row = row
                self.render_case()
                self.action_show_view("intervention")
            elif name == "/controls":
                self.action_show_view("controls")
            elif name == "/direction":
                self.action_direction()
            elif name == "/evidence":
                self.action_show_view("evidence")
            elif name == "/export":
                self.action_export()
            elif name == "/replay":
                self.action_replay()
            elif name in ("/help", "/?"):
                self.action_help()
            elif name == "/condition":
                options = (STUDY_INFO[self.study]["primary"],) + CONTROL_CONDITIONS
                if len(args) != 1 or args[0] not in options:
                    raise EvidenceError("unsupported", f"Condition {' '.join(args) or '(none)'} was not measured in {self.study}.",
                                        "Measured: " + ", ".join(options))
                self.condition = args[0]
                self.render_case()
                self.action_show_view("intervention")
            elif name == "/role":
                if len(args) != 1 or args[0] not in ("candidate", "control"):
                    raise EvidenceError("unsupported", "Usage: /role candidate | control",
                                        "candidate = forward repair; control = reverse insertion into the clean reference")
                self.role = args[0]
                self.render_case()
                self.action_show_view("intervention")
            elif name in ("/quit", "/q"):
                self.exit()
            else:
                raise EvidenceError("unsupported", f"Unknown command {name}.",
                                    "Commands: " + "  ".join(COMMANDS))
        except EvidenceError as err:
            self.notify(f"{err.message}\n{err.hint}".strip(), title=err.title, severity="error", timeout=8)
        self.render_inspector()


def run(evidence_root=None, transcript_dir=None, study="fixed_dev_mean", export_dir=None) -> None:
    AuditorApp(Auditor(evidence_root, transcript_dir, export_dir), study=study).run()

"""Record a short demo of the real TUI.

Drives the actual app headlessly with Textual's Pilot, pressing the keys listed
in SCRIPT, and saves one SVG frame per step. If Chrome and ffmpeg are present
it rasterizes the frames and writes tui/recording/model-auditor-demo.mp4 (+ .gif).
Every frame is the app's real screen after the keys in its caption; nothing is
mocked. Replay is stepped manually with `n`, so frame timing is not presented
as recorded timing.

    .venv/Scripts/python.exe -m tui.record_demo
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .app import AuditorApp
from .evidence import Auditor

HERE = Path(__file__).resolve().parent
OUT = HERE / "recording"
FRAMES = OUT / "frames"
SIZE = (168, 46)
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

# (keys to press, seconds to hold the frame, caption)
SCRIPT: list[tuple[list[str], float, str]] = [
    ([], 3.5, "Opens directly into the recorded fixed-direction investigation"),
    (["r"], 1.6, "r · recorded replay starts (RECORDED REPLAY badge stays visible)"),
    *[(["n"], 1.0, "n · next recorded event") for _ in range(9)],
    (["escape"], 1.5, "esc · full transcript"),
    (["down", "down", "enter"], 3.5, "enter · expand a real tool call: receipt hash, local re-read, input, output"),
    (["2"], 5.0, "2 · Intervention: same prompt, same weights; only block 19 changes. APPROVE → REFER"),
    (["down"], 2.0, "↓ · ordinary twin is preserved"),
    (["down"], 2.0, "↓ · legitimate approval is preserved"),
    (["c"], 3.5, "c · generic norm-matched arm: repairs nothing, malformed trigger answer"),
    (["c", "c", "c", "c", "v"], 3.5, "v · reverse insertion into the clean reference: approves the trigger"),
    (["v", "right"], 2.5, "→ · next generated case (row 5)"),
    (["3"], 5.0, "3 · Controls: 12/12 beside the FAILED frozen gate"),
    (["pagedown"], 3.5, "reverse insertion is nonspecific: 10/12 triggers AND 12/12 twins"),
    (["pagedown"], 3.5, "why the gate is closed, criterion by criterion"),
    (["pagedown", "pagedown"], 3.0, "48/48 is decision-prefix scoring, not 48 generated answers"),
    (["d"], 3.5, "d · saved residual direction and the development layer sweep"),
    (["escape", "4"], 2.5, "4 · Evidence: every file hashed against its manifest"),
    (["down", "down", "down", "down", "down"], 2.5, "raw generated answers, SHA-256 verified"),
    (["enter"], 3.0, "enter · open the record"),
    (["escape", "e"], 3.5, "e · export report.md + evidence.json"),
    (["slash", *"study matched", "enter"], 3.5, "/study matched · supporting study, separate denominators"),
    (["3"], 4.0, "matched: 24/24 generated repair, gate also FAILED (29 malformed)"),
    (["slash", *"case 2", "enter"], 3.5, "/case 2 · actionable error: no generated answer for that row"),
    (["question_mark"], 3.0, "? · keys and commands"),
]

KEYMAP = {" ": "space", "/": "slash"}


async def capture(export_dir: Path) -> list[tuple[Path, float, str]]:
    app = AuditorApp(Auditor(export_dir=export_dir))
    frames = []
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause(0.6)
        for i, (keys, hold, caption) in enumerate(SCRIPT):
            for key in keys:
                await pilot.press(KEYMAP.get(key, key))
                await pilot.pause(0.05)
            await pilot.pause(0.4)
            path = FRAMES / f"{i:03d}.svg"
            app.save_screenshot(filename=path.name, path=str(FRAMES))
            frames.append((path, hold, caption))
    return frames


def rasterize(svg: Path) -> Path | None:
    if not CHROME.exists():
        return None
    png = svg.with_suffix(".png")
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg.read_text(encoding="utf-8"))
    w, h = int(float(m.group(1))), int(float(m.group(2)))
    profile = Path(tempfile.gettempdir()) / "model-auditor-record-chrome"
    subprocess.run([str(CHROME), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                    f"--user-data-dir={profile}", "--force-device-scale-factor=1", f"--window-size={w},{h}",
                    f"--screenshot={png}", svg.resolve().as_uri()], timeout=60, capture_output=True)
    return png if png.exists() else None


def encode(frames: list[tuple[Path, float, str]]) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    pngs = [(rasterize(svg), hold) for svg, hold, _ in frames]
    if not ffmpeg or any(p is None for p, _ in pngs):
        return []
    concat = OUT / "frames" / "concat.txt"
    lines = []
    for png, hold in pngs:
        lines += [f"file '{png.as_posix()}'", f"duration {hold}"]
    lines.append(f"file '{pngs[-1][0].as_posix()}'")
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mp4, gif = OUT / "model-auditor-demo.mp4", OUT / "model-auditor-demo.gif"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-r", "10", "-c:v", "libx264",
                    "-crf", "20", "-movflags", "+faststart", str(mp4)], check=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                    "fps=5,scale=1280:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse",
                    str(gif)], check=True)
    return [mp4, gif]


def main() -> None:
    FRAMES.mkdir(parents=True, exist_ok=True)
    for old in FRAMES.glob("*"):
        old.unlink()
    export_dir = Path(tempfile.mkdtemp(prefix="model-auditor-demo-export-"))
    frames = asyncio.run(capture(export_dir))
    (OUT / "script.json").write_text(json.dumps(
        [{"frame": p.name, "hold_seconds": h, "caption": c} for p, h, c in frames], indent=2), encoding="utf-8")
    outputs = encode(frames)
    print(f"{len(frames)} frames in {FRAMES}")
    for path in outputs:
        print(path)
    if not outputs:
        print("Chrome or ffmpeg not found: SVG frames only.")


if __name__ == "__main__":
    main()

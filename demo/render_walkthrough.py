"""Encode captioned, verified browser captures as a local silent walkthrough."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPTURES = HERE / 'captures'
OUTPUTS = Path('C:/Users/sarta/Documents/Codex/2026-10-07/i-saved-the-project-overview-to/outputs')
DURATIONS = [10, 7, 10, 10, 7, 8, 9, 8, 7, 12]


def run(argv):
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-5000:])
    return result.stdout


def main():
    manifest = json.loads((CAPTURES / 'walkthrough-captures.json').read_text())
    shots = manifest['shots']
    assert len(shots) == len(DURATIONS) == 10
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    assert ffmpeg and ffprobe
    work = HERE / 'video-work'
    work.mkdir(exist_ok=True)
    font = work / 'caption-font.ttf'
    shutil.copyfile('C:/Windows/Fonts/segoeui.ttf', font)
    clips = []
    facts = []
    for index, (shot, duration) in enumerate(zip(shots, DURATIONS)):
        source = Path(shot['path'])
        size = json.loads(run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                              '-show_entries', 'stream=width,height', '-of', 'json', str(source)]))['streams'][0]
        width, height = size['width'], size['height']
        caption_lines = shot['caption'].splitlines()
        assert len(caption_lines) == 2
        captions = []
        for line_index, text in enumerate(caption_lines):
            caption = work / f'{index:02d}-caption-{line_index}.txt'
            caption.write_bytes(text.encode('utf-8'))
            captions.append(caption)
        clip = work / f'{index:02d}-clip.mp4'
        # Relative filter paths avoid Windows-drive colon ambiguity in FFmpeg filters.
        filter_text = (f'pad=ceil(iw/2)*2:ceil((ih+92)/2)*2:0:0:color=0x173154,'
                       f'drawbox=x=0:y={height}:w=iw:h=3:color=0x244fd5:t=fill,'
                       f"drawtext=fontfile='{font.name}':textfile='{captions[0].name}':"
                       f'fontcolor=white:fontsize=22:x=26:y={height+15},'
                       f"drawtext=fontfile='{font.name}':textfile='{captions[1].name}':"
                       f'fontcolor=white:fontsize=22:x=26:y={height+50}')
        result = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                                 '-loop', '1', '-framerate', '24', '-i', str(source),
                                 '-t', str(duration), '-vf', filter_text, '-c:v', 'libx264',
                                 '-threads', '4', '-preset', 'medium', '-crf', '24',
                                 '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(clip)],
                                cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-5000:])
        clips.append(clip)
        facts.append({**shot, 'duration_seconds': duration, 'width': width, 'height': height,
                      'capture_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        print(json.dumps({'clip': index+1, 'seconds': duration, 'bytes': clip.stat().st_size}), flush=True)
    listing = work / 'concat.txt'
    listing.write_text(''.join("file '" + p.name + "'\n" for p in clips), encoding='utf-8')
    destination = OUTPUTS / 'Model-Auditor-walkthrough.mp4'
    run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
         '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(destination)])
    probe = json.loads(run([ffprobe, '-v', 'error', '-show_entries', 'format=duration,size:stream=codec_name,width,height,r_frame_rate',
                           '-of', 'json', str(destination)]))
    receipt = {'status': 'encoded', 'capture_method': manifest['capture_method'],
               'audio': 'None; captions are burned into every frame.',
               'scope': 'Recorded evidence replay, not a live inference session.',
               'output': str(destination), 'bytes': destination.stat().st_size,
               'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
               'probe': probe, 'shots': facts}
    (OUTPUTS / 'Model-Auditor-walkthrough-receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    assert 45 <= float(probe['format']['duration']) <= 90
    assert destination.stat().st_size < 10_000_000
    print(json.dumps({k:v for k,v in receipt.items() if k!='shots'}), flush=True)


if __name__ == '__main__':
    main()

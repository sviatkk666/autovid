"""Audio mix stage — realize the Sound Designer's blueprint into sound.

The director's Sound Designer leaves each scene a list of SFX cues with timing
(`scene.sfx = [{cue, at_sec}]`) and a music mood (`project.music`). This stage
turns those into actual audio and mixes them under the narration of the finished
montage video:

  - SFX: synthesized procedurally with ffmpeg (no sound library or API needed) —
    a small palette (chime, whoosh/air, impact, click) keyed off the cue text —
    each placed at its absolute timeline position.
  - Music: a bed from a file (montage.music_file, or a `music/` folder matched to
    the mood) if available, else a subtle synthesized ambient pad. Mixed low.

Everything is amix'd with the video's existing narration and muxed back onto the
video in place (video stream copied, audio re-encoded).

Entry point: mix_project(project, cfg, force=False) -> str | None  (video_path)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..config import ROOT
from ..project import Project
from .montage import _require_ffmpeg, _scene_duration, ffmpeg_bin

WORK_SUBDIR = "_audiomix"
_AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac")


# --- SFX palette (procedural, via ffmpeg lavfi) ------------------------------

def _sfx_args(cue: str) -> tuple[list[str], str]:
    """Return (ffmpeg input+filter args, kind) synthesizing a one-shot for a cue."""
    c = cue.lower()

    def has(*words):
        return any(w in c for w in words)

    if has("chime", "bell", "ding", "ring", "shimmer", "sparkle"):
        # two decaying partials -> a soft bell
        return (["-f", "lavfi", "-i", "sine=f=880:d=1.4",
                 "-f", "lavfi", "-i", "sine=f=1320:d=1.4",
                 "-filter_complex",
                 "[0][1]amix=2:normalize=0,afade=t=out:st=0.15:d=1.2,volume=1.6[a]",
                 "-map", "[a]"], "chime")
    if has("impact", "thud", "boom", "hit", "drop", "weight", "footstep", "step", "stomp"):
        # low body + short noise transient
        return (["-f", "lavfi", "-i", "sine=f=66:d=0.5",
                 "-f", "lavfi", "-i", "anoisesrc=d=0.12:c=brown:a=0.5",
                 "-filter_complex",
                 "[0]afade=t=out:st=0.05:d=0.45,volume=3[b];"
                 "[1]lowpass=f=900,afade=t=out:d=0.11[n];"
                 "[b][n]amix=2:normalize=0[a]",
                 "-map", "[a]"], "impact")
    if has("click", "tick", "tap", "pop", "snap"):
        return (["-f", "lavfi", "-i", "anoisesrc=d=0.05:c=white:a=0.4",
                 "-af", "highpass=f=1200,afade=t=out:d=0.045,volume=1.4"], "click")
    # default: airy noise swell — covers whoosh/wind/breath/exhale/release/swell/riser
    longer = has("swell", "riser", "rise", "build", "wind", "drone")
    d = 1.6 if longer else 0.9
    return (["-f", "lavfi", "-i", f"anoisesrc=d={d}:c=pink:a=0.6",
             "-af", f"highpass=f=380,lowpass=f=3600,"
                    f"afade=t=in:d={d*0.55:.2f},afade=t=out:st={d*0.55:.2f}:d={d*0.45:.2f},volume=1.4"],
            "air")


def _synth_sfx(cue: str, out: Path, binary: str) -> bool:
    args, _ = _sfx_args(cue)
    try:
        subprocess.run([binary, "-y", *args, "-ac", "2", "-ar", "44100", str(out)],
                       check=True, capture_output=True)
        return out.exists()
    except subprocess.CalledProcessError as e:
        print(f"[audiomix] sfx '{cue}' synth failed ({e.stderr[:120] if e.stderr else e})", file=sys.stderr)
        return False


# --- music bed ---------------------------------------------------------------

def _resolve_music_file(project: Project, cfg: dict) -> Path | None:
    mcfg = cfg.get("montage", {})
    explicit = mcfg.get("music_file")
    if explicit and Path(explicit).exists():
        return Path(explicit)
    # a `music/` folder at the repo root: pick a file matching the mood, else first
    mdir = ROOT / "music"
    if mdir.is_dir():
        files = [f for f in sorted(mdir.iterdir()) if f.suffix.lower() in _AUDIO_EXTS]
        if files:
            mood = set((project.music or "").lower().split())
            best = max(files, key=lambda f: len(mood & set(f.stem.lower().replace("-", " ").split())))
            return best
    return None


def _prep_music_file(src: Path, out: Path, total: float, binary: str) -> bool:
    try:
        subprocess.run(
            [binary, "-y", "-stream_loop", "-1", "-i", str(src), "-t", f"{total:.2f}",
             "-af", f"afade=t=in:d=2,afade=t=out:st={max(0, total-3):.2f}:d=3",
             "-ac", "2", "-ar", "44100", str(out)],
            check=True, capture_output=True)
        return out.exists()
    except subprocess.CalledProcessError:
        return False


def _synth_pad(out: Path, total: float, binary: str) -> bool:
    """A subtle warm ambient drone (A minor-ish), for use as a low underscore."""
    try:
        subprocess.run(
            [binary, "-y",
             "-f", "lavfi", "-i", f"sine=f=110:d={total:.2f}",
             "-f", "lavfi", "-i", f"sine=f=164.81:d={total:.2f}",
             "-f", "lavfi", "-i", f"sine=f=220:d={total:.2f}",
             "-filter_complex",
             "[0][1][2]amix=3:normalize=0,lowpass=f=1100,tremolo=f=0.2:d=0.3,"
             f"afade=t=in:d=3,afade=t=out:st={max(0, total-4):.2f}:d=4[a]",
             "-map", "[a]", "-ac", "2", "-ar", "44100", str(out)],
            check=True, capture_output=True)
        return out.exists()
    except subprocess.CalledProcessError:
        return False


# --- entry point -------------------------------------------------------------

def mix_project(project: Project, cfg: dict, force: bool = False) -> str | None:
    if not project.video_path or not (project.dir / project.video_path).exists():
        raise RuntimeError("no video yet — run montage before audiomix.")
    binary = ffmpeg_bin(cfg)
    _require_ffmpeg(binary)
    mcfg = cfg.get("montage", {})
    do_sfx = mcfg.get("sfx", True)
    do_music = mcfg.get("music", True)
    sfx_gain = float(mcfg.get("sfx_gain", 0.5))
    music_gain = float(mcfg.get("music_gain", 0.12))
    fallback = mcfg.get("fallback_scene_seconds", 4)

    # absolute timeline positions for each sfx cue
    events: list[tuple[float, str]] = []
    t = 0.0
    for s in project.scenes:
        dur = _scene_duration(s, project, fallback)
        if do_sfx:
            for fx in (s.sfx or []):
                cue = (fx.get("cue") or "").strip()
                if not cue:
                    continue
                at = max(0.0, min(float(fx.get("at_sec", 0) or 0), max(0.0, dur - 0.1)))
                if t + at < t + dur:
                    events.append((round(t + at, 2), cue))
        t += dur
    total = max(t, 1.0)

    music_src = _resolve_music_file(project, cfg) if do_music else None
    want_music = do_music and (music_src is not None or mcfg.get("music_synth", True))
    if not events and not want_music:
        print("[audiomix] nothing to mix (no sfx cues, no music)", file=sys.stderr)
        return project.video_path

    work = project.dir / WORK_SUBDIR
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # synth each sfx one-shot
    sfx_files: list[tuple[float, Path]] = []
    for i, (start, cue) in enumerate(events):
        f = work / f"sfx_{i:02d}.wav"
        if _synth_sfx(cue, f, binary):
            sfx_files.append((start, f))

    # music bed
    bed: Path | None = None
    if want_music:
        bed = work / "bed.wav"
        ok = _prep_music_file(music_src, bed, total, binary) if music_src else _synth_pad(bed, total, binary)
        if not ok:
            bed = None

    # final mix: video narration + delayed sfx + low music bed
    video = project.dir / project.video_path
    inputs: list[str] = ["-i", str(video)]
    parts = ["[0:a]aformat=channel_layouts=stereo[nar]"]
    labels = ["[nar]"]
    idx = 1
    for start, f in sfx_files:
        inputs += ["-i", str(f)]
        ms = int(start * 1000)
        parts.append(f"[{idx}:a]aformat=channel_layouts=stereo,adelay={ms}:all=1,volume={sfx_gain}[s{idx}]")
        labels.append(f"[s{idx}]")
        idx += 1
    if bed is not None:
        inputs += ["-i", str(bed)]
        parts.append(f"[{idx}:a]aformat=channel_layouts=stereo,volume={music_gain}[bed]")
        labels.append("[bed]")
        idx += 1

    if len(labels) == 1:
        print("[audiomix] nothing synthesized — leaving video as-is", file=sys.stderr)
        shutil.rmtree(work, ignore_errors=True)
        return project.video_path

    parts.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0:duration=first[aout]")
    out = work / "mixed.mp4"
    cmd = [binary, "-y", *inputs, "-filter_complex", ";".join(parts),
           "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"audiomix failed: {(e.stderr or b'')[-300:].decode(errors='ignore')}")

    os.replace(out, video)
    shutil.rmtree(work, ignore_errors=True)
    print(f"[audiomix] mixed {len(sfx_files)} sfx"
          f"{' + music bed' if bed is not None else ''} -> {project.video_path}", file=sys.stderr)
    return project.video_path

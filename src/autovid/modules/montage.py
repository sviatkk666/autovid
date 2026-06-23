"""Montage stage — combine per-scene images + audio into the final video.

For each scene it renders a clip: the scene's still image, held for the length
of the scene's audio (or its estimated duration if there's no audio), with that
audio as the soundtrack. Clips are letterboxed to the project's aspect and then
concatenated into projects/<slug>/output/<slug>.mp4.

Uses ffmpeg directly (no extra Python deps). Per-scene clips are kept under
output/clips/ so each step can be inspected.

Entry point: build_video(project, cfg, force=False, dry_run=False) -> Path | None
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from ..config import env
from ..project import Project
from ..util import audio_duration

OUTPUT_SUBDIR = "output"
CLIPS_SUBDIR = "output/clips"

# short edge (in px) by quality label
_QUALITY = {"720p": 720, "1080p": 1080, "1440p": 1440, "2160p": 2160}


def ffmpeg_bin(cfg: dict) -> str:
    return cfg.get("montage", {}).get("ffmpeg_binary") or env("FFMPEG_BINARY", "ffmpeg")


def _require_ffmpeg(binary: str) -> None:
    if not shutil.which(binary) and not Path(binary).exists():
        raise RuntimeError(
            f"ffmpeg binary '{binary}' not found. Install ffmpeg "
            "(https://ffmpeg.org/download.html) or set montage.ffmpeg_binary."
        )


def resolution(aspect: str, quality: str) -> tuple[int, int]:
    """(width, height) for the given aspect and quality (both even)."""
    short = _QUALITY.get(quality, 1080)
    long = round(short * 16 / 9)
    long += long % 2  # keep even for yuv420p
    return (short, long) if aspect == "9:16" else (long, short)


def _scene_duration(scene, project: Project, fallback: float) -> float:
    if scene.audio_path:
        dur = audio_duration(project.dir / scene.audio_path)
        if dur:
            return dur
    return scene.est_duration_sec if scene.est_duration_sec > 0 else fallback


def _fit_filter(w: int, h: int, bg: str) -> str:
    return (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={bg},setsar=1"
    )


def _kenburns_filter(w: int, h: int, fps: int, duration: float,
                     zoom: float, index: int) -> str:
    """A slow zoom (Ken Burns) filter for a still image filling the frame.

    Upscales+crops the image to cover the frame (no letterbox), then `zoompan`
    eases the zoom from 1.0 to 1+zoom across the clip. Direction alternates per
    scene (zoom-in / zoom-out) for visual variety. Upscaling first keeps the
    motion smooth (zoompan jitters on small inputs).
    """
    frames = max(2, round(duration * fps))
    inc = zoom / (frames - 1)
    sw, sh = w * 4, h * 4  # supersample so the slow zoom stays crisp
    if index % 2 == 0:  # zoom in: 1.0 -> 1+zoom
        zexpr = f"1+{inc:.6f}*on"
    else:               # zoom out: 1+zoom -> 1.0
        zexpr = f"{1 + zoom:.6f}-{inc:.6f}*on"
    return (
        f"scale={sw}:{sh}:force_original_aspect_ratio=increase,crop={sw}:{sh},"
        f"zoompan=z='{zexpr}':d={frames}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"s={w}x{h}:fps={fps},setsar=1"
    )


def scene_clip_cmd(
    binary: str, image: Path, audio: Path | None, out: Path,
    w: int, h: int, fps: int, bg: str, duration: float,
    ken_burns: bool = False, kb_zoom: float = 0.12, kb_index: int = 0,
) -> list[str]:
    """ffmpeg command to render one scene clip (still image + audio/silence)."""
    cmd = [binary, "-y", "-loop", "1", "-i", str(image)]
    if audio is not None:
        cmd += ["-i", str(audio)]
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
    cmd += ["-c:v", "libx264", "-preset", "medium"]
    # A still image encodes leaner with -tune stillimage; motion (Ken Burns) doesn't.
    if ken_burns:
        vf = _kenburns_filter(w, h, fps, duration, kb_zoom, kb_index)
    else:
        vf = _fit_filter(w, h, bg)
        cmd += ["-tune", "stillimage"]
    cmd += [
        "-pix_fmt", "yuv420p", "-r", str(fps),
        "-c:a", "aac", "-b:a", "192k",
        "-vf", vf,
    ]
    # With real audio, let -shortest stop at the audio's end; otherwise cap with -t.
    cmd += ["-shortest"] if audio is not None else ["-t", f"{duration:.3f}"]
    cmd += [str(out)]
    return cmd


def concat_cmd(binary: str, list_file: Path, out: Path) -> list[str]:
    return [binary, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c", "copy", str(out)]


def _run(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print("  " + " ".join(cmd), file=sys.stderr)
        return
    subprocess.run(cmd, check=True, capture_output=True)


def build_video(
    project: Project,
    cfg: dict,
    force: bool = False,
    dry_run: bool = False,
) -> Path | None:
    mcfg = cfg.get("montage", {})
    binary = ffmpeg_bin(cfg)
    if not dry_run:
        _require_ffmpeg(binary)

    w, h = resolution(project.aspect, mcfg.get("quality", "1080p"))
    fps = int(mcfg.get("fps", 30))
    bg = mcfg.get("background", "black")
    fallback = float(mcfg.get("fallback_scene_seconds", 4))
    ken_burns = bool(mcfg.get("ken_burns", True))
    kb_zoom = float(mcfg.get("ken_burns_zoom", 0.12))

    out_dir = project.dir / OUTPUT_SUBDIR
    clips_dir = project.dir / CLIPS_SUBDIR
    final = out_dir / f"{project.slug}.mp4"

    if final.exists() and not force and not dry_run:
        print(f"[montage] {final} exists (use --force to rebuild)", file=sys.stderr)
        return final

    if not dry_run:
        clips_dir.mkdir(parents=True, exist_ok=True)

    clip_paths: list[Path] = []
    kb_i = 0
    for scene in project.scenes:
        if not scene.image_path:
            print(f"[montage] scene {scene.id}: skip (no image)", file=sys.stderr)
            continue
        image = project.dir / scene.image_path
        audio = (project.dir / scene.audio_path) if scene.audio_path else None
        duration = _scene_duration(scene, project, fallback)
        clip = clips_dir / f"scene_{scene.id:02d}.mp4"

        motion = "ken-burns" if ken_burns else "static"
        print(f"[montage] scene {scene.id}: {scene.image_path} "
              f"+ {'audio' if audio else 'silence'} ({duration:.1f}s, {motion})", file=sys.stderr)
        _run(scene_clip_cmd(binary, image, audio, clip, w, h, fps, bg, duration,
                            ken_burns=ken_burns, kb_zoom=kb_zoom, kb_index=kb_i), dry_run)
        clip_paths.append(clip)
        kb_i += 1

    if not clip_paths:
        print("[montage] no renderable scenes (need at least an image).", file=sys.stderr)
        return None

    list_file = clips_dir / "clips.txt"
    list_text = "".join(f"file '{c.resolve()}'\n" for c in clip_paths)
    if dry_run:
        print(f"[montage] concat list ({len(clip_paths)} clips) -> {list_file}", file=sys.stderr)
    else:
        list_file.write_text(list_text, encoding="utf-8")

    print(f"[montage] concatenating {len(clip_paths)} clips -> {final}", file=sys.stderr)
    _run(concat_cmd(binary, list_file, final), dry_run)

    if not dry_run:
        project.video_path = final.relative_to(project.dir).as_posix()
        if project.status == "draft":
            project.status = "ready"   # rendered → ready to review/publish
        project.save()
    return final

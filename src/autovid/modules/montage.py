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


# Per-scene camera moves the montage can apply to a still (scene.animation). The
# art director / producer assign these for visual rhythm; "auto" alternates a slow
# zoom in/out per scene (the classic Ken Burns) when nothing is specified.
ANIMATIONS = ("auto", "kenburns-in", "kenburns-out", "zoom-in", "zoom-out",
              "pan-left", "pan-right", "pan-up", "pan-down", "static")


def _motion_filter(animation: str, w: int, h: int, fps: int, duration: float,
                   zoom: float, index: int) -> str:
    """Build a ffmpeg filter that animates a still image to fill the WxH frame.

    Supersamples + crops to cover (no letterbox), then `zoompan` applies the
    requested move: a slow zoom (in/out), a directional pan, or none ("static").
    "auto" alternates zoom in/out per scene for variety. Upscaling first keeps
    the slow motion crisp (zoompan jitters on small inputs).
    """
    a = (animation or "auto").strip().lower().replace("_", "-").replace("ken-burns", "kenburns")
    frames = max(2, round(duration * fps))
    n1 = max(1, frames - 1)
    sw, sh = w * 4, h * 4
    cover = f"scale={sw}:{sh}:force_original_aspect_ratio=increase,crop={sw}:{sh}"
    inc = zoom / n1
    zin, zout = f"1+{inc:.6f}*on", f"{1 + zoom:.6f}-{inc:.6f}*on"
    pz = 1 + max(zoom, 0.18)  # constant zoom for pans, so there's room to move

    def zp(z: str, x: str = "iw/2-(iw/zoom/2)", y: str = "ih/2-(ih/zoom/2)") -> str:
        return (f"{cover},zoompan=z='{z}':d={frames}:x='{x}':y='{y}':"
                f"s={w}x{h}:fps={fps},setsar=1")

    if a in ("static", "none"):
        return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1"
    if a == "auto":
        return zp(zin if index % 2 == 0 else zout)
    if a in ("kenburns-in", "kenburns", "zoom-in", "zoom"):
        return zp(zin)
    if a in ("kenburns-out", "zoom-out"):
        return zp(zout)
    if a == "pan-left":
        return zp(f"{pz:.4f}", x=f"(iw-iw/zoom)*(1-on/{n1})")
    if a == "pan-right":
        return zp(f"{pz:.4f}", x=f"(iw-iw/zoom)*(on/{n1})")
    if a == "pan-up":
        return zp(f"{pz:.4f}", y=f"(ih-ih/zoom)*(1-on/{n1})")
    if a == "pan-down":
        return zp(f"{pz:.4f}", y=f"(ih-ih/zoom)*(on/{n1})")
    return zp(zin)  # unknown → gentle zoom in


# Scene-to-scene transitions (scene.transition), realized with ffmpeg's xfade.
TRANSITIONS = ("cut", "fade", "fadeblack", "fadewhite", "dissolve",
               "slide-left", "slide-right", "slide-up", "slide-down",
               "wipe-left", "wipe-right", "wipe-up", "wipe-down", "zoom", "circle")
_XFADE = {
    "fade": "fade", "crossfade": "fade", "dissolve": "dissolve",
    "fadeblack": "fadeblack", "fade-black": "fadeblack",
    "fadewhite": "fadewhite", "fade-white": "fadewhite",
    "slide-left": "slideleft", "slide-right": "slideright",
    "slide-up": "slideup", "slide-down": "slidedown",
    "wipe-left": "wipeleft", "wipe-right": "wiperight",
    "wipe-up": "wipeup", "wipe-down": "wipedown",
    "zoom": "zoomin", "zoom-in": "zoomin", "circle": "circleopen",
    "circleopen": "circleopen", "radial": "radial", "smooth": "smoothleft",
}


def _xfade_name(t: str) -> str:
    return _XFADE.get((t or "").strip().lower().replace("_", "-"), "")


def normalize_animation(s: str) -> str:
    """Canonicalize a camera-move name to one of ANIMATIONS (default 'auto')."""
    a = (s or "auto").strip().lower().replace("_", "-").replace(" ", "-").replace("ken-burns", "kenburns")
    return a if a in ANIMATIONS else "auto"


def normalize_transition(s: str) -> str:
    """Canonicalize a transition name to one of TRANSITIONS ('' = use the default)."""
    t = (s or "").strip().lower().replace("_", "-").replace(" ", "-")
    return t if t in TRANSITIONS else ""


def scene_clip_cmd(
    binary: str, image: Path, audio: Path | None, out: Path,
    w: int, h: int, fps: int, bg: str, duration: float,
    motion: bool = False, animation: str = "auto", kb_zoom: float = 0.12, kb_index: int = 0,
) -> list[str]:
    """ffmpeg command to render one scene clip (still image + audio/silence).

    `motion` toggles camera movement globally; `animation` picks the move for this
    scene (see ANIMATIONS). With motion off the still is letterboxed and static.
    """
    cmd = [binary, "-y", "-loop", "1", "-i", str(image)]
    if audio is not None:
        cmd += ["-i", str(audio)]
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
    cmd += ["-c:v", "libx264", "-preset", "medium"]
    # A truly static still encodes leaner with -tune stillimage; any move does not.
    if motion and (animation or "auto").strip().lower() not in ("static", "none"):
        vf = _motion_filter(animation, w, h, fps, duration, kb_zoom, kb_index)
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


def xfade_concat_cmd(binary: str, clips: list[Path], durations: list[float],
                     transitions: list[str], out: Path, fps: int = 30,
                     xd: float = 0.5, default: str = "fade") -> list[str] | None:
    """Assemble clips with scene-to-scene transitions via xfade (+ audio acrossfade).

    transitions[i] is the transition INTO clip i (transitions[0] is ignored).
    Returns None if no real transition is requested (caller uses plain concat).
    A boundary's duration is clamped so it never exceeds either neighbour clip.
    """
    if len(clips) < 2:
        return None
    names, durs = [], []
    any_real = False
    for j in range(1, len(clips)):
        t = (transitions[j] if j < len(transitions) else "") or ""
        tl = t.strip().lower()
        if tl == "cut":
            names.append("fade"); durs.append(0.06)           # near-instant = hard cut
        elif _xfade_name(tl):
            names.append(_xfade_name(tl)); durs.append(xd); any_real = True
        else:
            names.append(_xfade_name(default) or "fade")      # unset → the default look
            durs.append(xd if default != "cut" else 0.06)
            any_real = any_real or default != "cut"
    if not any_real:
        return None
    # clamp each boundary so it can never exceed ~half of either neighbour clip
    # (xfade/acrossfade error out if the overlap is longer than an input)
    for j in range(len(durs)):
        durs[j] = max(0.05, min(durs[j], 0.45 * durations[j], 0.45 * durations[j + 1]))

    inputs: list[str] = []
    for c in clips:
        inputs += ["-i", str(c)]
    parts: list[str] = []
    vprev, aprev = "0:v", "0:a"
    acc = durations[0]
    for j in range(1, len(clips)):
        d, name = durs[j - 1], names[j - 1]
        off = max(0.0, acc - d)
        vout, aout = f"vx{j}", f"ax{j}"
        parts.append(f"[{vprev}][{j}:v]xfade=transition={name}:duration={d:.3f}:offset={off:.3f}[{vout}]")
        parts.append(f"[{aprev}][{j}:a]acrossfade=d={d:.3f}[{aout}]")
        vprev, aprev = vout, aout
        acc = acc + durations[j] - d
    return [binary, "-y", *inputs, "-filter_complex", ";".join(parts),
            "-map", f"[{vprev}]", "-map", f"[{aprev}]",
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-c:a", "aac", "-b:a", "192k", str(out)]


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
    ken_burns = bool(mcfg.get("ken_burns", True))     # global camera-motion toggle
    kb_zoom = float(mcfg.get("ken_burns_zoom", 0.12))
    do_transitions = bool(mcfg.get("transitions", True))
    xd = float(mcfg.get("transition_seconds", 0.5))
    default_transition = str(mcfg.get("default_transition", "fade"))

    out_dir = project.dir / OUTPUT_SUBDIR
    clips_dir = project.dir / CLIPS_SUBDIR
    final = out_dir / f"{project.slug}.mp4"

    if final.exists() and not force and not dry_run:
        print(f"[montage] {final} exists (use --force to rebuild)", file=sys.stderr)
        return final

    if not dry_run:
        clips_dir.mkdir(parents=True, exist_ok=True)

    clip_paths: list[Path] = []
    durations: list[float] = []
    transitions: list[str] = []
    kb_i = 0
    for scene in project.scenes:
        if not scene.image_path:
            print(f"[montage] scene {scene.id}: skip (no image)", file=sys.stderr)
            continue
        image = project.dir / scene.image_path
        audio = (project.dir / scene.audio_path) if scene.audio_path else None
        duration = _scene_duration(scene, project, fallback)
        clip = clips_dir / f"scene_{scene.id:02d}.mp4"

        anim = (scene.animation or "auto") if ken_burns else "static"
        trans = (scene.transition or "").strip()
        print(f"[montage] scene {scene.id}: {scene.image_path} "
              f"+ {'audio' if audio else 'silence'} ({duration:.1f}s, motion={anim}"
              f"{', into=' + trans if trans else ''})", file=sys.stderr)
        _run(scene_clip_cmd(binary, image, audio, clip, w, h, fps, bg, duration,
                            motion=ken_burns, animation=anim, kb_zoom=kb_zoom, kb_index=kb_i), dry_run)
        clip_paths.append(clip)
        durations.append(duration)
        transitions.append(trans)
        kb_i += 1

    if not clip_paths:
        print("[montage] no renderable scenes (need at least an image).", file=sys.stderr)
        return None

    # Assembly: scene-to-scene transitions via xfade when requested, else a fast
    # stream-copy concat. xfade re-encodes, so it's only used when it adds something;
    # any xfade failure falls back to the plain concat so a render never breaks.
    xcmd = (xfade_concat_cmd(binary, clip_paths, durations, transitions, final,
                             fps=fps, xd=xd, default=default_transition)
            if do_transitions else None)
    if xcmd is not None:
        print(f"[montage] assembling {len(clip_paths)} clips with transitions -> {final}", file=sys.stderr)
        try:
            _run(xcmd, dry_run)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"")[-200:]
            print(f"[montage] xfade assembly failed ({err!r}); plain concat instead", file=sys.stderr)
            xcmd = None
    if xcmd is None:
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

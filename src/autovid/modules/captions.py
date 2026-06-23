"""Captions stage — transcribe the video's narration into subtitles.

Produces projects/<slug>/captions.srt and .vtt (upload the SRT to YouTube, or burn
it into the video for shorts). Two backends, auto-picked:

  whisper  — OpenAI Whisper API (whisper-1) on the rendered audio. Accurate, timed
             at the segment level. Needs OPENAI_API_KEY (no heavy local deps).
  script   — deterministic fallback: each scene's narration over its time window,
             from the scene timeline. Free, always available, text is 100% exact
             (it IS our script), just coarser timing.

Optionally burns the captions into the video (config captions.burn).

Entry point: make_captions(project, cfg, force=False) -> Project
"""

from __future__ import annotations

import subprocess
import sys

from ..config import env
from ..project import Project
from .montage import _require_ffmpeg, _scene_duration, ffmpeg_bin


def _ts(t: float, sep: str) -> str:
    t = max(0.0, t)
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s += 1; ms = 0
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _write_srt(segs: list[tuple[float, float, str]], path) -> None:
    lines = []
    for i, (st, en, tx) in enumerate(segs, 1):
        if not tx:
            continue
        lines.append(f"{i}\n{_ts(st, ',')} --> {_ts(en, ',')}\n{tx}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_vtt(segs: list[tuple[float, float, str]], path) -> None:
    out = ["WEBVTT", ""]
    for st, en, tx in segs:
        if not tx:
            continue
        out.append(f"{_ts(st, '.')} --> {_ts(en, '.')}")
        out.append(tx)
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def _extract_audio(video, out_mp3, binary: str) -> None:
    """Pull a small mono 16 kHz mp3 from the video for transcription (<25 MB API cap)."""
    cmd = [binary, "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
           "-b:a", "64k", str(out_mp3)]
    subprocess.run(cmd, check=True, capture_output=True)


def _whisper_segments(audio_path, cfg) -> list[tuple[float, float, str]]:
    from openai import OpenAI  # raises ImportError if the SDK isn't installed
    key = env("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("no OPENAI_API_KEY for Whisper")
    model = cfg.get("captions", {}).get("model", "whisper-1")
    client = OpenAI(api_key=key)
    with open(audio_path, "rb") as f:
        r = client.audio.transcriptions.create(model=model, file=f, response_format="verbose_json")
    raw = getattr(r, "segments", None)
    if raw is None and isinstance(r, dict):
        raw = r.get("segments")
    segs: list[tuple[float, float, str]] = []
    for s in (raw or []):
        get = (lambda k, d=None: s.get(k, d)) if isinstance(s, dict) else (lambda k, d=None: getattr(s, k, d))
        segs.append((float(get("start", 0) or 0), float(get("end", 0) or 0),
                     str(get("text", "") or "").strip()))
    if not segs:
        raise RuntimeError("Whisper returned no segments")
    return segs


def _script_segments(project: Project, cfg) -> list[tuple[float, float, str]]:
    fallback = float(cfg.get("montage", {}).get("fallback_scene_seconds", 4))
    out, t = [], 0.0
    for s in project.scenes:
        d = _scene_duration(s, project, fallback)
        if s.text.strip():
            out.append((t, t + d, s.text.strip()))
        t += d
    return out


def _burn(project: Project, cfg, srt, binary: str) -> None:
    """Hard-burn the subtitles into the video, in place (re-encode)."""
    import os
    style = cfg.get("captions", {}).get("style",
                                        "FontName=Arial,Fontsize=22,Bold=1,PrimaryColour=&H00FFFFFF,"
                                        "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=40")
    video = project.dir / project.video_path
    out = video.with_suffix(".captioned.mp4")
    # Subtitles filter needs a path it can parse; run from the project dir with a relative name.
    cmd = [binary, "-y", "-i", str(video),
           "-vf", f"subtitles={srt.name}:force_style='{style}'",
           "-c:a", "copy", str(out.name)]
    subprocess.run(cmd, check=True, capture_output=True, cwd=str(project.dir))
    os.replace(out, video)
    print(f"[captions] burned into {project.video_path}", file=sys.stderr)


def make_captions(project: Project, cfg: dict, force: bool = False) -> Project:
    ccfg = cfg.get("captions", {})
    provider = (ccfg.get("provider") or "auto").lower()
    srt = project.dir / "captions.srt"
    vtt = project.dir / "captions.vtt"
    if srt.exists() and not force:
        print("[captions] exist (use force to regenerate)", file=sys.stderr)
        return project

    segs = None
    has_video = bool(project.video_path) and (project.dir / project.video_path).exists()
    if provider in ("auto", "openai", "whisper") and has_video:
        try:
            binary = ffmpeg_bin(cfg)
            _require_ffmpeg(binary)
            tmp = project.dir / "_cap_audio.mp3"
            _extract_audio(project.dir / project.video_path, tmp, binary)
            segs = _whisper_segments(tmp, cfg)
            tmp.unlink(missing_ok=True)
            print(f"[captions] whisper: {len(segs)} segments", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — degrade to deterministic script timing
            print(f"[captions] whisper unavailable ({e}); using script timing", file=sys.stderr)
            segs = None
    if not segs:
        if not project.scenes:
            raise ValueError("nothing to caption — render the video or write scenes first.")
        segs = _script_segments(project, cfg)
        print(f"[captions] script timing: {len(segs)} segments", file=sys.stderr)

    _write_srt(segs, srt)
    _write_vtt(segs, vtt)
    project.captions = {"srt": "captions.srt", "vtt": "captions.vtt"}

    if ccfg.get("burn") and has_video:
        try:
            _burn(project, cfg, srt, ffmpeg_bin(cfg))
            project.captions["burned"] = True
        except Exception as e:  # noqa: BLE001
            print(f"[captions] burn failed ({e})", file=sys.stderr)

    project.save()
    return project

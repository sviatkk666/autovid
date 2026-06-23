"""TTS stage — synthesize voiceover audio for each scene.

For every scene it generates an audio file under projects/<slug>/audio/ and
records the relative path in scene.audio_path. When the produced audio's real
duration is measurable (always for .wav, and for .mp3 if mutagen is installed)
it replaces the rough word-count estimate in scene.est_duration_sec.

Entry point: synthesize_project(project, cfg, tts=None, force=False) -> Project
"""

from __future__ import annotations

import sys

from ..project import Project
from ..providers.tts import TTS, get_tts
from ..util import audio_duration

AUDIO_SUBDIR = "audio"


def synthesize_project(
    project: Project,
    cfg: dict,
    tts: TTS | None = None,
    force: bool = False,
    only: set[int] | None = None,
) -> Project:
    tts = tts or get_tts(cfg)
    audio_dir = project.dir / AUDIO_SUBDIR

    for scene in project.scenes:
        if only is not None and scene.id not in only:
            continue
        out_path = audio_dir / f"scene_{scene.id:02d}.{tts.ext}"
        rel = out_path.relative_to(project.dir).as_posix()

        if scene.audio_path and out_path.exists() and not force:
            print(f"[tts] scene {scene.id}: skip (exists)", file=sys.stderr)
            continue

        text = scene.text.strip()
        if not text:
            print(f"[tts] scene {scene.id}: skip (empty)", file=sys.stderr)
            continue

        # Per-scene voice casting: the scene's voice wins, else the project voice.
        voice = (scene.voice or project.voice or "").strip() or None
        tts.synthesize(text, out_path, voice=voice)
        scene.audio_path = rel

        dur = audio_duration(out_path)
        if dur:
            scene.est_duration_sec = dur
        vtag = f" [{voice}]" if voice else ""
        print(
            f"[tts] scene {scene.id}: {rel} ({dur:.1f}s){vtag}" if dur
            else f"[tts] scene {scene.id}: {rel}{vtag}",
            file=sys.stderr,
        )

    project.save()
    return project

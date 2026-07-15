"""Images stage (generate approach) — synthesize one image per scene.

The counterpart to `images.py` (which searches the web). For every scene it
sends the scene's image prompt to an image generator and writes the result to
projects/<slug>/images/scene_NN.<ext>, recording the path plus a "generated"
provenance on the scene so CREDITS.md notes how each image was made. Either
approach fills the same `image_path` slot, so montage works with either.

Entry point: generate_project(project, cfg, generator=None, force=False) -> Project
"""

from __future__ import annotations

import sys
import time

from ..project import Project
from ..providers.imagegen import ImageGenerator, get_image_generator
from .images import _write_credits

IMAGES_SUBDIR = "images"


def _build_prompt(scene, cfg: dict) -> str:
    """Use the full descriptive image prompt (falls back to narration text)."""
    field = cfg.get("images", {}).get("generate", {}).get("prompt_from", "image_prompt")
    return (getattr(scene, field, "") or scene.image_prompt or scene.text).strip()


def _generate_with_retry(generator, prompt, dest, orientation, *, attempts, backoff):
    """Generate one image, retrying transient failures with exponential backoff.

    Image gateways (ai33 especially) fail intermittently — soft rate-limits,
    slow tasks, the odd 5xx. Rather than drop the scene (a hole in the video),
    we retry so a batch run stays continuous. The final failure is re-raised so
    the caller decides whether to keep going or stop.
    """
    for attempt in range(1, attempts + 1):
        try:
            generator.generate(prompt, dest, orientation=orientation)
            return
        except Exception as e:  # noqa: BLE001 — any provider/network error is retryable
            if attempt >= attempts:
                raise
            wait = backoff * (2 ** (attempt - 1))
            print(f"[imagegen]   attempt {attempt}/{attempts} failed ({e}); "
                  f"retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)


def generate_project(
    project: Project,
    cfg: dict,
    generator: ImageGenerator | None = None,
    force: bool = False,
    only: set[int] | None = None,
) -> Project:
    generator = generator or get_image_generator(cfg)
    orientation = "portrait" if project.aspect == "9:16" else "landscape"
    images_dir = project.dir / IMAGES_SUBDIR

    gcfg = cfg.get("images", {}).get("generate", {})
    attempts = max(1, int(gcfg.get("max_retries", 4)))
    backoff = float(gcfg.get("retry_backoff", 5.0))
    # Persist after each success so a long batch that's interrupted keeps its
    # finished scenes (idempotent: a re-run skips scenes that already have an image).
    failures: list[int] = []
    produced = 0

    for scene in project.scenes:
        if only is not None and scene.id not in only:
            continue
        if scene.image_path and (project.dir / scene.image_path).exists() and not force:
            print(f"[imagegen] scene {scene.id}: skip (exists)", file=sys.stderr)
            continue

        prompt = _build_prompt(scene, cfg)
        if not prompt:
            print(f"[imagegen] scene {scene.id}: skip (empty prompt)", file=sys.stderr)
            continue

        dest = images_dir / f"scene_{scene.id:02d}.{generator.ext}"
        try:
            _generate_with_retry(generator, prompt, dest, orientation,
                                  attempts=attempts, backoff=backoff)
        except Exception as e:  # noqa: BLE001 — exhausted retries for this scene
            failures.append(scene.id)
            print(f"[imagegen] scene {scene.id}: generation failed after {attempts} "
                  f"attempts ({e})", file=sys.stderr)
            continue

        scene.image_path = dest.relative_to(project.dir).as_posix()
        scene.image_source = f"generated:{generator.name}"
        scene.image_license = "AI-generated"
        scene.image_attribution = ""
        scene.image_credit_url = ""
        project.save()
        produced += 1
        print(f"[imagegen] scene {scene.id}: {scene.image_path} [{generator.name}]",
              file=sys.stderr)

    _write_credits(project)
    if failures:
        # Keep the run CONTINUOUS: a scene that exhausted its retries keeps its
        # prior image (montage skips a truly imageless scene), so one flaky
        # generation never aborts the whole production. Only a total outage —
        # nothing produced at all — is worth raising, since that means the
        # generator is down and every scene would be missing.
        msg = (f"imagegen: {len(failures)} scene(s) failed after {attempts} "
               f"attempts: {failures}")
        if produced == 0:
            raise RuntimeError(msg + " — image generation appears to be down")
        print(f"[imagegen] WARNING: {msg}; kept prior images, continuing",
              file=sys.stderr)
    return project

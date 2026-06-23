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

from ..project import Project
from ..providers.imagegen import ImageGenerator, get_image_generator
from .images import _write_credits

IMAGES_SUBDIR = "images"


def _build_prompt(scene, cfg: dict) -> str:
    """Use the full descriptive image prompt (falls back to narration text)."""
    field = cfg.get("images", {}).get("generate", {}).get("prompt_from", "image_prompt")
    return (getattr(scene, field, "") or scene.image_prompt or scene.text).strip()


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
            generator.generate(prompt, dest, orientation=orientation)
        except Exception as e:
            print(f"[imagegen] scene {scene.id}: generation failed ({e})", file=sys.stderr)
            continue

        scene.image_path = dest.relative_to(project.dir).as_posix()
        scene.image_source = f"generated:{generator.name}"
        scene.image_license = "AI-generated"
        scene.image_attribution = ""
        scene.image_credit_url = ""
        print(f"[imagegen] scene {scene.id}: {scene.image_path} [{generator.name}]",
              file=sys.stderr)

    project.save()
    _write_credits(project)
    return project

"""Visuals decoder — realize each scene's image the way its blueprint says.

The director's Art Director tags every scene with a `visual_type`. This module
"decodes" that: it groups scenes by type and calls the right maker once per
group (so providers are built once, not per scene):

  search      -> images.fetch_project          (real stock/internet photo)
  photo_edit  -> photo_edit.photo_edit_project  (stock photo + light HTML polish)
  chart       -> chartgen.chart_project         (LLM HTML/SVG data-viz)
  text_card   -> textcard.textcard_project      (LLM typographic quote/stat card)
  generate    -> imagegen.generate_project      (AI image; kept <=20% by director)
  animation   -> (motion lives in montage via scene.animation) falls back to a stock photo

All makers fill the same scene.image_path slot, so montage is unchanged. (Per-scene
camera MOTION — Ken Burns / pans / zooms — and scene-to-scene TRANSITIONS are a
separate axis applied by montage from scene.animation / scene.transition.)

Entry point: realize_visuals(project, cfg, force=False, only=None) -> Project
"""

from __future__ import annotations

import sys
from collections import defaultdict

from ..project import Project
from .chartgen import chart_project
from .imagegen import generate_project
from .images import fetch_project
from .photo_edit import photo_edit_project
from .textcard import textcard_project

# visual_type -> maker function. "animation" is not a still-maker (motion is applied
# by montage), so it degrades to a real photo (the safe, on-policy default).
_MAKERS = {
    "search": fetch_project,
    "photo_edit": photo_edit_project,
    "chart": chart_project,
    "text_card": textcard_project,
    "generate": generate_project,
    "animation": fetch_project,
}


def realize_visuals(
    project: Project,
    cfg: dict,
    force: bool = False,
    only: set[int] | None = None,
) -> Project:
    scenes = [s for s in project.scenes if only is None or s.id in only]

    # generate_all: every frame is an AI image (via images.generate.provider,
    # e.g. ai33) — no stock/internet search, no charts/cards. One code path,
    # with per-scene retry inside generate_project so the run stays continuous.
    if cfg.get("images", {}).get("generate_all"):
        ids = {s.id for s in scenes}
        if ids:
            print(f"[visuals] generate_all: {len(ids)} scene(s) via AI image gen",
                  file=sys.stderr)
            generate_project(project, cfg, force=force, only=ids)
        return project

    groups: dict[str, set[int]] = defaultdict(set)
    for s in scenes:
        # Tolerate loose stored values ("Text Card" / "text-card") -> canonical key.
        vt = (s.visual_type or "search").strip().lower().replace(" ", "_").replace("-", "_")
        groups[vt].add(s.id)

    for vtype, ids in groups.items():
        maker = _MAKERS.get(vtype, fetch_project)
        if maker is fetch_project and vtype != "search":
            print(f"[visuals] '{vtype}' has no maker — using a stock photo", file=sys.stderr)
        print(f"[visuals] {vtype}: {len(ids)} scene(s)", file=sys.stderr)
        try:
            maker(project, cfg, force=force, only=ids)
        except Exception as e:  # noqa: BLE001 — render/provider/network: degrade, never abort
            print(f"[visuals] {vtype} failed ({e})", file=sys.stderr)
            # An HTML-render type (chart/text_card/photo_edit) failing — e.g. no
            # Chrome — would leave those scenes imageless; fall back to a stock photo
            # so the scene survives into the montage instead of disappearing.
            if maker is not fetch_project:
                print(f"[visuals] falling back to stock photos for {len(ids)} {vtype} scene(s)", file=sys.stderr)
                try:
                    fetch_project(project, cfg, force=force, only=ids)
                except Exception as e2:  # noqa: BLE001
                    print(f"[visuals] stock-photo fallback also failed ({e2})", file=sys.stderr)
    return project

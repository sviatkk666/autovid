"""Visuals decoder — realize each scene's image the way its blueprint says.

The director's Art Director tags every scene with a `visual_type`. This module
"decodes" that: it groups scenes by type and calls the right maker once per
group (so providers are built once, not per scene):

  search      -> images.fetch_project        (real stock/internet photo)
  photo_edit  -> photo_edit.photo_edit_project (stock photo + light HTML polish)
  chart       -> chartgen.chart_project        (LLM HTML/SVG data-viz)
  generate    -> imagegen.generate_project     (AI image; kept <=20% by director)
  animation   -> (no renderer yet) falls back to a stock photo

All makers fill the same scene.image_path slot, so montage is unchanged.

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

# visual_type -> (maker function, label). animation has no realizer yet, so it
# degrades to a real photo (the safe, on-policy default).
_MAKERS = {
    "search": fetch_project,
    "photo_edit": photo_edit_project,
    "chart": chart_project,
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
    groups: dict[str, set[int]] = defaultdict(set)
    for s in scenes:
        groups[(s.visual_type or "search").lower()].add(s.id)

    for vtype, ids in groups.items():
        maker = _MAKERS.get(vtype, fetch_project)
        if vtype == "animation":
            print("[visuals] animation has no renderer yet — using a stock photo",
                  file=sys.stderr)
        print(f"[visuals] {vtype}: {len(ids)} scene(s)", file=sys.stderr)
        try:
            maker(project, cfg, force=force, only=ids)
        except (RuntimeError, ValueError) as e:
            print(f"[visuals] {vtype} failed ({e})", file=sys.stderr)
    return project

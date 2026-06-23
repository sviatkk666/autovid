"""Images stage (chart approach) — visualize a scene as an LLM-authored chart.

The third way to fill a scene's image slot (besides `images.py` search and
`imagegen.py` diffusion): for beats that are about data, structure, steps or
comparisons, a clean chart/diagram beats a stock photo. The LLM writes ONE
self-contained HTML/CSS/SVG visual, and we screenshot it to PNG with headless
Chrome — the exact technique the thumbnail stage uses, so text stays razor sharp
and it costs one cheap LLM call per scene.

Output fills the same `image_path` slot, so montage is unchanged. The HTML
source is kept under projects/<slug>/charts/ for inspection / hand-tweaking.

Entry point: chart_project(project, cfg, force=False, only=None) -> Project
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..providers.llm import get_llm
from ..project import Project
from ..util import strip_code_fence
from .images import _write_credits
from .thumbnail import _chrome_binary, render_html_to_png

IMAGES_SUBDIR = "images"
CHARTS_SUBDIR = "charts"

# Full video-frame resolution per aspect (montage scales / Ken-Burns from here).
_SIZE = {"16:9": (1920, 1080), "9:16": (1080, 1920)}

_SYSTEM = """You are a data-visualization designer who codes. For ONE beat of a \
video's narration you output ONE complete, self-contained HTML document that \
renders a single striking visual — a chart, diagram, infographic, timeline, \
comparison or big-number stat — that makes the idea land.

HARD REQUIREMENTS:
- Output ONLY HTML. No markdown, no commentary, no code fences.
- Exactly {w}x{h} pixels: <body> and the root element must be {w}px by {h}px, \
margin:0, overflow:hidden.
- Inline CSS and inline SVG ONLY. NO external resources of any kind (no <img>, \
no web fonts, no URLs) — it must render fully offline. Use system fonts \
(Arial/Helvetica/sans-serif), CSS gradients, SVG shapes/paths, and emoji.
- Choose the visual that best fits the beat from a WIDE repertoire — vary it across \
the video: bar / column / line / area / pie or donut / progress or gauge / \
scatter; or a conceptual diagram — flowchart, step/process, timeline, comparison \
or VS split, before/after, hierarchy / pyramid, Venn, quadrant / 2x2 matrix, \
mind-map, cycle, funnel, labelled illustration, or a big-number stat block. If the \
narration has concrete numbers, chart them honestly; otherwise draw the structure.
- Cinematic and high-contrast, full-bleed background suited to video. ONE focal \
visual with minimal, large, legible text — not a slide full of bullet points.
- Do NOT restate the narration as a paragraph. Visualize it."""

_USER = """Video title: {title}

This scene's narration:
{text}
{hint}
Design the single best visual for this beat. Return only the HTML document."""


def _resolution(aspect: str) -> tuple[int, int]:
    return _SIZE.get(aspect, _SIZE["16:9"])


def _design_html(scene, project: Project, cfg, llm, w: int, h: int) -> str:
    hint = ""
    prompt = (scene.image_prompt or "").strip()
    if prompt:
        hint = f"\nVisual hint (optional): {prompt}\n"
    html = llm.complete(
        _SYSTEM.format(w=w, h=h),
        _USER.format(title=project.title or project.slug, text=scene.text.strip(), hint=hint),
    )
    return strip_code_fence(html)


def chart_project(
    project: Project,
    cfg: dict,
    force: bool = False,
    only: set[int] | None = None,
) -> Project:
    binary = _chrome_binary(cfg)  # fail early if no headless renderer
    w, h = _resolution(project.aspect)
    llm = get_llm(cfg)
    images_dir = project.dir / IMAGES_SUBDIR
    charts_dir = project.dir / CHARTS_SUBDIR

    for scene in project.scenes:
        if only is not None and scene.id not in only:
            continue
        if scene.image_path and (project.dir / scene.image_path).exists() and not force:
            print(f"[chart] scene {scene.id}: skip (exists)", file=sys.stderr)
            continue

        if not scene.text.strip():
            print(f"[chart] scene {scene.id}: skip (empty)", file=sys.stderr)
            continue

        try:
            html = _design_html(scene, project, cfg, llm, w, h)
        except (RuntimeError, ValueError) as e:
            print(f"[chart] scene {scene.id}: design failed ({e})", file=sys.stderr)
            continue

        html_file = charts_dir / f"scene_{scene.id:02d}.html"
        html_file.parent.mkdir(parents=True, exist_ok=True)
        html_file.write_text(html, encoding="utf-8")

        png = images_dir / f"scene_{scene.id:02d}.png"
        try:
            render_html_to_png(binary, html_file, png, w, h)
        except RuntimeError as e:
            print(f"[chart] scene {scene.id}: render failed ({e})", file=sys.stderr)
            continue

        scene.image_path = png.relative_to(project.dir).as_posix()
        scene.image_source = "chart:llm"
        scene.image_license = "AI-generated (chart)"
        scene.image_attribution = ""
        scene.image_credit_url = ""
        print(f"[chart] scene {scene.id}: {scene.image_path} ({w}x{h})", file=sys.stderr)

    project.save()
    _write_credits(project)
    return project

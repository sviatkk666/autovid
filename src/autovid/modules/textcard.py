"""Text-card stage — a striking typographic card for a beat (no photo, no chart).

A fourth HTML->PNG treatment alongside `chart`: for beats that land hardest as
WORDS — a punchy quote, a single bold statement, a definition, a key statistic,
a short numbered/checklist, or a section title. The LLM writes one self-contained
HTML document (kinetic-poster typography, gradients, shapes — no data plotting)
and we screenshot it to PNG with headless Chrome, so text is razor-sharp.

Output fills the same `image_path` slot, so montage is unchanged. The HTML source
is kept under projects/<slug>/cards/ for inspection / hand-tweaking.

Entry point: textcard_project(project, cfg, force=False, only=None) -> Project
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
CARDS_SUBDIR = "cards"

_SIZE = {"16:9": (1920, 1080), "9:16": (1080, 1920)}

_SYSTEM = """You are a motion-graphics typographer who codes. For ONE beat of a \
video's narration you output ONE complete, self-contained HTML document that \
renders a striking TYPOGRAPHIC card — words as the hero, not a photo or a chart.

Pick the form that lands the beat hardest:
- a short punchy QUOTE or one bold statement (the line the viewer remembers),
- a single huge WORD or NUMBER / key statistic with a tiny caption,
- a crisp DEFINITION (term + meaning),
- a short numbered list or checklist (max 4 items),
- a section TITLE / chapter card.

HARD REQUIREMENTS:
- Output ONLY HTML. No markdown, no commentary, no code fences.
- Exactly {w}x{h} pixels: <body> and the root element are {w}px by {h}px, \
margin:0, overflow:hidden.
- Inline CSS and inline SVG ONLY. NO external resources (no <img>, no web fonts, \
no URLs) — render fully offline. System fonts (Arial/Helvetica/Georgia/sans-serif), \
CSS gradients, shapes, emoji.
- Cinematic, high-contrast, full-bleed background suited to video. ONE focal idea, \
huge legible type, generous spacing — not a slide of bullets.
- Keep the words SHORT and faithful to the beat; do NOT dump the whole narration."""

_USER = """Video title: {title}

This scene's narration:
{text}
{hint}
Design the single best typographic card for this beat. Return only the HTML document."""


def _resolution(aspect: str) -> tuple[int, int]:
    return _SIZE.get(aspect, _SIZE["16:9"])


def _design_html(scene, project: Project, llm, w: int, h: int) -> str:
    prompt = (scene.image_prompt or "").strip()
    hint = f"\nVisual hint (optional): {prompt}\n" if prompt else ""
    html = llm.complete(
        _SYSTEM.format(w=w, h=h),
        _USER.format(title=project.title or project.slug, text=scene.text.strip(), hint=hint),
    )
    return strip_code_fence(html)


def textcard_project(
    project: Project,
    cfg: dict,
    force: bool = False,
    only: set[int] | None = None,
) -> Project:
    binary = _chrome_binary(cfg)  # fail early if no headless renderer
    w, h = _resolution(project.aspect)
    llm = get_llm(cfg)
    images_dir = project.dir / IMAGES_SUBDIR
    cards_dir = project.dir / CARDS_SUBDIR

    for scene in project.scenes:
        if only is not None and scene.id not in only:
            continue
        if scene.image_path and (project.dir / scene.image_path).exists() and not force:
            print(f"[text_card] scene {scene.id}: skip (exists)", file=sys.stderr)
            continue
        if not scene.text.strip():
            print(f"[text_card] scene {scene.id}: skip (empty)", file=sys.stderr)
            continue

        try:
            html = _design_html(scene, project, llm, w, h)
        except (RuntimeError, ValueError) as e:
            print(f"[text_card] scene {scene.id}: design failed ({e})", file=sys.stderr)
            continue

        html_file = cards_dir / f"scene_{scene.id:02d}.html"
        html_file.parent.mkdir(parents=True, exist_ok=True)
        html_file.write_text(html, encoding="utf-8")

        png = images_dir / f"scene_{scene.id:02d}.png"
        try:
            render_html_to_png(binary, html_file, png, w, h)
        except RuntimeError as e:
            print(f"[text_card] scene {scene.id}: render failed ({e})", file=sys.stderr)
            continue

        scene.image_path = png.relative_to(project.dir).as_posix()
        scene.image_source = "text_card:llm"
        scene.image_license = "AI-generated (text card)"
        scene.image_attribution = ""
        scene.image_credit_url = ""
        print(f"[text_card] scene {scene.id}: {scene.image_path} ({w}x{h})", file=sys.stderr)

    project.save()
    _write_credits(project)
    return project

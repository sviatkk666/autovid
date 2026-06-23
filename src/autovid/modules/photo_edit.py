"""Images stage (photo_edit approach) — a real stock photo with a light HTML edit.

The user's "minimal photoshop by Claude": fetch a real stock/internet photo (so
visuals stay authentic, not over-AI'd), then have the LLM author a small HTML
document that places that photo full-bleed and adds a tasteful cinematic polish
(subtle grade, vignette, an optional short text label). We screenshot it to PNG
with headless Chrome — the same renderer charts/thumbnails use.

The photo is injected AFTER the LLM call: the model writes the composition
referencing the literal placeholder string IMAGE_PLACEHOLDER, and we swap in the
base64 data URI. That keeps the (large) image out of the prompt and guarantees
the real photo is used. Stock-photo license/attribution is preserved on the scene
so CREDITS.md still credits it.

Entry point: photo_edit_project(project, cfg, force=False, only=None) -> Project
"""

from __future__ import annotations

import base64
import sys

import requests

from ..project import Project
from ..providers.images import get_image_source
from ..providers.llm import get_llm
from ..util import strip_code_fence
from .images import _build_query, _pick, _search_with_fallback, _write_credits
from .thumbnail import _chrome_binary, render_html_to_png

IMAGES_SUBDIR = "images"
EDITS_SUBDIR = "edits"
_SIZE = {"16:9": (1920, 1080), "9:16": (1080, 1920)}
_PLACEHOLDER = "IMAGE_PLACEHOLDER"

_SYSTEM = """You are a motion-graphics designer doing a LIGHT edit ("minimal \
photoshop") of a REAL photo for a video frame. You output ONE complete, \
self-contained HTML document that places the photo full-bleed and adds tasteful \
cinematic polish over it.

HARD REQUIREMENTS:
- Output ONLY HTML. No markdown, no commentary, no code fences.
- Exactly {w}x{h} pixels: <body> and the root element must be {w}px by {h}px, \
margin:0, overflow:hidden.
- The PHOTO is the base layer and must stay the star. Put it on a full-bleed \
element EXACTLY like:
  background-image:url('IMAGE_PLACEHOLDER'); background-size:cover; background-position:center;
  Use the literal string IMAGE_PLACEHOLDER as the URL — do NOT invent any other \
image URL or <img> source.
- Keep edits MINIMAL and classy: a subtle color grade or gradient overlay, a \
soft vignette, and OPTIONALLY a short on-screen label (<= 6 words) pulled from \
the narration, in a bold legible system font with strong contrast. No heavy \
redrawing — light polish only.
- Inline CSS + inline SVG only. No external resources besides the placeholder."""

_USER = """Scene narration:
{text}

Compose a lightly-polished frame around the photo (use the literal placeholder \
for it). A short text label is optional. Return only the HTML document."""


def _resolution(aspect: str) -> tuple[int, int]:
    return _SIZE.get(aspect, _SIZE["16:9"])


def _ensure_photo(html: str, data_uri: str) -> str:
    """Guarantee the photo shows even if the model forgot the placeholder."""
    if _PLACEHOLDER in html:
        return html.replace(_PLACEHOLDER, data_uri)
    layer = (f'<div style="position:absolute;inset:0;z-index:-1;'
             f"background:url('{data_uri}') center/cover;\"></div>")
    if "</body>" in html:
        return html.replace("</body>", layer + "</body>")
    return html + layer


def photo_edit_project(
    project: Project,
    cfg: dict,
    force: bool = False,
    only: set[int] | None = None,
) -> Project:
    source = get_image_source(cfg)
    binary = _chrome_binary(cfg)  # fail early if no renderer
    w, h = _resolution(project.aspect)
    orientation = "portrait" if project.aspect == "9:16" else "landscape"
    llm = get_llm(cfg)
    http = requests.Session()
    http.headers["User-Agent"] = "autovid/0.1 photo-edit"
    images_dir = project.dir / IMAGES_SUBDIR
    edits_dir = project.dir / EDITS_SUBDIR

    for scene in project.scenes:
        if only is not None and scene.id not in only:
            continue
        if scene.image_path and (project.dir / scene.image_path).exists() and not force:
            print(f"[photo_edit] scene {scene.id}: skip (exists)", file=sys.stderr)
            continue

        query = _build_query(scene, cfg)
        if not query:
            print(f"[photo_edit] scene {scene.id}: skip (empty query)", file=sys.stderr)
            continue
        try:
            results, _used = _search_with_fallback(source, query, orientation)
        except Exception as e:  # noqa: BLE001
            print(f"[photo_edit] scene {scene.id}: search failed ({e})", file=sys.stderr)
            continue
        pick = _pick(results)
        if pick is None:
            print(f"[photo_edit] scene {scene.id}: no stock photo for '{query}'", file=sys.stderr)
            continue

        try:
            r = http.get(pick.url, timeout=30)
            r.raise_for_status()
            ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if not ctype.startswith("image/"):
                ctype = "image/jpeg"  # some hosts mislabel; the bytes are still an image
            data_uri = f"data:{ctype};base64," + base64.b64encode(r.content).decode("ascii")
        except Exception as e:  # noqa: BLE001
            print(f"[photo_edit] scene {scene.id}: photo download failed ({e})", file=sys.stderr)
            continue

        try:
            html = strip_code_fence(llm.complete(
                _SYSTEM.format(w=w, h=h), _USER.format(text=scene.text.strip())))
        except (RuntimeError, ValueError) as e:
            print(f"[photo_edit] scene {scene.id}: compose failed ({e})", file=sys.stderr)
            continue
        html = _ensure_photo(html, data_uri)

        html_file = edits_dir / f"scene_{scene.id:02d}.html"
        html_file.parent.mkdir(parents=True, exist_ok=True)
        html_file.write_text(html, encoding="utf-8")
        png = images_dir / f"scene_{scene.id:02d}.png"
        try:
            render_html_to_png(binary, html_file, png, w, h)
        except RuntimeError as e:
            print(f"[photo_edit] scene {scene.id}: render failed ({e})", file=sys.stderr)
            continue

        scene.image_path = png.relative_to(project.dir).as_posix()
        scene.image_source = f"photo_edit:{pick.source}"
        scene.image_license = pick.license
        scene.image_attribution = pick.attribution
        scene.image_credit_url = pick.credit_url
        print(f"[photo_edit] scene {scene.id}: {scene.image_path} "
              f"[{pick.source} · {pick.license or 'n/a'} + HTML polish]", file=sys.stderr)

    http.close()
    project.save()
    _write_credits(project)
    return project

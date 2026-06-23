"""Thumbnail stage — design a YouTube thumbnail as HTML, render it to PNG.

Claude (or whatever LLM is configured) reads the project's title and script and
writes a self-contained HTML poster (inline CSS, no network resources). We then
screenshot that HTML at the target size with headless Chrome to get a crisp PNG.
HTML is a far better thumbnail medium than a diffusion model: text is razor sharp
and on-message, layout is controllable, and it costs one cheap LLM call.

Entry point: make_thumbnail(project, cfg, force=False) -> Project
Outputs: projects/<slug>/thumbnail.html and projects/<slug>/thumbnail.png
"""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ..providers.images import get_image_source
from ..providers.llm import get_llm
from ..project import Project
from ..util import strip_code_fence
from .images import _pick, _search_with_fallback

# Thumbnail pixel size by project aspect.
_SIZE = {"16:9": (1280, 720), "9:16": (1080, 1920)}
_CHROME_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")

_SYSTEM = """You are a senior YouTube thumbnail designer who codes. You output ONE \
complete, self-contained HTML document that renders a click-worthy thumbnail.

HARD REQUIREMENTS:
- Output ONLY HTML. No markdown, no commentary, no code fences.
- Exactly {w}x{h} pixels: <body> and the root element are {w}px by {h}px, \
margin:0, overflow:hidden.
- Inline CSS only. The ONLY external resource allowed is the photo placeholder \
described below (when one is provided). Otherwise NO external resources — system \
fonts (Arial/Helvetica/sans-serif), CSS gradients, shapes, emoji.
- A short, punchy headline (3-6 words) in a HUGE, bold, high-contrast font — \
readable as a tiny phone thumbnail. Not the title verbatim; hook the eye.
- Strong contrast, one clear focal point. Avoid walls of text.
- Match the requested STYLE / clickbait level and this variant's CONCEPT.
- If a PHOTO is provided: use it as a full-bleed background EXACTLY via \
background-image:url('IMAGE_PLACEHOLDER');background-size:cover;background-position:center; \
and lay a dark gradient scrim under the text so it stays legible."""

_USER = """Title: {title}

Script (context):
{script}

Thumbnail style / clickbait level: {style}
This variant's concept: {concept}
{photo_line}

Design this thumbnail now. Return only the HTML document."""

# Distinct concepts so the variants don't look alike. Concepts mentioning "photo"
# get a real stock photo embedded as the background.
_CONCEPTS = [
    "Bold text-driven: a 3-5 word shock hook, dramatic gradient, no photo.",
    "Photo hero: a real PHOTO background with a punchy text overlay and a bright arrow.",
    "Curiosity gap: a provocative question, very high contrast, one emoji.",
    "Big word / number: one huge word or number, minimal, striking color block.",
    "Photo + contrast: a real PHOTO with a VS / before-after split and bold caps.",
]


def _resolution(aspect: str) -> tuple[int, int]:
    return _SIZE.get(aspect, _SIZE["16:9"])


def _concepts(n: int) -> list[str]:
    return [_CONCEPTS[i % len(_CONCEPTS)] for i in range(n)]


def _thumb_query(project: Project) -> str:
    text = (project.title or "") + " " + (project.theme or "")
    words = [w for w in re.sub(r"[^\w\s]", " ", text).split() if len(w) > 2]
    return " ".join(words[:5])


def _ensure_photo(html: str, data_uri: str) -> str:
    if "IMAGE_PLACEHOLDER" in html:
        return html.replace("IMAGE_PLACEHOLDER", data_uri)
    layer = (f'<div style="position:absolute;inset:0;z-index:-1;'
             f"background:url('{data_uri}') center/cover;\"></div>")
    return html.replace("</body>", layer + "</body>") if "</body>" in html else html + layer


def _fetch_photo_uri(project: Project, cfg: dict) -> str | None:
    """Find a stock photo for photo-based thumbnails, as a base64 data URI."""
    try:
        import requests
        source = get_image_source(cfg)
        orientation = "portrait" if project.aspect == "9:16" else "landscape"
        query = _thumb_query(project)
        if not query:
            return None
        results, _used = _search_with_fallback(source, query, orientation)
        pick = _pick(results)
        if pick is None:
            return None
        r = requests.get(pick.url, timeout=30)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if not ctype.startswith("image/"):
            ctype = "image/jpeg"
        return f"data:{ctype};base64," + base64.b64encode(r.content).decode("ascii")
    except Exception as e:  # noqa: BLE001
        print(f"[thumbnail] photo fetch skipped ({e})", file=sys.stderr)
        return None


def _design_one(project, script, style, concept, w, h, llm, photo_uri) -> str:
    photo_line = ("A real PHOTO is provided — use it as the full-bleed background via IMAGE_PLACEHOLDER."
                  if photo_uri else "No photo for this variant — use graphics, gradients and emoji.")
    html = llm.complete(
        _SYSTEM.format(w=w, h=h),
        _USER.format(title=project.title or project.slug, script=script[:3000],
                     style=style or "eye-catching but tasteful", concept=concept, photo_line=photo_line))
    html = strip_code_fence(html)
    if photo_uri:
        html = _ensure_photo(html, photo_uri)
    return html


def _chrome_binary(cfg: dict) -> str:
    cfgbin = cfg.get("thumbnail", {}).get("chrome_binary")
    if cfgbin:
        return cfgbin
    for name in _CHROME_NAMES:
        if shutil.which(name):
            return name
    raise RuntimeError(
        "No Chrome/Chromium found for HTML->PNG. Install Google Chrome or Chromium, "
        "or set thumbnail.chrome_binary in config.yaml."
    )


def render_html_to_png(binary: str, html_file: Path, png_file: Path, w: int, h: int) -> Path:
    """Screenshot an HTML file at WxH with headless Chrome."""
    png_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        binary, "--headless=new", "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage",
        "--hide-scrollbars", "--force-device-scale-factor=1",
        "--default-background-color=00000000",
        f"--window-size={w},{h}",
        f"--screenshot={png_file}",
        html_file.resolve().as_uri(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not png_file.exists():
        raise RuntimeError(f"Chrome screenshot failed: {proc.stderr[:300] or proc.stdout[:300]}")
    return png_file


def make_thumbnails(project: Project, cfg: dict, n: int | None = None,
                    style: str = "", force: bool = False) -> Project:
    """Generate several thumbnail variants (some photo-based) at the channel's
    clickbait/style level. Fills project.thumbnails[]; selects the first."""
    tcfg = cfg.get("thumbnail", {})
    n = int(n or tcfg.get("count", 3))
    use_photos = tcfg.get("use_photos", True)
    style = style or tcfg.get("style", "")
    w, h = _resolution(project.aspect)
    binary = _chrome_binary(cfg)  # fail early if no renderer

    if (project.thumbnails and len(project.thumbnails) >= n
            and all((project.dir / p).exists() for p in project.thumbnails) and not force):
        print(f"[thumbnail] skip ({len(project.thumbnails)} variants exist)", file=sys.stderr)
        return project

    llm = get_llm(cfg)
    script = project.script_human or "\n\n".join(s.text for s in project.scenes) or project.title
    concepts = _concepts(n)
    photo_uri = (_fetch_photo_uri(project, cfg)
                 if use_photos and any("photo" in c.lower() for c in concepts) else None)
    tdir = project.dir / "thumbnails"
    tdir.mkdir(parents=True, exist_ok=True)

    thumbs: list[str] = []
    for i in range(n):
        with_photo = bool(photo_uri) and "photo" in concepts[i].lower()
        try:
            html = _design_one(project, script, style, concepts[i], w, h, llm,
                               photo_uri if with_photo else None)
        except (RuntimeError, ValueError) as e:
            print(f"[thumbnail] variant {i+1}: design failed ({e})", file=sys.stderr)
            continue
        hf = tdir / f"thumb_{i+1}.html"
        hf.write_text(html, encoding="utf-8")
        png = tdir / f"thumb_{i+1}.png"
        try:
            render_html_to_png(binary, hf, png, w, h)
        except RuntimeError as e:
            print(f"[thumbnail] variant {i+1}: render failed ({e})", file=sys.stderr)
            continue
        thumbs.append(png.relative_to(project.dir).as_posix())
        print(f"[thumbnail] variant {i+1}/{n}{' (photo)' if with_photo else ''} -> {thumbs[-1]}", file=sys.stderr)

    if thumbs:
        project.thumbnails = thumbs
        if not project.thumbnail_path or project.thumbnail_path not in thumbs:
            project.thumbnail_path = thumbs[0]
    project.save()
    return project


def make_thumbnail(project: Project, cfg: dict, force: bool = False) -> Project:
    """Back-compat single entry — now produces the variant set (CLI `thumbnail`)."""
    return make_thumbnails(project, cfg, force=force)

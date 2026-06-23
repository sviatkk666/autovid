"""Images stage (search approach) — find & download one image per scene.

For every scene it searches an image source with a query derived from the
scene's image prompt, downloads the first usable result to
projects/<slug>/images/scene_NN.<ext>, and records the path plus license /
attribution on the scene. A human-readable CREDITS.md is written so the images
can be credited when the video is published.

Entry point: fetch_project(project, cfg, source=None, force=False) -> Project
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import requests

from ..project import Project
from ..providers.images import ImageResult, ImageSource, get_image_source

IMAGES_SUBDIR = "images"
_EXT_BY_TYPE = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif"}

# Filler that crowds out the meaningful nouns in a search query. Narration —
# especially humanized narration — leads with connectives ("moreover", "needless
# to say"); dropping them surfaces the words a search engine can actually match.
_STOPWORDS = frozenset("""
a an the this that these those it its is are was were be been being am
and or but so yet for nor as if then than also too very just only even still
of in on at to from by with without into onto over under up down out off about
we you they he she i me my your our their his her them us its it's
can could will would shall should may might must do does did done doing have has had
not no yes more most many much some any all each every few several
moreover furthermore however therefore additionally needless say worth noting
important note today todays world fast paced when comes things conclusion
""".split())


def _build_query(scene, cfg: dict) -> str:
    """Turn a scene's image prompt into short search keywords.

    Search engines want a few keywords, not a paragraph: drop the configured
    style suffix, strip punctuation, remove filler/stopwords, and cap the word
    count. If filtering removes everything, fall back to the raw leading words.
    """
    icfg = cfg.get("images", {})
    field = icfg.get("query_from", "image_prompt")
    text = getattr(scene, field, "") or scene.text

    style = (cfg.get("parser", {}).get("image_style") or "").strip()
    if style and text.lower().rstrip(". ").endswith(style.lower().rstrip(". ")):
        text = text[: text.lower().rfind(style.lower())]

    words = re.sub(r"[^\w\s]", " ", text).split()
    cap = icfg.get("query_max_words", 8)
    # Drop stopwords and stray 1-char tokens (e.g. the "s" left by "today's").
    keywords = [w for w in words if len(w) > 1 and w.lower() not in _STOPWORDS]
    chosen = keywords or words  # don't return an empty query
    return " ".join(chosen[:cap]).strip()


_SMART_SYSTEM = """You write STOCK-PHOTO search queries. For each scene, output 2-4 \
plain, LITERAL, concrete keywords that a stock site (Pexels) actually has photos of \
— a photographable subject, object, place or action. NOT metaphors, emotions, \
abstractions, made-up compounds, on-screen text, camera/lighting jargon, or full \
sentences. If the scene is abstract (e.g. "discipline", "success"), translate it to \
a concrete scene a photographer would shoot (e.g. "person running at dawn", \
"climber reaching summit", "tidy desk laptop"). Match the literal subject of what's \
being said. Output ONLY JSON: {"queries":[{"id":<int>,"q":"<keywords>"}]}"""


def _smart_queries(scenes, cfg: dict) -> dict[int, str]:
    """One LLM call → concrete, literal stock-photo keywords per scene id.

    Stock engines match nouns, not the art director's descriptive/metaphorical
    image_prompt — so we translate each beat into what a photographer would shoot.
    Best-effort: any failure returns {} and the caller falls back to keyword-strip.
    """
    if not scenes:
        return {}
    from ..providers.llm import get_llm
    from ..util import extract_json
    lines = []
    for s in scenes:
        vis = (getattr(s, "image_prompt", "") or "").strip()
        lines.append(f'{s.id}: says "{s.text.strip()[:180]}"' + (f' | wants "{vis[:160]}"' if vis else ""))
    try:
        raw = get_llm(cfg, "art_director").complete(_SMART_SYSTEM, "Scenes:\n" + "\n".join(lines), temperature=0.3)
        data = extract_json(raw)
        out: dict[int, str] = {}
        for it in (data.get("queries") if isinstance(data, dict) else []) or []:
            if isinstance(it, dict) and it.get("id") is not None and it.get("q"):
                out[int(it["id"])] = re.sub(r"[^\w\s]", " ", str(it["q"])).strip()
        return out
    except Exception as e:  # noqa: BLE001 — fall back to keyword extraction
        print(f"[images] smart-query skipped ({e})", file=sys.stderr)
        return {}


def _ext_for(url: str, content_type: str) -> str:
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in _EXT_BY_TYPE:
        return _EXT_BY_TYPE[ct]
    suffix = Path(url.split("?")[0]).suffix.lower().lstrip(".")
    return suffix if suffix in {"jpg", "jpeg", "png", "webp", "gif"} else "jpg"


def _download(url: str, dest_stem: Path, http: requests.Session) -> Path:
    r = http.get(url, timeout=30, stream=True)
    r.raise_for_status()
    dest = dest_stem.with_suffix("." + _ext_for(url, r.headers.get("Content-Type", "")))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return dest


def _pick(results: list[ImageResult]) -> ImageResult | None:
    return results[0] if results else None


def _search_with_fallback(source: ImageSource, query: str, orientation: str, min_words: int = 2):
    """Search the full query, then progressively drop trailing words.

    A long keyword query is precise but often matches nothing (sources AND the
    terms). Narrowing from the front keeps the most salient words and trades
    specificity for a hit. Returns (results, query_used).
    """
    words = query.split()
    for n in range(len(words), min_words - 1, -1):
        q = " ".join(words[:n])
        results = source.search(q, orientation=orientation)
        if results:
            return results, q
    return [], query


def fetch_project(
    project: Project,
    cfg: dict,
    source: ImageSource | None = None,
    force: bool = False,
    only: set[int] | None = None,
) -> Project:
    source = source or get_image_source(cfg)
    orientation = "portrait" if project.aspect == "9:16" else "landscape"
    images_dir = project.dir / IMAGES_SUBDIR
    http = requests.Session()
    http.headers["User-Agent"] = "autovid/0.1 image pipeline"

    # Scenes we'll actually search this run → one LLM call for literal stock queries.
    targets = [s for s in project.scenes
               if (only is None or s.id in only)
               and not (s.image_path and (project.dir / s.image_path).exists() and not force)]
    smart = _smart_queries(targets, cfg) if cfg.get("images", {}).get("smart_query", True) else {}

    for scene in project.scenes:
        if only is not None and scene.id not in only:
            continue
        if scene.image_path and (project.dir / scene.image_path).exists() and not force:
            print(f"[images] scene {scene.id}: skip (exists)", file=sys.stderr)
            continue

        query = smart.get(scene.id) or _build_query(scene, cfg)
        if not query:
            print(f"[images] scene {scene.id}: skip (empty query)", file=sys.stderr)
            continue

        try:
            results, query_used = _search_with_fallback(source, query, orientation)
        except Exception as e:
            print(f"[images] scene {scene.id}: search failed ({e})", file=sys.stderr)
            continue

        pick = _pick(results)
        if pick is None:
            print(f"[images] scene {scene.id}: no results for '{query}'", file=sys.stderr)
            continue
        if query_used != query:
            print(f"[images] scene {scene.id}: narrowed query -> '{query_used}'", file=sys.stderr)

        try:
            dest = _download(pick.url, images_dir / f"scene_{scene.id:02d}", http)
        except Exception as e:
            print(f"[images] scene {scene.id}: download failed ({e})", file=sys.stderr)
            continue

        scene.image_path = dest.relative_to(project.dir).as_posix()
        scene.image_source = pick.source
        scene.image_license = pick.license
        scene.image_attribution = pick.attribution
        scene.image_credit_url = pick.credit_url
        print(f"[images] scene {scene.id}: {scene.image_path} "
              f"[{pick.source} · {pick.license or 'n/a'}]", file=sys.stderr)

    project.save()
    _write_credits(project)
    return project


def _write_credits(project: Project) -> None:
    lines = [f"# Image credits — {project.title or project.slug}", ""]
    any_credit = False
    for scene in project.scenes:
        if not scene.image_path:
            continue
        any_credit = True
        parts = [f"- **Scene {scene.id}** (`{scene.image_path}`):"]
        if scene.image_attribution:
            parts.append(f" {scene.image_attribution};")
        if scene.image_license:
            parts.append(f" {scene.image_license};")
        if scene.image_source:
            parts.append(f" via {scene.image_source};")
        if scene.image_credit_url:
            parts.append(f" {scene.image_credit_url}")
        lines.append("".join(parts).rstrip(";"))
    if any_credit:
        (project.dir / "CREDITS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Publish stage — generate everything needed to POST the video to YouTube.

Given the finished blueprint (title, full script, scene timeline, channel profile),
one LLM call writes the complete publishing package:
  - several optimized TITLE options (you pick one),
  - an SEO DESCRIPTION (hook + summary + natural CTA),
  - TAGS / keywords and HASHTAGS,
  - CHAPTERS with real timestamps (computed from the scene timeline, so they're
    accurate, then labelled by the model),
  - a PINNED COMMENT to seed engagement.

Stored on project.publish (editable in the dashboard before you post).

Entry point: make_publish_kit(project, cfg, force=False) -> Project
"""

from __future__ import annotations

import sys

from ..providers.llm import get_llm
from ..project import Project
from ..util import extract_json
from .montage import _scene_duration

_SYSTEM = """You are a YouTube growth strategist who writes the full publishing \
package for a video. You optimize for click-through and watch-time WITHOUT \
clickbait that lies. You output ONLY JSON.

Return this exact shape:
{
 "titles": ["<5 distinct title options, <=70 chars, curiosity + clarity>"],
 "description": "<the full YouTube description: a strong first 1-2 lines (they \
show above the fold), then a tight summary of what the viewer gets, then a natural \
call to subscribe. Plain text with line breaks. If a CHAPTERS list is given below, \
weave it in verbatim under a 'Chapters:' heading. End with the hashtags on one line.>",
 "tags": ["<10-15 SEO tags/keywords, lowercase, no #>"],
 "hashtags": ["<3-5 hashtags WITH the # prefix>"],
 "chapters": [{"time":"M:SS","label":"<short chapter title>"}],
 "pinned_comment": "<one engaging pinned comment that invites replies>",
 "category": "<one YouTube category, e.g. Education / People & Blogs / Howto & Style>"
}

Rules: match the channel's voice and rules if given. Titles must be genuinely \
different angles, not rewordings. Use the EXACT timestamps provided for chapters \
(don't invent times); the first chapter must be 0:00. Keep it all in the video's \
language (English)."""


def _fmt_ts(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _timeline(project: Project, cfg: dict) -> list[tuple[str, str]]:
    """(timestamp, scene-text-snippet) for each scene, from cumulative durations."""
    fallback = float(cfg.get("montage", {}).get("fallback_scene_seconds", 4))
    out, t = [], 0.0
    for s in project.scenes:
        out.append((_fmt_ts(t), s.text.strip()[:80]))
        t += _scene_duration(s, project, fallback)
    return out


def _user(project: Project, channel_profile: str, timeline: list[tuple[str, str]]) -> str:
    script = project.script_human or project.script_raw or "\n".join(s.text for s in project.scenes)
    chan = f"Channel profile (match its voice/rules):\n{channel_profile}\n\n" if channel_profile else ""
    tl = "\n".join(f"{ts}  {txt}" for ts, txt in timeline) or "(no scenes yet)"
    return (
        f"{chan}Working title: {project.title or project.slug}\n\n"
        f"Scene timeline (use these EXACT timestamps for chapters; group them into "
        f"4-8 meaningful chapters, first at 0:00):\n{tl}\n\n"
        f"Full narration script:\n{script[:6000]}\n\n"
        f"Write the complete YouTube publishing package as JSON."
    )


def make_publish_kit(project: Project, cfg: dict, channel_profile: str = "",
                     force: bool = False) -> Project:
    """Generate the full YouTube posting package onto project.publish."""
    if project.publish.get("description") and not force:
        print("[publish] kit exists (use force to regenerate)", file=sys.stderr)
        return project
    if not (project.script_human or project.script_raw or project.scenes):
        raise ValueError("nothing to publish yet — write a script first.")

    llm = get_llm(cfg, "producer")
    timeline = _timeline(project, cfg)
    raw = llm.complete(_SYSTEM, _user(project, channel_profile, timeline),
                       temperature=min(cfg.get("llm", {}).get("temperature", 0.9), 0.7))
    data = extract_json(raw)
    if not isinstance(data, dict):
        raise ValueError("publish kit was not a JSON object")

    def _slist(v):
        return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []

    chapters = []
    for c in (data.get("chapters") or []):
        if isinstance(c, dict) and c.get("label"):
            chapters.append({"time": str(c.get("time", "0:00")).strip(),
                             "label": str(c["label"]).strip()})

    project.publish = {
        "titles": _slist(data.get("titles"))[:6],
        "description": str(data.get("description") or "").strip(),
        "tags": _slist(data.get("tags"))[:20],
        "hashtags": _slist(data.get("hashtags"))[:8],
        "chapters": chapters,
        "pinned_comment": str(data.get("pinned_comment") or "").strip(),
        "category": str(data.get("category") or "").strip(),
    }
    project.save()
    print(f"[publish] kit: {len(project.publish['titles'])} titles, "
          f"{len(project.publish['tags'])} tags, {len(chapters)} chapters", file=sys.stderr)
    return project

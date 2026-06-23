"""The director — a team of role-specialized agents that turn a theme/topic into
a full, editable production blueprint (a Project whose scenes carry not just
narration but visual treatment, voice/delivery, sound and motion).

Each agent is its OWN LLM call with its own system prompt and settings (model /
temperature, config-overridable). They run as a pipeline, each enriching the
blueprint the previous one produced:

  Screenwriter   -> narration script (reads content memory for series continuity)
  Humanizer      -> rewrite so it doesn't sound AI-written
  Art Director   -> split into scenes; per-scene visual_type + image_prompt
                    (hard rule: <=20% AI-generated; majority real stock photos)
  Voice Director -> project voice + per-scene delivery
  Sound Designer -> per-scene sfx cues (with timing) + music bed
  Showrunner     -> review the whole blueprint, write production notes

Core stages (screenwriter, humanizer, art director) are required; the rest are
best-effort (a failure leaves those blueprint fields empty, still editable in the
dashboard). The result is a Project ready to be saved and then "decoded" by the
asset modules.

Entry point: build_blueprint(brief, cfg, idea=None, seconds=None) -> Project
"""

from __future__ import annotations

import copy
import math

from ..project import Project, Scene
from ..providers.llm import LLM, agent_model, build_llm, get_llm
from ..providers.voices import list_voices
from ..util import extract_json
from . import memory_store
from .humanizer import humanize_text
from .montage import ANIMATIONS, TRANSITIONS
from .scriptgen import Idea, _niche_cfg, write_script

VISUAL_TYPES = ("search", "photo_edit", "chart", "text_card", "generate")
DEFAULT_AGENTS = ("screenwriter", "humanizer", "art_director",
                  "voice_director", "sound_designer", "showrunner")


def normalize_visual_type(s: str) -> str:
    """Canonicalize a visual_type to one of VISUAL_TYPES (default 'search').

    Tolerates spaces/hyphens (e.g. "Text Card" / "text-card" -> "text_card").
    """
    v = (s or "search").strip().lower().replace(" ", "_").replace("-", "_")
    return v if v in VISUAL_TYPES else "search"


# --- per-agent LLM (optional model/temperature override) ---------------------

def _agent_cfg(cfg: dict, role: str) -> dict:
    return cfg.get("director", {}).get("agents", {}).get(role, {}) or {}


def _agent_llm(cfg: dict, role: str, shared: LLM) -> LLM:
    # Per-agent model override lives in config.models.agents (or the dashboard ⚙).
    am = agent_model(cfg, role)
    return build_llm(am) if am else shared


def _agent_temp(cfg: dict, role: str, default: float) -> float:
    return float(_agent_cfg(cfg, role).get("temperature", default))


def _enabled(cfg: dict, role: str) -> bool:
    enabled = cfg.get("director", {}).get("enabled_agents", list(DEFAULT_AGENTS))
    return role in enabled


# --- Art Director ------------------------------------------------------------

_ART_SYSTEM = """You are the Art Director for a faceless YouTube channel. You \
split a finished narration script into scenes and design the COMPLETE visual \
treatment of each scene — what fills the frame, how the camera moves, and how it \
transitions in. Use the full toolbox and VARY it so the video has visual rhythm; \
never make every scene the same look or the same motion. You output ONLY a JSON array.

For every scene return an object:
  "text": the narration for this scene, VERBATIM from the script (concatenating \
all scene texts in order must reproduce the script; whitespace-only changes ok),
  "image_prompt": a concrete visual description of the still for this beat,
  "visual_type": how to MAKE the still — one of "search" | "photo_edit" | "chart" | "text_card" | "generate",
  "animation": the camera MOVE over the still — one of {animations},
  "transition": how this scene enters from the previous one — one of {transitions},
  "visual_reason": one short clause justifying the visual choice.

VISUAL TREATMENTS (pick the best per beat, keep the mix varied):
- "search" — a real stock/internet photo. This is the BACKBONE; the MAJORITY of \
scenes must be "search". Real photos read as authentic.
- "photo_edit" — a real stock photo with light polish (overlay text, color grade, \
vignette, framing) done later as HTML, not a redraw.
- "chart" — data, numbers, steps, comparisons, timelines, processes, hierarchies: \
anything better DRAWN than photographed.
- "text_card" — words as the hero: a punchy quote, one bold statement, a huge \
number/stat, a definition, a short list, or a section title. Great for hooks, \
mantras, key takeaways and chapter beats.
- "generate" (AI image) — ONLY for shots that genuinely cannot be found, charted \
or set as type (impossible/surreal/specific composite). Use SPARINGLY: at most one \
in five scenes. Never default to it.

CAMERA MOTION (animation): give each scene deliberate movement — "auto" (a gentle \
Ken Burns that alternates per scene), "kenburns-in"/"kenburns-out", \
"zoom-in"/"zoom-out", "pan-left"/"pan-right"/"pan-up"/"pan-down" (good for wide \
scenery / reveals), or "static" (no move — best for a clean text_card or chart). \
Vary the motion across neighbouring scenes.

TRANSITIONS (into the scene): "cut" (energetic, punchy), "fade"/"dissolve" \
(smooth, reflective), "fadeblack"/"fadewhite" (a beat / chapter break), \
"slide-*"/"wipe-*" (directional momentum), "zoom"/"circle" (dramatic). Match the \
transition to the pacing of the beat; the first scene is always a "cut".

Keep image_prompt free of on-screen text and narration restatement (text_card and \
chart handle their own words)."""


def _art_user(script: str, target_seconds: float, style: str, channel_profile: str = "") -> str:
    style_line = f'Visual style to keep consistent: "{style}".\n' if style else ""
    chan = f"Channel look to match:\n{channel_profile}\n\n" if channel_profile else ""
    return (
        f"{chan}Split this script into scenes of about {target_seconds} seconds of "
        f"narration each.\n{style_line}"
        f"Return ONLY a JSON array of scene objects per the schema.\n\nSCRIPT:\n{script}"
    )


def _art_director(script: str, cfg: dict, llm: LLM, channel_profile: str = "") -> list[Scene]:
    pcfg = cfg.get("parser", {})
    wps = float(pcfg.get("words_per_second", 2.5))
    target_seconds = pcfg.get("target_scene_seconds", 8)
    style = (pcfg.get("image_style") or "").strip()
    sys_prompt = _ART_SYSTEM.format(
        animations=" | ".join(f'"{a}"' for a in ANIMATIONS),
        transitions=" | ".join(f'"{t}"' for t in TRANSITIONS))
    raw = _agent_llm(cfg, "art_director", llm).complete(
        sys_prompt, _art_user(script, target_seconds, style, channel_profile),
        temperature=_agent_temp(cfg, "art_director", 0.6))
    data = extract_json(raw)
    if not isinstance(data, list) or not data:
        raise ValueError("art director did not return a scene array")

    scenes: list[Scene] = []
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        vtype = (item.get("visual_type") or "search").strip().lower()
        if vtype not in VISUAL_TYPES:
            vtype = "search"
        anim = (item.get("animation") or "auto").strip().lower().replace("_", "-")
        if anim not in ANIMATIONS:
            anim = "auto"
        trans = (item.get("transition") or "").strip().lower().replace("_", "-")
        if trans not in TRANSITIONS:
            trans = ""
        prompt = (item.get("image_prompt") or "").strip()
        if style and style.lower() not in prompt.lower():
            prompt = f"{prompt}. {style}" if prompt else style
        scenes.append(Scene(
            id=i, text=text, image_prompt=prompt, visual_type=vtype,
            animation=anim, transition=("cut" if i == 1 and not trans else trans),
            est_duration_sec=round(len(text.split()) / wps, 1) if wps else 0.0,
            notes=(item.get("visual_reason") or "").strip(),
        ))
    if not scenes:
        raise ValueError("art director produced zero usable scenes")
    return scenes


def enforce_ai_cap(scenes: list[Scene], max_fraction: float = 0.2) -> int:
    """Demote 'generate' scenes beyond the cap to 'photo_edit'. Returns #demoted."""
    n = len(scenes)
    cap = math.floor(n * max_fraction)
    gens = [s for s in scenes if s.visual_type == "generate"]
    demoted = 0
    for s in gens[cap:]:  # keep the first `cap`, demote the rest
        s.visual_type = "photo_edit"
        s.notes = (s.notes + " | AI-cap: demoted to photo_edit").strip(" |")
        demoted += 1
    return demoted


# --- Voice & Delivery Director ------------------------------------------------

_VOICE_SYSTEM = """You are the Voice & Delivery Director. You cast the narrator \
voice(s) for a video and write a short delivery note per scene.

- Choose ONE primary voice for the whole video, picking its id from the AVAILABLE \
VOICES list. Match it to the channel/voice character if one is given.
- Use that SAME voice for almost every scene. Only assign a DIFFERENT voice id to \
a scene when the content clearly calls for it — a quote from a named person, a \
second character/perspective, or a deliberate contrast. Keep switches rare.
- Use ONLY ids from the provided list.

Output ONLY JSON:
{"voice": "<primary voice id>", "scenes": [{"id": <int>, "voice": "<id, ONLY if \
different from primary>", "delivery": "<short delivery note>"}]}"""


def _voice_director(project: Project, cfg: dict, llm: LLM, voices: list[dict] | None = None) -> None:
    catalog = ", ".join(f"{v['id']} ({v['gender'] or '?'})" for v in (voices or [])) or "(no extra voices; one default voice only)"
    pref = (cfg.get("director", {}).get("default_voice") or project.voice or "none")
    scene_list = "\n".join(f'{s.id}: {s.text}' for s in project.scenes)
    raw = _agent_llm(cfg, "voice_director", llm).complete(
        _VOICE_SYSTEM,
        f"AVAILABLE VOICES (ids): {catalog}\nVoice character preference: {pref}\n\n"
        f"Scenes:\n{scene_list}\n\nReturn the JSON.",
        temperature=_agent_temp(cfg, "voice_director", 0.5))
    data = extract_json(raw)
    if isinstance(data, dict):
        project.voice = (data.get("voice") or project.voice or "").strip()
        by_id = {int(x["id"]): x for x in data.get("scenes", []) if isinstance(x, dict) and "id" in x}
        for s in project.scenes:
            d = by_id.get(s.id)
            if d:
                s.delivery = (d.get("delivery") or "").strip()
                sv = (d.get("voice") or "").strip()
                if sv and sv != project.voice:
                    s.voice = sv


# --- Sound Designer -----------------------------------------------------------

_SOUND_SYSTEM = """You are the Sound Designer. Given a script split into scenes, \
you choose a music bed mood for the video and, per scene, optional sound effects \
with a rough timing offset in seconds from the scene start. Be tasteful and \
sparse — silence is fine. Output ONLY JSON:
{"music": "<music bed mood>", "scenes": [{"id": <int>, "music": "<optional>", \
"sfx": [{"cue": "<short>", "at_sec": <number>}]}]}"""


def _sound_designer(project: Project, cfg: dict, llm: LLM) -> None:
    scene_list = "\n".join(f'{s.id} ({s.est_duration_sec}s): {s.text}' for s in project.scenes)
    raw = _agent_llm(cfg, "sound_designer", llm).complete(
        _SOUND_SYSTEM, f"Scenes:\n{scene_list}\n\nReturn the JSON.",
        temperature=_agent_temp(cfg, "sound_designer", 0.6))
    data = extract_json(raw)
    if isinstance(data, dict):
        project.music = (data.get("music") or "").strip()
        by_id = {int(x["id"]): x for x in data.get("scenes", []) if isinstance(x, dict) and "id" in x}
        for s in project.scenes:
            d = by_id.get(s.id)
            if not d:
                continue
            s.music = (d.get("music") or "").strip()
            sfx = d.get("sfx")
            if isinstance(sfx, list):
                s.sfx = [{"cue": str(c.get("cue", "")).strip(),
                          "at_sec": float(c.get("at_sec", 0) or 0)}
                         for c in sfx if isinstance(c, dict) and c.get("cue")]


# --- Showrunner / Critic ------------------------------------------------------

_SHOW_SYSTEM = """You are the Showrunner. You review a complete video blueprint \
(narration split into scenes, each with a visual treatment, delivery, sound) and \
write crisp production notes: what's strong, what to fix, and whether the visual \
mix respects 'mostly real photos, AI used sparingly'. Output ONLY JSON:
{"notes": "<your production notes>"}"""


def _showrunner(project: Project, cfg: dict, llm: LLM) -> None:
    mix = {}
    for s in project.scenes:
        mix[s.visual_type] = mix.get(s.visual_type, 0) + 1
    digest = "\n".join(
        f'{s.id}. [{s.visual_type}] {s.text[:80]}' for s in project.scenes)
    raw = _agent_llm(cfg, "showrunner", llm).complete(
        _SHOW_SYSTEM,
        f"Title: {project.title}\nVisual mix: {mix}\nVoice: {project.voice}\n"
        f"Music: {project.music}\n\nScenes:\n{digest}\n\nReturn the JSON.",
        temperature=_agent_temp(cfg, "showrunner", 0.4))
    data = extract_json(raw)
    if isinstance(data, dict):
        project.blueprint_notes = (data.get("notes") or "").strip()


# --- orchestrator ------------------------------------------------------------

def build_blueprint(
    brief: str,
    cfg: dict,
    idea: Idea | None = None,
    seconds: float | None = None,
    llm: LLM | None = None,
    log=print,
    channel_profile: str = "",
    channel_slug: str = "",
    channel_signature: str = "",
) -> Project:
    """Run the agent team and return a fully-populated (unsaved) Project.

    channel_profile/_signature (greetings, sign-offs, CTAs) keep every video for
    a channel on-brand; channel_slug scopes the content memory to that channel.
    """
    shared = llm or get_llm(cfg)
    niche, _, _ = _niche_cfg(cfg)
    title = (idea.title if idea else brief).strip()
    theme = brief.strip()
    max_ai = float(cfg.get("director", {}).get("max_ai_fraction", 0.2))

    # 0. content memory — scoped to this channel so the series evolves, no repeats.
    mem_ctx = memory_store.context_for_prompt(theme, channel=channel_slug or None)

    # 1. Screenwriter (required) — reuse scriptgen, seeded with channel identity,
    #    its recurring greeting/sign-off/CTA, and past-video context.
    log("[director] screenwriter: writing narration...")
    parts = [brief]
    if channel_profile:
        parts.append(f"Channel profile (stay on-brand):\n{channel_profile}")
    if channel_signature:
        parts.append(channel_signature)
    if mem_ctx:
        parts.append(mem_ctx)
    seed = "\n\n".join(parts)
    script = write_script(seed, cfg, idea=idea, seconds=seconds, llm=_agent_llm(cfg, "screenwriter", shared))

    # 2. Humanizer (best-effort: a failure just falls back to the raw script).
    human = ""
    if _enabled(cfg, "humanizer"):
        log("[director] humanizer: de-robotizing...")
        try:
            human = humanize_text(script, cfg, llm=_agent_llm(cfg, "humanizer", shared))
        except Exception as e:  # noqa: BLE001 — provider/network errors must not abort the build
            log(f"[director] humanizer skipped ({e})")
    final_script = human or script

    # 3. Art Director (required) — scenes + visual treatment.
    log("[director] art director: shot-listing + visual treatment...")
    scenes = _art_director(final_script, cfg, shared, channel_profile=channel_profile)
    demoted = enforce_ai_cap(scenes, max_ai)
    gen = sum(1 for s in scenes if s.visual_type == "generate")
    log(f"[director] {len(scenes)} scenes; {gen} AI-generated "
        f"({demoted} demoted to honor <={int(max_ai*100)}% cap)")
    # Sanity: the art director must keep narration verbatim. Warn loudly if the
    # concatenated scene text drifts materially from the script (lost/added lines).
    _norm = lambda s: " ".join((s or "").split()).lower()
    joined, src = _norm(" ".join(s.text for s in scenes)), _norm(final_script)
    if abs(len(joined) - len(src)) > max(40, 0.15 * len(src)):
        log(f"[director] WARNING: scene text diverges from script "
            f"({len(joined)} vs {len(src)} chars) — narration may have been altered.")

    project = Project(
        slug="", title=title, aspect=cfg.get("_aspect", "16:9"),
        script_raw=script, script_human=human, scenes=scenes,
        channel=channel_slug, theme=theme,
        source_idea={"title": idea.title, "hook": idea.hook, "angle": idea.angle, "theme": theme}
        if idea else {"theme": theme},
    )

    # 4-6. Best-effort enrichment agents.
    if _enabled(cfg, "voice_director"):
        log("[director] voice director: casting voice + delivery...")
        try:
            _voice_director(project, cfg, shared, voices=list_voices(cfg))
        except Exception as e:  # noqa: BLE001 — best-effort: never abort the build
            log(f"[director] voice director skipped ({e})")
    if _enabled(cfg, "sound_designer"):
        log("[director] sound designer: sfx + music...")
        try:
            _sound_designer(project, cfg, shared)
        except Exception as e:  # noqa: BLE001
            log(f"[director] sound designer skipped ({e})")
    if _enabled(cfg, "showrunner"):
        log("[director] showrunner: final review...")
        try:
            _showrunner(project, cfg, shared)
        except Exception as e:  # noqa: BLE001
            log(f"[director] showrunner skipped ({e})")

    return project

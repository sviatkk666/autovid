"""The producer-agent — drive the whole video pipeline by chatting.

Instead of clicking step buttons, the creator talks to this agent ("write the
script", "make scene 3 a chart", "use the female voice for the quote", "render
it"). Given the current project state, the channel profile and the conversation,
the agent replies briefly AND emits a list of actions, which we execute against
the project. Heavy actions (rendering) stream progress through the job log.

It maps natural language onto the existing pipeline — it doesn't reimplement it.

Entry point: run_agent(project, messages, cfg, channel_*, log) -> {reply, actions}
"""

from __future__ import annotations

from ..project import Scene, invalidate_stale_assets
from ..providers.llm import get_llm
from ..util import extract_json
from .chartgen import chart_project  # noqa: F401  (kept for parity / future tools)
from . import memory_store
from .director import _art_director, build_blueprint, enforce_ai_cap, normalize_visual_type
from .montage import normalize_animation, normalize_transition
from .humanizer import humanize_text  # noqa: F401
from .images import fetch_project  # noqa: F401
from .montage import build_video
from .audiomix import mix_project
from .captions import make_captions
from .publish import make_publish_kit
from .scriptgen import write_script
from .thumbnail import make_thumbnail
from .tts import synthesize_project
from .visuals import realize_visuals

_SYSTEM = """You are the AI producer for ONE faceless-YouTube video. You chat \
with the creator and DO the work by emitting actions. Be brief and natural — like \
a sharp collaborator, not a chatbot. You can also just talk (brainstorm, advise) \
with no actions.

Rules you always follow:
- Honor the CHANNEL PROFILE if one is given (voice, rules, visual style, the
  recurring intro/sign-off/CTA).
- When splitting a script into scenes, keep the narration VERBATIM.
- TARGETED EDITS: when the creator is talking about SPECIFIC scenes ("change scene
  3", "add a scene about X after scene 5", "cut the part about Y", "make scene 2
  punchier"), touch ONLY those scenes via edit_scene / add_scene / delete_scene —
  one action per scene. Do NOT rewrite the whole script or re-split everything. Only
  use rewrite / split_scenes / generate_script / blueprint when they clearly want to
  redo the ENTIRE video from scratch.
- If the creator asks for a LENGTH (e.g. "9 minutes", "90 seconds", "a 5-min
  video"), put it on generate_script/blueprint as "seconds" (minutes x 60: 9 min
  = 540). The writer targets ~150 words per minute. Default ~60s if unspecified.
- Always design with the FULL visual toolbox below and keep the mix VARIED — a
  video should never be all one look or one motion.
- COHESION: the result must feel like ONE continuous video, NOT a slideshow of
  disconnected stills. Keep a consistent look, give nearly every scene motion, and
  use transitions that connect the beats (not random). Vary for rhythm, but keep
  the through-line.

VISUAL TOOLBOX (you can set these per scene via edit_scene):
- visual_type — how the still is MADE:
  "search" (real stock photo; the BACKBONE / most scenes) · "photo_edit" (stock
  photo + light HTML polish) · "chart" (data/steps/comparison/timeline drawn) ·
  "text_card" (words as hero: quote, big stat, definition, list, title card) ·
  "generate" (AI image — SPARINGLY, at most ~20% of scenes; never the default).
- animation — the camera MOVE, for PHOTOS only: "auto" (gentle alternating Ken
  Burns) · "kenburns-in"/"kenburns-out" · "zoom-in"/"zoom-out" ·
  "pan-left"/"pan-right"/"pan-up"/"pan-down" · "static". Charts and text_cards must
  ALWAYS be "static" — motion on a graph or poster looks terrible. Be deliberate;
  not every photo needs a big move.
- transition — how a scene ENTERS: "cut" · "fade"/"dissolve" ·
  "fadeblack"/"fadewhite" · "slide-left/right/up/down" · "wipe-left/right/up/down"
  · "zoom" · "circle". Match it to the pacing.

Output ONLY JSON: {"reply": "<short reply>", "actions": [ <action>, ... ]}.
Emit actions ONLY for concrete changes the creator asked for; otherwise [].

Actions:
- {"tool":"generate_script","topic":"...","seconds":<optional total length in seconds>}   write the full narration from a topic/brief (clears scenes)
- {"tool":"rewrite","instruction":"..."}     rewrite the current script per the instruction
- {"tool":"blueprint","topic":"...","seconds":<optional total length in seconds>}   run the FULL director at once (script+scenes+visuals+motion+voice+sound). Use when they want "make the whole thing".
- {"tool":"split_scenes"}                     split the current script into scenes + full visual treatment
- {"tool":"edit_scene","id":N,"text":"...","visual_type":"search|photo_edit|chart|text_card|generate","animation":"<see toolbox>","transition":"<see toolbox>","voice":"...","delivery":"...","image_prompt":"..."}  (change ONE existing scene — include only the fields to change)
- {"tool":"add_scene","after":<scene id, or null for the end>,"text":"<narration>","visual_type":"...","animation":"...","transition":"...","image_prompt":"...","voice":"...","delivery":"..."}  (insert ONE new scene)
- {"tool":"delete_scene","id":N}   (remove ONE scene)
- {"tool":"set_project","title":"...","voice":"...","music":"...","aspect":"16:9|9:16"}
- {"tool":"run","step":"visuals|voice|montage|audiomix|captions|thumbnail|publish|all"}   produce assets / render / caption / write the YouTube publish kit ("all" does everything for posting)
"""


def _state_summary(project, channel_profile: str) -> str:
    lines = [f"CURRENT VIDEO: title={project.title or '(untitled)'} aspect={project.aspect}"]
    if channel_profile:
        lines.append("CHANNEL PROFILE:\n" + channel_profile)
    script = project.script_human or project.script_raw
    lines.append("SCRIPT: " + (script[:500] + "…" if script else "(none yet)"))
    if project.scenes:
        lines.append(f"SCENES ({len(project.scenes)}):")
        for s in project.scenes[:30]:
            v = f" voice={s.voice}" if s.voice else ""
            mo = f"/{s.animation}" if s.animation and s.animation != "auto" else ""
            tr = f" >{s.transition}" if s.transition else ""
            lines.append(f"  {s.id} [{s.visual_type}{mo}]{tr}{v}: {s.text[:60]}")
    else:
        lines.append("SCENES: (none yet)")
    lines.append(f"voice={project.voice or '(default)'} | music={project.music or '(none)'} | "
                 f"rendered={'yes' if project.video_path else 'no'}")
    return "\n".join(lines)


def _decide(project, messages, channel_profile, cfg, llm) -> dict:
    convo = "\n".join(
        f"{'CREATOR' if (m.get('role') in ('user', 'creator')) else 'PRODUCER'}: {m.get('content', '').strip()}"
        for m in messages)
    user = f"{_state_summary(project, channel_profile)}\n\nConversation:\n{convo}\n\nReturn the JSON."
    temp = min(cfg.get("llm", {}).get("temperature", 0.9), 0.6)
    raw = llm.complete(_SYSTEM, user, temperature=temp)
    try:
        data = extract_json(raw)
        if isinstance(data, dict):
            data.setdefault("reply", "")
            data.setdefault("actions", [])
            return data
    except (ValueError, Exception):  # noqa: BLE001
        pass
    return {"reply": raw.strip()[:1500], "actions": []}


# --- action executors --------------------------------------------------------

def _action_seconds(a: dict) -> float | None:
    """Total target length in seconds from an action (accepts seconds or minutes)."""
    for key, mult in (("seconds", 1), ("minutes", 60)):
        v = a.get(key)
        if v is not None:
            try:
                s = float(v) * mult
                return s if s > 0 else None
            except (TypeError, ValueError):
                return None
    return None


def _seed(topic, channel_profile, channel_signature):
    parts = [topic]
    if channel_profile:
        parts.append("Channel profile (stay on-brand):\n" + channel_profile)
    if channel_signature:
        parts.append(channel_signature)
    return "\n\n".join(parts)


def run_agent(project, messages, cfg, *, channel_profile="", channel_signature="",
              channel_slug="", log=print) -> dict:
    llm = get_llm(cfg, "producer")
    decision = _decide(project, messages, channel_profile, cfg, llm)
    actions = decision.get("actions") or []
    summary: list[str] = []
    acfg = dict(cfg)
    acfg["_aspect"] = project.aspect

    for a in actions:
        if not isinstance(a, dict):
            continue
        tool = a.get("tool", "")
        try:
            if tool == "generate_script":
                topic = (a.get("topic") or project.title or project.theme or "").strip()
                secs = _action_seconds(a)
                raw = write_script(_seed(topic, channel_profile, channel_signature), cfg,
                                   llm=get_llm(cfg, "screenwriter"), seconds=secs)
                project.script_raw, project.script_human, project.scenes = raw, "", []
                if topic and (not project.title or project.title == project.slug):
                    project.title = topic[:70]
                r = f"wrote script ({len(raw.split())} words{', ~%ds' % secs if secs else ''})"

            elif tool == "rewrite":
                cur = project.script_human or project.script_raw
                if not cur:
                    r = "nothing to rewrite (no script yet)"
                else:
                    out = llm.complete(
                        "You rewrite a video's spoken narration per the instruction. "
                        "Output ONLY the rewritten narration — no commentary, no headings.",
                        f"Instruction: {a.get('instruction', '')}\n\nScript:\n{cur}").strip()
                    if project.script_human:
                        project.script_human = out
                    else:
                        project.script_raw = out
                    r = "rewrote the script"

            elif tool == "blueprint":
                topic = (a.get("topic") or project.title or project.theme or "").strip()
                bp = build_blueprint(topic, acfg, seconds=_action_seconds(a), log=log,
                                     channel_profile=channel_profile,
                                     channel_signature=channel_signature, channel_slug=channel_slug)
                for f in ("title", "script_raw", "script_human", "scenes", "theme",
                          "voice", "music", "style", "blueprint_notes"):
                    setattr(project, f, getattr(bp, f) or getattr(project, f))
                if bp.source_idea:
                    project.source_idea = bp.source_idea
                try:   # remember it so the channel's content memory learns from chat-built videos too
                    memory_store.remember(project)
                except Exception as e:  # noqa: BLE001 — best-effort
                    log(f"[agent] memory skipped ({e})")
                r = f"built the full blueprint ({len(project.scenes)} scenes)"

            elif tool == "split_scenes":
                script = project.script_human or project.script_raw
                if not script:
                    r = "no script to split yet"
                else:
                    scenes = _art_director(script, cfg, get_llm(cfg, "art_director"), channel_profile=channel_profile)
                    enforce_ai_cap(scenes, float(cfg.get("director", {}).get("max_ai_fraction", 0.5)))
                    project.scenes = scenes
                    try:   # this storyline now feeds the channel's content memory
                        memory_store.remember(project)
                    except Exception as e:  # noqa: BLE001
                        log(f"[agent] memory skipped ({e})")
                    gen = sum(1 for s in scenes if s.visual_type == "generate")
                    r = f"split into {len(scenes)} scenes ({gen} AI-generated, {len(scenes) - gen} found)"

            elif tool == "edit_scene":
                sc = project.scene_by_id(int(a.get("id", 0)))
                if not sc:
                    r = f"no scene {a.get('id')}"
                else:
                    # Normalize the controlled-vocabulary fields so a loose LLM value
                    # ("Zoom In", "Photo Edit", "glitch") can't persist a bogus blueprint.
                    _norm = {"visual_type": normalize_visual_type,
                             "animation": normalize_animation, "transition": normalize_transition}
                    changed = set()
                    for k in ("text", "image_prompt", "visual_type", "animation",
                              "transition", "voice", "delivery", "music", "notes"):
                        if a.get(k) is not None:
                            v = _norm[k](a[k]) if k in _norm else a[k]
                            if getattr(sc, k, None) != v:
                                setattr(sc, k, v)
                                changed.add(k)
                    if sc.visual_type in ("chart", "text_card"):
                        sc.animation = "static"
                    cleared = invalidate_stale_assets(project, sc, changed)
                    r = f"edited scene {sc.id}" + (f" (regenerating its {', '.join(cleared)})" if cleared else "")

            elif tool == "add_scene":
                text = (a.get("text") or "").strip()
                if not text:
                    r = "add_scene needs narration text"
                else:
                    vt = normalize_visual_type(a.get("visual_type") or "search")
                    new = Scene(
                        id=project.next_scene_id(), text=text,
                        image_prompt=(a.get("image_prompt") or "").strip(),
                        visual_type=vt,
                        animation="static" if vt in ("chart", "text_card") else normalize_animation(a.get("animation") or "auto"),
                        transition=normalize_transition(a.get("transition") or ""),
                        voice=(a.get("voice") or "").strip(),
                        delivery=(a.get("delivery") or "").strip())
                    after = a.get("after")
                    idx = next((i for i, s in enumerate(project.scenes) if s.id == int(after)), None) if after not in (None, "") else None
                    if idx is not None:
                        project.scenes.insert(idx + 1, new)
                    else:
                        project.scenes.append(new)
                    r = f"added scene {new.id}" + (f" after {after}" if idx is not None else " at the end")

            elif tool == "delete_scene":
                sid = int(a.get("id", 0))
                sc = project.scene_by_id(sid)
                if not sc:
                    r = f"no scene {sid}"
                else:
                    for rel in (sc.audio_path, sc.image_path):   # clean up its assets
                        if rel and (project.dir / rel).exists():
                            try:
                                (project.dir / rel).unlink()
                            except OSError:
                                pass
                    project.scenes = [s for s in project.scenes if s.id != sid]
                    r = f"removed scene {sid}"

            elif tool == "set_project":
                for k in ("title", "voice", "music", "style", "aspect", "theme", "status", "blueprint_notes"):
                    if a.get(k) is not None:
                        setattr(project, k, a[k])
                r = "updated project settings"

            elif tool == "run":
                step = a.get("step", "all")
                seq = (["visuals", "voice", "montage", "audiomix", "captions", "thumbnail", "publish"]
                       if step == "all" else [step])
                done = []
                for s in seq:
                    if s == "visuals":
                        realize_visuals(project, cfg)
                    elif s == "voice":
                        synthesize_project(project, cfg)
                    elif s == "montage":
                        build_video(project, cfg)
                    elif s == "audiomix":
                        mix_project(project, cfg)
                    elif s == "captions":
                        make_captions(project, cfg)
                    elif s == "thumbnail":
                        make_thumbnail(project, cfg)
                    elif s == "publish":
                        make_publish_kit(project, cfg, channel_profile=channel_profile)
                    else:
                        continue
                    done.append(s)
                r = "ran " + ", ".join(done) if done else "nothing to run"

            else:
                r = f"(skipped unknown action '{tool}')"

            summary.append(r)
            log(f"[agent] {r}")
        except Exception as e:  # noqa: BLE001 — one bad action shouldn't kill the turn
            summary.append(f"{tool} failed: {e}")
            log(f"[agent] {tool} failed: {e}")

    project.save()
    return {"reply": decision.get("reply", ""), "actions": summary}

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

from ..providers.llm import get_llm
from ..util import extract_json
from .chartgen import chart_project  # noqa: F401  (kept for parity / future tools)
from .director import _art_director, build_blueprint, enforce_ai_cap
from .humanizer import humanize_text  # noqa: F401
from .images import fetch_project  # noqa: F401
from .montage import build_video
from .audiomix import mix_project
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
- Keep at most ~20% of scenes AI-generated ("generate"); prefer "search" (real
  stock photo), "photo_edit" (stock photo + light edit) or "chart".

Output ONLY JSON: {"reply": "<short reply>", "actions": [ <action>, ... ]}.
Emit actions ONLY for concrete changes the creator asked for; otherwise [].

Actions:
- {"tool":"generate_script","topic":"..."}   write the full narration from a topic/brief (clears scenes)
- {"tool":"rewrite","instruction":"..."}     rewrite the current script per the instruction
- {"tool":"blueprint","topic":"..."}         run the FULL director at once (script+scenes+voice+sound). Use when they want "make the whole thing".
- {"tool":"split_scenes"}                     split the current script into scenes + visual types
- {"tool":"edit_scene","id":N,"text":"...","visual_type":"search|photo_edit|chart|generate","voice":"...","delivery":"...","image_prompt":"..."}  (include only the fields to change)
- {"tool":"set_project","title":"...","voice":"...","music":"...","aspect":"16:9|9:16"}
- {"tool":"run","step":"visuals|voice|montage|audiomix|thumbnail|all"}   produce assets / render the video
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
            lines.append(f"  {s.id} [{s.visual_type}]{v}: {s.text[:60]}")
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
                raw = write_script(_seed(topic, channel_profile, channel_signature), cfg,
                                   llm=get_llm(cfg, "screenwriter"))
                project.script_raw, project.script_human, project.scenes = raw, "", []
                if topic and (not project.title or project.title == project.slug):
                    project.title = topic[:70]
                r = f"wrote script ({len(raw.split())} words)"

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
                bp = build_blueprint(topic, acfg, log=log, channel_profile=channel_profile,
                                     channel_signature=channel_signature, channel_slug=channel_slug)
                for f in ("title", "script_raw", "script_human", "scenes", "theme",
                          "voice", "music", "style", "blueprint_notes"):
                    setattr(project, f, getattr(bp, f) or getattr(project, f))
                if bp.source_idea:
                    project.source_idea = bp.source_idea
                r = f"built the full blueprint ({len(project.scenes)} scenes)"

            elif tool == "split_scenes":
                script = project.script_human or project.script_raw
                if not script:
                    r = "no script to split yet"
                else:
                    scenes = _art_director(script, cfg, get_llm(cfg, "art_director"), channel_profile=channel_profile)
                    demoted = enforce_ai_cap(scenes, float(cfg.get("director", {}).get("max_ai_fraction", 0.2)))
                    project.scenes = scenes
                    r = f"split into {len(scenes)} scenes ({demoted} AI demoted to honor the cap)"

            elif tool == "edit_scene":
                sc = project.scene_by_id(int(a.get("id", 0)))
                if not sc:
                    r = f"no scene {a.get('id')}"
                else:
                    for k in ("text", "image_prompt", "visual_type", "voice", "delivery", "music", "notes"):
                        if a.get(k) is not None:
                            setattr(sc, k, a[k])
                    r = f"edited scene {sc.id}"

            elif tool == "set_project":
                for k in ("title", "voice", "music", "style", "aspect", "theme", "status", "blueprint_notes"):
                    if a.get(k) is not None:
                        setattr(project, k, a[k])
                r = "updated project settings"

            elif tool == "run":
                step = a.get("step", "all")
                seq = ["visuals", "voice", "montage", "audiomix"] if step == "all" else [step]
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
                    elif s == "thumbnail":
                        make_thumbnail(project, cfg)
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

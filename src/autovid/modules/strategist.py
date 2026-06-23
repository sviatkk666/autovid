"""The strategist — the single human-facing "head analyzer & brainstormer".

This is the one chat the user talks to. Behind it sits the whole multi-agent
director, but here the user just converses with a content strategist that knows
the channel's profile and its past storylines, helps analyze what's worked,
brainstorms new angles, and shapes ONE topic to send to production.

Also drafts a channel profile (description + rules) from a few notes, so setting
up a channel is "tell me the vibe, I'll write the rules".

Entry points:
  draft_profile(name, niche, notes, cfg) -> dict
  chat_reply(channel, messages, cfg, memory_ctx="") -> str
"""

from __future__ import annotations

import json

from ..providers.llm import LLM, get_llm
from ..util import extract_json

# --- channel profile drafting -----------------------------------------------

_PROFILE_SYSTEM = """You are a YouTube channel strategist. From a channel name, a \
niche, and a few notes, you write a tight channel profile other AI agents will \
follow to keep every video on-brand — including the recurring "signature" lines \
(greeting, sign-off, like/subscribe CTA, catchphrase) that build a loyal, engaged \
audience. Output ONLY JSON:
{"description": "<2-3 sentences: what the channel is and its promise>",
 "rules": "<concrete do's and don'ts for tone, structure, pacing, what to avoid>",
 "audience": "<who it's for, one line>",
 "voice": "<the narrator voice character, one line>",
 "visual_notes": "<the look/feel: palette, imagery style, one line>",
 "intro": "<a recurring greeting / cold-open line for every video>",
 "outro": "<a recurring sign-off line>",
 "cta": "<a natural, on-brand like/subscribe call to action>",
 "catchphrase": "<a short signature phrase the channel becomes known for>"}"""


def draft_profile(name: str, niche: str, notes: str, cfg: dict, llm: LLM | None = None) -> dict:
    llm = llm or get_llm(cfg, "strategist")
    user = (f"Channel name: {name}\nNiche: {niche}\nNotes from the creator: {notes or '(none)'}\n\n"
            f"Write the profile JSON, including engaging recurring intro/outro/cta/catchphrase.")
    data = extract_json(llm.complete(_PROFILE_SYSTEM, user, temperature=0.7))
    if not isinstance(data, dict):
        raise ValueError("profile draft was not a JSON object")
    keys = ("description", "rules", "audience", "voice", "visual_notes",
            "intro", "outro", "cta", "catchphrase")
    return {k: (data.get(k) or "").strip() for k in keys}


# --- the strategist chat -----------------------------------------------------

_CHAT_SYSTEM = """You are the head content strategist and brainstorming partner \
for a faceless YouTube channel. You talk with the creator like a sharp, concise \
collaborator — not a chatbot. You know the channel's profile and its past videos.

Your job in this chat:
- Analyze what the channel has done and what's worked; spot gaps and fresh angles.
- Brainstorm specific, bingeable video ideas that fit the channel's rules.
- Help the creator converge on ONE storyline for the next video.
- When they settle on it, restate it as a single crisp production brief on its own \
final line, prefixed exactly with "PRODUCTION BRIEF: " so it can be sent to \
production.

Be brief and concrete. Don't over-explain. Don't invent past performance data you \
weren't given."""


def _conversation(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        who = "CREATOR" if role in ("user", "creator") else "STRATEGIST"
        lines.append(f"{who}: {m.get('content', '').strip()}")
    lines.append("STRATEGIST:")
    return "\n".join(lines)


def chat_reply(channel_profile: str, messages: list[dict], cfg: dict,
               memory_ctx: str = "", llm: LLM | None = None) -> str:
    """One strategist turn, given the channel profile, past storylines, and history."""
    llm = llm or get_llm(cfg, "strategist")
    system = _CHAT_SYSTEM
    if channel_profile:
        system += f"\n\n--- CHANNEL PROFILE ---\n{channel_profile}"
    if memory_ctx:
        system += f"\n\n--- PAST VIDEOS ---\n{memory_ctx}"
    temperature = cfg.get("llm", {}).get("temperature", 0.9)
    return llm.complete(system, _conversation(messages), temperature=min(temperature, 0.85)).strip()


def extract_brief(text: str) -> str:
    """Pull a 'PRODUCTION BRIEF: ...' line out of a strategist reply, if present."""
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("PRODUCTION BRIEF:"):
            return s.split(":", 1)[1].strip()
    return ""


# --- chat-first channel setup ------------------------------------------------

_SETUP_FIELDS = ("name", "handle", "niche", "description", "rules", "audience", "voice",
                 "visual_notes", "aspect", "intro", "outro", "cta", "catchphrase")

_SETUP_SYSTEM = """You help a creator set up a new faceless-YouTube channel by \
chatting. Ask for what you need (or infer it), and progressively fill the channel \
PROFILE: name, niche (motivational|educational|storytelling), description, rules \
(do's/don'ts), audience, voice (narrator character), visual_notes, and the \
recurring engagement signature — intro (greeting), outro (sign-off), cta \
(like/subscribe), catchphrase.

Each turn: reply briefly, and return your CURRENT best profile. PROACTIVELY \
PROPOSE concrete values for EVERY field — including a channel name and the \
signature lines — rather than leaving them blank; the creator will edit what they \
don't like. Keep prior values unless you're improving them. Ask at most one quick \
question; prefer proposing over interrogating. Set ready=true once name + niche + \
description + the signature (intro/outro/cta) are filled and the creator seems happy.

Output ONLY JSON (aspect is "16:9" or "9:16"):
{"reply":"<short reply>","profile":{"name":"","handle":"","niche":"","description":"","rules":"","audience":"","voice":"","visual_notes":"","aspect":"16:9","intro":"","outro":"","cta":"","catchphrase":""},"ready":false}"""


def channel_setup_turn(draft: dict, messages: list[dict], cfg: dict, llm: LLM | None = None) -> dict:
    """One turn of the channel-creation chat: reply + an updated profile draft."""
    llm = llm or get_llm(cfg, "strategist")
    convo = "\n".join(
        f"{'CREATOR' if (m.get('role') in ('user', 'creator')) else 'SETUP'}: {m.get('content', '').strip()}"
        for m in messages)
    user = f"Current draft profile: {json.dumps(draft or {})}\n\nConversation:\n{convo}\n\nReturn the JSON."
    merged = dict(draft or {})
    try:
        data = extract_json(llm.complete(_SETUP_SYSTEM, user, temperature=0.6))
    except ValueError:
        return {"reply": "(couldn't parse that — try rephrasing)", "profile": merged, "ready": False}
    except Exception as e:  # noqa: BLE001 — provider/network error: surface it, don't mislabel
        return {"reply": f"(LLM error: {e})", "profile": merged, "ready": False}
    if not isinstance(data, dict):
        return {"reply": "", "profile": merged, "ready": False}
    for k, v in (data.get("profile") or {}).items():
        if k in _SETUP_FIELDS and v:
            merged[k] = v
    return {"reply": (data.get("reply") or "").strip(), "profile": merged, "ready": bool(data.get("ready"))}


_EDIT_SYSTEM = """You help a creator EDIT an existing faceless-YouTube channel's \
profile by chatting. Apply their requested changes to the profile — name, niche \
(motivational|educational|storytelling), description, rules, audience, voice, \
visual_notes, handle, the default aspect ("16:9" or "9:16"), and the recurring \
signature (intro/outro/cta/catchphrase). Reply briefly confirming WHAT you \
changed (or ask one quick question only if truly ambiguous). Return the FULL \
updated profile, keeping fields they didn't touch exactly as-is.

Output ONLY JSON: {"reply":"<short reply>","profile":{"name":"","handle":"","niche":"","description":"","rules":"","audience":"","voice":"","visual_notes":"","aspect":"16:9","intro":"","outro":"","cta":"","catchphrase":""}}"""


def channel_edit_turn(draft: dict, messages: list[dict], cfg: dict, llm: LLM | None = None) -> dict:
    """One turn of editing an EXISTING channel's profile by chat: reply + updated profile."""
    llm = llm or get_llm(cfg, "strategist")
    convo = "\n".join(
        f"{'CREATOR' if (m.get('role') in ('user', 'creator')) else 'EDITOR'}: {m.get('content', '').strip()}"
        for m in messages)
    user = f"Current profile: {json.dumps(draft or {})}\n\nConversation:\n{convo}\n\nApply the edits and return the full JSON."
    merged = dict(draft or {})
    try:
        data = extract_json(llm.complete(_EDIT_SYSTEM, user, temperature=0.5))
    except ValueError:
        return {"reply": "(couldn't parse that — try rephrasing)", "profile": merged}
    except Exception as e:  # noqa: BLE001 — provider/network error: surface it, don't mislabel
        return {"reply": f"(LLM error: {e})", "profile": merged}
    if isinstance(data, dict):
        for k, v in (data.get("profile") or {}).items():
            if k in _SETUP_FIELDS and v is not None:
                merged[k] = v
        return {"reply": (data.get("reply") or "").strip(), "profile": merged}
    return {"reply": "", "profile": merged}

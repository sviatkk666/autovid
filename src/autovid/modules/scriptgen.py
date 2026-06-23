"""Storyline generator — invent video ideas and write full narration scripts.

This is the new *first* stage: it turns a topic (or a whole niche/theme) into a
ready-to-narrate script, which then flows through the existing pipeline
(humanizer -> parser -> tts -> images -> montage).

Two entry points:
  - brainstorm_ideas(theme, n, cfg, llm) -> list[Idea]
        From a niche/theme, propose N distinct video storylines (title + hook +
        angle). Used by the `ideas` and `batch` commands.
  - write_script(topic, cfg, llm, idea=None) -> str
        Write a complete spoken-narration script for one topic/idea. Output is
        plain narration (no titles, no scene labels, no markdown) so the
        humanizer and parser can consume it directly.

The default voice is "faceless motivational" (stoicism/discipline/mindset), the
style of the scripts already in this repo. Change scriptgen.niche / .tone in
config.yaml for a different channel.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..providers.llm import LLM, get_llm
from ..util import extract_json

# --- niche presets -----------------------------------------------------------

# Each preset shapes both brainstorming and script-writing. "motivational" is the
# default; the others are here so the same engine drives a different channel by
# flipping scriptgen.niche in config.
NICHE_GUIDE = {
    "motivational": (
        "Faceless motivational / self-improvement (stoicism, discipline, focus, "
        "mindset). Calm, direct, second-person ('you'). Hard truths, no fluff. "
        "Think Stoic wisdom applied to a modern viewer's daily struggle."
    ),
    "educational": (
        "Educational explainer. Curious, clear, informative third-person. Teach "
        "one idea well with concrete examples; spark 'huh, I didn't know that'."
    ),
    "storytelling": (
        "Narrative storytelling with a dramatic arc: setup, rising tension, turn, "
        "payoff. Vivid, cinematic, emotionally engaging."
    ),
}

DEFAULT_NICHE = "motivational"


@dataclass
class Idea:
    """One proposed video, before it becomes a full script."""
    title: str
    hook: str = ""      # the first-line grab; what stops the scroll
    angle: str = ""     # the specific take / what makes this one distinct

    def as_topic(self) -> str:
        """A one-line brief to seed write_script()."""
        bits = [self.title]
        if self.angle:
            bits.append(self.angle)
        if self.hook:
            bits.append(f"Open on: {self.hook}")
        return " — ".join(bits)


def _niche_cfg(cfg: dict) -> tuple[str, str, str]:
    """Return (niche_key, niche_guide, tone) from config with sane defaults."""
    scfg = cfg.get("scriptgen", {})
    niche = (scfg.get("niche") or DEFAULT_NICHE).strip().lower()
    guide = NICHE_GUIDE.get(niche, NICHE_GUIDE[DEFAULT_NICHE])
    tone = (scfg.get("tone") or "").strip()
    return niche, guide, tone


def _target_words(cfg: dict, seconds: float | None = None) -> int:
    """How long the script should be, in words, from a target duration.

    Uses scriptgen.target_seconds (or an override) and the parser's words-per-
    second so script length lines up with the rest of the pipeline's pacing.
    """
    scfg = cfg.get("scriptgen", {})
    secs = seconds if seconds is not None else float(scfg.get("target_seconds", 60))
    secs = max(secs, float(scfg.get("min_seconds", 0)))   # enforce the minimum video length
    wps = float(cfg.get("parser", {}).get("words_per_second", 2.5))
    return max(40, int(secs * wps))


# --- brainstorming -----------------------------------------------------------

_IDEAS_SYSTEM = """You are a YouTube content strategist for a faceless channel. \
You generate distinct, bingeable video ideas that fit one niche and tone. Each \
idea must be genuinely different from the others (different angle, not a reword). \
You output ONLY a JSON array — no prose, no markdown fences."""


def _ideas_user(theme: str, n: int, guide: str, tone: str) -> str:
    tone_line = f"Channel tone: {tone}\n" if tone else ""
    return (
        f"Niche: {guide}\n{tone_line}"
        f"Theme / direction from the creator: {theme}\n\n"
        f"Propose {n} video ideas. Each must be a distinct angle on the theme — "
        f"no two interchangeable. For each idea return an object with:\n"
        f'  "title": a punchy YouTube title (<= 70 chars),\n'
        f'  "hook": the spoken first line that stops the scroll (1 sentence),\n'
        f'  "angle": one sentence on the specific take / why it is different.\n\n'
        f"Return ONLY a JSON array of {n} such objects."
    )


def brainstorm_ideas(
    theme: str,
    n: int,
    cfg: dict,
    llm: LLM | None = None,
) -> list[Idea]:
    """Propose N distinct video storylines for a niche/theme."""
    if not theme.strip():
        raise ValueError("brainstorm needs a non-empty theme.")
    n = max(1, int(n))
    _, guide, tone = _niche_cfg(cfg)
    temperature = cfg.get("llm", {}).get("temperature", 0.9)

    llm = llm or get_llm(cfg)
    raw = llm.complete(_IDEAS_SYSTEM, _ideas_user(theme, n, guide, tone),
                       temperature=min(temperature, 0.8))
    data = extract_json(raw)
    if not isinstance(data, list):
        raise ValueError("brainstorm: LLM did not return a JSON array of ideas.")

    ideas: list[Idea] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        if not title:
            continue
        ideas.append(Idea(
            title=title,
            hook=(item.get("hook") or "").strip(),
            angle=(item.get("angle") or "").strip(),
        ))
    if not ideas:
        raise ValueError("brainstorm produced zero usable ideas.")
    return ideas[:n]


# --- script writing ----------------------------------------------------------

_SCRIPT_SYSTEM = """You are a scriptwriter for a faceless YouTube channel. You \
write the spoken voiceover narration for a single video — the words a narrator \
reads aloud, nothing else.

Hard rules:
- Output ONLY the narration. No title, no scene numbers, no stage directions, \
no '[music]' cues, no markdown, no headings, no speaker labels.
- Open with a hook in the first one or two sentences. Earn the next line every \
line. No throat-clearing ("In this video...", "Today we'll talk about...").
- Write for the ear: short, punchy sentences. Vary rhythm. Be concrete.
- One clear through-line from start to a strong closing thought.
- Keep it in English. Plain words, real human cadence — not an essay."""


def _script_user(brief: str, guide: str, tone: str, target_words: int,
                 audience: str, cta: str) -> str:
    tone_line = f"Channel tone: {tone}\n" if tone else ""
    aud_line = f"Audience: {audience}\n" if audience else ""
    cta_line = (
        f"End with a natural, non-cheesy call to action: {cta}\n" if cta else ""
    )
    return (
        f"Niche: {guide}\n{tone_line}{aud_line}"
        f"Write the voiceover narration for this video:\n{brief}\n\n"
        f"Target length: about {target_words} words "
        f"(a tight, well-paced script). {cta_line}"
        f"Output only the narration text."
    )


def write_script(
    topic: str,
    cfg: dict,
    llm: LLM | None = None,
    idea: Idea | None = None,
    seconds: float | None = None,
) -> str:
    """Write a full spoken-narration script for a topic (or a brainstormed idea).

    Returns plain narration text, ready to flow into humanize -> parse.
    """
    brief = idea.as_topic() if idea is not None else topic.strip()
    if not brief:
        raise ValueError("write_script needs a topic or an idea.")

    scfg = cfg.get("scriptgen", {})
    _, guide, tone = _niche_cfg(cfg)
    audience = (scfg.get("audience") or "").strip()
    cta = (scfg.get("cta") or "").strip()
    target_words = _target_words(cfg, seconds)
    temperature = cfg.get("llm", {}).get("temperature", 0.9)

    llm = llm or get_llm(cfg)
    user = _script_user(brief, guide, tone, target_words, audience, cta)
    script = llm.complete(_SCRIPT_SYSTEM, user, temperature=temperature).strip()

    # Long-form scripts: a single pass almost always underdelivers on length, so
    # expand the draft (deepening beats, not padding) until it's near the target.
    import sys
    tries = 0
    while len(script.split()) < 0.8 * target_words and tries < 3:
        tries += 1
        have = len(script.split())
        print(f"[scriptgen] expanding script: {have}/{target_words} words (pass {tries})", file=sys.stderr)
        script = llm.complete(
            _SCRIPT_SYSTEM,
            f"The narration below is too SHORT — it has {have} words but needs about "
            f"{target_words} for this video's length. EXPAND it: go deeper on each existing "
            f"beat with concrete specifics, examples, evidence and natural pacing/pauses. Do "
            f"NOT add a new topic, do NOT repeat yourself, do NOT pad with filler or empty "
            f"phrases. Keep the same voice and one through-line. Output ONLY the full script.\n\n"
            f"{script}", temperature=temperature).strip()
    return script

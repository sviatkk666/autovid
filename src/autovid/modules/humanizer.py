"""Humanizer — rewrite a script so it reads like a real person wrote it.

Two layers:
  1. LLM rewrite guided by a prompt that targets common "AI tells".
  2. Deterministic cleanup that strips residual tells the model may leave in.

Entry point: humanize_text(text, cfg, llm) -> str
"""

from __future__ import annotations

import re

from ..providers.llm import LLM, get_llm

# --- Layer 2: deterministic cleanup -----------------------------------------

# Phrases/openers that scream "LLM". Removed when they start a sentence/clause.
BANNED_OPENERS = [
    "it's important to note that",
    "it is important to note that",
    "it's worth noting that",
    "it is worth noting that",
    "it's worth mentioning that",
    "needless to say,",
    "in today's world,",
    "in today's fast-paced world,",
    "in conclusion,",
    "in summary,",
    "to sum up,",
    "at the end of the day,",
    "when it comes to",
    "as we all know,",
    "without a doubt,",
]

# Words/phrases LLMs overuse -> plainer swaps. Case-insensitive, word-boundary.
# NOTE: longer/multi-word patterns must come before their shorter variants so
# they win (dict order is preserved and applied top-to-bottom).
WORD_SWAPS = {
    r"\bdelves into\b": "digs into",
    r"\bdelved into\b": "dug into",
    r"\bdelve into\b": "dig into",
    r"\bdelves\b": "digs",
    r"\bdelve\b": "dig",
    r"\bfurthermore\b": "also",
    r"\bmoreover\b": "and",
    r"\badditionally\b": "also",
    r"\bnevertheless\b": "still",
    r"\bnonetheless\b": "still",
    r"\butilizes\b": "uses",
    r"\butilize\b": "use",
    r"\bleverage\b": "use",
    r"\ba plethora of\b": "plenty of",
    r"\bplethora\b": "plenty",
    r"\ba myriad of\b": "countless",
    r"\bmyriad\b": "countless",
    r"\brobust\b": "solid",
    r"\bseamlessly\b": "smoothly",
    r"\bseamless\b": "smooth",
    r"\bnavigate the\b": "handle the",
    r"\bin order to\b": "to",
    r"\ba testament to\b": "proof of",
    r"\btestament to\b": "proof of",
    r"\bgame-changer\b": "big deal",
    r"\bunlock\b": "open up",
    r"\btapestry\b": "mix",
}


# A sentence/clause boundary: start of text, after end punctuation + space,
# or after a line break. Captured so it can be re-inserted on removal.
_BOUNDARY = r"(\A|[.!?][\"')\]]?[ \t]+|\n[ \t]*)"


def _strip_openers(text: str) -> str:
    out = text
    for opener in BANNED_OPENERS:
        pat = _BOUNDARY + re.escape(opener) + r"[ \t]*"
        out = re.sub(pat, lambda m: m.group(1), out, flags=re.IGNORECASE)
    return out


def deterministic_cleanup(text: str) -> str:
    out = text

    # Em/en dashes -> comma+space (LLMs overuse "—").
    out = re.sub(r"\s*[—–]\s*", ", ", out)

    # Word-level swaps (preserve leading capital where possible).
    for pattern, repl in WORD_SWAPS.items():
        def _sub(m: re.Match, r=repl) -> str:
            return r.capitalize() if m.group(0)[0].isupper() else r
        out = re.sub(pattern, _sub, out, flags=re.IGNORECASE)

    # Strip banned openers. Repeat to a fixpoint so stacked openers in one
    # sentence ("In today's world, it's important to note that ...") all go.
    for _ in range(5):
        stripped = _strip_openers(out)
        if stripped == out:
            break
        out = stripped

    out = _recapitalize_sentences(out)

    # Collapse triple+ spaces and excess blank lines.
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _recapitalize_sentences(text: str) -> str:
    def cap(m: re.Match) -> str:
        return m.group(1) + m.group(2).upper()
    return re.sub(r"(^|[.!?]\s+)([a-z])", cap, text)


# --- Layer 1: LLM rewrite ----------------------------------------------------

_STRENGTH_GUIDE = {
    "light": "Polish lightly. Keep the wording close to the original; only fix robotic rhythm and obvious AI phrasing.",
    "medium": "Rephrase freely to sound natural. MAXIMIZE burstiness (wildly varied sentence lengths) and pick vivid, specific, slightly-unexpected words. Break every predictable structure.",
    "heavy": "Fully rewrite in a relaxed, spoken human voice, as if telling it to a friend — maximum burstiness and unexpected, concrete word choice, real personality. Keep all facts and meaning.",
}

SYSTEM_PROMPT = """You are a top-tier ghostwriter rewriting a spoken YouTube \
voiceover so it reads as unmistakably HUMAN-WRITTEN — to a person AND to an \
AI-content detector — while becoming genuinely BETTER writing. You keep every fact \
and the original meaning, but you replace the statistical fingerprint of machine \
text with the irregular, vivid rhythm of how real people actually write.

What AI detectors flag (so kill exactly these):
- LOW BURSTINESS — uniform sentence length. FIX: vary length wildly. Slam a \
two-word sentence against a long, winding one. Then a medium. Never an even beat.
- LOW PERPLEXITY — predictable, average word choice. FIX: choose the specific, \
slightly-unexpected word over the safe one. Strong verbs, concrete nouns, real \
images and numbers — never generic abstractions or the "most likely next word".
- TEMPLATED STRUCTURE — tidy intros, balanced "not only... but also", triple \
lists, formulaic transitions ("furthermore", "moreover", "in conclusion", "it's \
important to note", "in today's world", "when it comes to"). FIX: break the \
pattern on purpose.
- AI VOCAB — "delve", "utilize", "leverage", "plethora", "robust", "seamless", \
"tapestry", "navigate the". FIX: plain human words.
- Em dashes as a crutch, and constant hedging. Be direct.

Write like a real person:
- A point of view. The occasional aside, a blunt opinion, a rhetorical question, \
a touch of dry humor. Talk straight to "you".
- Human imperfection: a fragment for emphasis. Starting a line with "And" or \
"But". A trailing afterthought. Natural contractions (it's, you're, don't).
- Rhythm over symmetry. Read it aloud in your head; if it sounds metronomic, \
break it.

QUALITY BAR: every line earns its place — sharp, concrete, worth hearing aloud. \
The rewrite must be more engaging than the original, not just different.

Hard rules:
- Do NOT add new facts, claims, or opinions that weren't in the source.
- Do NOT add commentary, headings, or notes. Output ONLY the rewritten text.
- Preserve the paragraph breaks of the source unless told otherwise.
- Keep it in the same language as the input."""


def _build_user_prompt(text: str, strength: str, preserve_paragraphs: bool) -> str:
    guide = _STRENGTH_GUIDE.get(strength, _STRENGTH_GUIDE["medium"])
    para = (
        "Keep the same paragraph breaks."
        if preserve_paragraphs
        else "You may re-paragraph for better flow."
    )
    return (
        f"Rewrite the script below. {guide} {para}\n\n"
        f"Output only the rewritten script.\n\n"
        f"---\n{text}\n---"
    )


def humanize_text(text: str, cfg: dict, llm: LLM | None = None) -> str:
    """Run the humanizer on raw script text and return the cleaned result."""
    if not text.strip():
        return text

    hcfg = cfg.get("humanizer", {})
    strength = hcfg.get("strength", "medium")
    preserve = hcfg.get("preserve_paragraphs", True)
    do_cleanup = hcfg.get("deterministic_cleanup", True)
    temperature = cfg.get("llm", {}).get("temperature", 0.9)

    llm = llm or get_llm(cfg)
    user = _build_user_prompt(text, strength, preserve)
    rewritten = llm.complete(SYSTEM_PROMPT, user, temperature=temperature)

    if do_cleanup:
        rewritten = deterministic_cleanup(rewritten)
    return rewritten

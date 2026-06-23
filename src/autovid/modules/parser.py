"""Parser — split a script into scenes (narration + image prompt per scene).

Two modes:
  - "llm": the model segments the script into natural visual beats and writes
    an image prompt for each scene.
  - "deterministic": split by paragraph (then by sentence to hit a target
    length); the image prompt falls back to the scene text. No LLM needed.

Entry point: parse_script(text, cfg, llm=None) -> list[Scene]
"""

from __future__ import annotations

import re

from ..project import Scene
from ..providers.llm import LLM, get_llm
from ..util import extract_json


def _estimate_seconds(text: str, wps: float) -> float:
    words = len(text.split())
    return round(words / wps, 1) if wps else 0.0


def _join_style(prompt: str, style: str) -> str:
    """Append a style suffix without doubling end punctuation."""
    prompt = prompt.strip()
    if not style:
        return prompt
    if not prompt:
        return style
    sep = " " if prompt[-1] in ".!?," else ". "
    return f"{prompt}{sep}{style}"


# --- deterministic mode ------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _chunk_to_target(text: str, target_words: int) -> list[str]:
    """Group sentences into chunks of roughly target_words."""
    sentences = _split_sentences(text)
    chunks: list[str] = []
    cur: list[str] = []
    count = 0
    for s in sentences:
        cur.append(s)
        count += len(s.split())
        if count >= target_words:
            chunks.append(" ".join(cur))
            cur, count = [], 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _parse_deterministic(text: str, cfg: dict) -> list[Scene]:
    pcfg = cfg.get("parser", {})
    wps = pcfg.get("words_per_second", 2.5)
    target_words = max(1, int(pcfg.get("target_scene_seconds", 8) * wps))
    style = pcfg.get("image_style", "").strip()

    scenes: list[Scene] = []
    sid = 1
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for chunk in _chunk_to_target(paragraph, target_words):
            prompt = _join_style(chunk, style)
            scenes.append(
                Scene(
                    id=sid,
                    text=chunk,
                    image_prompt=prompt,
                    est_duration_sec=_estimate_seconds(chunk, wps),
                )
            )
            sid += 1
    return scenes


# --- LLM mode ----------------------------------------------------------------

_SYSTEM = """You are a video director's assistant. You split a narration script \
into scenes for an automated video. For each scene you keep the exact narration \
text and write a vivid image prompt that visualizes that moment.

Rules:
- Keep the narration VERBATIM. Concatenating all scene "text" in order must \
reproduce the original script (you may only adjust whitespace).
- Each scene should be one visual idea, roughly the length given by the user.
- The "image_prompt" must be a concrete, visual description of a single still \
image (subject, setting, composition, lighting, mood). No text overlays, no \
narration restated. Write it in English.
- Output ONLY a JSON array. No prose, no markdown fences."""


def _build_user(text: str, target_seconds: float, style: str) -> str:
    style_line = f'Append this style to every image prompt: "{style}".\n' if style else ""
    return (
        f"Split the script into scenes of about {target_seconds} seconds of "
        f"narration each.\n{style_line}"
        f'Return a JSON array; each item: {{"text": "...", "image_prompt": "..."}}.\n\n'
        f"SCRIPT:\n{text}"
    )


def _parse_llm(text: str, cfg: dict, llm: LLM) -> list[Scene]:
    pcfg = cfg.get("parser", {})
    wps = pcfg.get("words_per_second", 2.5)
    target_seconds = pcfg.get("target_scene_seconds", 8)
    style = pcfg.get("image_style", "").strip()
    temperature = cfg.get("llm", {}).get("temperature", 0.9)

    raw = llm.complete(_SYSTEM, _build_user(text, target_seconds, style), temperature=min(temperature, 0.7))
    data = extract_json(raw)
    if not isinstance(data, list):
        raise ValueError("LLM did not return a JSON array of scenes.")

    scenes: list[Scene] = []
    for i, item in enumerate(data, start=1):
        scene_text = (item.get("text") or "").strip()
        prompt = (item.get("image_prompt") or "").strip()
        if not scene_text:
            continue
        if style and style.lower() not in prompt.lower():
            prompt = _join_style(prompt, style)
        scenes.append(
            Scene(
                id=i,
                text=scene_text,
                image_prompt=prompt,
                est_duration_sec=_estimate_seconds(scene_text, wps),
            )
        )
    if not scenes:
        raise ValueError("Parsed zero scenes from LLM response.")
    return scenes


# --- entry point -------------------------------------------------------------

def parse_script(text: str, cfg: dict, llm: LLM | None = None) -> list[Scene]:
    mode = cfg.get("parser", {}).get("mode", "auto")

    if mode == "deterministic":
        return _parse_deterministic(text, cfg)

    if mode == "llm":
        return _parse_llm(text, cfg, llm or get_llm(cfg))

    # auto: LLM if one is available, else deterministic.
    try:
        llm = llm or get_llm(cfg)
    except RuntimeError:
        return _parse_deterministic(text, cfg)
    return _parse_llm(text, cfg, llm)

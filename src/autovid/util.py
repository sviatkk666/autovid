"""Small shared helpers."""

from __future__ import annotations

import contextlib
import json
import re
import wave
from pathlib import Path
from typing import Any


def extract_json(text: str) -> Any:
    """Parse JSON from an LLM reply, tolerating ```code fences``` and prose.

    Tries, in order: the whole string, a fenced block, the first balanced
    array/object found. Raises ValueError if nothing parses.
    """
    text = text.strip()

    # Try the raw string first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ```json ... ``` or ``` ... ``` fenced block.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    # First balanced [...] or {...} span. String-aware so brackets that appear
    # inside JSON string values don't throw off the depth count.
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError("No valid JSON found in LLM response.")


def strip_code_fence(text: str) -> str:
    """Return the body of a ```fenced``` block, or the trimmed text if unfenced.

    LLMs often wrap HTML/code in ```html ... ``` even when asked not to.
    """
    text = text.strip()
    fence = re.search(r"```(?:[a-zA-Z]+)?\s*(.*?)```", text, re.DOTALL)
    return fence.group(1).strip() if fence else text


def audio_duration(path: str | Path) -> float | None:
    """Real duration of an audio file in seconds, or None if unmeasurable.

    .wav is read with the stdlib (no deps). Other formats (e.g. mp3) need
    mutagen; if it isn't installed we return None and callers keep the estimate.
    """
    path = Path(path)
    if path.suffix.lower() == ".wav":
        with contextlib.suppress(Exception), wave.open(str(path), "rb") as w:
            frames, rate = w.getnframes(), w.getframerate()
            if rate:
                return round(frames / rate, 2)
        return None
    try:
        from mutagen import File as MutagenFile  # type: ignore

        audio = MutagenFile(str(path))
        if audio and audio.info and audio.info.length:
            return round(float(audio.info.length), 2)
    except Exception:
        pass
    return None


def slugify(text: str, max_len: int = 50) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:max_len].strip("-") or "project"

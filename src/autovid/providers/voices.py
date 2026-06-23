"""Voice catalog + resolver — turn a scene's voice (an id or a description) into
a concrete Piper model so each scene can be narrated by a different voice.

Voices are discovered two ways (config wins):
  - tts.piper_voices: {id: /path/to/model.onnx, ...} in config.yaml, and
  - auto-scan: every *.onnx next to tts.piper_model.

The Voice Director picks exact ids from list_voices(); resolve() also accepts a
free-text description ("calm female") as a best-effort fallback for hand-edits.
"""

from __future__ import annotations

from pathlib import Path

from ..config import env

# Rough gender hints by the speaker name embedded in a voice id (Piper voices are
# named en_US-<speaker>-<quality>). Used for description-based fallback matching.
_GENDER = {
    "amy": "female", "jenny": "female", "kathleen": "female", "kristin": "female",
    "lessac": "female", "hfc_female": "female", "libritts": "female",
    "ryan": "male", "joe": "male", "kusal": "male", "alan": "male",
    "danny": "male", "hfc_male": "male", "arctic": "male",
}


def _gender_for(vid: str) -> str:
    low = vid.lower()
    for key, g in _GENDER.items():
        if key in low:
            return g
    return ""


def piper_voices(cfg: dict) -> dict[str, str]:
    """{voice_id: model_path} from config + auto-scan of the model directory."""
    tcfg = cfg.get("tts", {}) or {}
    voices: dict[str, str] = dict(tcfg.get("piper_voices") or {})
    model = tcfg.get("piper_model") or env("PIPER_MODEL")
    if model:
        d = Path(model).parent
        if d.is_dir():
            for onnx in sorted(d.glob("*.onnx")):
                voices.setdefault(onnx.stem, str(onnx))
    return voices


def list_voices(cfg: dict) -> list[dict]:
    """A UI/agent-facing list: [{id, label, gender}]."""
    out = []
    for vid in piper_voices(cfg):
        # A friendly label: the speaker name if we can find it, else the id.
        parts = vid.replace("_", "-").split("-")
        label = parts[1] if len(parts) >= 2 and vid.startswith("en") else vid
        out.append({"id": vid, "label": label, "gender": _gender_for(vid)})
    return out


def resolve(voice: str | None, cfg: dict) -> str | None:
    """Map a voice id or description to a model path, or None for the default."""
    if not voice:
        return None
    vs = voice.strip().lower()
    voices = piper_voices(cfg)
    if not voices:
        return None
    # 1. exact id
    for vid, path in voices.items():
        if vid.lower() == vs:
            return path
    # 2. a voice's speaker token appears in the string (e.g. "ryan" in "ryan, calm")
    for vid, path in voices.items():
        for tok in vid.lower().replace("_", "-").split("-"):
            if len(tok) > 2 and tok in vs:
                return path
    # 3. gender heuristic for free-text descriptions
    want = ("female" if any(w in vs for w in ("female", "woman", "she ", "her ")) else
            "male" if any(w in vs for w in ("male", "man", " he ", "his ", "guy")) else "")
    if want:
        for vid, path in voices.items():
            if _gender_for(vid) == want:
                return path
    return None

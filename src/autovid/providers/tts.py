"""Unified text-to-speech interface over ElevenLabs, ai33.pro and Piper.

Every provider implements `synthesize(text, out_path) -> Path` and exposes an
`ext` (the file extension it writes, without the dot). Use `get_tts(cfg)` to
build one from config; provider "auto" prefers ElevenLabs when a key is present,
else local Piper. ai33.pro is NOT auto-selected for TTS — its public v1 voice
endpoints are sunset and v3 needs a web-session token, not the API key — so pick
provider "ai33" explicitly only if you have a v3-capable credential.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Protocol

from ..config import env


class TTS(Protocol):
    name: str
    ext: str

    # `voice` optionally overrides the default voice for THIS call (per-scene
    # casting): a Piper voice id / model for PiperTTS, a voice_id for the cloud
    # providers. None keeps the provider's default voice.
    def synthesize(self, text: str, out_path: Path, voice: str | None = None) -> Path: ...


class ElevenLabsTTS:
    name = "elevenlabs"
    ext = "mp3"

    def __init__(self, cfg: dict):
        from elevenlabs.client import ElevenLabs

        self.client = ElevenLabs(api_key=env("ELEVENLABS_API_KEY"))
        self.voice_id = cfg.get("elevenlabs_voice_id", "21m00Tcm4TlvDq8ikWAM")  # "Rachel"
        self.model_id = cfg.get("elevenlabs_model", "eleven_multilingual_v2")
        self.output_format = cfg.get("elevenlabs_format", "mp3_44100_128")

    def synthesize(self, text: str, out_path: Path, voice: str | None = None) -> Path:
        audio = self.client.text_to_speech.convert(
            voice_id=voice or self.voice_id,
            model_id=self.model_id,
            text=text,
            output_format=self.output_format,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)
        return out_path


class Ai33TTS:
    """ElevenLabs-compatible TTS via the ai33.pro async task gateway."""

    name = "ai33"
    ext = "mp3"

    def __init__(self, cfg: dict):
        from .ai33 import Ai33Client

        self.client = Ai33Client(cfg)
        self.voice_id = cfg.get("ai33_voice_id") or cfg.get(
            "elevenlabs_voice_id", "21m00Tcm4TlvDq8ikWAM")  # "Rachel"
        self.model_id = cfg.get("ai33_model", "eleven_multilingual_v2")
        self.output_format = cfg.get("ai33_format", "mp3_44100_128")

    def synthesize(self, text: str, out_path: Path, voice: str | None = None) -> Path:
        audio = self.client.run(
            f"/v1/text-to-speech/{voice or self.voice_id}?output_format={self.output_format}",
            prefer="audio_url",
            json={"text": text, "model_id": self.model_id, "with_transcript": False},
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio)
        return out_path


class PiperTTS:
    name = "piper"
    ext = "wav"

    def __init__(self, cfg: dict):
        self._cfg = {"tts": cfg}  # for the voice resolver (which reads cfg["tts"])
        self.binary = cfg.get("piper_binary") or env("PIPER_BINARY", "piper")
        self.model = cfg.get("piper_model") or env("PIPER_MODEL")
        if not self.model:
            raise RuntimeError(
                "Piper needs a voice model. Set tts.piper_model in config.yaml "
                "or PIPER_MODEL in .env to a .onnx voice file "
                "(download from https://github.com/rhasspy/piper/releases)."
            )
        if not shutil.which(self.binary) and not Path(self.binary).exists():
            raise RuntimeError(
                f"Piper binary '{self.binary}' not found. Install piper "
                "(https://github.com/rhasspy/piper) or set tts.piper_binary."
            )

    def synthesize(self, text: str, out_path: Path, voice: str | None = None) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model = self.model
        if voice:
            from .voices import resolve
            resolved = resolve(voice, self._cfg)
            if resolved:
                model = resolved
        subprocess.run(
            [self.binary, "--model", model, "--output_file", str(out_path)],
            input=text.encode("utf-8"),
            check=True,
            capture_output=True,
        )
        return out_path


def get_tts(cfg: dict) -> TTS:
    tts_cfg = cfg.get("tts", {})
    provider = tts_cfg.get("provider", "auto")

    if provider == "auto":
        # NOTE: ai33.pro is intentionally NOT auto-selected for TTS. Its public
        # v1 text-to-speech endpoints are sunset ("please use api v3 ...") and
        # the v3 endpoints reject the public xi-api-key (they need the web
        # app's session token). ai33 still works for IMAGES. Choose provider
        # "ai33" explicitly if you later get a v3-capable token.
        if env("ELEVENLABS_API_KEY"):
            provider = "elevenlabs"
        elif shutil.which(tts_cfg.get("piper_binary") or env("PIPER_BINARY", "piper")):
            provider = "piper"
        else:
            raise RuntimeError(
                "No TTS available. Set ELEVENLABS_API_KEY in .env for cloud voices, "
                "or install Piper (https://github.com/rhasspy/piper) and set "
                "tts.piper_model. (ai33.pro TTS now needs API v3, not reachable with "
                "the public key; ai33 still works for images.)"
            )

    if provider == "elevenlabs":
        return ElevenLabsTTS(tts_cfg)
    if provider == "ai33":
        return Ai33TTS(tts_cfg)
    if provider == "piper":
        return PiperTTS(tts_cfg)
    raise ValueError(f"Unknown tts.provider: {provider}")

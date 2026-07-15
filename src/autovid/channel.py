"""Channel — a reusable profile that keeps a YouTube channel's content on-style.

A Channel captures who the channel is and the rules its content follows (niche,
description, do/don't rules, audience, voice character, visual preferences). The
director injects this profile into its agents so every video for the channel
shares one voice and look, and the content memory is scoped per channel so the
strategist can analyze "what we've done on THIS channel".

Persisted through `storage` (channels/<slug>/channel.json on the filesystem
backend, a JSONB row on PostgreSQL), mirroring Project.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from . import storage
from .config import DATA_DIR

CHANNELS_DIR = DATA_DIR / "channels"


@dataclass
class Channel:
    slug: str
    name: str = ""
    handle: str = ""              # e.g. "@stoicpath"
    niche: str = "motivational"   # seeds scriptgen.niche for this channel
    description: str = ""         # what the channel is about (LLM-assisted)
    rules: str = ""               # content/style do's and don'ts (LLM-assisted)
    audience: str = ""            # who it's for
    voice: str = ""               # default voice character
    visual_notes: str = ""        # look/feel guidance for the Art Director
    thumbnail_style: str = ""     # thumbnail vibe / clickbait level (e.g. "high-energy clickbait, big arrows" / "clean & minimal")
    aspect: str = "16:9"          # default format for new videos
    # Max share of scenes that may be full AI-generated images, for THIS channel.
    # -1 = inherit the global default (director.max_ai_fraction). 0 = none, 1 = no
    # cap. Lets each channel pick its own creative latitude (stock-heavy ↔ AI-heavy).
    max_ai_fraction: float = -1.0
    # --- recurring engagement "signature" (woven into every video's script) ---
    intro: str = ""               # recurring greeting / cold-open line
    outro: str = ""               # recurring sign-off
    cta: str = ""                 # like / subscribe call to action
    catchphrase: str = ""         # signature phrase the channel is known for
    # --- YouTube account-creation brand kit (filled by modules/brandkit.py) ----
    # Everything the CRM needs to stand up the channel/account. Identity/creds
    # (password, recovery, the real Google signup) stay in the CRM vault — autovid
    # only prepares the brand + a suggested persona.
    language: str = "English"     # primary spoken/UI language
    region: str = ""              # target region, ISO-ish (e.g. "US", "UA")
    keywords: list[str] = field(default_factory=list)  # channel SEO keywords
    persona_name: str = ""        # the persona/creator name behind the account
    suggested_email: str = ""     # suggested gmail handle (CRM finalizes the real one)
    default_category: str = ""    # default YouTube category for uploads
    avatar_path: str = ""         # generated profile picture (relative to channel dir)
    banner_path: str = ""         # generated channel art / banner (relative to dir)

    @property
    def dir(self) -> Path:
        return CHANNELS_DIR / self.slug

    @property
    def json_path(self) -> Path:
        return self.dir / "channel.json"

    def save(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)  # asset dir (avatar/banner)
        storage.store().put("channel", self.slug, asdict(self))
        return self.json_path

    @classmethod
    def from_dict(cls, data: dict) -> "Channel":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, slug: str) -> "Channel":
        data = storage.store().get("channel", slug)
        if data is None:
            raise FileNotFoundError(f"no channel '{slug}'")
        return cls.from_dict(data)

    @classmethod
    def exists(cls, slug: str) -> bool:
        return storage.store().exists("channel", slug)

    @classmethod
    def list(cls) -> list["Channel"]:
        out = []
        for data in storage.store().all("channel"):
            try:
                out.append(cls.from_dict(data))
            except Exception:  # noqa: BLE001 — skip an unreadable channel
                continue
        return out

    @classmethod
    def delete(cls, slug: str) -> None:
        """Remove the document AND the channel's asset directory."""
        storage.store().delete("channel", slug)
        shutil.rmtree(CHANNELS_DIR / slug, ignore_errors=True)

    def profile_text(self) -> str:
        """A compact block injected into the director/strategist prompts."""
        bits = [f"Channel: {self.name or self.slug}" + (f" ({self.handle})" if self.handle else "")]
        if self.niche:
            bits.append(f"Niche: {self.niche}")
        if self.description:
            bits.append(f"About: {self.description}")
        if self.rules:
            bits.append(f"Content rules: {self.rules}")
        if self.audience:
            bits.append(f"Audience: {self.audience}")
        if self.voice:
            bits.append(f"Voice character: {self.voice}")
        if self.visual_notes:
            bits.append(f"Visual style: {self.visual_notes}")
        if self.catchphrase:
            bits.append(f"Signature phrase: {self.catchphrase}")
        return "\n".join(bits)

    def signature_text(self) -> str:
        """An explicit directive telling the screenwriter to weave in the channel's
        recurring greeting / sign-off / CTA so every video feels part of the series."""
        if not any((self.intro, self.outro, self.cta, self.catchphrase)):
            return ""
        bits = ["Channel signature — weave these in naturally (vary the wording slightly, "
                "keep the intent recurring):"]
        if self.intro:
            bits.append(f"- Open with this greeting: {self.intro}")
        if self.catchphrase:
            bits.append(f"- Work in the signature phrase where it fits: {self.catchphrase}")
        if self.outro:
            bits.append(f"- Close with this sign-off: {self.outro}")
        if self.cta:
            bits.append(f"- End with this like/subscribe call: {self.cta}")
        return "\n".join(bits)

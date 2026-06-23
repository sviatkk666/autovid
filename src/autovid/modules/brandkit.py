"""Brand-kit stage — generate everything the CRM needs to STAND UP a channel.

Given a Channel profile (name, niche, description, audience…), one LLM call writes
the YouTube account-creation brand kit — persona, suggested email handle, language,
region, SEO keywords, default category — plus prompts for the two channel images.
Then we render them:

  - avatar  → AI image (imagegen), square (YouTube profile picture)
  - banner  → HTML → PNG (headless Chrome), 2560x1440 channel art with the key
              content kept inside YouTube's centered "safe area" (1546x423)

Identity/credentials (password, recovery, the real Google signup) are NOT made
here — they stay in the CRM vault. autovid only prepares the brand + a persona.

Entry point: make_brand_kit(channel, cfg, force=False) -> Channel
Outputs: channels/<slug>/avatar.png and channels/<slug>/banner.{html,png}
"""

from __future__ import annotations

import sys

from ..channel import Channel
from ..providers.imagegen import get_image_generator
from ..providers.llm import LLM, get_llm
from ..util import extract_json, strip_code_fence
from .thumbnail import _chrome_binary, render_html_to_png

# YouTube channel art: full canvas + the area visible on every device.
BANNER_W, BANNER_H = 2560, 1440
SAFE_W, SAFE_H = 1546, 423

_KIT_SYSTEM = """You are a YouTube channel brand strategist. From a channel profile \
you produce the brand kit needed to create the channel + its Google account persona. \
Output ONLY JSON, no commentary:
{
 "persona_name": "<a realistic human creator name behind the channel>",
 "suggested_email": "<a gmail local-part suggestion (letters/numbers/dots, no @), \
on-brand and plausible, e.g. 'mindset.daily.alex'>",
 "language": "<primary language, e.g. English>",
 "region": "<2-letter target region, e.g. US>",
 "keywords": ["<10-15 channel SEO keywords, lowercase, no #>"],
 "default_category": "<one YouTube category, e.g. Education / People & Blogs / Howto & Style>",
 "avatar_prompt": "<a vivid image-generation prompt for a SQUARE channel avatar/logo: \
iconic, simple, high-contrast, recognizable at tiny size; no text unless a single letter/monogram>",
 "banner_concept": "<a one-line art-direction brief for the channel banner: mood, \
palette, the channel name + tagline to show>"
}"""

_BANNER_SYSTEM = """You are a senior brand designer who codes. You output ONE complete, \
self-contained HTML document that renders a YouTube channel BANNER (channel art).

HARD REQUIREMENTS:
- Output ONLY HTML. No markdown, no commentary, no code fences.
- Exactly {w}x{h} pixels: <body> and the root element are {w}px by {h}px, margin:0, \
overflow:hidden.
- Inline CSS only. NO external resources — system fonts (Arial/Helvetica/sans-serif), \
CSS gradients, shapes, emoji only.
- CRITICAL: all essential content (channel name, tagline, logo) MUST sit inside the \
centered SAFE AREA of {sw}x{sh}px (this is all that shows on phones). Treat the outer \
band as ambient background only — it gets cropped on small screens.
- The channel NAME is the hero: large, bold, high-contrast, instantly legible. Add a \
short tagline under it. Match the channel's style/palette and this concept.
- One clear focal composition; avoid clutter."""

_BANNER_USER = """Channel name: {name}
Niche: {niche}
About: {about}
Visual style: {visual}
Banner concept: {concept}

Design the channel banner now. Return only the HTML document."""


def _kit_text(channel: Channel, cfg: dict, llm: LLM) -> dict:
    about = channel.description or channel.rules or channel.niche
    user = (
        f"Channel name: {channel.name or channel.slug}\n"
        f"Handle: {channel.handle or '(none)'}\n"
        f"Niche: {channel.niche}\n"
        f"About: {about}\n"
        f"Audience: {channel.audience or '(general)'}\n"
        f"Visual style: {channel.visual_notes or '(open)'}\n\n"
        f"Write the brand-kit JSON."
    )
    data = extract_json(llm.complete(_KIT_SYSTEM, user, temperature=0.6))
    if not isinstance(data, dict):
        raise ValueError("brand kit was not a JSON object")
    return data


def _make_avatar(channel: Channel, cfg: dict, prompt: str) -> str | None:
    if not prompt:
        return None
    try:
        gen = get_image_generator(cfg)
        out = channel.dir / "avatar.png"
        gen.generate(prompt, out, orientation="square")
        if out.exists():
            print(f"[brandkit] avatar -> avatar.png [{getattr(gen, 'name', '?')}]", file=sys.stderr)
            return "avatar.png"
    except Exception as e:  # noqa: BLE001 — avatar is best-effort, never block the kit
        print(f"[brandkit] avatar skipped ({e})", file=sys.stderr)
    return None


def _make_banner(channel: Channel, cfg: dict, concept: str, llm: LLM) -> str | None:
    try:
        binary = _chrome_binary(cfg)
    except RuntimeError as e:
        print(f"[brandkit] banner skipped (no Chrome: {e})", file=sys.stderr)
        return None
    try:
        html = strip_code_fence(llm.complete(
            _BANNER_SYSTEM.format(w=BANNER_W, h=BANNER_H, sw=SAFE_W, sh=SAFE_H),
            _BANNER_USER.format(
                name=channel.name or channel.slug,
                niche=channel.niche,
                about=channel.description or channel.niche,
                visual=channel.visual_notes or "(open)",
                concept=concept or "clean, bold channel name on a striking gradient",
            ),
        ))
        hf = channel.dir / "banner.html"
        hf.parent.mkdir(parents=True, exist_ok=True)
        hf.write_text(html, encoding="utf-8")
        png = channel.dir / "banner.png"
        render_html_to_png(binary, hf, png, BANNER_W, BANNER_H)
        if png.exists():
            print("[brandkit] banner -> banner.png", file=sys.stderr)
            return "banner.png"
    except (RuntimeError, ValueError) as e:
        print(f"[brandkit] banner failed ({e})", file=sys.stderr)
    return None


def _slist(v) -> list[str]:
    return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []


def make_brand_kit(channel: Channel, cfg: dict, force: bool = False) -> Channel:
    """Generate the channel's account-creation brand kit onto the Channel."""
    if channel.persona_name and not force:
        print("[brandkit] kit exists (use force to regenerate)", file=sys.stderr)
        return channel

    llm = get_llm(cfg, "strategist")
    data = _kit_text(channel, cfg, llm)

    channel.persona_name = str(data.get("persona_name") or channel.persona_name or "").strip()
    channel.suggested_email = str(data.get("suggested_email") or channel.suggested_email or "").strip()
    channel.language = str(data.get("language") or channel.language or "English").strip()
    channel.region = str(data.get("region") or channel.region or "").strip()
    channel.default_category = str(data.get("default_category") or channel.default_category or "").strip()
    kw = _slist(data.get("keywords"))[:15]
    if kw:
        channel.keywords = kw

    avatar = _make_avatar(channel, cfg, str(data.get("avatar_prompt") or "").strip())
    if avatar:
        channel.avatar_path = avatar
    banner = _make_banner(channel, cfg, str(data.get("banner_concept") or "").strip(), llm)
    if banner:
        channel.banner_path = banner

    channel.save()
    print(
        f"[brandkit] kit: persona='{channel.persona_name}', {len(channel.keywords)} keywords, "
        f"avatar={'yes' if channel.avatar_path else 'no'}, banner={'yes' if channel.banner_path else 'no'}",
        file=sys.stderr,
    )
    return channel

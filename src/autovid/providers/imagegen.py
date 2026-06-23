"""Unified image-generation interface — synthesize an image per scene.

This is the "generate" approach (the search-and-download approach lives in
`images.py`). Every generator implements
`generate(prompt, out_path, orientation) -> Path` and exposes an `ext` (the file
extension it writes, without the dot). Use `get_image_generator(cfg)` to build
one from config; provider "auto" prefers a cloud key (OpenAI, then Replicate/
Flux, then Stability), and finally a local Stable Diffusion WebUI if reachable.

Providers:
  - openai      DALL·E 3 / gpt-image-1.        Needs OPENAI_API_KEY.
  - flux        Flux via Replicate.            Needs REPLICATE_API_TOKEN.
  - stability   Stable Image (Stability AI).   Needs STABILITY_API_KEY.
  - ai33        Imagen-style gen via ai33.pro. Needs AI33_API_KEY.
  - local       Stable Diffusion WebUI API.    No key (AUTOMATIC1111 / Forge).

Generated images carry no third-party license, but the model's terms apply; the
module records a "generated" provenance so CREDITS.md notes how each was made.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Protocol

import requests

from ..config import env

USER_AGENT = "autovid/0.1 (image generation)"
_TIMEOUT = 120

# Per-orientation pixel sizes (16:9 / 9:16). Used by providers that take w/h.
_SIZE = {"landscape": (1344, 768), "portrait": (768, 1344)}
# Aspect-ratio strings for providers that take a ratio instead of pixels.
_RATIO = {"landscape": "16:9", "portrait": "9:16"}


class ImageGenerator(Protocol):
    name: str
    ext: str

    def generate(self, prompt: str, out_path: Path, orientation: str = "landscape") -> Path: ...


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _write_bytes(out_path: Path, data: bytes) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return out_path


class OpenAIImage:
    name = "openai"
    ext = "png"
    # DALL·E 3 and gpt-image-1 take different size strings.
    _SIZES = {
        "dall-e-3": {"landscape": "1792x1024", "portrait": "1024x1792"},
        "gpt-image-1": {"landscape": "1536x1024", "portrait": "1024x1536"},
    }

    def __init__(self, cfg: dict):
        from openai import OpenAI

        if not env("OPENAI_API_KEY"):
            raise RuntimeError("OpenAI image generation needs OPENAI_API_KEY in .env.")
        self.client = OpenAI(api_key=env("OPENAI_API_KEY"))
        self.model = cfg.get("openai_model", "gpt-image-1")

    def _size(self, orientation: str) -> str:
        sizes = self._SIZES.get(self.model, self._SIZES["gpt-image-1"])
        return sizes.get(orientation, "1024x1024")

    def generate(self, prompt, out_path, orientation="landscape"):
        # dall-e-3 needs response_format to return b64; gpt-image-1 always does.
        kwargs = dict(model=self.model, prompt=prompt, size=self._size(orientation), n=1)
        if self.model == "dall-e-3":
            kwargs["response_format"] = "b64_json"
        resp = self.client.images.generate(**kwargs)
        return _write_bytes(out_path, base64.b64decode(resp.data[0].b64_json))


class FluxImage:
    """Flux via Replicate's HTTP API (no SDK dependency)."""

    name = "flux"
    ext = "png"

    def __init__(self, cfg: dict):
        self.token = env("REPLICATE_API_TOKEN")
        if not self.token:
            raise RuntimeError(
                "Flux needs REPLICATE_API_TOKEN in .env (get one at replicate.com)."
            )
        self.model = cfg.get("flux_model", "black-forest-labs/flux-schnell")
        self.http = _session()
        self.http.headers["Authorization"] = f"Bearer {self.token}"

    def generate(self, prompt, out_path, orientation="landscape"):
        # `Prefer: wait` blocks until the prediction finishes (up to ~60s).
        r = self.http.post(
            f"https://api.replicate.com/v1/models/{self.model}/predictions",
            headers={"Prefer": "wait"},
            json={"input": {
                "prompt": prompt,
                "aspect_ratio": _RATIO.get(orientation, "1:1"),
                "output_format": "png",
            }},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        pred = r.json()
        # If it didn't finish synchronously, poll the status URL.
        get_url = (pred.get("urls") or {}).get("get")
        while pred.get("status") in ("starting", "processing") and get_url:
            time.sleep(1.5)
            pred = self.http.get(get_url, timeout=_TIMEOUT).json()
        if pred.get("status") != "succeeded":
            raise RuntimeError(f"Flux prediction failed: {pred.get('error') or pred.get('status')}")
        output = pred.get("output")
        img_url = output[0] if isinstance(output, list) else output
        img = self.http.get(img_url, timeout=_TIMEOUT)
        img.raise_for_status()
        return _write_bytes(out_path, img.content)


class StabilityImage:
    name = "stability"
    ext = "png"

    def __init__(self, cfg: dict):
        self.key = env("STABILITY_API_KEY")
        if not self.key:
            raise RuntimeError(
                "Stability needs STABILITY_API_KEY in .env (platform.stability.ai)."
            )
        self.endpoint = cfg.get(
            "stability_endpoint",
            "https://api.stability.ai/v2beta/stable-image/generate/core",
        )
        self.http = _session()
        self.http.headers.update({"Authorization": f"Bearer {self.key}", "Accept": "image/*"})

    def generate(self, prompt, out_path, orientation="landscape"):
        r = self.http.post(
            self.endpoint,
            files={"none": ""},  # forces multipart/form-data
            data={
                "prompt": prompt,
                "aspect_ratio": _RATIO.get(orientation, "1:1"),
                "output_format": "png",
            },
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Stability error {r.status_code}: {r.text[:200]}")
        return _write_bytes(out_path, r.content)


class LocalSDImage:
    """Local Stable Diffusion via the AUTOMATIC1111 / Forge WebUI API."""

    name = "local"
    ext = "png"

    def __init__(self, cfg: dict):
        self.url = (cfg.get("local_url") or env("SD_WEBUI_URL", "http://localhost:7860")).rstrip("/")
        self.steps = int(cfg.get("local_steps", 25))
        self.http = _session()

    def generate(self, prompt, out_path, orientation="landscape"):
        w, h = _SIZE.get(orientation, (1024, 1024))
        r = self.http.post(
            f"{self.url}/sdapi/v1/txt2img",
            json={"prompt": prompt, "width": w, "height": h, "steps": self.steps},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        images = r.json().get("images") or []
        if not images:
            raise RuntimeError("Local SD returned no image (is a model loaded?).")
        # WebUI may prefix with a data URI; strip it before decoding.
        b64 = images[0].split(",", 1)[-1]
        return _write_bytes(out_path, base64.b64decode(b64))


class Ai33Image:
    """Imagen-style image generation via the ai33.pro async task gateway.

    Each model only accepts a fixed set of aspect ratios / resolutions (and the
    catalogue changes), so we read the model's spec from GET /v1i/models once and
    snap the requested orientation to the nearest ratio the model allows. This
    keeps generation working when ai33 rotates its model line-up.
    """

    name = "ai33"
    ext = "png"  # ffmpeg sniffs by content, so a jpg payload is still fine here

    # Preferred aspect ratios per orientation, widest-match first.
    _PREFER = {
        "landscape": ["16:9", "3:2", "4:3", "5:4", "2:1", "1:1"],
        "portrait": ["9:16", "2:3", "3:4", "4:5", "1:2", "1:1"],
        "square": ["1:1"],
    }

    def __init__(self, cfg: dict):
        import json as _json

        from .ai33 import Ai33Client

        self._json = _json
        self.client = Ai33Client(cfg)
        self.model_id = cfg.get("ai33_model", "gpt-image-2")
        self.resolution = cfg.get("ai33_resolution", "2K")
        self._ratios: list[str] | None = None
        self._default_ratio: str | None = None
        self._resolutions: list[str] | None = None
        self._default_res: str | None = None
        self._load_spec()

    def _load_spec(self) -> None:
        try:
            r = self.client.http.get(self.client._url("/v1i/models"), timeout=20)
            spec = next((m for m in r.json().get("models", [])
                         if m.get("model_id") == self.model_id), None)
            if spec:
                self._ratios = spec.get("aspect_ratios")
                self._default_ratio = spec.get("default_aspect_ratio")
                self._resolutions = spec.get("resolutions")
                self._default_res = spec.get("default_resolution")
        except Exception:  # noqa: BLE001 — fall back to raw config values
            pass

    def _pick_ratio(self, orientation: str) -> str:
        prefer = self._PREFER.get(orientation, self._PREFER["square"])
        if self._ratios:
            for ratio in prefer:
                if ratio in self._ratios:
                    return ratio
            return self._default_ratio or self._ratios[0]
        return _RATIO.get(orientation, "1:1")

    def _pick_resolution(self) -> str:
        if self._resolutions and self.resolution not in self._resolutions:
            return self._default_res or self._resolutions[-1]
        return self.resolution

    def generate(self, prompt, out_path, orientation="landscape"):
        params = {"aspect_ratio": self._pick_ratio(orientation), "resolution": self._pick_resolution()}
        img = self.client.run(
            "/v1i/task/generate-image",
            prefer="image_url",
            data={
                "prompt": prompt,
                "model_id": self.model_id,
                "generations_count": 1,
                "model_parameters": self._json.dumps(params),
            },
        )
        return _write_bytes(out_path, img)


_GENERATORS = {
    "openai": OpenAIImage,
    "flux": FluxImage,
    "stability": StabilityImage,
    "local": LocalSDImage,
    "ai33": Ai33Image,
}


def _local_up(cfg: dict) -> bool:
    url = (cfg.get("local_url") or env("SD_WEBUI_URL", "http://localhost:7860")).rstrip("/")
    try:
        requests.get(f"{url}/sdapi/v1/sd-models", timeout=2).raise_for_status()
        return True
    except Exception:
        return False


def get_image_generator(cfg: dict) -> ImageGenerator:
    """Build a generator from the `images.generate` config section."""
    gcfg = cfg.get("images", {}).get("generate", {})
    provider = gcfg.get("provider", "auto")

    if provider == "auto":
        if env("OPENAI_API_KEY"):
            provider = "openai"
        elif env("REPLICATE_API_TOKEN"):
            provider = "flux"
        elif env("STABILITY_API_KEY"):
            provider = "stability"
        elif env("AI33_API_KEY"):
            provider = "ai33"
        elif _local_up(gcfg):
            provider = "local"
        else:
            raise RuntimeError(
                "No image generator available. Set OPENAI_API_KEY, REPLICATE_API_TOKEN, "
                "STABILITY_API_KEY, or AI33_API_KEY in .env, or run a Stable Diffusion "
                "WebUI locally (AUTOMATIC1111 / Forge) and set images.generate.local_url."
            )

    cls = _GENERATORS.get(provider)
    if cls is None:
        raise ValueError(f"Unknown images.generate.provider: {provider}")
    return cls(gcfg)

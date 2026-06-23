"""Text embeddings for the RAG content memory.

get_embedder(cfg) auto-picks a backend: OpenAI (text-embedding-3-small) when an
OpenAI key is set, else local Ollama (nomic-embed-text) if its server is up, else
a deterministic offline hash fallback (low quality, but never fails / no setup).

Every embedder exposes embed(list[str]) -> list[list[float]].
"""

from __future__ import annotations

import hashlib
import math
import re

from ..config import env


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=env("OPENAI_API_KEY"))
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self.client.embeddings.create(model=self.model, input=[t[:8000] for t in texts])
        return [d.embedding for d in resp.data]


class OllamaEmbedder:
    name = "ollama"

    def __init__(self, model: str = "nomic-embed-text"):
        import requests
        self._requests = requests
        self.model = model
        host = env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.host = host if host.startswith("http") else "http://" + host

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            r = self._requests.post(f"{self.host}/api/embeddings",
                                    json={"model": self.model, "prompt": t[:8000]}, timeout=120)
            r.raise_for_status()
            out.append(r.json()["embedding"])
        return out


class HashEmbedder:
    """Deterministic offline fallback: hashed bag-of-words into a fixed dim."""
    name = "hash"
    dim = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in re.findall(r"[a-z0-9]{2,}", (t or "").lower()):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            out.append(v)
        return out


def _ollama_up() -> bool:
    try:
        from .llm import _ollama_up as up
        return up()
    except Exception:  # noqa: BLE001
        return False


def get_embedder(cfg: dict | None = None):
    mcfg = (cfg or {}).get("memory", {}) or {}
    prov = mcfg.get("embed_provider", "auto")
    if prov == "auto":
        if env("OPENAI_API_KEY"):
            prov = "openai"
        elif _ollama_up():
            prov = "ollama"
        else:
            prov = "hash"
    if prov == "openai":
        try:
            return OpenAIEmbedder(mcfg.get("openai_embed_model", "text-embedding-3-small"))
        except Exception:  # noqa: BLE001 — sdk missing / no key → degrade
            return HashEmbedder()
    if prov == "ollama":
        return OllamaEmbedder(mcfg.get("ollama_embed_model", "nomic-embed-text"))
    return HashEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

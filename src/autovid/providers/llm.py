"""Unified LLM interface over Anthropic, OpenAI and Ollama.

Every provider implements `complete(system, user) -> str`.
Use `get_llm(cfg)` to build one from config; provider "auto" picks the
first available backend (cloud key present, else local Ollama).
"""

from __future__ import annotations

from typing import Protocol

from .. import usage
from ..config import env


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, temperature: float = 0.9) -> str: ...


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(api_key=env("ANTHROPIC_API_KEY"))

    def complete(self, system: str, user: str, temperature: float = 0.9) -> str:
        # NOTE: Opus 4.8 / 4.7 reject `temperature`/`top_p`/`budget_tokens` (400).
        # Variation comes from prompting, not sampling params. `temperature` is
        # accepted here for interface parity with the other providers but not sent.
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        u = getattr(resp, "usage", None)
        if u:
            usage.record("anthropic", self.model, getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))
        return "".join(b.text for b in resp.content if b.type == "text").strip()


class OpenAILLM:
    name = "openai"

    def __init__(self, model: str):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=env("OPENAI_API_KEY"))

    def complete(self, system: str, user: str, temperature: float = 0.9) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        u = getattr(resp, "usage", None)
        if u:
            usage.record("openai", self.model, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
        return (resp.choices[0].message.content or "").strip()


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str):
        import requests

        self._requests = requests
        self.model = model
        self.host = env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    def complete(self, system: str, user: str, temperature: float = 0.9) -> str:
        r = self._requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": temperature},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=600,
        )
        r.raise_for_status()
        data = r.json()
        usage.record("ollama", self.model, data.get("prompt_eval_count", 0), data.get("eval_count", 0))
        return data["message"]["content"].strip()


def _ollama_up() -> bool:
    try:
        import requests

        host = env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        requests.get(f"{host}/api/tags", timeout=2).raise_for_status()
        return True
    except Exception:
        return False


def provider_for(model: str) -> str:
    """Infer the provider from a model id."""
    m = (model or "").lower()
    if m.startswith(("claude", "anthropic")):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "o4", "chatgpt", "text-")):
        return "openai"
    return "ollama"


def build_llm(model: str) -> LLM:
    """Build an LLM for a specific model id (provider inferred from the name)."""
    p = provider_for(model)
    if p == "anthropic":
        return AnthropicLLM(model)
    if p == "openai":
        return OpenAILLM(model)
    return OllamaLLM(model)


def agent_model(cfg: dict, role: str | None) -> str:
    """The configured model id for an agent role ('' = use the default)."""
    if not role:
        return ""
    return ((cfg.get("models", {}) or {}).get("agents", {}) or {}).get(role, "") or ""


def available_models() -> list[dict]:
    """Models the dashboard offers per agent (default + whatever's reachable)."""
    out = [{"id": "", "label": "Default (auto)", "provider": "default"}]
    if env("ANTHROPIC_API_KEY"):
        out += [{"id": "claude-opus-4-8", "label": "Claude Opus 4.8", "provider": "anthropic"},
                {"id": "claude-sonnet-4-6", "label": "Claude Sonnet 4.6", "provider": "anthropic"},
                {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5", "provider": "anthropic"}]
    if env("OPENAI_API_KEY"):
        out += [{"id": "gpt-4o", "label": "GPT-4o", "provider": "openai"},
                {"id": "gpt-4o-mini", "label": "GPT-4o mini", "provider": "openai"}]
    if _ollama_up():
        try:
            import requests
            host = env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
            host = host if host.startswith("http") else "http://" + host
            for m in requests.get(f"{host}/api/tags", timeout=3).json().get("models", []):
                name = m.get("name", "")
                if name:
                    out.append({"id": name, "label": name + " (local)", "provider": "ollama"})
        except Exception:  # noqa: BLE001
            pass
    return out


def get_llm(cfg: dict, role: str | None = None) -> LLM:
    # Per-agent model override (set in config.models.agents or the dashboard).
    am = agent_model(cfg, role)
    if am:
        return build_llm(am)
    llm_cfg = cfg.get("llm", {})
    # LLM_PROVIDER / LLM_MODEL env vars override config for a quick switch, e.g.
    #   LLM_PROVIDER=ollama autovid direct "..."   (run fully local, offline)
    provider = env("LLM_PROVIDER") or llm_cfg.get("provider", "auto")
    model_override = env("LLM_MODEL")

    if provider == "auto":
        if env("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif env("OPENAI_API_KEY"):
            provider = "openai"
        elif _ollama_up():
            provider = "ollama"
        else:
            raise RuntimeError(
                "No LLM available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env, "
                "or start Ollama (https://ollama.com) and `ollama pull llama3.2:3b`."
            )

    if provider == "anthropic":
        return AnthropicLLM(model_override or llm_cfg.get("anthropic_model", "claude-opus-4-8"))
    if provider == "openai":
        return OpenAILLM(model_override or llm_cfg.get("openai_model", "gpt-4o"))
    if provider == "ollama":
        return OllamaLLM(model_override or llm_cfg.get("ollama_model", "llama3.2:3b"))
    raise ValueError(f"Unknown llm.provider: {provider}")

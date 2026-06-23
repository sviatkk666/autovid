"""Load YAML config + .env into a simple dict-like object."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# Writable studio data — projects, channels, the RAG content memory, usage and
# settings — lives under DATA_DIR. Defaults to the repo root for local dev. On a
# host with an ephemeral filesystem (e.g. Railway) set DATA_DIR to a mounted
# volume (e.g. /data) so channels + RAG memory survive redeploys.
DATA_DIR = Path(os.environ.get("DATA_DIR") or ROOT)
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    DATA_DIR = ROOT


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config.yaml and load .env from the project root."""
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default

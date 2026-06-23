"""LLM token-usage tracking + cost estimation.

Every LLM call records (provider, model, in/out tokens) here, tagged with the
current "kind" (which operation — director, producer, chat, …) via a thread-local
the server binds per job. The dashboard shows recent requests, per-model totals,
estimated spend, and (where available) real provider balance.

Estimated cost only — providers don't bill from here; treat as a guide.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque

from .config import DATA_DIR

USAGE_FILE = DATA_DIR / "usage.json"
_LOCK = threading.Lock()
_local = threading.local()
RECENT: deque = deque(maxlen=400)
TOTALS: dict = {}   # model -> {in, out, cost, calls}

# Rough USD price per 1M tokens (input, output). Best-effort; edit as needed.
PRICES = {
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (15.0, 75.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.5, 10.0),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def bind(kind: str) -> None:
    _local.kind = kind


def unbind() -> None:
    _local.kind = None


def _price(model: str) -> tuple[float, float]:
    for key, p in PRICES.items():
        if model and model.startswith(key):
            return p
    return (0.0, 0.0)


def record(provider: str, model: str, tin: int, tout: int) -> None:
    pin, pout = _price(model)
    cost = (tin / 1e6) * pin + (tout / 1e6) * pout
    rec = {"ts": time.time(), "kind": getattr(_local, "kind", None) or "other",
           "provider": provider, "model": model, "in": int(tin), "out": int(tout),
           "cost": round(cost, 6)}
    with _LOCK:
        RECENT.appendleft(rec)
        t = TOTALS.setdefault(model, {"in": 0, "out": 0, "cost": 0.0, "calls": 0})
        t["in"] += int(tin); t["out"] += int(tout); t["cost"] += cost; t["calls"] += 1
        _save()


def _save() -> None:
    try:
        USAGE_FILE.write_text(json.dumps({"totals": TOTALS}), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _load() -> None:
    global TOTALS
    try:
        TOTALS = json.loads(USAGE_FILE.read_text(encoding="utf-8")).get("totals", {})
    except Exception:  # noqa: BLE001
        TOTALS = {}


_load()


def summary() -> dict:
    with _LOCK:
        tin = sum(t["in"] for t in TOTALS.values())
        tout = sum(t["out"] for t in TOTALS.values())
        tcost = sum(t["cost"] for t in TOTALS.values())
        return {
            "recent": list(RECENT)[:60],
            "by_model": {m: {**t, "cost": round(t["cost"], 5)} for m, t in TOTALS.items()},
            "totals": {"in": tin, "out": tout, "cost": round(tcost, 4),
                       "calls": sum(t["calls"] for t in TOTALS.values())},
        }

"""POST prepared assets to the YouTube-Farm CRM ingest endpoints.

Service-to-service auth via the x-service-token header (shared secret =
AUTOVID_SERVICE_TOKEN, must match the CRM's env). The CRM resolves the channel
by slug, creates the draft records, and (for video) the PENDING publish schedule
an operator later approves.

Env (autovid .env): CRM_BASE_URL, AUTOVID_SERVICE_TOKEN
"""

from __future__ import annotations

import sys

import requests

from ..config import env


def _base_and_token() -> tuple[str, str]:
    base = env("CRM_BASE_URL").rstrip("/")
    token = env("AUTOVID_SERVICE_TOKEN")
    if not base or not token:
        raise RuntimeError(
            "CRM_BASE_URL / AUTOVID_SERVICE_TOKEN not set — configure the CRM bridge in autovid .env"
        )
    return base, token


def _post(path: str, payload: dict, timeout: int) -> dict:
    base, token = _base_and_token()
    url = f"{base}{path}"
    resp = requests.post(url, json=payload, headers={"x-service-token": token}, timeout=timeout)
    if not resp.ok:
        raise RuntimeError(f"CRM {path} HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        return resp.json()
    except ValueError:
        return {"ok": True}


def ingest_channel(payload: dict) -> dict:
    """Create/update the draft channel + account from the brand kit."""
    res = _post("/api/ingest/channel", payload, timeout=60)
    print(f"[crm] ingest/channel -> {res}", file=sys.stderr)
    return res


def ingest_video(payload: dict) -> dict:
    """Register a finished video → VideoFile + PublishSchedule(PENDING)."""
    res = _post("/api/ingest/video", payload, timeout=120)
    print(f"[crm] ingest/video -> {res}", file=sys.stderr)
    return res

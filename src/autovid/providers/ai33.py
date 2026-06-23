"""Shared client for ai33.pro — an async, task-based AI media gateway.

ai33.pro proxies ElevenLabs-style TTS and Imagen-style image generation behind
one key. Every "create" call returns a task_id; you then poll GET /v1/task/{id}
until `status == "done"` and read the result URL off the finished task.

Auth is the header `xi-api-key: <AI33_API_KEY>`. Base URL defaults to
https://api.ai33.pro (override with AI33_BASE_URL or *_base_url in config).

This module is provider-agnostic plumbing; the TTS and image providers in
tts.py / imagegen.py build on it.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from ..config import env

_DEFAULT_BASE = "https://api.ai33.pro"
_DONE = {"done", "success", "completed", "complete"}
_FAILED = {"error", "failed"}
# Result fields a finished task may expose a media URL under, in priority order.
_URL_FIELDS = ("audio_url", "image_url", "url", "output_url", "file_url", "result_url")


class Ai33Error(RuntimeError):
    pass


def ai33_credits() -> int | None:
    """Current ai33.pro credit balance (GET /v1/credits), or None if unavailable."""
    key = env("AI33_API_KEY")
    if not key:
        return None
    try:
        r = requests.get(f"{_DEFAULT_BASE}/v1/credits", headers={"xi-api-key": key}, timeout=10)
        if r.status_code < 400:
            return r.json().get("credits")
    except Exception:  # noqa: BLE001
        pass
    return None


class Ai33Client:
    def __init__(self, cfg: dict):
        self.key = env("AI33_API_KEY")
        if not self.key:
            raise Ai33Error("ai33.pro needs AI33_API_KEY in .env (https://ai33.pro).")
        self.base = (cfg.get("ai33_base_url") or env("AI33_BASE_URL", _DEFAULT_BASE)).rstrip("/")
        # Poll no faster than ~5s: tighter intervals get rate-limited to empty
        # 200 bodies. Image tasks routinely take a few minutes, so allow 10 min.
        self.poll_interval = float(cfg.get("ai33_poll_interval", 5.0))
        self.poll_timeout = float(cfg.get("ai33_poll_timeout", 600.0))
        self.http = requests.Session()
        self.http.headers.update({"xi-api-key": self.key, "User-Agent": "autovid/0.1"})

    # --- low-level ----------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.base}/{path.lstrip('/')}"

    def create_task(self, path: str, *, json: dict | None = None,
                    data: dict | None = None, files: dict | None = None) -> str:
        """POST a create endpoint and return its task_id (raises on failure)."""
        r = self.http.post(self._url(path), json=json, data=data, files=files, timeout=60)
        if r.status_code >= 400:
            raise Ai33Error(f"ai33 {path} -> HTTP {r.status_code}: {r.text[:200]}")
        body = r.json()
        # A 200 with success:false is a rejected request (e.g. the v1 TTS
        # endpoints now answer "please use api v3 for this endpoint"). Surface
        # the message instead of a confusing "no task_id".
        if body.get("success") is False:
            raise Ai33Error(f"ai33 {path}: {body.get('message') or 'request rejected'}")
        task_id = body.get("task_id") or body.get("id")
        if not task_id:
            raise Ai33Error(f"ai33 {path}: no task_id in response ({str(body)[:200]})")
        return task_id

    def poll(self, task_id: str) -> dict[str, Any]:
        """Poll GET /v1/task/{id} until the task finishes; return the task dict.

        The gateway intermittently answers a poll with an empty 200 body (a soft
        rate-limit) and tasks sit in "doing"/"processing" for minutes. Treat
        empty/non-JSON/transient responses as "keep waiting" rather than failing.
        """
        waited = 0.0
        status = ""
        while True:
            task: dict[str, Any] | None = None
            try:
                r = self.http.get(self._url(f"/v1/task/{task_id}"), timeout=60)
                if r.text.strip():
                    task = r.json()
            except (ValueError, requests.RequestException):
                task = None
            if task is not None:
                status = str(task.get("status", "")).lower()
                if status in _DONE:
                    return task
                if status in _FAILED:
                    raise Ai33Error(f"ai33 task {task_id} failed: {task.get('error') or status}")
            if waited >= self.poll_timeout:
                raise Ai33Error(f"ai33 task {task_id} timed out after {self.poll_timeout:.0f}s (status={status or 'unknown'})")
            time.sleep(self.poll_interval)
            waited += self.poll_interval

    def result_url(self, task: dict, prefer: str = "") -> str:
        """Find the media URL on a finished task, tolerating field-name variants."""
        fields = ((prefer,) if prefer else ()) + _URL_FIELDS
        for f in fields:
            val = task.get(f)
            if isinstance(val, str) and val.startswith("http"):
                return val
        # Some tasks nest the result under a dict (e.g. metadata) or a list
        # (images/assets/output). Search both shapes for a media URL.
        for key in ("metadata", "result", "output", "data"):
            sub = task.get(key)
            if isinstance(sub, dict):
                for f in fields:
                    if isinstance(sub.get(f), str) and sub[f].startswith("http"):
                        return sub[f]
        for key in ("images", "assets", "output", "result"):
            val = task.get(key)
            if isinstance(val, list) and val:
                first = val[0]
                if isinstance(first, str) and first.startswith("http"):
                    return first
                if isinstance(first, dict):
                    for f in fields:
                        if isinstance(first.get(f), str) and first[f].startswith("http"):
                            return first[f]
        raise Ai33Error(f"ai33 task done but no media URL found ({str(task)[:200]})")

    def download(self, url: str) -> bytes:
        r = self.http.get(url, timeout=120)
        r.raise_for_status()
        return r.content

    def run(self, path: str, *, prefer: str = "", **create_kwargs) -> bytes:
        """create_task -> poll -> download, the common end-to-end flow."""
        task_id = self.create_task(path, **create_kwargs)
        task = self.poll(task_id)
        return self.download(self.result_url(task, prefer=prefer))

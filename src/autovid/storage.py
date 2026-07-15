"""Pluggable persistence for the studio's metadata documents.

Channel and Project *metadata* (the JSON documents) go through one tiny
store interface with two interchangeable backends:

- FsStore — JSON files under DATA_DIR (channels/<slug>/channel.json,
  projects/<slug>/project.json). The default: zero setup, human-readable,
  great for local dev.
- PgStore — PostgreSQL, selected by setting DATABASE_URL. Documents live in
  JSONB tables (`channels`, `projects`); `projects.channel` is kept as a
  real column so per-channel queries don't need to unpack JSON.

Binary media (audio, images, video) is NOT stored here — it stays on the
filesystem (DATA_DIR, a mounted volume in hosted deploys). Media is big,
streamed by the web server, and consumed by ffmpeg; a database adds nothing
for those files, so the split is: metadata → store, assets → disk.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

from .config import DATA_DIR, ROOT

# Kind → (filesystem dir name, json file name, table name).
_KINDS = {
    "channel": ("channels", "channel.json", "channels"),
    "project": ("projects", "project.json", "projects"),
}


class Store(Protocol):
    def get(self, kind: str, slug: str) -> dict[str, Any] | None: ...
    def put(self, kind: str, slug: str, data: dict[str, Any]) -> None: ...
    def delete(self, kind: str, slug: str) -> None: ...
    def exists(self, kind: str, slug: str) -> bool: ...
    def all(self, kind: str) -> list[dict[str, Any]]: ...


# --- filesystem backend (the default) -----------------------------------------

class FsStore:
    """JSON documents as files — exactly the layout autovid always used."""

    name = "fs"

    def _path(self, kind: str, slug: str) -> Path:
        d, fname, _ = _KINDS[kind]
        return DATA_DIR / d / slug / fname

    def get(self, kind, slug):
        try:
            return json.loads(self._path(kind, slug).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def put(self, kind, slug, data):
        path = self._path(kind, slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: a concurrent reader (the dashboard) never sees a torn file.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def delete(self, kind, slug):
        try:
            self._path(kind, slug).unlink()
        except FileNotFoundError:
            pass

    def exists(self, kind, slug):
        return self._path(kind, slug).exists()

    def all(self, kind):
        d, fname, _ = _KINDS[kind]
        root = DATA_DIR / d
        out: list[dict[str, Any]] = []
        if root.exists():
            for sub in sorted(root.iterdir()):
                f = sub / fname
                if f.exists():
                    try:
                        out.append(json.loads(f.read_text(encoding="utf-8")))
                    except Exception:  # noqa: BLE001 — skip an unreadable doc
                        continue
        return out


# --- PostgreSQL backend --------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
  slug        TEXT PRIMARY KEY,
  data        JSONB NOT NULL,
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS projects (
  slug        TEXT PRIMARY KEY,
  channel     TEXT NOT NULL DEFAULT '',
  data        JSONB NOT NULL,
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS projects_channel_idx ON projects (channel);
"""


class PgStore:
    """JSONB document tables. The dashboard runs jobs on a thread pool, so all
    access goes through a small connection pool."""

    name = "postgres"

    def __init__(self, url: str):
        from psycopg_pool import ConnectionPool  # lazy: only needed with DATABASE_URL

        self.pool = ConnectionPool(url, min_size=1, max_size=5, open=True)
        with self.pool.connection() as conn:
            conn.execute(_SCHEMA)

    @staticmethod
    def _table(kind: str) -> str:
        return _KINDS[kind][2]

    def get(self, kind, slug):
        with self.pool.connection() as conn:
            row = conn.execute(
                f"SELECT data FROM {self._table(kind)} WHERE slug = %s", (slug,)
            ).fetchone()
        return row[0] if row else None

    def put(self, kind, slug, data):
        cols, vals = "slug, data", [slug, json.dumps(data, ensure_ascii=False)]
        sets = "data = EXCLUDED.data, updated_at = now()"
        if kind == "project":
            cols, sets = cols + ", channel", sets + ", channel = EXCLUDED.channel"
            vals.append(str(data.get("channel") or ""))
        with self.pool.connection() as conn:
            conn.execute(
                f"INSERT INTO {self._table(kind)} ({cols}) "
                f"VALUES ({', '.join(['%s'] * len(vals))}) "
                f"ON CONFLICT (slug) DO UPDATE SET {sets}",
                vals,
            )

    def delete(self, kind, slug):
        with self.pool.connection() as conn:
            conn.execute(f"DELETE FROM {self._table(kind)} WHERE slug = %s", (slug,))

    def exists(self, kind, slug):
        with self.pool.connection() as conn:
            row = conn.execute(
                f"SELECT 1 FROM {self._table(kind)} WHERE slug = %s", (slug,)
            ).fetchone()
        return row is not None

    def all(self, kind):
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"SELECT data FROM {self._table(kind)} ORDER BY slug"
            ).fetchall()
        return [r[0] for r in rows]


# --- backend selection ----------------------------------------------------------

_STORE: Store | None = None
_STORE_LOCK = threading.Lock()


def store() -> Store:
    """The active backend: PgStore when DATABASE_URL is set, else FsStore."""
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                load_dotenv(ROOT / ".env")  # storage may be imported before load_config()
                url = os.environ.get("DATABASE_URL", "")
                _STORE = PgStore(url) if url else FsStore()
    return _STORE

"""One-shot boot seeding for hosted deploys (`python -m autovid.seed`).

Two idempotent steps, both safe to run on every boot:

1. Demo media — if SEED_URL is set and DATA_DIR hasn't been seeded yet,
   download the tarball (channels/ + projects/ trees: JSON docs + rendered
   media) and unpack it into DATA_DIR. A `.seeded` marker prevents
   re-downloading on later boots.

2. Metadata import — if the active store is PostgreSQL (DATABASE_URL set),
   copy every channel/project document found on the filesystem into the
   database, *only where the slug isn't there yet*. Existing rows are never
   overwritten, so edits made in the dashboard survive redeploys. This also
   doubles as a local filesystem→Postgres migration tool.

Failures are logged and swallowed: a broken seed must never keep the
dashboard from starting.
"""

from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .config import DATA_DIR
from .storage import FsStore, store


def _log(msg: str) -> None:
    print(f"[seed] {msg}", file=sys.stderr)


def fetch_demo_bundle() -> None:
    url = os.environ.get("SEED_URL", "").strip()
    marker = DATA_DIR / ".seeded"
    if not url:
        return
    if marker.exists():
        _log(f"media already present ({marker}), skipping download")
        return
    _log(f"downloading demo bundle: {url}")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with urllib.request.urlopen(url, timeout=120) as resp:
            while chunk := resp.read(1 << 20):
                tmp.write(chunk)
        bundle = Path(tmp.name)
    try:
        with tarfile.open(bundle, "r:gz") as tar:
            tar.extractall(DATA_DIR, filter="data")  # refuses absolute paths/'..'
        marker.write_text(url, encoding="utf-8")
        _log(f"unpacked into {DATA_DIR}")
    finally:
        bundle.unlink(missing_ok=True)


def import_fs_docs() -> None:
    active = store()
    if isinstance(active, FsStore):
        return  # filesystem IS the store — nothing to import into
    fs = FsStore()
    for kind in ("channel", "project"):
        added = skipped = 0
        for data in fs.all(kind):
            slug = data.get("slug", "")
            if not slug:
                continue
            if active.exists(kind, slug):
                skipped += 1
                continue
            active.put(kind, slug, data)
            added += 1
        _log(f"{kind}s: {added} imported into postgres, {skipped} already there")


def main() -> None:
    for step in (fetch_demo_bundle, import_fs_docs):
        try:
            step()
        except Exception as e:  # noqa: BLE001 — seeding must never block startup
            _log(f"{step.__name__} failed (continuing): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

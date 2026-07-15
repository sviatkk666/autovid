"""Storage backends: the same contract must hold for FsStore and PgStore.

PgStore tests need a real database: set PG_TEST_URL (CI provides a postgres
service container); locally e.g.
  docker run -d -e POSTGRES_PASSWORD=test -p 55432:5432 postgres:16-alpine
  PG_TEST_URL=postgresql://postgres:test@localhost:55432/postgres pytest
"""

import os

import pytest

from autovid.storage import FsStore

PG_TEST_URL = os.environ.get("PG_TEST_URL", "")


def _roundtrip(store):
    doc = {"slug": "s1", "name": "One", "nested": {"a": [1, 2]}}
    store.put("channel", "s1", doc)
    assert store.get("channel", "s1") == doc
    assert store.exists("channel", "s1")
    assert not store.exists("channel", "missing")
    assert store.get("channel", "missing") is None

    store.put("project", "p1", {"slug": "p1", "channel": "s1", "title": "T"})
    store.put("project", "p2", {"slug": "p2", "channel": "", "title": "U"})
    # all() is slug-ordered; don't assume the store is otherwise empty.
    slugs = [d["slug"] for d in store.all("project")]
    assert [s for s in slugs if s in ("p1", "p2")] == ["p1", "p2"]

    # Overwrite wins; delete is idempotent.
    store.put("project", "p1", {"slug": "p1", "channel": "", "title": "T2"})
    assert store.get("project", "p1")["title"] == "T2"
    store.delete("project", "p1")
    store.delete("project", "p1")
    assert not store.exists("project", "p1")

    store.delete("project", "p2")
    store.delete("channel", "s1")
    assert not any(d["slug"] in ("p1", "p2") for d in store.all("project"))


def test_fs_roundtrip():
    _roundtrip(FsStore())


@pytest.mark.skipif(not PG_TEST_URL, reason="PG_TEST_URL not set")
def test_pg_roundtrip():
    from autovid.storage import PgStore

    _roundtrip(PgStore(PG_TEST_URL))


@pytest.mark.skipif(not PG_TEST_URL, reason="PG_TEST_URL not set")
def test_seed_import_is_idempotent(monkeypatch):
    """import_fs_docs copies FS docs into the DB once and never overwrites."""
    from autovid import seed, storage
    from autovid.storage import PgStore

    fs, pg = FsStore(), PgStore(PG_TEST_URL)
    fs.put("channel", "seeded", {"slug": "seeded", "name": "From FS"})
    monkeypatch.setattr(storage, "_STORE", pg)
    try:
        seed.import_fs_docs()
        assert pg.get("channel", "seeded")["name"] == "From FS"

        # A dashboard edit in the DB must survive a re-seed.
        pg.put("channel", "seeded", {"slug": "seeded", "name": "Edited in DB"})
        seed.import_fs_docs()
        assert pg.get("channel", "seeded")["name"] == "Edited in DB"
    finally:
        pg.delete("channel", "seeded")
        fs.delete("channel", "seeded")
        monkeypatch.setattr(storage, "_STORE", None)

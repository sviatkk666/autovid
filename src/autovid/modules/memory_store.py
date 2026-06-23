"""Content memory — what the studio remembers about past storylines.

A small local corpus the director reads BEFORE writing a new storyline, so the
agents can build a series on a theme and avoid repeating themselves. Each record
captures the durable essence of one produced video (title, theme, hook, summary,
key points) and leaves room for YouTube analytics to be folded in later.

Stored as a single JSON file at the repo root (content_memory.json), keyed by
project slug. Intentionally dependency-free and human-readable.

Entry points:
  remember(project, ...) -> dict        # upsert a record from a Project
  recall(theme=None, limit=N) -> list    # recent records, theme-ranked
  context_for_prompt(theme, limit) -> str  # compact block to inject into agents
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from ..config import DATA_DIR
from ..project import Project
from ..providers.embeddings import cosine, get_embedder

MEMORY_PATH = DATA_DIR / "content_memory.json"
# The dashboard runs director jobs concurrently (and they all write this one
# file), so serialize the whole read-modify-write.
_LOCK = threading.Lock()

_EMBEDDER = None


def _embedder():
    """Lazily build (and cache) the embedding backend (auto: OpenAI/Ollama/hash)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = get_embedder()
    return _EMBEDDER


def _doc_text(rec: dict) -> str:
    """The text we embed for a record (what the director should be able to recall)."""
    return " ".join(filter(None, [rec.get("title", ""), rec.get("theme", ""),
                                  rec.get("angle", ""), rec.get("hook", ""), rec.get("summary", "")]))


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save(records: list[dict], path: Path) -> None:
    # Unique temp name so concurrent writers never clobber a shared .tmp.
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _summarize(project: Project) -> str:
    """A one-paragraph gist of the video from its script (no LLM)."""
    text = (project.script_human or project.script_raw
            or " ".join(s.text for s in project.scenes)).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:400]


def record_from_project(project: Project) -> dict:
    idea = project.source_idea or {}
    return {
        "slug": project.slug,
        "title": project.title,
        "channel": project.channel,
        "theme": project.theme or idea.get("theme", ""),
        "hook": idea.get("hook", ""),
        "angle": idea.get("angle", ""),
        "summary": _summarize(project),
        "n_scenes": len(project.scenes),
        # Placeholder for YouTube analytics, wired later ("потім підключу").
        "analytics": {},
    }


def remember(project: Project, path: Path = MEMORY_PATH) -> dict:
    """Upsert a memory record for a project (keyed by slug). Thread-safe.

    Also embeds the record so future storylines can be retrieved semantically.
    """
    rec = record_from_project(project)
    try:
        vec = _embedder().embed([_doc_text(rec)])
        if vec:
            rec["vec"] = vec[0]
    except Exception as e:  # noqa: BLE001 — embedding is best-effort; keyword recall still works
        print(f"[memory] embed skipped ({e})", file=__import__("sys").stderr)
    with _LOCK:
        records = [r for r in _load(path) if r.get("slug") != rec["slug"]]
        records.append(rec)
        _save(records, path)
    return rec


def recall_semantic(query: str | None, k: int = 12, channel: str | None = None,
                    path: Path = MEMORY_PATH) -> list[dict] | None:
    """Top-k records by embedding similarity to `query`. None → caller falls back."""
    if not query:
        return None
    records = _load(path)
    if channel is not None:
        records = [r for r in records if r.get("channel", "") == channel]
    withvec = [r for r in records if r.get("vec")]
    if not withvec:
        return None
    try:
        qv = _embedder().embed([query])[0]
    except Exception:  # noqa: BLE001
        return None
    return sorted(withvec, key=lambda r: cosine(qv, r["vec"]), reverse=True)[:k]


def _score(rec: dict, theme: str) -> int:
    """Cheap relevance: count shared lowercase word stems between theme & record."""
    if not theme:
        return 0
    want = set(re.findall(r"[a-z]{3,}", theme.lower()))
    hay = set(re.findall(r"[a-z]{3,}", f"{rec.get('theme','')} {rec.get('title','')} "
                                       f"{rec.get('angle','')}".lower()))
    return len(want & hay)


def recall(theme: str | None = None, limit: int = 12, channel: str | None = None,
           path: Path = MEMORY_PATH) -> list[dict]:
    """Most relevant (if theme given) else most recent records, optional channel filter."""
    records = _load(path)
    if channel is not None:
        records = [r for r in records if r.get("channel", "") == channel]
    if theme:
        records = sorted(records, key=lambda r: _score(r, theme), reverse=True)
    else:
        records = list(reversed(records))
    return records[:limit]


def context_for_prompt(theme: str | None = None, limit: int = 12,
                       channel: str | None = None, path: Path = MEMORY_PATH) -> str:
    """A compact block the director injects so agents don't repeat past videos.

    Uses semantic (embedding) recall when records are indexed, else keyword/recency.
    """
    recs = recall_semantic(theme, limit, channel, path)
    if recs is None:
        recs = recall(theme, limit, channel, path)
    if not recs:
        return ""
    lines = ["Previously produced videos (do NOT repeat these; build on/around them):"]
    for r in recs:
        bit = f"- {r.get('title','(untitled)')}"
        if r.get("angle"):
            bit += f" — {r['angle']}"
        lines.append(bit)
    return "\n".join(lines)


def all_records(path: Path = MEMORY_PATH) -> list[dict]:
    return _load(path)

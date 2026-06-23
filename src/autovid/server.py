"""Local web dashboard for the autovid pipeline.

A thin FastAPI layer over the existing modules so you can drive and inspect the
whole pipeline from a browser: brainstorm/generate storylines, run each step,
preview audio/images/video, and edit intermediates (script, per-scene text and
image prompt, re-voice / regen single scenes, reorder/delete) before continuing.

Heavy steps (tts, image search/gen/chart, montage, scriptgen, batch) run as
background jobs; the page polls /api/jobs/{id} for status + live log. Each job
captures the modules' stderr via a thread-local router, and writes to one
project are serialized with a per-slug lock so concurrent jobs never clobber
project.json.

Launch with `python -m autovid.cli serve` (or `uvicorn autovid.server:app`).
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import usage
from .channel import CHANNELS_DIR, Channel
from .config import ROOT, env, load_config
from .modules import memory_store
from .providers.ai33 import ai33_credits
from .providers.llm import available_models
from .modules.agent import run_agent
from .modules.audiomix import audiomix_done, mix_project
from .modules.chartgen import chart_project
from .modules.director import build_blueprint
from .modules.strategist import (channel_edit_turn, channel_setup_turn, chat_reply,
                                 draft_profile, extract_brief)
from .modules.humanizer import humanize_text
from .modules.imagegen import generate_project
from .modules.images import fetch_project
from .modules.montage import build_video
from .modules.parser import parse_script
from .modules.photo_edit import photo_edit_project
from .modules.scriptgen import brainstorm_ideas, write_script
from .modules.thumbnail import make_thumbnail
from .modules.tts import synthesize_project
from .modules.visuals import realize_visuals
from .providers.voices import list_voices
from .project import PROJECTS_DIR, Project, Scene

WEB_DIR = Path(__file__).parent / "web"
CFG = load_config()
SETTINGS_FILE = ROOT / "settings.json"
AGENT_KEYS = ["screenwriter", "humanizer", "art_director", "voice_director",
              "sound_designer", "showrunner", "producer", "strategist"]


def _apply_settings() -> None:
    """Overlay dashboard-saved per-agent model choices onto CFG."""
    try:
        s = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        CFG.setdefault("models", {}).setdefault("agents", {}).update(s.get("agents", {}))
    except Exception:  # noqa: BLE001
        pass


_apply_settings()


# --- per-job stderr capture --------------------------------------------------

class _LogRouter:
    """A sys.stderr stand-in that routes a thread's writes to its job log.

    Background worker threads bind their job's log list for the run; everything
    else (uvicorn's own logging) falls through to the real stderr.
    """

    def __init__(self, real):
        self.real = real
        self._local = threading.local()

    def bind(self, sink: list[str]) -> None:
        self._local.sink = sink

    def unbind(self) -> None:
        self._local.sink = None

    def write(self, s: str) -> int:
        sink = getattr(self._local, "sink", None)
        if sink is not None:
            sink.append(s)
            if len(sink) > 4000:
                del sink[: len(sink) - 2000]
            return len(s)
        return self.real.write(s)

    def flush(self) -> None:
        self.real.flush()

    def isatty(self) -> bool:
        return False


LOG_ROUTER = _LogRouter(sys.stderr)
sys.stderr = LOG_ROUTER


# --- job registry ------------------------------------------------------------

@dataclass
class Job:
    id: str
    kind: str
    slug: str
    status: str = "queued"          # queued | running | done | error
    log: list[str] = field(default_factory=list)
    result: Any = None
    error: str = ""
    created: float = 0.0

    def view(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "slug": self.slug,
            "status": self.status, "error": self.error, "result": self.result,
            "log": "".join(self.log)[-8000:],
        }


EXECUTOR = ThreadPoolExecutor(max_workers=4)
JOBS: dict[str, Job] = {}
_JOBS_GUARD = threading.Lock()
_SLUG_LOCKS: dict[str, threading.Lock] = defaultdict(threading.Lock)
_LOCKS_GUARD = threading.Lock()
_SLUG_RE = __import__("re").compile(r"^[a-z0-9][a-z0-9-]*$")


def _safe_slug(slug: str) -> str:
    """Reject slugs that aren't a plain direct child of PROJECTS_DIR (no traversal)."""
    if not _SLUG_RE.match(slug or "") or \
            (PROJECTS_DIR / slug).resolve().parent != PROJECTS_DIR.resolve():
        raise HTTPException(400, f"invalid slug '{slug}'")
    return slug


def _slug_lock(slug: str):
    if not slug:
        return nullcontext()
    with _LOCKS_GUARD:
        return _SLUG_LOCKS[slug]


def _unique_slug(base: str) -> str:
    """Return base, or base-2/-3/... if a project with that slug already exists."""
    base = base or "project"
    slug, i = base, 2
    while Project.exists(slug):
        slug, i = f"{base}-{i}", i + 1
    return slug


def _run_job(job: Job, fn: Callable[[], Any]) -> None:
    job.status = "running"
    LOG_ROUTER.bind(job.log)
    usage.bind(job.kind)
    try:
        with _slug_lock(job.slug):
            job.result = fn()
        job.status = "done"
    except Exception as e:  # noqa: BLE001 — surface any failure to the UI
        job.error = f"{type(e).__name__}: {e}"
        job.log.append(f"\n[error] {job.error}\n")
        job.status = "error"
    finally:
        LOG_ROUTER.unbind()
        usage.unbind()


def _submit(kind: str, slug: str, fn: Callable[[], Any]) -> Job:
    job = Job(id=uuid.uuid4().hex, kind=kind, slug=slug, created=time.time())
    with _JOBS_GUARD:
        JOBS[job.id] = job
        # Keep the registry from growing forever.
        if len(JOBS) > 200:
            for jid in sorted(JOBS, key=lambda j: JOBS[j].created)[:50]:
                JOBS.pop(jid, None)
    EXECUTOR.submit(_run_job, job, fn)
    return job


# --- config helpers ----------------------------------------------------------

def _cfg(niche: str | None = None) -> dict:
    c = copy.deepcopy(CFG)
    if niche:
        c.setdefault("scriptgen", {})["niche"] = niche
    return c


def _provider_status() -> dict:
    keys = {
        "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
        "ai33": "AI33_API_KEY", "elevenlabs": "ELEVENLABS_API_KEY",
        "pexels": "PEXELS_API_KEY", "pixabay": "PIXABAY_API_KEY",
        "replicate": "REPLICATE_API_TOKEN", "stability": "STABILITY_API_KEY",
    }
    return {name: bool(env(var)) for name, var in keys.items()}


# --- project views -----------------------------------------------------------

def _asset_url(slug: str, rel: str) -> str:
    """URL for a project-relative asset, cache-busted by mtime."""
    if not rel:
        return ""
    f = PROJECTS_DIR / slug / rel
    if not f.exists():
        return ""
    return f"/projects/{slug}/{rel}?t={int(f.stat().st_mtime)}"


def _step_status(p: Project) -> dict:
    n = len(p.scenes)
    voiced = sum(1 for s in p.scenes if s.audio_path and (p.dir / s.audio_path).exists())
    imaged = sum(1 for s in p.scenes if s.image_path and (p.dir / s.image_path).exists())
    return {
        "script": bool(p.script_raw),
        "humanize": bool(p.script_human),
        "parse": {"done": n > 0, "scenes": n},
        "tts": {"done": n > 0 and voiced == n, "have": voiced, "total": n},
        "images": {"done": n > 0 and imaged == n, "have": imaged, "total": n},
        "montage": bool(p.video_path) and (p.dir / p.video_path).exists(),
        "audiomix": audiomix_done(p),
        "thumbnail": bool(p.thumbnail_path) and (p.dir / p.thumbnail_path).exists(),
    }


def _project_view(p: Project, *, full: bool = True) -> dict:
    data = asdict(p)
    data["steps"] = _step_status(p)
    data["video_url"] = _asset_url(p.slug, p.video_path)
    data["thumbnail_url"] = _asset_url(p.slug, p.thumbnail_path)
    data["duration_sec"] = round(sum(s.est_duration_sec for s in p.scenes), 1)
    # All generated thumbnail variants, with which one is currently selected, so
    # the dashboard can show a picker (POST .../select-thumbnail to choose one).
    data["thumbnail_variants"] = [
        {"path": rel, "url": _asset_url(p.slug, rel), "selected": rel == p.thumbnail_path}
        for rel in (p.thumbnails or []) if _asset_url(p.slug, rel)
    ]
    if full:
        for s, sv in zip(p.scenes, data["scenes"]):
            sv["audio_url"] = _asset_url(p.slug, s.audio_path)
            sv["image_url"] = _asset_url(p.slug, s.image_path)
    else:
        data.pop("scenes", None)
        data.pop("script_raw", None)
        data.pop("script_human", None)
    return data


def _load(slug: str) -> Project:
    _safe_slug(slug)
    try:
        return Project.load(slug)
    except FileNotFoundError:
        raise HTTPException(404, f"no project '{slug}'")


# --- app ---------------------------------------------------------------------

app = FastAPI(title="autovid dashboard")

if PROJECTS_DIR.exists():
    app.mount("/projects", StaticFiles(directory=str(PROJECTS_DIR)), name="projects")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/config")
def get_config():
    sg = CFG.get("scriptgen", {})
    return {
        "providers": _provider_status(),
        "niches": ["motivational", "educational", "storytelling"],
        "defaults": {
            "niche": sg.get("niche", "motivational"),
            "target_seconds": sg.get("target_seconds", 60),
            "aspect": "16:9",
        },
        "image_modes": ["search", "generate", "chart", "photo_edit"],
        "visual_types": ["search", "photo_edit", "chart", "generate", "animation"],
        "create_modes": ["director", "topic", "idea", "script", "batch"],
        "steps": ["humanize", "parse", "tts", "visuals", "images", "imagegen",
                  "chart", "photo_edit", "montage", "audiomix", "thumbnail"],
        "tts_providers": ["auto", "elevenlabs", "ai33", "piper"],
        "voices": list_voices(CFG),
        "director_agents": list(CFG.get("director", {}).get(
            "enabled_agents", ["screenwriter", "humanizer", "art_director",
                               "voice_director", "sound_designer", "showrunner"])),
        "max_ai_fraction": CFG.get("director", {}).get("max_ai_fraction", 0.2),
    }


@app.get("/api/projects")
def list_projects():
    out = []
    if PROJECTS_DIR.exists():
        for d in sorted(PROJECTS_DIR.iterdir()):
            if (d / "project.json").exists():
                try:
                    out.append(_project_view(Project.load(d.name), full=False))
                except Exception:  # noqa: BLE001 — skip an unreadable project
                    continue
    return {"projects": out}


@app.get("/api/projects/{slug}")
def get_project(slug: str):
    return _project_view(_load(slug))


@app.delete("/api/projects/{slug}")
def delete_project(slug: str):
    _safe_slug(slug)
    with _slug_lock(slug):
        d = PROJECTS_DIR / slug
        if not (d / "project.json").exists():
            raise HTTPException(404, f"no project '{slug}'")
        shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}


# --- storyline generation ----------------------------------------------------

@app.post("/api/ideas")
def post_ideas(body: dict = Body(...)):
    theme = (body.get("theme") or "").strip()
    if not theme:
        raise HTTPException(400, "theme is required")
    n = int(body.get("n") or 5)
    niche = body.get("niche")

    def work():
        ideas = brainstorm_ideas(theme, n, _cfg(niche))
        return {"ideas": [asdict(i) for i in ideas]}

    return _submit("ideas", "", work).view()


def _create_project(*, title: str, raw: str, aspect: str, cfg: dict) -> str:
    """Create a project from a ready script: parse into scenes and save."""
    from .util import slugify
    slug = _unique_slug(slugify(title))
    scenes = parse_script(raw, cfg)
    Project(slug=slug, title=title, aspect=aspect, script_raw=raw, scenes=scenes).save()
    print(f"[create] {slug}: {len(scenes)} scenes", file=sys.stderr)
    return slug


@app.post("/api/projects")
def create_project(body: dict = Body(...)):
    mode = body.get("mode", "topic")
    aspect = body.get("aspect") or "16:9"
    niche = body.get("niche")
    seconds = body.get("seconds")
    cfg = _cfg(niche)

    if mode == "script":
        raw = (body.get("script") or "").strip()
        title = (body.get("title") or "").strip() or "untitled"
        if not raw:
            raise HTTPException(400, "script is required")

        def work_script():
            return {"slugs": [_create_project(title=title, raw=raw, aspect=aspect, cfg=cfg)]}

        return _submit("create", "", work_script).view()

    if mode == "topic":
        topic = (body.get("topic") or "").strip()
        if not topic:
            raise HTTPException(400, "topic is required")
        title = (body.get("title") or "").strip() or topic

        def work_topic():
            raw = write_script(topic, cfg, seconds=seconds)
            return {"slugs": [_create_project(title=title, raw=raw, aspect=aspect, cfg=cfg)]}

        return _submit("create", "", work_topic).view()

    if mode == "idea":
        from .modules.scriptgen import Idea
        idea_d = body.get("idea") or {}
        if not idea_d.get("title"):
            raise HTTPException(400, "idea.title is required")
        idea = Idea(title=idea_d.get("title", ""), hook=idea_d.get("hook", ""),
                    angle=idea_d.get("angle", ""))

        def work_idea():
            raw = write_script(idea.title, cfg, idea=idea, seconds=seconds)
            return {"slugs": [_create_project(title=idea.title, raw=raw, aspect=aspect, cfg=cfg)]}

        return _submit("create", "", work_idea).view()

    if mode == "batch":
        theme = (body.get("theme") or "").strip()
        if not theme:
            raise HTTPException(400, "theme is required")
        n = int(body.get("n") or 3)

        def work_batch():
            ideas = brainstorm_ideas(theme, n, cfg)
            slugs = []
            for i, idea in enumerate(ideas, 1):
                print(f"[batch] ({i}/{len(ideas)}) {idea.title}", file=sys.stderr)
                try:
                    raw = write_script(idea.title, cfg, idea=idea, seconds=seconds)
                    slugs.append(_create_project(title=idea.title, raw=raw, aspect=aspect, cfg=cfg))
                except Exception as e:  # noqa: BLE001
                    print(f"[batch] '{idea.title}' failed ({e})", file=sys.stderr)
            return {"slugs": slugs}

        return _submit("batch", "", work_batch).view()

    if mode == "director":
        topic = (body.get("topic") or body.get("theme") or "").strip()
        if not topic:
            raise HTTPException(400, "topic/theme is required")
        from .util import slugify
        chan_slug = (body.get("channel") or "").strip()
        chan = Channel.load(chan_slug) if chan_slug and Channel.exists(chan_slug) else None
        dcfg = copy.deepcopy(cfg)
        if chan:
            dcfg.setdefault("scriptgen", {})["niche"] = chan.niche
            dcfg["_aspect"] = body.get("aspect") or chan.aspect
        else:
            dcfg["_aspect"] = aspect
        prof = chan.profile_text() if chan else ""
        sig = chan.signature_text() if chan else ""

        def work_director():
            project = build_blueprint(
                topic, dcfg, seconds=seconds, log=lambda m: print(m, file=sys.stderr),
                channel_profile=prof, channel_signature=sig,
                channel_slug=(chan.slug if chan else ""))
            project.slug = _unique_slug((body.get("slug") or "").strip() or slugify(project.title))
            project.save()
            memory_store.remember(project)
            return {"slugs": [project.slug]}

        return _submit("director", "", work_director).view()

    if mode == "blank":
        from .util import slugify
        title = (body.get("title") or "Untitled video").strip()
        chan = (body.get("channel") or "").strip()
        asp = (Channel.load(chan).aspect if chan and Channel.exists(chan) else aspect)

        def work_blank():
            slug = _unique_slug(slugify(title))
            Project(slug=slug, title=title, channel=chan, aspect=asp).save()
            return {"slugs": [slug]}

        return _submit("create", "", work_blank).view()

    raise HTTPException(400, f"unknown mode '{mode}'")


# --- pipeline steps ----------------------------------------------------------

def _step_cfg(step: str, provider: str | None) -> dict:
    cfg = copy.deepcopy(CFG)
    if not provider or provider == "auto":
        return cfg
    if step == "tts":
        cfg.setdefault("tts", {})["provider"] = provider
    elif step == "images":
        cfg.setdefault("images", {})["provider"] = provider
    elif step == "imagegen":
        cfg.setdefault("images", {}).setdefault("generate", {})["provider"] = provider
    return cfg


@app.post("/api/projects/{slug}/steps/{step}")
def run_step(slug: str, step: str, body: dict = Body(default={})):
    _load(slug)  # 404 if missing
    force = bool(body.get("force"))
    only_list = body.get("only")
    only = set(int(x) for x in only_list) if only_list else None
    provider = body.get("provider")
    cfg = _step_cfg(step, provider)

    def work():
        p = Project.load(slug)  # reload fresh inside the lock
        if step == "humanize":
            p.script_human = humanize_text(p.script_raw, cfg)
            p.save()
        elif step == "parse":
            src = p.script_human or p.script_raw
            p.scenes = parse_script(src, cfg)
            p.save()
        elif step == "tts":
            synthesize_project(p, cfg, force=force, only=only)
        elif step == "images":
            fetch_project(p, cfg, force=force, only=only)
        elif step == "imagegen":
            generate_project(p, cfg, force=force, only=only)
        elif step == "chart":
            chart_project(p, cfg, force=force, only=only)
        elif step == "photo_edit":
            photo_edit_project(p, cfg, force=force, only=only)
        elif step == "visuals":
            realize_visuals(p, cfg, force=force, only=only)
        elif step == "montage":
            build_video(p, cfg, force=force)
        elif step == "audiomix":
            mix_project(p, cfg, force=force)
        elif step == "thumbnail":
            make_thumbnail(p, cfg, force=force)
        else:
            raise ValueError(f"unknown step '{step}'")
        return {"slug": slug}

    return _submit(step, slug, work).view()


# --- intermediate editing (synchronous) --------------------------------------

_PROJECT_FIELDS = ("title", "aspect", "script_raw", "script_human", "status",
                   "theme", "voice", "music", "style", "blueprint_notes")
# Scene blueprint fields the dashboard may edit (text/prompt + the director's plan).
_SCENE_FIELDS = ("text", "image_prompt", "visual_type", "voice", "delivery",
                 "music", "transition", "animation", "notes")


@app.patch("/api/projects/{slug}")
def patch_project(slug: str, body: dict = Body(...)):
    with _slug_lock(slug):
        p = _load(slug)
        for key in _PROJECT_FIELDS:
            if key in body and body[key] is not None:
                setattr(p, key, body[key])
        p.save()
        return _project_view(p)


@app.patch("/api/projects/{slug}/scenes/{sid}")
def patch_scene(slug: str, sid: int, body: dict = Body(...)):
    with _slug_lock(slug):
        p = _load(slug)
        scene = p.scene_by_id(sid)
        if scene is None:
            raise HTTPException(404, f"no scene {sid}")
        for key in _SCENE_FIELDS:
            if key in body and body[key] is not None:
                setattr(scene, key, body[key])
        if "est_duration_sec" in body and body["est_duration_sec"] is not None:
            scene.est_duration_sec = float(body["est_duration_sec"])
        if isinstance(body.get("sfx"), list):
            scene.sfx = [{"cue": str(c.get("cue", "")), "at_sec": float(c.get("at_sec", 0) or 0)}
                         for c in body["sfx"] if isinstance(c, dict)]
        p.save()
        return _project_view(p)


@app.post("/api/projects/{slug}/scenes")
def add_scene(slug: str, body: dict = Body(default={})):
    with _slug_lock(slug):
        p = _load(slug)
        new = Scene(id=p.next_scene_id(), text=(body.get("text") or "").strip(),
                    image_prompt=(body.get("image_prompt") or "").strip())
        after = body.get("after")
        if after is not None and p.scene_by_id(int(after)) is not None:
            idx = next(i for i, s in enumerate(p.scenes) if s.id == int(after))
            p.scenes.insert(idx + 1, new)
        else:
            p.scenes.append(new)
        p.save()
        return _project_view(p)


@app.post("/api/projects/{slug}/scenes/reorder")
def reorder_scenes(slug: str, body: dict = Body(...)):
    order = [int(x) for x in (body.get("order") or [])]
    with _slug_lock(slug):
        p = _load(slug)
        by_id = {s.id: s for s in p.scenes}
        if set(order) != set(by_id):
            raise HTTPException(400, "order must be a permutation of existing scene ids")
        p.scenes = [by_id[i] for i in order]
        p.save()
        return _project_view(p)


@app.delete("/api/projects/{slug}/scenes/{sid}")
def delete_scene(slug: str, sid: int):
    with _slug_lock(slug):
        p = _load(slug)
        scene = p.scene_by_id(sid)
        if scene is None:
            raise HTTPException(404, f"no scene {sid}")
        for rel in (scene.audio_path, scene.image_path):
            if rel:
                f = p.dir / rel
                if f.exists():
                    f.unlink()
        p.scenes = [s for s in p.scenes if s.id != sid]
        p.save()
        return _project_view(p)


@app.post("/api/projects/{slug}/select-thumbnail")
def select_thumbnail(slug: str, body: dict = Body(...)):
    """Pick one of the generated thumbnail variants as the project's thumbnail."""
    path = (body.get("path") or "").strip()
    with _slug_lock(slug):
        p = _load(slug)
        if path not in (p.thumbnails or []):
            raise HTTPException(400, f"'{path}' is not one of this project's thumbnails")
        if not (p.dir / path).exists():
            raise HTTPException(404, f"thumbnail file '{path}' is missing")
        p.thumbnail_path = path
        p.save()
        return _project_view(p)


# --- jobs --------------------------------------------------------------------

@app.get("/api/memory")
def get_memory():
    return {"records": memory_store.all_records()}


# --- settings (per-agent models) + usage / balance -------------------------

@app.get("/api/settings")
def get_settings():
    return {
        "agent_keys": AGENT_KEYS,
        "models": available_models(),
        "agents": CFG.get("models", {}).get("agents", {}),
        "default_model": CFG.get("llm", {}).get("anthropic_model", "auto"),
    }


@app.put("/api/settings")
def put_settings(body: dict = Body(...)):
    agents = body.get("agents") or {}
    chosen = {k: v for k, v in agents.items() if k in AGENT_KEYS and isinstance(v, str)}
    CFG.setdefault("models", {}).setdefault("agents", {}).update(chosen)
    try:
        SETTINGS_FILE.write_text(json.dumps({"agents": CFG["models"]["agents"]}, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"could not save settings: {e}")
    return {"ok": True, "agents": CFG["models"]["agents"]}


@app.get("/api/usage")
def get_usage():
    s = usage.summary()
    s["balances"] = {"ai33_credits": ai33_credits()}
    return s


# --- channels ----------------------------------------------------------------

def _safe_chan(slug: str) -> str:
    if not _SLUG_RE.match(slug or "") or \
            (CHANNELS_DIR / slug).resolve().parent != CHANNELS_DIR.resolve():
        raise HTTPException(400, f"invalid channel '{slug}'")
    return slug


def _channel_view(ch: Channel) -> dict:
    d = asdict(ch)
    n = 0
    if PROJECTS_DIR.exists():
        for p in PROJECTS_DIR.iterdir():
            j = p / "project.json"
            if j.exists():
                try:
                    if json.loads(j.read_text(encoding="utf-8")).get("channel") == ch.slug:
                        n += 1
                except Exception:  # noqa: BLE001
                    pass
    d["n_projects"] = n
    return d


@app.get("/api/channels")
def list_channels():
    return {"channels": [_channel_view(c) for c in Channel.list()]}


@app.get("/api/channels/{slug}")
def get_channel(slug: str):
    _safe_chan(slug)
    if not Channel.exists(slug):
        raise HTTPException(404, f"no channel '{slug}'")
    return _channel_view(Channel.load(slug))


@app.post("/api/channels")
def create_channel(body: dict = Body(...)):
    from .util import slugify
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    slug = _safe_chan(body.get("slug") or slugify(name))
    if Channel.exists(slug):
        raise HTTPException(409, f"channel '{slug}' already exists")
    ch = Channel(slug=slug, name=name)
    # Accept any Channel profile field (the setup chat sends the whole profile).
    for f in fields(Channel):
        if f.name not in ("slug", "name") and body.get(f.name) is not None:
            setattr(ch, f.name, body[f.name])
    ch.save()
    return _channel_view(ch)


@app.patch("/api/channels/{slug}")
def patch_channel(slug: str, body: dict = Body(...)):
    _safe_chan(slug)
    if not Channel.exists(slug):
        raise HTTPException(404, f"no channel '{slug}'")
    ch = Channel.load(slug)
    for f in fields(Channel):
        if f.name != "slug" and f.name in body and body[f.name] is not None:
            setattr(ch, f.name, body[f.name])
    ch.save()
    return _channel_view(ch)


@app.delete("/api/channels/{slug}")
def delete_channel(slug: str):
    _safe_chan(slug)
    d = CHANNELS_DIR / slug
    if not (d / "channel.json").exists():
        raise HTTPException(404, f"no channel '{slug}'")
    # Release this channel's videos (so they become Unfiled, not orphaned/invisible).
    if PROJECTS_DIR.exists():
        for pd in PROJECTS_DIR.iterdir():
            if (pd / "project.json").exists():
                try:
                    p = Project.load(pd.name)
                    if p.channel == slug:
                        p.channel = ""
                        p.save()
                except Exception:  # noqa: BLE001
                    pass
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}


@app.post("/api/channels/{slug}/draft")
def draft_channel(slug: str, body: dict = Body(default={})):
    """LLM-draft the channel's profile + signature lines (fills empty fields)."""
    _safe_chan(slug)
    if not Channel.exists(slug):
        raise HTTPException(404, f"no channel '{slug}'")
    overwrite = bool(body.get("overwrite"))
    notes = body.get("notes", "")

    def work():
        ch = Channel.load(slug)
        drafted = draft_profile(ch.name, ch.niche, notes, copy.deepcopy(CFG))
        for k, v in drafted.items():
            if v and (overwrite or not getattr(ch, k, "")):
                setattr(ch, k, v)
        ch.save()
        return {"slug": slug}

    return _submit("draft", "", work).view()


@app.post("/api/channels/{slug}/chat")
def channel_chat(slug: str, body: dict = Body(...)):
    """One strategist turn: analyze past videos + brainstorm for this channel."""
    _safe_chan(slug)
    if not Channel.exists(slug):
        raise HTTPException(404, f"no channel '{slug}'")
    ch = Channel.load(slug)
    messages = body.get("messages") or []
    mem = memory_store.context_for_prompt(None, 20, channel=slug)
    reply = chat_reply(ch.profile_text(), messages, copy.deepcopy(CFG), memory_ctx=mem)
    return {"reply": reply, "brief": extract_brief(reply)}


@app.post("/api/channel-agent")
def channel_agent(body: dict = Body(...)):
    """One turn of the chat-first channel setup: reply + an updated profile draft."""
    return channel_setup_turn(body.get("draft") or {}, body.get("messages") or [], copy.deepcopy(CFG))


@app.post("/api/channels/{slug}/edit")
def edit_channel(slug: str, body: dict = Body(...)):
    """Edit an existing channel's profile by chat — applies + saves the changes."""
    _safe_chan(slug)
    if not Channel.exists(slug):
        raise HTTPException(404, f"no channel '{slug}'")
    draft = {f.name: getattr(Channel.load(slug), f.name) for f in fields(Channel) if f.name != "slug"}
    # Run the (slow) LLM turn WITHOUT the lock; only hold it for the load/apply/save.
    res = channel_edit_turn(draft, body.get("messages") or [], copy.deepcopy(CFG))
    with _slug_lock(slug):
        ch = Channel.load(slug)
        for f in fields(Channel):
            if f.name != "slug" and f.name in res["profile"] and res["profile"][f.name] is not None:
                setattr(ch, f.name, res["profile"][f.name])
        ch.save()
        return {"reply": res["reply"], "channel": _channel_view(ch)}


# --- the producer-agent (drive a video by chatting) --------------------------

@app.post("/api/projects/{slug}/agent")
def project_agent(slug: str, body: dict = Body(...)):
    _load(slug)
    messages = body.get("messages") or []

    def work():
        p = Project.load(slug)
        prof = sig = ""
        if p.channel and Channel.exists(p.channel):
            ch = Channel.load(p.channel)
            prof, sig = ch.profile_text(), ch.signature_text()
        return run_agent(p, messages, copy.deepcopy(CFG), channel_profile=prof,
                         channel_signature=sig, channel_slug=p.channel,
                         log=lambda m: print(m, file=sys.stderr))

    return _submit("agent", slug, work).view()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job.view()


@app.get("/api/jobs")
def list_jobs():
    with _JOBS_GUARD:
        snapshot = list(JOBS.values())
    recent = sorted(snapshot, key=lambda j: j.created, reverse=True)[:30]
    return {"jobs": [{"id": j.id, "kind": j.kind, "slug": j.slug,
                      "status": j.status, "error": j.error} for j in recent]}


@app.exception_handler(Exception)
def _unhandled(_request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})

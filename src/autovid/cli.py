"""Command-line entry for the pipeline.

Usage:
  python -m autovid.cli ideas THEME [-n N] [--niche motivational|educational|storytelling]
  python -m autovid.cli script TOPIC [-o OUTPUT] [--niche ...] [--seconds N]
  python -m autovid.cli humanize INPUT [-o OUTPUT] [--strength light|medium|heavy]
  python -m autovid.cli humanize INPUT --cleanup-only      # no LLM, deterministic pass only
  python -m autovid.cli parse INPUT [--humanize] [--deterministic] [--aspect 16:9|9:16]
  python -m autovid.cli tts SLUG [--provider elevenlabs|ai33|piper] [--force]
  python -m autovid.cli images SLUG [--provider openverse|wikimedia|pexels|pixabay|internet_archive] [--force]
  python -m autovid.cli imagegen SLUG [--provider openai|flux|stability|ai33|local] [--force]
  python -m autovid.cli montage SLUG [--force] [--dry-run]
  python -m autovid.cli thumbnail SLUG [--force]
  python -m autovid.cli run (INPUT | --topic TOPIC) [--no-humanize] [--no-tts] [--images search|generate]
  python -m autovid.cli batch THEME [-n N] [--images search|generate] [--aspect 16:9|9:16]
  python -m autovid.cli serve [--host H] [--port P]      # web dashboard
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .channel import Channel
from .config import load_config
from .modules.audiomix import mix_project
from .modules.brandkit import make_brand_kit
from .modules.chartgen import chart_project
from .modules.director import build_blueprint
from .modules.humanizer import deterministic_cleanup, humanize_text
from .modules.imagegen import generate_project
from .modules.images import fetch_project
from .modules import memory_store
from .modules.visuals import realize_visuals
from .modules.montage import build_video
from .modules.parser import parse_script
from .modules.scriptgen import brainstorm_ideas, write_script
from .modules.thumbnail import make_thumbnail
from .modules.tts import synthesize_project
from .project import Project
from .util import slugify


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _write(text: str, out: str | None) -> None:
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(f"[humanizer] wrote {out}", file=sys.stderr)
    else:
        sys.stdout.write(text + "\n")


def cmd_humanize(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.strength:
        cfg.setdefault("humanizer", {})["strength"] = args.strength

    text = _read(args.input)

    if args.cleanup_only:
        result = deterministic_cleanup(text)
    else:
        try:
            result = humanize_text(text, cfg)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            print("tip: run with --cleanup-only to test without an LLM.", file=sys.stderr)
            return 1

    _write(result, args.output)
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.deterministic:
        cfg.setdefault("parser", {})["mode"] = "deterministic"

    raw = _read(args.input)

    text = raw
    human = ""
    if args.humanize:
        try:
            human = humanize_text(raw, cfg)
            text = human
        except RuntimeError as e:
            print(f"error (humanize): {e}", file=sys.stderr)
            return 1

    try:
        scenes = parse_script(text, cfg)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    title = args.title or Path(args.input).stem
    slug = args.slug or slugify(title)
    project = Project(
        slug=slug,
        title=title,
        aspect=args.aspect,
        script_raw=raw,
        script_human=human,
        scenes=scenes,
    )
    path = project.save()

    total = sum(s.est_duration_sec for s in scenes)
    print(f"[parser] {len(scenes)} scenes, ~{total:.0f}s total", file=sys.stderr)
    print(f"[parser] saved {path}", file=sys.stderr)
    return 0


def cmd_tts(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.provider:
        cfg.setdefault("tts", {})["provider"] = args.provider

    try:
        project = Project.load(args.slug)
    except FileNotFoundError:
        print(f"error: no project '{args.slug}' (run `parse` first).", file=sys.stderr)
        return 1

    try:
        synthesize_project(project, cfg, force=args.force)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    done = sum(1 for s in project.scenes if s.audio_path)
    total = sum(s.est_duration_sec for s in project.scenes)
    print(f"[tts] {done}/{len(project.scenes)} scenes voiced, ~{total:.0f}s total",
          file=sys.stderr)
    print(f"[tts] saved {project.json_path}", file=sys.stderr)
    return 0


def cmd_images(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.provider:
        cfg.setdefault("images", {})["provider"] = args.provider

    try:
        project = Project.load(args.slug)
    except FileNotFoundError:
        print(f"error: no project '{args.slug}' (run `parse` first).", file=sys.stderr)
        return 1

    try:
        fetch_project(project, cfg, force=args.force)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    done = sum(1 for s in project.scenes if s.image_path)
    print(f"[images] {done}/{len(project.scenes)} scenes have an image", file=sys.stderr)
    print(f"[images] saved {project.json_path}", file=sys.stderr)
    return 0


def cmd_imagegen(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.provider:
        cfg.setdefault("images", {}).setdefault("generate", {})["provider"] = args.provider

    try:
        project = Project.load(args.slug)
    except FileNotFoundError:
        print(f"error: no project '{args.slug}' (run `parse` first).", file=sys.stderr)
        return 1

    try:
        generate_project(project, cfg, force=args.force)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    done = sum(1 for s in project.scenes if s.image_path)
    print(f"[imagegen] {done}/{len(project.scenes)} scenes have an image", file=sys.stderr)
    print(f"[imagegen] saved {project.json_path}", file=sys.stderr)
    return 0


def cmd_audiomix(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    try:
        project = Project.load(args.slug)
    except FileNotFoundError:
        print(f"error: no project '{args.slug}' (run `montage` first).", file=sys.stderr)
        return 1
    try:
        mix_project(project, cfg, force=args.force)
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_chart(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)

    try:
        project = Project.load(args.slug)
    except FileNotFoundError:
        print(f"error: no project '{args.slug}' (run `parse` first).", file=sys.stderr)
        return 1

    try:
        chart_project(project, cfg, force=args.force)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    done = sum(1 for s in project.scenes if s.image_path)
    print(f"[chart] {done}/{len(project.scenes)} scenes have an image", file=sys.stderr)
    print(f"[chart] saved {project.json_path}", file=sys.stderr)
    return 0


def cmd_montage(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)

    try:
        project = Project.load(args.slug)
    except FileNotFoundError:
        print(f"error: no project '{args.slug}' (run `parse` first).", file=sys.stderr)
        return 1

    try:
        final = build_video(project, cfg, force=args.force, dry_run=args.dry_run)
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if final is None:
        return 1
    print(f"[montage] {'would write' if args.dry_run else 'wrote'} {final}", file=sys.stderr)
    return 0


def cmd_thumbnail(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)

    try:
        project = Project.load(args.slug)
    except FileNotFoundError:
        print(f"error: no project '{args.slug}' (run `parse` first).", file=sys.stderr)
        return 1

    try:
        make_thumbnail(project, cfg, force=args.force)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"[thumbnail] saved {project.json_path}", file=sys.stderr)
    return 0


def cmd_brandkit(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)

    try:
        channel = Channel.load(args.slug)
    except FileNotFoundError:
        print(f"error: no channel '{args.slug}' (create it in the dashboard first).", file=sys.stderr)
        return 1

    try:
        make_brand_kit(channel, cfg, force=args.force)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"[brandkit] saved {channel.json_path}", file=sys.stderr)
    return 0


def cmd_push_channel(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)

    try:
        channel = Channel.load(args.slug)
    except FileNotFoundError:
        print(f"error: no channel '{args.slug}' (create it in the dashboard first).", file=sys.stderr)
        return 1

    # Ensure a brand kit exists (persona + avatar) before pushing.
    if not channel.persona_name or not channel.avatar_path:
        try:
            channel = make_brand_kit(channel, cfg, force=False)
        except (RuntimeError, ValueError) as e:
            print(f"error: brand kit generation failed: {e}", file=sys.stderr)
            return 1

    from .modules.crm_push import ingest_channel
    from .modules.r2_upload import upload_file

    try:
        avatar_url = (upload_file(channel.dir / channel.avatar_path, f"channels/{channel.slug}/avatar.png")
                      if channel.avatar_path else None)
        banner_url = (upload_file(channel.dir / channel.banner_path, f"channels/{channel.slug}/banner.png")
                      if channel.banner_path else None)
        payload = {
            "channelSlug": channel.slug,
            "name": channel.name or channel.slug,
            "handle": channel.handle,
            "niche": channel.niche,
            "language": channel.language,
            "region": channel.region,
            "description": channel.description,
            "rules": channel.rules,
            "audience": channel.audience,
            "keywords": channel.keywords,
            "personaName": channel.persona_name,
            "suggestedEmail": channel.suggested_email,
            "defaultCategory": channel.default_category,
            "voice": channel.voice,
            "visualNotes": channel.visual_notes,
            "thumbnailStyle": channel.thumbnail_style,
            "avatarUrl": avatar_url,
            "bannerUrl": banner_url,
            "signature": {
                "intro": channel.intro,
                "outro": channel.outro,
                "cta": channel.cta,
                "catchphrase": channel.catchphrase,
            },
        }
        ingest_channel(payload)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"[push-channel] pushed '{channel.slug}' to the CRM", file=sys.stderr)
    return 0


def _apply_run_overrides(cfg: dict, args: argparse.Namespace) -> None:
    """Apply --tts-provider / --image-provider onto the loaded config in place."""
    if getattr(args, "tts_provider", None):
        cfg.setdefault("tts", {})["provider"] = args.tts_provider
    if getattr(args, "image_provider", None):
        if args.images == "generate":
            cfg.setdefault("images", {}).setdefault("generate", {})["provider"] = args.image_provider
        else:
            cfg.setdefault("images", {})["provider"] = args.image_provider


def _run_pipeline(raw: str, *, title: str, slug: str, aspect: str, cfg: dict,
                  humanize: bool, do_tts: bool, images_mode: str,
                  thumbnail: bool, force: bool, dry_run: bool,
                  tag: str = "run") -> int:
    """Shared pipeline: humanize -> parse -> tts -> images -> montage [-> thumb].

    Used by `run` and `batch`. Humanize/tts are non-fatal (warn, continue);
    parse and montage are required. Returns a process exit code.
    """
    # 1. humanize (non-fatal)
    text, human = raw, ""
    if humanize:
        try:
            human = humanize_text(raw, cfg)
            text = human
            print(f"[{tag}] humanized script", file=sys.stderr)
        except RuntimeError as e:
            print(f"[{tag}] humanize skipped ({e})", file=sys.stderr)

    # 2. parse (required)
    try:
        scenes = parse_script(text, cfg)
    except (RuntimeError, ValueError) as e:
        print(f"error (parse): {e}", file=sys.stderr)
        return 1

    project = Project(slug=slug, title=title, aspect=aspect,
                      script_raw=raw, script_human=human, scenes=scenes)
    project.save()
    total = sum(s.est_duration_sec for s in scenes)
    print(f"[{tag}] parsed {len(scenes)} scenes (~{total:.0f}s) -> {slug}", file=sys.stderr)

    # 3. tts (non-fatal: montage handles silent scenes)
    if do_tts:
        try:
            synthesize_project(project, cfg, force=force)
        except (RuntimeError, ValueError) as e:
            print(f"[{tag}] tts skipped ({e}) — video will be silent", file=sys.stderr)

    # 4. images: search (default) | generate | chart (non-fatal)
    try:
        if images_mode == "generate":
            generate_project(project, cfg, force=force)
        elif images_mode == "chart":
            chart_project(project, cfg, force=force)
        else:
            fetch_project(project, cfg, force=force)
    except (RuntimeError, ValueError) as e:
        print(f"[{tag}] images skipped ({e})", file=sys.stderr)
    if not any(s.image_path for s in project.scenes):
        print(f"error ({slug}): no scene has an image — cannot render a video.", file=sys.stderr)
        return 1

    # 5. montage (required)
    try:
        final = build_video(project, cfg, force=force, dry_run=dry_run)
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as e:
        print(f"error (montage): {e}", file=sys.stderr)
        return 1
    if final is None:
        return 1

    # 5b. audiomix — realize SFX cues + a music bed onto the video (non-fatal).
    if not dry_run:
        try:
            mix_project(project, cfg, force=force)
        except (RuntimeError, ValueError, subprocess.CalledProcessError) as e:
            print(f"[{tag}] audiomix skipped ({e})", file=sys.stderr)

    # 6. thumbnail (opt-in; non-fatal)
    if thumbnail and not dry_run:
        try:
            make_thumbnail(project, cfg, force=force)
        except (RuntimeError, ValueError) as e:
            print(f"[{tag}] thumbnail skipped ({e})", file=sys.stderr)

    print(f"[{tag}] {'would write' if dry_run else 'done ->'} {final}", file=sys.stderr)
    return 0


def cmd_script(args: argparse.Namespace) -> int:
    """Generate a full narration script from a topic (writes a file or stdout)."""
    cfg = load_config(args.config)
    if args.niche:
        cfg.setdefault("scriptgen", {})["niche"] = args.niche
    try:
        script = write_script(args.topic, cfg, seconds=args.seconds)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    _write(script, args.output) if args.output else sys.stdout.write(script + "\n")
    if args.output:
        print(f"[script] wrote {args.output} ({len(script.split())} words)", file=sys.stderr)
    return 0


def cmd_ideas(args: argparse.Namespace) -> int:
    """Brainstorm N video storylines for a niche/theme and print them."""
    cfg = load_config(args.config)
    if args.niche:
        cfg.setdefault("scriptgen", {})["niche"] = args.niche
    try:
        ideas = brainstorm_ideas(args.theme, args.number, cfg)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for i, idea in enumerate(ideas, 1):
        print(f"{i}. {idea.title}")
        if idea.hook:
            print(f"   hook:  {idea.hook}")
        if idea.angle:
            print(f"   angle: {idea.angle}")
    print(f"[ideas] {len(ideas)} ideas for '{args.theme}'", file=sys.stderr)
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Brainstorm N storylines for a theme, then run each through the pipeline."""
    cfg = load_config(args.config)
    if args.niche:
        cfg.setdefault("scriptgen", {})["niche"] = args.niche
    _apply_run_overrides(cfg, args)

    try:
        ideas = brainstorm_ideas(args.theme, args.number, cfg)
    except (RuntimeError, ValueError) as e:
        print(f"error (brainstorm): {e}", file=sys.stderr)
        return 1
    print(f"[batch] {len(ideas)} ideas for '{args.theme}'", file=sys.stderr)

    failures = 0
    for i, idea in enumerate(ideas, 1):
        print(f"\n[batch] ({i}/{len(ideas)}) {idea.title}", file=sys.stderr)
        try:
            raw = write_script(idea.title, cfg, idea=idea, seconds=args.seconds)
        except (RuntimeError, ValueError) as e:
            print(f"[batch] script failed ({e}) — skipping", file=sys.stderr)
            failures += 1
            continue
        slug = slugify(idea.title)
        rc = _run_pipeline(raw, title=idea.title, slug=slug, aspect=args.aspect, cfg=cfg,
                           humanize=args.humanize, do_tts=args.tts, images_mode=args.images,
                           thumbnail=args.thumbnail, force=args.force, dry_run=args.dry_run,
                           tag="batch")
        failures += rc != 0
    done = len(ideas) - failures
    print(f"\n[batch] {done}/{len(ideas)} videos produced", file=sys.stderr)
    return 0 if failures == 0 else 1


def cmd_run(args: argparse.Namespace) -> int:
    """All-in-one: [generate ->] humanize -> parse -> tts -> images -> montage.

    Source the script from a file (INPUT) or generate it from --topic. Stages
    that need an unavailable provider (humanize/tts) are non-fatal.
    """
    cfg = load_config(args.config)
    if args.niche:
        cfg.setdefault("scriptgen", {})["niche"] = args.niche
    _apply_run_overrides(cfg, args)

    if args.topic:
        try:
            raw = write_script(args.topic, cfg, seconds=args.seconds)
            print(f"[run] generated script from topic ({len(raw.split())} words)", file=sys.stderr)
        except (RuntimeError, ValueError) as e:
            print(f"error (scriptgen): {e}", file=sys.stderr)
            return 1
        title = args.title or args.topic
    elif args.input:
        raw = _read(args.input)
        title = args.title or Path(args.input).stem
    else:
        print("error: give an INPUT script file or --topic to generate one.", file=sys.stderr)
        return 1

    slug = args.slug or slugify(title)
    return _run_pipeline(raw, title=title, slug=slug, aspect=args.aspect, cfg=cfg,
                         humanize=args.humanize, do_tts=args.tts, images_mode=args.images,
                         thumbnail=args.thumbnail, force=args.force, dry_run=args.dry_run,
                         tag="run")


def cmd_direct(args: argparse.Namespace) -> int:
    """Multi-agent director: a theme/topic -> a full, editable production blueprint."""
    cfg = load_config(args.config)
    if args.niche:
        cfg.setdefault("scriptgen", {})["niche"] = args.niche
    cfg["_aspect"] = args.aspect

    try:
        project = build_blueprint(args.topic, cfg, seconds=args.seconds,
                                  log=lambda m: print(m, file=sys.stderr))
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    project.slug = args.slug or slugify(project.title)
    project.save()
    memory_store.remember(project)

    mix: dict[str, int] = {}
    for s in project.scenes:
        mix[s.visual_type] = mix.get(s.visual_type, 0) + 1
    print(f"[direct] {project.slug}: {len(project.scenes)} scenes; visual mix {mix}", file=sys.stderr)
    print(f"[direct] voice: {project.voice or '(none)'} | music: {project.music or '(none)'}", file=sys.stderr)
    print(f"[direct] saved {project.json_path} — edit it, then `visuals` + `montage` (or use the dashboard).",
          file=sys.stderr)
    return 0


def cmd_visuals(args: argparse.Namespace) -> int:
    """Decode each scene's visual_type (search/photo_edit/chart/generate) into an image."""
    cfg = load_config(args.config)
    try:
        project = Project.load(args.slug)
    except FileNotFoundError:
        print(f"error: no project '{args.slug}' (run `direct`/`parse` first).", file=sys.stderr)
        return 1
    realize_visuals(project, cfg, force=args.force)
    done = sum(1 for s in project.scenes if s.image_path)
    print(f"[visuals] {done}/{len(project.scenes)} scenes have an image", file=sys.stderr)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report which providers/tools are available for the pipeline."""
    import shutil

    from .config import env
    cfg = load_config(args.config)

    def line(label, ok, note=""):
        mark = "OK " if ok else "-- "
        print(f"  [{mark}] {label}" + (f"  ({note})" if note else ""))

    print("LLM / agents:")
    for name, var in [("Anthropic", "ANTHROPIC_API_KEY"), ("OpenAI", "OPENAI_API_KEY")]:
        line(name, bool(env(var)))
    from .providers.llm import _ollama_up
    ollama_installed = shutil.which("ollama") or (Path.home() / ".local/bin/ollama").exists()
    up = _ollama_up()
    models = ""
    if up:
        try:
            import requests
            host = env("OLLAMA_HOST", "http://localhost:11434")
            host = host if host.startswith("http") else "http://" + host
            tags = requests.get(host.rstrip("/") + "/api/tags", timeout=3).json().get("models", [])
            models = ", ".join(m.get("name", "") for m in tags[:5])
        except Exception:  # noqa: BLE001
            pass
    note = (f"server up; models: {models or 'none — `ollama pull llama3.2:3b`'}" if up
            else "installed; server not running (`ollama serve`)" if ollama_installed
            else "not installed (https://ollama.com)")
    line("Ollama (local LLM)", bool(up and models), note)
    active = env("LLM_PROVIDER") or cfg.get("llm", {}).get("provider", "auto")
    print(f"  active llm.provider: {active}  (override per-run with LLM_PROVIDER=ollama)")

    print("Voice (TTS):")
    line("ElevenLabs", bool(env("ELEVENLABS_API_KEY")))
    line("ai33 voice", False, "unavailable: ai33 TTS needs API v3 (public key = images only)")
    line("Piper (local)", shutil.which(cfg.get("tts", {}).get("piper_binary") or "piper") is not None)

    print("Images:")
    line("Stock search", True, "Openverse/Wikimedia keyless; Pexels/Pixabay if keyed")
    line("ai33 image gen", bool(env("AI33_API_KEY")))
    line("OpenAI image gen", bool(env("OPENAI_API_KEY")))
    line("Replicate/Flux", bool(env("REPLICATE_API_TOKEN")))

    print("Render tools:")
    line("ffmpeg (montage)", shutil.which(cfg.get("montage", {}).get("ffmpeg_binary") or "ffmpeg") is not None)
    chrome = any(shutil.which(b) for b in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"))
    line("Chrome (charts/photo_edit/thumbnail)", chrome)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Launch the local web dashboard (FastAPI + uvicorn)."""
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print("error: the dashboard needs fastapi + uvicorn.\n"
              "  pip install --user fastapi uvicorn", file=sys.stderr)
        return 1
    import uvicorn
    print(f"[serve] dashboard on http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run("autovid.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autovid")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    h = sub.add_parser("humanize", help="rewrite a script to sound human")
    h.add_argument("input", help="input script (.txt/.md)")
    h.add_argument("-o", "--output", default=None, help="output file (default: stdout)")
    h.add_argument("--strength", choices=["light", "medium", "heavy"], default=None)
    h.add_argument("--cleanup-only", action="store_true",
                   help="run only deterministic cleanup, no LLM call")
    h.set_defaults(func=cmd_humanize)

    p = sub.add_parser("parse", help="split a script into scenes -> project.json")
    p.add_argument("input", help="input script (.txt/.md)")
    p.add_argument("--humanize", action="store_true", help="run humanizer before parsing")
    p.add_argument("--deterministic", action="store_true", help="split without an LLM")
    p.add_argument("--title", default=None, help="project title (default: filename)")
    p.add_argument("--slug", default=None, help="project slug (default: from title)")
    p.add_argument("--aspect", choices=["16:9", "9:16"], default="16:9")
    p.set_defaults(func=cmd_parse)

    t = sub.add_parser("tts", help="synthesize per-scene voiceover for a project")
    t.add_argument("slug", help="project slug (folder under projects/)")
    t.add_argument("--provider", choices=["elevenlabs", "ai33", "piper"], default=None,
                   help="override tts.provider from config")
    t.add_argument("--force", action="store_true",
                   help="re-synthesize scenes that already have audio")
    t.set_defaults(func=cmd_tts)

    im = sub.add_parser("images", help="find & download one image per scene")
    im.add_argument("slug", help="project slug (folder under projects/)")
    im.add_argument("--provider",
                    choices=["openverse", "wikimedia", "pexels", "pixabay", "internet_archive"],
                    default=None, help="override images.provider from config")
    im.add_argument("--force", action="store_true",
                    help="re-fetch scenes that already have an image")
    im.set_defaults(func=cmd_images)

    ig = sub.add_parser("imagegen", help="generate one image per scene with an AI model")
    ig.add_argument("slug", help="project slug (folder under projects/)")
    ig.add_argument("--provider", choices=["openai", "flux", "stability", "ai33", "local"],
                    default=None, help="override images.generate.provider from config")
    ig.add_argument("--force", action="store_true",
                    help="re-generate scenes that already have an image")
    ig.set_defaults(func=cmd_imagegen)

    ch = sub.add_parser("chart", help="visualize each scene as an LLM-authored chart/diagram (HTML->PNG)")
    ch.add_argument("slug", help="project slug (folder under projects/)")
    ch.add_argument("--force", action="store_true",
                    help="re-render scenes that already have an image")
    ch.set_defaults(func=cmd_chart)

    am = sub.add_parser("audiomix", help="synth + mix the blueprint's SFX + music onto the video")
    am.add_argument("slug", help="project slug (folder under projects/)")
    am.add_argument("--force", action="store_true")
    am.set_defaults(func=cmd_audiomix)

    mo = sub.add_parser("montage", help="render images + audio into the final mp4")
    mo.add_argument("slug", help="project slug (folder under projects/)")
    mo.add_argument("--force", action="store_true", help="rebuild even if the video exists")
    mo.add_argument("--dry-run", action="store_true",
                    help="print the ffmpeg commands without running them")
    mo.set_defaults(func=cmd_montage)

    tb = sub.add_parser("thumbnail", help="design an HTML thumbnail and render it to PNG")
    tb.add_argument("slug", help="project slug (folder under projects/)")
    tb.add_argument("--force", action="store_true", help="regenerate even if one exists")
    tb.set_defaults(func=cmd_thumbnail)

    bk = sub.add_parser("brandkit", help="generate the channel's account-creation brand kit (fields + avatar + banner)")
    bk.add_argument("slug", help="channel slug (folder under channels/)")
    bk.add_argument("--force", action="store_true", help="regenerate even if a kit exists")
    bk.set_defaults(func=cmd_brandkit)

    pc = sub.add_parser("push-channel", help="upload the channel brand kit (avatar/banner→R2) + register it in the CRM")
    pc.add_argument("slug", help="channel slug (folder under channels/)")
    pc.set_defaults(func=cmd_push_channel)

    r = sub.add_parser("run", help="all-in-one: [generate ->] humanize -> parse -> tts -> images -> montage")
    r.add_argument("input", nargs="?", default=None, help="input script (.txt/.md); omit if using --topic")
    r.add_argument("--topic", default=None, help="generate the script from this topic instead of a file")
    r.add_argument("--niche", choices=["motivational", "educational", "storytelling"],
                   default=None, help="override scriptgen.niche (with --topic)")
    r.add_argument("--seconds", type=float, default=None, help="target script length in seconds (with --topic)")
    r.add_argument("--title", default=None, help="project title (default: filename/topic)")
    r.add_argument("--slug", default=None, help="project slug (default: from title)")
    r.add_argument("--aspect", choices=["16:9", "9:16"], default="16:9")
    r.add_argument("--no-humanize", dest="humanize", action="store_false",
                   help="skip the humanizer pass (on by default)")
    r.add_argument("--no-tts", dest="tts", action="store_false",
                   help="skip voiceover (produce a silent video)")
    r.add_argument("--images", choices=["search", "generate", "chart"], default="search",
                   help="image approach: search the web (default), AI-generate, or chart")
    r.add_argument("--thumbnail", action="store_true",
                   help="also design an HTML thumbnail -> PNG after montage")
    r.add_argument("--image-provider", default=None,
                   help="override the image provider for the chosen approach")
    r.add_argument("--tts-provider", choices=["elevenlabs", "ai33", "piper"], default=None,
                   help="override tts.provider from config")
    r.add_argument("--force", action="store_true",
                   help="rebuild assets that already exist")
    r.add_argument("--dry-run", action="store_true",
                   help="print the ffmpeg commands without running them")
    r.set_defaults(func=cmd_run, humanize=True, tts=True)

    sc = sub.add_parser("script", help="generate a full narration script from a topic")
    sc.add_argument("topic", help="what the video is about")
    sc.add_argument("-o", "--output", default=None, help="output file (default: stdout)")
    sc.add_argument("--niche", choices=["motivational", "educational", "storytelling"],
                    default=None, help="override scriptgen.niche")
    sc.add_argument("--seconds", type=float, default=None, help="target length in seconds")
    sc.set_defaults(func=cmd_script)

    idp = sub.add_parser("ideas", help="brainstorm N video storylines for a niche/theme")
    idp.add_argument("theme", help="the niche / direction to brainstorm around")
    idp.add_argument("-n", "--number", type=int, default=5, help="how many ideas (default 5)")
    idp.add_argument("--niche", choices=["motivational", "educational", "storytelling"],
                     default=None, help="override scriptgen.niche")
    idp.set_defaults(func=cmd_ideas)

    b = sub.add_parser("batch", help="brainstorm N storylines, then render each to a video")
    b.add_argument("theme", help="the niche / direction to brainstorm around")
    b.add_argument("-n", "--number", type=int, default=3, help="how many videos (default 3)")
    b.add_argument("--niche", choices=["motivational", "educational", "storytelling"],
                   default=None, help="override scriptgen.niche")
    b.add_argument("--seconds", type=float, default=None, help="target script length in seconds")
    b.add_argument("--aspect", choices=["16:9", "9:16"], default="16:9")
    b.add_argument("--no-humanize", dest="humanize", action="store_false")
    b.add_argument("--no-tts", dest="tts", action="store_false")
    b.add_argument("--images", choices=["search", "generate", "chart"], default="search")
    b.add_argument("--thumbnail", action="store_true")
    b.add_argument("--image-provider", default=None)
    b.add_argument("--tts-provider", choices=["elevenlabs", "ai33", "piper"], default=None)
    b.add_argument("--force", action="store_true")
    b.add_argument("--dry-run", action="store_true")
    b.set_defaults(func=cmd_batch, humanize=True, tts=True)

    dr = sub.add_parser("direct", help="multi-agent director: theme/topic -> editable blueprint")
    dr.add_argument("topic", help="the theme or topic to build a video blueprint from")
    dr.add_argument("--seconds", type=float, default=None, help="target script length in seconds")
    dr.add_argument("--niche", choices=["motivational", "educational", "storytelling"],
                    default=None, help="override scriptgen.niche")
    dr.add_argument("--slug", default=None, help="project slug (default: from title)")
    dr.add_argument("--aspect", choices=["16:9", "9:16"], default="16:9")
    dr.set_defaults(func=cmd_direct)

    vs = sub.add_parser("visuals", help="decode each scene's visual_type into an image")
    vs.add_argument("slug", help="project slug (folder under projects/)")
    vs.add_argument("--force", action="store_true", help="re-make scenes that already have an image")
    vs.set_defaults(func=cmd_visuals)

    dc = sub.add_parser("doctor", help="report which providers/tools are available")
    dc.set_defaults(func=cmd_doctor)

    sv = sub.add_parser("serve", help="launch the local web dashboard")
    sv.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    sv.add_argument("--port", type=int, default=8000, help="bind port (default 8000)")
    sv.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev)")
    sv.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

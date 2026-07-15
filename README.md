# autovid — multi-agent video studio

Automated pipeline that turns a topic or a ready-made script into a finished
video: a team of role-specialized LLM agents plans the production, then the
pipeline humanizes the text, voices it, generates visuals, and edits it all
together with ffmpeg — driven from a chat-first web dashboard.

Built as independent modules so each step can be run and inspected on its own,
then chained into a full pipeline. Hybrid stack: cloud APIs for quality, local
free tools as a fallback. Every provider (LLM, TTS, image search, image gen)
sits behind one interface, so backends swap via config, not code changes.

## Pipeline

```
script.txt
  → [1. humanizer]  make the text read like a human wrote it   ← DONE
  → [2. parser]     split into scenes (voiceover + image prompts) → project.json   ← DONE
  → [3. tts]        voice each scene to audio   ← DONE
  → [4. images]     find/download or generate a visual per scene   ← DONE (search + gen)
  → [5. montage]    ffmpeg: images + audio → mp4   ← DONE
```

State for steps 2+ lives in `projects/<slug>/project.json` (see `project.py`).
Each `Scene` holds narration text, an image prompt, a duration estimate, and
asset paths that later stages fill in.

## Studio: multi-agent director + dashboard

Beyond running a ready script through the pipeline, autovid can **invent the
storyline and plan the whole production**, then let you drive and edit every step
from a browser.

**Step 0 — the director.** A team of role-specialized agents (each its own
prompt/model/temperature, see `director:` in `config.yaml`) turns a theme/topic
into a full, editable *blueprint*:

```
Screenwriter → Humanizer → Art Director → Voice Director → Sound Designer → Showrunner
```

The blueprint is the `Scene`/`Project` model enriched with `visual_type`
(search/photo_edit/chart/generate), `voice`, `delivery`, `sfx[]`, `music`,
`transition`, `animation`, and notes. **Visual policy:** the Art Director
targets a configurable AI/stock image ratio (`director.max_ai_fraction`,
default 50/50; override per channel) and a deterministic pass enforces it.
A **content memory** (`content_memory.json`, embedding-ranked) lets the agents
build a series and avoid repeats.

**Four visual treatments** fill the same image slot:
`search` (stock photo) · `photo_edit` (stock photo + a light Claude HTML edit) ·
`chart` (LLM HTML/SVG data-viz) · `generate` (AI image).

**Sound design** (`audiomix`, after montage): the Sound Designer's per-scene SFX
cues and music mood get realized into audio — SFX are **synthesized with ffmpeg**
(a small palette keyed off the cue text, no sound library needed) and placed at
their timeline position; the music bed comes from `montage.music_file` / a
`music/` folder, or a subtle synthesized ambient pad — all mixed low under the
narration. **Voice** is per-scene: the Voice Director casts from the installed
Piper voices (primary voice + the occasional switch for a quote/character).

**The dashboard** (`autovid serve` → http://localhost:8000) is **chat-first**:
- **New channel = a setup chat.** You describe the channel; the assistant fills a
  live, editable profile (incl. the recurring intro/sign-off/CTA); "Create" makes it.
- **New video = a producer chat.** In a channel, hit "＋ New video" and just talk:
  *"write a 30s script on X and split into scenes"*, *"scene 3 → chart"*, *"use the
  female voice for the quote"*, *"render it"*. The agent performs each pipeline
  action (honoring the channel's style) and shows the editable result.
- **Manual controls stay as a fallback** — the pipeline step buttons + per-scene
  cards (edit narration, swap `visual_type`, voice selector, regen one visual,
  reorder/delete, preview audio/images, watch the video) are right below the chat.

```bash
autovid doctor                       # what providers/tools are available
autovid ideas "stoic discipline" -n 5
autovid direct "why discipline beats motivation"   # theme -> full blueprint
autovid visuals <slug>               # decode each scene's visual_type -> images
autovid audiomix <slug>              # synth + mix the blueprint's SFX + music bed
autovid serve                        # the web dashboard
```

### Channels & the director chat

A **Channel** is a reusable profile (`channels/<slug>/channel.json`) that keeps a
YouTube channel's output consistent: niche, description, content rules, audience,
voice + visual style, and a **recurring engagement signature** — a greeting,
sign-off, like/subscribe CTA, and catchphrase that get woven into every video's
script. Click "✦ Draft profile with AI" and the strategist writes the whole
profile (including the signature) from the name + niche.

Each channel has **one chat** — the *director chat*. You talk to a single head
strategist that knows the channel's profile and its past videos (content memory
is scoped per channel), helps you analyze what's worked and brainstorm, and when
you settle on a topic it emits a `PRODUCTION BRIEF`. Hit **Send to production**
and the full six-agent director builds the blueprint behind the scenes — using
the channel's style and signature. You never juggle a dozen LLM chats; the
multi-agent machinery stays hidden.

> **Voice:** ai33.pro exposes **images** via the public API but **not TTS** (it
> moved voice to an "API v3" that needs a web-session login, not the API key).
> Voice runs on **local Piper** (free, offline) — configured in `config.yaml`
> (`tts.piper_binary` + `tts.piper_model`, voice `en_US-ryan-medium`). Swap in
> another model from https://huggingface.co/rhasspy/piper-voices, or set
> `ELEVENLABS_API_KEY` to use ElevenLabs instead.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in keys you have (optional for cleanup-only)
```

LLM backend (writes the scripts/blueprints/chat — note: the *voice* is separate,
done locally by Piper TTS):
- **Cloud (best quality):** set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in `.env`.
- **Local/free/offline:** install [Ollama](https://ollama.com), `ollama serve`,
  `ollama pull llama3.2:3b`. Then either set `llm.provider: ollama` in
  `config.yaml`, or switch per-run with `LLM_PROVIDER=ollama autovid …`.

`llm.provider: auto` picks the first available (cloud key, else a running Ollama).
`autovid doctor` shows what's active. Quality of a small local model is well below
Claude for the multi-agent director, but it runs fully offline.

## Usage — humanizer (step 1)

```bash
# Full humanize (LLM rewrite + deterministic cleanup)
PYTHONPATH=src python3 -m autovid.cli humanize scripts/sample.txt -o out.txt

# Pick rewrite strength
PYTHONPATH=src python3 -m autovid.cli humanize scripts/sample.txt --strength heavy

# Deterministic cleanup only — no LLM needed, good for a quick test
PYTHONPATH=src python3 -m autovid.cli humanize scripts/sample.txt --cleanup-only
```

The humanizer works in two layers:
1. **LLM rewrite** — a prompt targeting AI tells (uniform rhythm, stock
   transitions, over-formal words, em-dash overuse). See `modules/humanizer.py`.
2. **Deterministic cleanup** — strips residual banned phrases / em-dashes as a
   safety net. Tune the lists in `BANNED_OPENERS` / `WORD_SWAPS`.

Settings live in `config.yaml` under `humanizer:` and `llm:`.

## Usage — parser (step 2)

```bash
# Split a script into scenes -> projects/<slug>/project.json
PYTHONPATH=src python3 -m autovid.cli parse scripts/sample.txt --title "My Video"

# Humanize first, then parse, in one go
PYTHONPATH=src python3 -m autovid.cli parse scripts/sample.txt --humanize

# No LLM (split by paragraph/sentence, image prompt = scene text)
PYTHONPATH=src python3 -m autovid.cli parse scripts/sample.txt --deterministic

# Vertical Shorts/Reels project
PYTHONPATH=src python3 -m autovid.cli parse scripts/sample.txt --aspect 9:16
```

In `llm` mode the model segments the script into visual beats and writes a
concrete image prompt per scene (narration is kept verbatim). Tune scene length
and visual style in `config.yaml` under `parser:`.

## Usage — tts (step 3)

```bash
# Voice every scene of a parsed project -> projects/<slug>/audio/scene_NN.*
PYTHONPATH=src python3 -m autovid.cli tts productivity-tips

# Force a re-synthesis (otherwise scenes that already have audio are skipped)
PYTHONPATH=src python3 -m autovid.cli tts productivity-tips --force

# Force a specific backend
PYTHONPATH=src python3 -m autovid.cli tts productivity-tips --provider piper
```

Audio paths are written back into `project.json`, and each scene's
`est_duration_sec` is updated to the real audio length when measurable (always
for Piper `.wav`; for ElevenLabs `.mp3` if `mutagen` is installed).

TTS backend (pick one):
- **Cloud:** set `ELEVENLABS_API_KEY` in `.env` (voice/model in `config.yaml`).
- **Local/free:** install [Piper](https://github.com/rhasspy/piper) and set
  `tts.piper_model` to a `.onnx` voice file.

`tts.provider: auto` prefers ElevenLabs, else Piper. The
[ai33.pro](https://ai33.pro) gateway is **images-only** now: its public v1
text-to-speech endpoint is sunset and v3 rejects the public API key, so it is
never auto-selected for voice (set `tts.provider: ai33` manually only if you
have a v3-capable token).

## Usage — images (step 4)

Two approaches fill the same per-scene image slot — pick either (or mix):
**search** the internet for a ready image (`images`), or **generate** one with an
AI model (`imagegen`). Montage works with whichever filled it.

```bash
# Download one image per scene -> projects/<slug>/images/scene_NN.*
PYTHONPATH=src python3 -m autovid.cli images productivity-tips

# Pick a specific source
PYTHONPATH=src python3 -m autovid.cli images productivity-tips --provider wikimedia

# Re-fetch scenes that already have an image
PYTHONPATH=src python3 -m autovid.cli images productivity-tips --force
```

Search sources (set in `config.yaml` under `images:` or `--provider`):
- **Openverse** — Creative Commons aggregator. No key. License-filtered to
  commercial + modification. The safe default.
- **Wikimedia Commons** — mostly public-domain / CC. No key.
- **Internet Archive** — archive.org media. No key. Mixed licensing.
- **Pexels / Pixabay** — stock photos. Free, need `PEXELS_API_KEY` /
  `PIXABAY_API_KEY` in `.env`.

`images.provider: auto` uses Pexels/Pixabay if a key is set, else Openverse.
Each image's license and attribution are stored on the scene and written to
`projects/<slug>/CREDITS.md` — **credit these when you publish.**

### Generation (step 4b)

```bash
# Generate one image per scene from its image prompt
PYTHONPATH=src python3 -m autovid.cli imagegen productivity-tips

# Pick a generator, or re-generate
PYTHONPATH=src python3 -m autovid.cli imagegen productivity-tips --provider flux
PYTHONPATH=src python3 -m autovid.cli imagegen productivity-tips --force
```

Generators (set under `images.generate:` or `--provider`):
- **openai** — DALL·E 3 / `gpt-image-1`. Needs `OPENAI_API_KEY`.
- **flux** — Flux via Replicate. Needs `REPLICATE_API_TOKEN`.
- **stability** — Stable Image (Stability AI). Needs `STABILITY_API_KEY`.
- **ai33** — Imagen-style gen via [ai33.pro](https://ai33.pro) (same key as TTS).
  Needs `AI33_API_KEY`; model via `images.generate.ai33_model`.
- **local** — Stable Diffusion WebUI (AUTOMATIC1111 / Forge) at `SD_WEBUI_URL`.
  No key.

`images.generate.provider: auto` prefers OpenAI → Flux → Stability → ai33 → local SD.
Generated images carry no third-party license (the model's terms apply) and are
recorded in `CREDITS.md` as `generated:<provider>`.

## Usage — montage (step 5)

Renders each scene (its image held for the length of its audio, or its estimated
duration if silent) and concatenates them into the final mp4. Needs **ffmpeg**.

```bash
# Inspect the exact ffmpeg commands without running them (no ffmpeg needed)
PYTHONPATH=src python3 -m autovid.cli montage productivity-tips --dry-run

# Build projects/<slug>/output/<slug>.mp4
PYTHONPATH=src python3 -m autovid.cli montage productivity-tips

# Rebuild over an existing video
PYTHONPATH=src python3 -m autovid.cli montage productivity-tips --force
```

By default each still gets a subtle **Ken Burns** zoom (alternating in/out) so the
video feels alive instead of a static slideshow; this fills the frame (crops to
cover). Set `montage.ken_burns: false` to keep the letterboxed static look
(16:9 → 1920×1080, 9:16 → 1080×1920), or tune `ken_burns_zoom`. Per-scene clips
are kept under `output/clips/` for inspection. Tune resolution, fps and pad color
in `config.yaml` under `montage:`. Install ffmpeg from
[ffmpeg.org](https://ffmpeg.org/download.html) (e.g. `sudo apt install ffmpeg`).

## Usage — thumbnail

Designs a YouTube thumbnail as **HTML** (the LLM reads the title + script and
writes a self-contained poster) and screenshots it to PNG with headless Chrome.
HTML beats a diffusion model here: text is razor-sharp and on-message, and it's
one cheap LLM call. Needs an LLM key and Google Chrome / Chromium.

```bash
# -> projects/<slug>/thumbnail.html + thumbnail.png
PYTHONPATH=src python3 -m autovid.cli thumbnail productivity-tips

# Regenerate
PYTHONPATH=src python3 -m autovid.cli thumbnail productivity-tips --force
```

Size matches the project aspect (16:9 → 1280×720, 9:16 → 1080×1920). The PNG is a
standalone asset (upload it as the YouTube thumbnail); the video isn't changed.
Set `thumbnail.chrome_binary` in `config.yaml` if Chrome isn't on `PATH`.

## Usage — run (whole pipeline)

One command chains every stage from a script to the final mp4:
**humanize → parse → tts → images → montage.**

```bash
# Script in, video out (projects/<slug>/output/<slug>.mp4)
PYTHONPATH=src python3 -m autovid.cli run scripts/sample.txt --slug my-video

# Shorts/Reels, AI-generated visuals, no voiceover
PYTHONPATH=src python3 -m autovid.cli run scripts/sample.txt \
    --aspect 9:16 --images generate --no-tts
```

Stages that need an unavailable provider (humanize/tts) are **non-fatal**: they
warn and the run continues, so you still get a (silent) video. Parse and montage
are required. Add `--thumbnail` to also design the HTML thumbnail after montage.
Flags: `--no-humanize`, `--no-tts`, `--images search|generate`, `--thumbnail`,
`--image-provider`, `--tts-provider`, `--title/--slug/--aspect`, `--force`,
`--dry-run`.

## Storage: filesystem or PostgreSQL

Channel and Project **metadata** (the JSON documents) go through one small
store interface (`src/autovid/storage.py`) with two backends:

- **Filesystem** (default, zero setup): `channels/<slug>/channel.json`,
  `projects/<slug>/project.json` under `DATA_DIR`.
- **PostgreSQL** — set `DATABASE_URL` and the same documents live in JSONB
  tables (`channels`, `projects`); `projects.channel` is a real indexed column.

**Media assets** (audio, images, video) always stay on the filesystem under
`DATA_DIR` — they're big, streamed by the web server, and consumed by ffmpeg.
So a hosted deploy is: Postgres for metadata + a mounted volume for media.

`python -m autovid.seed` (run automatically on container boot) is idempotent:
it downloads a demo bundle into `DATA_DIR` if `SEED_URL` is set, and imports
any filesystem documents into Postgres *only where missing* — dashboard edits
are never overwritten, and it doubles as an FS→Postgres migration tool.

## Deploying (Railway or any Docker host)

The `Dockerfile` ships everything the pipeline needs: ffmpeg, headless
Chromium (HTML→PNG charts/thumbnails), and Piper with two voices for free
local TTS. On Railway:

1. Create a service from this repo (it builds the Dockerfile).
2. Add a **PostgreSQL** database and reference its `DATABASE_URL`.
3. Mount a **volume** at `/data` and set `DATA_DIR=/data`.
4. Optionally set `SEED_URL` to a demo-bundle tarball, and provider API keys
   (without keys the dashboard is a read-only tour of seeded content —
   generation steps explain what's missing instead of crashing).

## Tests

```bash
pip install pytest && pytest            # storage contract, models, text utils
```

The PostgreSQL backend tests run when `PG_TEST_URL` points at a database
(CI spins up a `postgres:16` service and runs both backends on every push).

## Layout

```
config.yaml              global config (per-module sections)
scripts/                 input scripts
tests/                   pytest suite (storage contract, models, text utils)
src/autovid/
  config.py              loads config.yaml + .env; DATA_DIR resolution
  cli.py                 command-line entry (every step + `run` + `serve`)
  server.py              FastAPI dashboard: jobs, chat agents, editing API
  storage.py             metadata store: filesystem (default) / PostgreSQL
  seed.py                idempotent boot seeding (demo bundle + FS->PG import)
  project.py             Project/Scene state (documents via storage.py)
  channel.py             Channel profiles (documents via storage.py)
  providers/llm.py       Anthropic / OpenAI / Ollama behind one interface
  providers/tts.py       ElevenLabs / ai33.pro / Piper behind one interface
  providers/ai33.py      ai33.pro async task gateway (shared by tts + imagegen)
  providers/images.py    Openverse / Wikimedia / Pexels / Pixabay / Archive
  providers/imagegen.py  OpenAI / Flux / Stability / ai33 / local SD
  modules/director.py    the six-agent blueprint builder
  modules/strategist.py  channel setup / director chat agents
  modules/humanizer.py   step 1
  modules/parser.py      step 2
  modules/tts.py         step 3
  modules/images.py      step 4 (search/download)
  modules/imagegen.py    step 4b (AI generation)
  modules/montage.py     step 5 (ffmpeg: Ken Burns, transitions, concat)
  modules/audiomix.py    SFX synthesis + music bed mixing
  modules/thumbnail.py   HTML thumbnail -> PNG (headless Chrome)
```

## Status

- [x] Project skeleton + config + LLM provider abstraction
- [x] Step 1: humanizer
- [x] Step 2: scene parser (+ project.json state)
- [x] Step 3: TTS (ElevenLabs cloud / Piper local)
- [x] Step 4: images — search/download (Openverse/Wikimedia/Pexels/Pixabay/Archive)
- [x] Step 4b: images — generation (OpenAI/Flux/Stability cloud / Stable Diffusion local)
- [x] Step 5: montage (ffmpeg — images + audio → mp4)
- [x] `run` — all-in-one pipeline command (script → mp4)
- [x] Thumbnail — LLM designs HTML poster → PNG (headless Chrome)
- [x] Director — six-agent blueprint (screenwriter → … → showrunner) + channels
- [x] Web dashboard — chat-first producer/strategist agents + manual controls
- [x] Storage backends — filesystem / PostgreSQL behind one interface
- [x] CI — pytest (both storage backends) + ruff on every push

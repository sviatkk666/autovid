#!/usr/bin/env bash
# Build the demo-data bundle uploaded as a GitHub Release asset and pointed to by
# SEED_URL on the hosted deploy. On first boot the app downloads and unpacks it
# into DATA_DIR (channels/ + projects/ trees + the RAG content memory), then
# imports the channel/project documents into Postgres. See src/autovid/seed.py.
#
# Usage:  scripts/make_seed_bundle.sh [output.tar.gz]
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="${1:-autovid-demo-seed.tar.gz}"

# The curated demo set: one polished, finished video per niche (plus a second on
# the flagship channel). Explicit list so half-finished experiments never ship.
SLUGS=(
  director-demo        # the-cold-discipline — "Why Discipline Beats Motivation"
  untitled-video-3     # the-cold-discipline — "How to Stop Reacting…"
  untitled-video-2     # bro-mode-gaming     — "One Epic Headshot in Battlefield 6"
  untitled-video-4     # horse-world         — Ukrainian: ponies
  untitled-video-5     # fire-line           — Ukrainian: fire hydrants
)

PATHS=(channels)
[ -f content_memory.json ] && PATHS+=(content_memory.json)   # the RAG corpus
videos=0
for s in "${SLUGS[@]}"; do
  if [ -f "projects/$s/output/$s.mp4" ]; then
    PATHS+=("projects/$s")
    videos=$((videos + 1))
  else
    echo "WARN: projects/$s has no finished video — skipping" >&2
  fi
done

echo "Channels: $(ls -d channels/*/ 2>/dev/null | wc -l)"
echo "Videos:   $videos"

# Drop the regenerable per-scene clips (the final mp4 already bakes them in) and
# any scratch files to keep the bundle small.
tar --exclude='*/output/clips' --exclude='*.tmp' --exclude='*.json.tmp' \
    -czf "$OUT" "${PATHS[@]}"

echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "Upload it as a Release asset, then set SEED_URL to its download URL."

#!/usr/bin/env bash
# Build the demo-data bundle uploaded as a GitHub Release asset and pointed to by
# SEED_URL on the hosted deploy. It contains the channels/ and projects/ trees
# (JSON documents + rendered media), minus the heavy per-scene clip/ intermediates
# that the final mp4 already bakes in.
#
# Usage:  scripts/make_seed_bundle.sh [output.tar.gz]
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="${1:-autovid-demo-seed.tar.gz}"

# Only ship projects that actually have a finished video (a good demo), plus all
# channels. Exclude output/clips (regenerable, large) and any *.tmp scratch.
INCLUDE=()
for p in projects/*/; do
  slug="$(basename "$p")"
  if [ -f "$p/output/$slug.mp4" ]; then
    INCLUDE+=("$p")
  fi
done

echo "Channels: $(ls -d channels/*/ 2>/dev/null | wc -l)"
echo "Projects with a finished video: ${#INCLUDE[@]}"

tar --exclude='*/output/clips' --exclude='*.tmp' --exclude='*.json.tmp' \
    -czf "$OUT" channels "${INCLUDE[@]}"

echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "Upload it as a Release asset, then set SEED_URL to its download URL."

#!/bin/bash
# Build the standalone Neuroglancer client from a pinned commit into build/ng-dist (about 5 MB without source maps).
# Needs Node >= 22.18. Output is NOT committed to main; assemble_pages.sh puts it on the Pages branch under /ng/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; PIN="${NG_COMMIT:-547848a}"; SRC="${NG_SRC:-$ROOT/build/ng-src}"
[ -d "$SRC/.git" ] || git clone -q https://github.com/google/neuroglancer "$SRC"
git -C "$SRC" fetch -q --depth 1 origin "$PIN" 2>/dev/null || git -C "$SRC" fetch -q origin; git -C "$SRC" checkout -q "$PIN"
(cd "$SRC" && npm ci --no-audit --no-fund --loglevel=error && npm run build -- --output "$ROOT/build/ng-dist")
echo "built neuroglancer $PIN -> $ROOT/build/ng-dist"

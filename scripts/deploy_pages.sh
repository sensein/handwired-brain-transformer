#!/bin/bash
# Publish demo/ (minus tests) to the gh-pages branch of origin. Needs push access; uses a throwaway clone.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="$(git -C "$ROOT" remote get-url origin)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
"$ROOT/scripts/assemble_pages.sh" "$T"
cd "$T" && git init -q -b gh-pages && git add -A
git -c user.name="$(git -C "$ROOT" config user.name)" -c user.email="$(git -C "$ROOT" config user.email)" commit -q -m "Deploy demo from $(git -C "$ROOT" rev-parse --short HEAD)"
git push -q --force "$REMOTE" gh-pages
echo "deployed gh-pages from $(git -C "$ROOT" rev-parse --short HEAD)"

#!/bin/bash
# Assemble the GitHub Pages tree into $1: demo/ (without tests) and, if built, Neuroglancer under ng/ with our hook injected.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; OUT="$1"; NG="${NG_DIST:-$ROOT/build/ng-dist}"
"$ROOT/scripts/sync_browser_py.sh" >/dev/null
mkdir -p "$OUT" && cp -r "$ROOT/demo/." "$OUT/" && rm -rf "$OUT/test" && touch "$OUT/.nojekyll"
if [ -d "$NG" ]; then
  mkdir -p "$OUT/ng" && rsync -a --exclude='*.map' "$NG/" "$OUT/ng/"
  python3 - "$OUT/ng/index.html" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
hook = '<script type="module" src="../ng-hook.js"></script>'
if hook not in s: s = s.replace('</body>', hook + '</body>', 1) if '</body>' in s else s + hook
open(p, 'w').write(s)
PY
  echo "assembled with Neuroglancer under ng/ ($(du -sh "$OUT/ng" | cut -f1))"
else echo "assembled without Neuroglancer (run scripts/build_neuroglancer.sh)"; fi

#!/bin/bash
# Copy the torch-free hwseg modules into demo/py/hwseg so the browser (Pyodide) can import them.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/demo/py/hwseg" && : > "$ROOT/demo/py/hwseg/__init__.py"
for f in program wiring programs exposure; do cp "$ROOT/src/hwseg/$f.py" "$ROOT/demo/py/hwseg/$f.py"; done
echo "synced browser python modules"

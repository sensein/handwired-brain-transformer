"""Export every program to sparse JSON for the WebGPU demo and print size stats (dense vs nonzero params)."""
import json, os, sys
from hwseg.programs import cells, layers, tissue, vessels

OUT = sys.argv[1] if len(sys.argv) > 1 else "demo/models"
os.makedirs(OUT, exist_ok=True)
models = {
    "tissue": tissue(),
    "layers": layers(),
    "cells": cells(),
    "vessels_13dir": vessels(n_dirs=13),
    "vessels_26dir": vessels(n_dirs=26),
}
index = {}
print(f"{'model':16s} {'layers':>6s} {'heads':>5s} {'d':>3s} {'dense':>8s} {'nonzero':>8s} {'json_kB':>8s}")
for k, p in models.items():
    js = p.to_json()
    open(f"{OUT}/{k}.json", "w").write(js)
    tot, nz = p.n_params()
    nh = sum(len(L.heads) for L in p.layers)
    print(f"{k:16s} {len(p.layers):6d} {nh:5d} {p.d:3d} {tot:8d} {nz:8d} {len(js)/1024:8.1f}")
    index[k] = dict(file=f"{k}.json", kind=p.meta["kind"], dense=tot, nonzero=nz, heads=nh, layers=len(p.layers), tile=p.meta.get('tile', 128))
json.dump(index, open(f"{OUT}/index.json", "w"), indent=1)

"""Export every program to sparse JSON for the WebGPU demo and print size stats (dense vs nonzero params)."""
import json, os, sys
from hwseg.programs import cells, exposure, layers, tissue, vessels

OUT = sys.argv[1] if len(sys.argv) > 1 else "demo/models"
os.makedirs(OUT, exist_ok=True)
SCALES = {"cells": (1, 2, 4, 8), "vessels_13dir": (1, 2, 4, 8), "vessels_26dir": (1, 2, 4, 8), "tissue": (1, 2, 4)}
models = {"layers": layers(), "exposure": exposure()}
for fam, scs in SCALES.items():
    for sc in scs:
        mk = {"cells": cells, "tissue": tissue}.get(fam) or (lambda scale, n=int(fam[8:-3]) if fam.startswith("vessels") else 0: vessels(n_dirs=n, scale=scale))
        models[fam if sc == 1 else f"{fam}_x{sc}"] = mk(scale=sc)
index = {}
print(f"{'model':16s} {'layers':>6s} {'heads':>5s} {'d':>3s} {'dense':>8s} {'nonzero':>8s} {'json_kB':>8s}")
for k, p in models.items():
    js = p.to_json()
    open(f"{OUT}/{k}.json", "w").write(js)
    tot, nz = p.n_params()
    nh = sum(len(L.heads) for L in p.layers)
    print(f"{k:16s} {len(p.layers):6d} {nh:5d} {p.d:3d} {tot:8d} {nz:8d} {len(js)/1024:8.1f}")
    index[k] = dict(file=f"{k}.json", kind=p.meta["kind"], dense=tot, nonzero=nz, heads=nh, layers=len(p.layers), tile=p.meta.get('tile', 128),
                  family=k.split('_x')[0], scale=p.meta.get('scale', 1), sigma0=p.meta.get('sigma0'))
json.dump(index, open(f"{OUT}/index.json", "w"), indent=1)

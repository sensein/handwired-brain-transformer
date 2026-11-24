import json, sys, glob, collections
import numpy as np
rows = []
for f in sys.argv[1:]: rows += json.load(open(f))
by = collections.defaultdict(list)
for r in rows: by[(r["program"], r["level"], r["variant"])].append(r)
progs = ["tissue", "layers", "cells", "vessels"]
print("stains:", sorted({r["stain"] for r in rows}), "slabs:", len({(r["stain"], r["slab"]) for r in rows}))
print("\nBASELINE signal per level  (cells/vessels: positive-voxel fraction %, objects in centre plane | tissue: gray%/white%/bg% | layers: dominant class %)")
print("level  spacing_um |  " + " | ".join(f"{p:>22s}" for p in progs))
for lv in range(7):
    sp = next((r["spacing_um"][2] for r in rows if r["level"] == lv), 0)
    cells_ = []
    for p in progs:
        b = by.get((p, lv, "base"), [])
        if not b: cells_.append(f"{'-':>22s}"); continue
        if p in ("cells", "vessels"): cells_.append(f"{100*np.mean([x['positive'] for x in b]):6.2f}%  n={np.mean([x['count'] for x in b]):6.1f}   ")
        elif p == "tissue": f = {k: np.mean([x['fractions'][k] for x in b]) for k in ("bg", "gray", "white")}; cells_.append(f"g{100*f['gray']:3.0f} w{100*f['white']:3.0f} b{100*f['bg']:3.0f}       ")
        else: f = {k: np.mean([x['fractions'][k] for x in b]) for k in b[0]['fractions']}; k = max(f, key=f.get); cells_.append(f"{k:>3s} {100*f[k]:3.0f}%  (WM {100*f['WM']:2.0f}%)   ")
    print(f"  {lv}    {sp:7.1f}   |  " + " | ".join(cells_))
print("\nSENSITIVITY: mean agreement with baseline (IoU for cells/vessels, pixel agreement for tissue/layers); lower = more sensitive")
for p in progs:
    print(f"\n{p}:  variant -> by level 0..6")
    for vn in ("gamma0.5", "gamma2", "contrast0.5", "contrast2", "bright-0.15", "bright+0.15"):
        vals = []
        for lv in range(7):
            v = by.get((p, lv, vn), [])
            vals.append(f"{np.mean([x['iou'] for x in v]):5.2f}" if v else "  -  ")
        print(f"  {vn:12s} " + "  ".join(vals))

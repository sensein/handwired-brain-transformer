"""Real-crop check of the cells floor: object count in the centre plane vs gamma / contrast / brightness, per tau0.
Wanted: stable counts for reasonable exposure changes, zero for flat or saturated views."""
import json, numpy as np, torch
from scipy import ndimage
from hwseg.exposure import display, window
from hwseg.model import WiredTransformer
from hwseg.programs import cells
bank = np.load("/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/bank.npz"); meta = [m for m in json.load(open("/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/bank.json")) if m["level"] == 0 and m["stain"] in ("YO", "NN")]
V = {"base": (0, 1, 1), "g0.6": (0, 1, .6), "g0.8": (0, 1, .8), "g1.5": (0, 1, 1.5), "g2": (0, 1, 2), "c0.5": (0, .5, 1), "c2": (0, 2, 1), "b-.2": (-.2, 1, 1), "b+.2": (.2, 1, 1), "flat c0.1": (0, .1, 1), "sat b+.5": (.5, 1, 1), "dark b-.5": (-.5, 1, 1)}
print("tau0  " + "  ".join(f"{k:>9s}" for k in V))
for tau0 in (0.02, 0.03, 0.04):
    m = WiredTransformer(cells(tau0=tau0), torch.float32, fast=True); row = {k: [] for k in V}
    for mt in meta:
        v = bank[mt["key"]].astype(np.float32); lo, hi = window(v); sp = tuple(mt["spacing_um"])
        for k, (b, c, g) in V.items():
            seg = (m.forward(display(v, lo, hi, b, c, g), sp)["seg"] > .5).numpy()
            row[k].append(ndimage.label(seg[seg.shape[0] // 2])[1])
    print(f"{tau0:.2f}  " + "  ".join(f"{np.mean(row[k]):9.1f}" for k in V), flush=True)

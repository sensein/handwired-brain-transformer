"""What does each program pick up at each pyramid level, and how much do brightness / contrast / gamma
adjustments of the model input change it? Streams a window per (slab, level); runs all four programs on it.

Adjustment (identical to demo/seg.js): u = clamp((logI-lo)/(hi-lo)); u' = clamp((u-.5)*c + .5 + b)^g; logI' = lo + u'(hi-lo),
with [lo,hi] the 2nd..99.7th percentile of the window's logI."""
import json, sys, time
import numpy as np, torch
from scipy import ndimage
from scipy.ndimage import uniform_filter
from hwseg.data import StreamedZarr, list_zarrs, s3_url, to_logI
from hwseg.model import WiredTransformer
from hwseg.programs import cells, layers, tissue, vessels

stain = sys.argv[1]; out = sys.argv[2]; n_slabs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
VARIANTS = {"base": (0, 1, 1), "gamma0.5": (0, 1, .5), "gamma2": (0, 1, 2), "contrast0.5": (0, .5, 1),
            "contrast2": (0, 2, 1), "bright-0.15": (-.15, 1, 1), "bright+0.15": (.15, 1, 1)}
PROGS = {"tissue": tissue(), "layers": layers(), "cells": cells(), "vessels": vessels(n_dirs=13)}
KIND = {"tissue": ["bg", "gray", "white"], "layers": ["bg", "L1", "L2", "L3", "L4", "L5", "L6", "WM"]}
MODELS = {k: WiredTransformer(p, torch.float32, fast=True) for k, p in PROGS.items()}


def adjust(v, lo, hi, b, c, g):
    u = np.clip((v - lo) / (hi - lo), 0, 1)
    u = np.clip((u - .5) * c + .5 + b, 0, 1) ** g
    return (lo + u * (hi - lo)).astype(np.float32)


def run(name, v, sp):
    o = MODELS[name].forward(torch.as_tensor(v), sp)
    if name in KIND:
        return torch.stack([o[k] for k in KIND[name]]).argmax(0).numpy()
    return (o["seg"] > 0.5).numpy()


rng = np.random.default_rng(3)
assets = [(a, p) for a, p in list_zarrs() if f"stain-{stain}_" in p]
rows = []
for aid, path in [assets[i] for i in rng.choice(len(assets), n_slabs, replace=False)]:
    z = StreamedZarr(s3_url(aid)); Ls = z.levels; top = len(Ls) - 1
    slab = path.split("sample-")[1].split("_")[0]
    a = z.crop(top, *[(0, s) for s in Ls[top].shape[2:]]).astype(np.float32)
    idx = np.unravel_index(np.argmax(uniform_filter(a, 7)), a.shape)
    for lv in range(len(Ls)):
        L = Ls[lv]; sp = L.spacing_um; f = 2 ** (top - lv); N = 96
        rg = []
        for ax, ii in enumerate(idx):
            n = L.shape[2 + ax]; c = int(np.clip(ii * f, N // 2, max(N // 2, n - N // 2))); rg.append((max(0, c - N // 2), min(n, max(0, c - N // 2) + N)))
        v = to_logI(z.crop(lv, *rg))
        if min(v.shape) < 8: continue
        lo, hi = np.quantile(v, [.02, .997]); hi = max(hi, lo + 1)
        base = {}
        t0 = time.time()
        for vn, (b, c, g) in VARIANTS.items():
            vv = v if vn == "base" else adjust(v, lo, hi, b, c, g)
            for pn in PROGS:
                m = run(pn, vv, sp); zc = m.shape[0] // 2; pl = m[zc]
                if vn == "base": base[pn] = m
                bm = base[pn]
                if pn in KIND:
                    agree = float((m == bm).mean()); frac = {k: float((m == i).mean()) for i, k in enumerate(KIND[pn])}
                    r = dict(positive=None, count=None, iou=agree, fractions=frac)
                else:
                    cnt = int(ndimage.label(pl)[1]); inter = (m & bm).sum(); uni = (m | bm).sum()
                    r = dict(positive=float(m.mean()), count=cnt, iou=float(inter / uni) if uni else 1.0, fractions=None)
                rows.append(dict(stain=stain, slab=slab, level=lv, spacing_um=list(sp), shape=list(v.shape), program=pn, variant=vn, **r))
        print(f"{stain} slab {slab} level {lv} shape {v.shape} sp {tuple(round(s,1) for s in sp)} {time.time()-t0:.0f}s", flush=True)
json.dump(rows, open(out, "w"))

"""Checks used while hand-tuning (no fitting: only prints numbers / images to look at).

  fable_check.py stats  DEBUG.json [--levels 0,2,4] [--stains YO,NN]   internal channel statistics on the real crop bank
  fable_check.py robust W.json [--levels 0,2,4] [--stains YO,NN]       IoU of seg vs the default view under b/c/g changes
  fable_check.py sizes  W.json --kind balls|tubes [--level 0]          class assignment vs object size on clean synthetic objects
  fable_check.py dark   W.json [--stains YO,NN]                        detection fraction over a brightness x contrast grid (all L0 crops)
"""
import argparse
import json

import numpy as np
import torch

from hwseg.exposure import display, window
from hwseg.model import Program, WiredTransformer

ap = argparse.ArgumentParser()
ap.add_argument("mode"); ap.add_argument("weights")
ap.add_argument("--bank", default="/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/bank")
ap.add_argument("--levels", default="0,2,4"); ap.add_argument("--stains", default="YO,NN,LEC")
ap.add_argument("--kind", default="balls"); ap.add_argument("--level", type=int, default=0)
ap.add_argument("--out", default="seg"); ap.add_argument("--sizes", default="")
a = ap.parse_args()
prog = Program.from_json(open(a.weights).read()); model = WiredTransformer(prog, torch.float32, fast=True)
bank = np.load(a.bank + ".npz"); meta = json.load(open(a.bank + ".json"))
levels = [int(x) for x in a.levels.split(",")]
sel = [m for m in meta if m["stain"] in a.stains.split(",") and m["level"] in levels]


def run(u, sp):
    return {k: v.numpy() for k, v in model.forward(torch.as_tensor(np.ascontiguousarray(u, dtype=np.float32)), tuple(sp)).items()}


def iou(p, q):
    return (p & q).sum() / max((p | q).sum(), 1)


if a.mode == "stats":
    names = [n for n in prog.out]
    for m in sel:
        o = run(display(bank[m["key"]].astype(np.float32)), m["spacing_um"])
        print(m["key"], " ".join(f"{n}:{100 * (o[n] > .5).mean():.2f}%" for n in names if n in ("seg", "nucleus", "soma", "cluster", "capillary", "vessel", "large")))
        line = []
        for n in names:
            if n.startswith("thr") or n in ("hbar", "mad", "ubar"):
                line.append(f"{n}={np.median(o[n]):.4g}")
        print("   ", " ".join(line))
        ks = sorted(int(n[1:]) for n in names if n[0] == "g" and n[1:].isdigit())
        for k in ks[:-1]:
            d = o[f"g{k}"] - o[f"g{k + 1}"]
            q = np.quantile(d, [0.5, 0.9, 0.99, 0.999, 1.0])
            print(f"    band{k}: sd={d.std():.4f} q50/90/99/99.9/max=" + "/".join(f"{x:.3f}" for x in q) + f"  >thr {100 * (d > o[f'thr{k}'] + o.get(f'pen{k}', 0)).mean():.2f}%  pen>0.01: {100 * (np.asarray(o.get(f'pen{k}', np.zeros(1))) > 0.01).mean():.0f}%")
        for n in names:
            if n.startswith("dev"):
                d = o[n]; q = np.quantile(d, [0.5, 0.9, 0.99, 0.999, 1.0])
                print(f"    {n}: q50/90/99/99.9/max=" + "/".join(f"{x:.4f}" for x in q))

elif a.mode == "robust":
    V = {"c0.5": (0, .5, 1), "c2": (0, 2, 1), "c4": (0, 4, 1), "g0.5": (0, 1, .5), "g2": (0, 1, 2), "b+.15": (.15, 1, 1), "b-.15": (-.15, 1, 1),
         "c.1": (0, .1, 1)}
    print(f"{'crop':12s} {'base%':>6s} " + " ".join(f"{k:>11s}" for k in V) + "   (IoU vs default / detected %)")
    acc = {k: [] for k in V}
    for m in sel:
        v = bank[m["key"]].astype(np.float32); lo, hi = window(v)
        base = run(display(v, lo, hi), m["spacing_um"])[a.out] > .5
        row = []
        for k, (b, c, g) in V.items():
            p = run(display(v, lo, hi, b, c, g), m["spacing_um"])[a.out] > .5
            j = iou(p, base); row.append(f"{j:.2f}/{100 * p.mean():5.2f}")
            if base.sum() > 0: acc[k].append(j)
        print(f"{m['key']:12s} {100 * base.mean():6.2f} " + " ".join(f"{r:>11s}" for r in row))
    print(f"{'mean IoU':12s} {'':6s} " + " ".join(f"{np.mean(acc[k]) if acc[k] else float('nan'):11.2f}" for k in V))

elif a.mode == "sizes":
    sp0 = np.array([2.564, 3.625, 2.564]) * 2 ** a.level
    shape = (64, 64, 64)
    ax = [(np.arange(n) - (n - 1) / 2) * s for n, s in zip(shape, sp0)]
    zz, yy, xx = np.meshgrid(*ax, indexing="ij")
    rng = np.random.default_rng(0)
    names = [n for n in prog.out]
    print("clean synthetic object, bg 0.3, fg 0.8, noise sd 0.02; fraction of object voxels in each output; level", a.level)
    sizes = np.array([4, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100] if a.kind == "balls" else [4, 6, 10, 16, 25, 40, 64]) * 2 ** a.level
    if a.sizes:
        sizes = np.array([float(x) for x in a.sizes.split(",")]) * 2 ** a.level
    names = [n for n in names if n in ("seg", "nucleus", "soma", "cluster", "capillary", "vessel", "large")]
    inner = (np.abs(zz) < 0.6 * ax[0][-1]) & (np.abs(yy) < 0.6 * ax[1][-1]) & (np.abs(xx) < 0.6 * ax[2][-1])   # ignore FOV borders
    oris = [(0.3, 0.5, 0.81), (0, 1, 0), (0, 0, 1), (0.6, 0.0, 0.8)] if a.kind == "tubes" else [None]
    for dia in sizes:
        R = dia / 2
        acc, segs, outs_ = [], [], []
        for dv in oris:
            if dv is None:
                obj = zz**2 + yy**2 + xx**2 <= R**2
            else:
                dv = np.array(dv, float); dv /= np.linalg.norm(dv)
                t = zz * dv[0] + yy * dv[1] + xx * dv[2]
                obj = ((zz**2 + yy**2 + xx**2 - t**2) <= R**2) & inner
            if obj.sum() == 0:
                continue
            u = np.clip(0.3 + 0.5 * obj + rng.normal(0, 0.02, shape), 0, 1)
            o = run(u, sp0)
            acc.append([100 * (o[n][obj] > .5).mean() for n in names]); segs.append(acc[-1][0])
            outs_.append(100 * (o["seg"][~obj & inner] > .5).mean())
        if not acc:
            continue
        m_ = np.mean(acc, 0)
        print(f"  {a.kind} diameter {dia:6.1f} um: " + " ".join(f"{n}={x:5.1f}%" for n, x in zip(names, m_))
              + f"   | seg min over orientations {min(segs):5.1f}%  outside seg {np.mean(outs_):5.2f}%")

elif a.mode == "phantom":
    from hwseg.phantom import make_phantom
    from scipy.ndimage import binary_dilation
    names = [n for n in prog.out if n in ("seg", "nucleus", "soma", "cluster", "capillary", "vessel", "large")]
    V = {"base": (0, 1, 1), "c0.5": (0, .5, 1), "c2": (0, 2, 1), "g0.5": (0, 1, .5), "g2": (0, 1, 2), "b+.15": (.15, 1, 1), "b-.15": (-.15, 1, 1), "c.1": (0, .1, 1)}
    kinds = [("cells", dict(n_cells=25, n_vessels=0)), ("vessels", dict(n_cells=15, n_vessels=3))]
    for kind, kw in kinds:
        if a.kind not in ("balls", kind): continue
        print(f"phantom {kind}: truth = {kind}; P/R/F1 of seg; where the positives fall (truth / other object / background, dilated 1 voxel); class %")
        for vn, (b, c, g) in V.items():
            acc = []
            for seed in range(3):
                vol, cells_, ves, sp = make_phantom(seed=seed, fg=800.0, cell_r_um=4.0, **kw)
                o = run(display(np.log(vol + 1), brightness=b, contrast=c, gamma=g), sp)
                p = o["seg"] > .5; t = cells_ if kind == "cells" else ves; oth = ves if kind == "cells" else cells_
                tp = (p & t).sum(); P = tp / max(p.sum(), 1); R = tp / max(t.sum(), 1)
                td, od = binary_dilation(t), binary_dilation(oth)
                acc.append([P, R, 2 * tp / max(p.sum() + t.sum(), 1), (p & td).sum() / max(p.sum(), 1), (p & od & ~td).sum() / max(p.sum(), 1),
                            (p & ~od & ~td).sum() / max(p.sum(), 1), 100 * p.mean(), 100 * t.mean()] + [100 * (o[n] > .5).mean() for n in names[1:]])
            m_ = np.mean(acc, 0)
            print(f"  {vn:6s} P {m_[0]:.2f} R {m_[1]:.2f} F1 {m_[2]:.2f} | on truth {m_[3]:.2f} other {m_[4]:.2f} bg {m_[5]:.2f} | seg {m_[6]:.2f}% truth {m_[7]:.2f}% | "
                  + " ".join(f"{n} {x:.2f}%" for n, x in zip(names[1:], m_[8:])))

elif a.mode == "exposure":
    print(f"{'crop':12s} {'lo(q2)':>7s} {'hi(q99.7)':>9s} | {'lo_net':>7s} {'hi_net':>7s} | pct of voxels below lo_net / above hi_net | min-width active")
    err = []
    for m in [m for m in meta if m["level"] in levels]:
        v = bank[m["key"]].astype(np.float32); lo, hi = window(v); q2, q997 = np.quantile(v, [0.02, 0.997])
        o = run(v, m["spacing_um"]); ln, hn = float(o["lo"].mean()), float(o["hi"].mean())
        assert o["lo"].std() < 1e-3 and o["hi"].std() < 1e-3
        err.append([ln - lo, hn - hi, 100 * (v < ln).mean(), 100 * (v > hn).mean()])
        print(f"{m['key']:12s} {lo:7.3f} {hi:9.3f} | {ln:7.3f} {hn:7.3f} | {100 * (v < ln).mean():5.2f}% (target 2) {100 * (v > hn).mean():5.2f}% (target 0.3) | {'yes' if q997 < q2 + 1 else 'no'}")
    e = np.array(err)
    print(f"mean abs error lo {np.abs(e[:, 0]).mean():.3f}  hi {np.abs(e[:, 1]).mean():.3f} (log units; window widths are >= 1)   median pct below lo {np.median(e[:, 2]):.2f}  above hi {np.median(e[:, 3]):.2f}")

elif a.mode == "phantomfig":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hwseg.phantom import make_phantom
    kw = dict(n_cells=25, n_vessels=0) if a.kind == "cells" else dict(n_cells=15, n_vessels=3)
    vol, cells_, ves, sp = make_phantom(seed=0, fg=800.0, cell_r_um=4.0, **kw)
    u = display(np.log(vol + 1)); o = run(u, sp); names = [n for n in prog.out if n in ("seg", "nucleus", "soma", "cluster", "capillary", "vessel", "large")]
    zs = [8, 16, 24, 32, 40]
    fig, ax = plt.subplots(3, len(zs), figsize=(3 * len(zs), 9))
    cols = [(1, .2, .8), (.2, .9, 1), (1, .8, .1), (.4, 1, .3)]
    for i, z in enumerate(zs):
        ax[0, i].imshow(u[z], cmap="gray", vmin=0, vmax=1); ax[0, i].set_title(f"u' z={z}", fontsize=8)
        t = np.stack([u[z]] * 3, -1); t[cells_[z]] = (.2, .9, 1); t[ves[z]] = (1, .8, .1); ax[1, i].imshow(t); ax[1, i].set_title("truth: cells cyan, vessels yellow", fontsize=8)
        r = np.stack([u[z]] * 3, -1)
        for k, n in enumerate(names):
            m_ = o[n][z] > .5; r[m_] = 0.3 * r[m_] + 0.7 * np.array(cols[k % 4])
        ax[2, i].imshow(r); ax[2, i].set_title(" ".join(names), fontsize=7)
        for x in ax[:, i]: x.axis("off")
    fig.tight_layout(); fig.savefig(a.out if a.out.endswith(".png") else "phantom.png", dpi=55); print("wrote", a.out)

elif a.mode == "dark":
    cs = (0.1, 0.25, 0.5, 1, 2, 4); bs = (-0.5, -0.3, -0.15, 0, 0.15, 0.3, 0.5)
    for m in [m for m in sel if m["level"] == levels[0]]:
        v = bank[m["key"]].astype(np.float32); lo, hi = window(v)
        print(m["key"], "  contrast: " + " ".join(f"{c:6.2f}" for c in cs))
        for b in bs:
            print(f"   brightness {b:+.2f}   " + " ".join(f"{100 * (run(display(v, lo, hi, b, c, 1.0), m['spacing_um'])[a.out] > .5).mean():6.2f}" for c in cs))

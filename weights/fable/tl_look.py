"""Look at tissue / cortex-wm / layer weight files on whole-slab windows (evaluation only; changes no weights).

  tl_look.py raw OUTDIR                                  display images + histograms of u' (no model)
  tl_look.py map WEIGHTS OUTDIR [--classes a,b,c] [--b --c --g] [--levels 6,5] [--stains NN,YO] [--tag T] [--chan name,..]
  tl_look.py robust WEIGHTS [--classes ..] [--levels ..] [--stains ..]     IoU of each class mask vs default view
  tl_look.py crop WEIGHTS OUTDIR --key K --yx y0,y1,x0,x1    run on a sub-FOV (FOV statistics from the crop alone)

The model is run on a depth slab of NZ planes around the centre plane (as the viewer does), mask = output > 0.5.
"""
import argparse, json, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hwseg.exposure import display, window
from hwseg.model import Program, WiredTransformer

BANK = "/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/slabbank"
COL = {"tissue": (.2, .9, .3), "bg": (.25, .25, .9), "cortex": (1, .55, .1), "wm": (.2, .8, 1), "sup": (1, .9, .2), "mid": (1, .5, .1),
       "deep": (.9, .15, .5), "VAL": (1, 0, 0), "l1": (1, 1, .4), "l2": (1, .75, .2), "l3": (1, .45, .1), "l4": (.9, .2, .3), "l5": (.7, .2, .8)}
ap = argparse.ArgumentParser()
ap.add_argument("mode"); ap.add_argument("a1", nargs="?"); ap.add_argument("a2", nargs="?")
ap.add_argument("--b", type=float, default=0.0); ap.add_argument("--c", type=float, default=1.0); ap.add_argument("--g", type=float, default=1.0)
ap.add_argument("--levels", default="6,5"); ap.add_argument("--stains", default="NN,YO"); ap.add_argument("--classes", default="")
ap.add_argument("--chan", default=""); ap.add_argument("--tag", default="d"); ap.add_argument("--nz", type=int, default=3)
ap.add_argument("--key", default=""); ap.add_argument("--yx", default=""); ap.add_argument("--keys", default="")
a = ap.parse_args()
bank = np.load(BANK + ".npz"); meta = json.load(open(BANK + ".json"))
levels = [int(x) for x in a.levels.split(",")]
sel = [m for m in meta if m["stain"] in a.stains.split(",") and m["level"] in levels]
if a.keys: sel = [m for m in meta if m["key"] in a.keys.split(",")]


def slab(m, nz=None):
    nz = nz or a.nz
    v = bank[m["key"]]; z = v.shape[0] // 2; h = nz // 2
    return v[max(0, z - h):z + h + 1].astype(np.float32)


def load(path):
    p = Program.from_json(open(path).read())
    return p, WiredTransformer(p, torch.float32, fast=True)


def run(model, u, sp):
    return {k: v.numpy() for k, v in model.forward(torch.as_tensor(u), sp).items()}


def overlay(u2, o, classes, z):
    rgb = np.stack([u2] * 3, -1).copy()
    for n in classes:
        mk = o[n][z] > 0.5
        rgb[mk] = 0.45 * rgb[mk] + 0.55 * np.array(COL.get(n, (1, .2, .8)))
    return rgb


def up2(v):   # pseudo finer level: 2x linear upsampling in-plane and z (spacing halves)
    t = torch.as_tensor(v)[None, None]
    return torch.nn.functional.interpolate(t, scale_factor=2, mode="trilinear", align_corners=False)[0, 0].numpy()


if a.mode == "raw":
    os.makedirs(a.a1, exist_ok=True)
    for m in sel:
        v = slab(m); u = display(v, brightness=a.b, contrast=a.c, gamma=a.g); z = u.shape[0] // 2
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw=dict(width_ratios=[3, 1]))
        ax[0].imshow(u[z], cmap="gray", vmin=0, vmax=1, aspect=m["spacing_um"][1] / m["spacing_um"][2]); ax[0].set_title(m["key"] + f" {u.shape} sp {np.round(m['spacing_um'], 0)}", fontsize=9)
        ax[1].hist(u.ravel(), bins=100, log=True); ax[1].set_title("u' hist  q10/50/90 " + " ".join(f"{q:.2f}" for q in np.quantile(u, [.1, .5, .9])), fontsize=8)
        fig.tight_layout(); fig.savefig(f"{a.a1}/raw_{m['key']}.png", dpi=70); plt.close(fig)
        print(m["key"], u.shape, "window", np.round(window(v), 2), "u' quantiles 5/25/50/75/95", np.round(np.quantile(u, [.05, .25, .5, .75, .95]), 3))
    sys.exit()

prog, model = load(a.a1)
classes = [c for c in a.classes.split(",") if c] or [n for n in prog.out if n in COL]
chans = [c for c in a.chan.split(",") if c]
print("program", prog.name, "d", prog.d, "layers", len(prog.layers), "nonzero", prog.n_params()[1], "classes", classes)

if a.mode in ("map", "crop"):
    os.makedirs(a.a2, exist_ok=True)
    if a.mode == "crop":
        sel = [m for m in meta if m["key"] == a.key]
    for m in sel:
        v = slab(m)
        if a.mode == "crop":
            y0, y1, x0, x1 = [int(t) for t in a.yx.split(",")]; v = v[:, y0:y1, x0:x1]
        sp = tuple(m["spacing_um"])
        if a.tag.startswith("up"): v = up2(v); sp = tuple(s / 2 for s in sp)
        u = display(v, brightness=a.b, contrast=a.c, gamma=a.g); z = u.shape[0] // 2
        o = run(model, u, sp); asp = m["spacing_um"][1] / m["spacing_um"][2]
        n = 2 + len(chans)
        wide = u.shape[2] * 1.0 > 1.6 * u.shape[1] * asp
        fig, ax = plt.subplots(n if wide else 1, 1 if wide else n, figsize=(13, 4.6 * n) if wide else (6.5 * n, 6.5), squeeze=False); ax = ax.ravel()
        ax[0].imshow(u[z], cmap="gray", vmin=0, vmax=1, aspect=asp); ax[0].set_title(f"{m['key']} u' b{a.b} c{a.c} g{a.g}", fontsize=9)
        ax[1].imshow(overlay(u[z], o, classes, z), aspect=asp)
        fr = " ".join(f"{c}:{100 * (o[c][z] > .5).mean():.1f}%" for c in classes)
        ax[1].set_title(fr, fontsize=9)
        for i, cn in enumerate(chans):
            im = ax[2 + i].imshow(o[cn][z], cmap="viridis", aspect=asp); ax[2 + i].set_title(f"{cn} [{o[cn][z].min():.3g},{o[cn][z].max():.3g}]", fontsize=9); plt.colorbar(im, ax=ax[2 + i], fraction=.02)
        for x in ax: x.axis("off")
        fig.tight_layout(); fn = f"{a.a2}/{a.tag}_{m['key']}.png"; fig.savefig(fn, dpi=62); plt.close(fig)
        print(m["key"], fr, " ".join(f"{cn}={o[cn][z].mean():.3f}" for cn in chans), "->", fn)

if a.mode == "sheet":      # compact montage: per level one PNG, [image | overlay] pairs
    os.makedirs(a.a2, exist_ok=True)
    for lev in levels:
        ms = [m for m in sel if m["level"] == lev]
        if not ms: continue
        per = 1 if lev == 6 else 2
        nr = int(np.ceil(len(ms) / per))
        fig, ax = plt.subplots(nr, 2 * per, figsize=(15, (3.1 if lev == 6 else 4.6) * nr), squeeze=False)
        for x in ax.ravel(): x.axis("off")
        for i, m in enumerate(ms):
            v = slab(m); u = display(v, brightness=a.b, contrast=a.c, gamma=a.g); z = u.shape[0] // 2
            o = run(model, u, tuple(m["spacing_um"])); asp = m["spacing_um"][1] / m["spacing_um"][2]
            r, c0 = i // per, 2 * (i % per)
            ax[r, c0].imshow(u[z], cmap="gray", vmin=0, vmax=1, aspect=asp); ax[r, c0].set_title(f"{m['key']} b{a.b} c{a.c} g{a.g}", fontsize=8)
            ax[r, c0 + 1].imshow(overlay(u[z], o, classes, z), aspect=asp)
            fr = " ".join(f"{c}:{100 * (o[c][z] > .5).mean():.0f}%" for c in classes); ax[r, c0 + 1].set_title(fr, fontsize=8)
            print(m["key"], fr, " ".join(f"{cn}={o[cn][z].mean():.3f}" for cn in chans))
        fig.tight_layout(); fn = f"{a.a2}/{a.tag}_L{lev}.png"; fig.savefig(fn, dpi=58); plt.close(fig); print("->", fn)

if a.mode == "robust":
    V = {"c0.5": (0, .5, 1), "c2": (0, 2, 1), "c4": (0, 4, 1), "g0.5": (0, 1, .5), "g2": (0, 1, 2), "b+.15": (.15, 1, 1), "b-.15": (-.15, 1, 1),
         "DARK b-.5": (-.5, 1, 1), "FLAT c.1": (0, .1, 1), "FLAT c.05": (0, .05, 1), "SAT b+1": (1., 1, 1), "SAT b+.6": (.6, 1, 1)}
    print("IoU vs default view per class (first 7 columns); detected % of FOV for DARK / FLAT / SAT views (must be ~0 for non-bg classes)")
    print(f"{'key':16s} {'class':7s} {'def%':>6s} " + " ".join(f"{k:>9s}" for k in V))
    acc = {c: [] for c in classes}
    if not a.keys and a.tag == "sub": sel = [m for m in sel if m["level"] == 6 or m["key"] in ("YO_117_L5_w2", "YO_90_L5_w0", "NN_86_L5_w0", "NN_161_L5_w0", "NN_86_L5_w2")]
    for m in sel:
        v = slab(m); lo, hi = window(v); sp = tuple(m["spacing_um"])
        base = run(model, display(v, lo, hi), sp)
        res = {k: run(model, display(v, lo, hi, *bcg), sp) for k, bcg in V.items()}
        for c in classes:
            g = base[c] > .5; row = []
            for k in V:
                p = res[k][c] > .5
                if k[0] in "DFS": row.append(100 * p.mean())
                else: row.append((p & g).sum() / max((p | g).sum(), 1))
            acc[c].append(row[:7])
            print(f"{m['key']:16s} {c:7s} {100 * g.mean():6.1f} " + " ".join(f"{x:9.2f}" for x in row))
    for c in classes:
        r = np.array(acc[c]); print(f"{'MEAN IoU':16s} {c:7s} {'':6s} " + " ".join(f"{x:9.2f}" for x in r.mean(0)), " min", f"{r.min():.2f}")

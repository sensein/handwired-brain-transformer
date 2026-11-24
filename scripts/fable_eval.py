"""Evaluate a hand-written weight file (Program JSON) by looking at images and phantom metrics.

  python scripts/fable_eval.py WEIGHTS.json OUTDIR [--bank PATH] [--b B --c C --g G] [--levels 0,1,2] [--outputs seg,nucleus]
                               [--phantom] [--progression]

* montage PNGs (OUTDIR/montage_*.png): per crop, the display image u' (what the model is fed) and the overlay of the
  requested outputs (seg-like channels, thresholded at 0.5; class channels coloured). Open them with the Read tool.
* --phantom: F1 on synthetic phantoms (cells/vessels) at several contrast/gamma/brightness settings.
* --progression: detection fraction on a real crop and phantom as brightness / contrast sweep from darkness to saturation.
Model input = display image u' in [0,1] (hwseg.exposure.display): window -> contrast/brightness -> gamma."""
import argparse, json, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hwseg.exposure import display, window
from hwseg.model import Program, WiredTransformer
from hwseg.phantom import make_phantom

ap = argparse.ArgumentParser()
ap.add_argument("weights"); ap.add_argument("out")
ap.add_argument("--bank", default=os.environ.get("FABLE_BANK", "/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/bank"))
ap.add_argument("--b", type=float, default=0.0); ap.add_argument("--c", type=float, default=1.0); ap.add_argument("--g", type=float, default=1.0)
ap.add_argument("--levels", default="0,2,4"); ap.add_argument("--outputs", default=""); ap.add_argument("--stains", default="YO,NN,LEC")
ap.add_argument("--phantom", action="store_true"); ap.add_argument("--progression", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
prog = Program.from_json(open(a.weights).read()); model = WiredTransformer(prog, torch.float32, fast=True)
outs = [o for o in a.outputs.split(",") if o] or [n for n in prog.out if n not in ("score",)]
print("program", prog.name, "d", prog.d, "layers", len(prog.layers), "nonzero params", prog.n_params()[1], "outputs", list(prog.out))


def run(u, sp):
    r = model.forward(torch.as_tensor(u), sp)
    return {k: v.numpy() for k, v in r.items()}


def f1(p, g): return 2 * (p & g).sum() / max(p.sum() + g.sum(), 1)


bank = np.load(a.bank + ".npz"); meta = json.load(open(a.bank + ".json"))
levels = [int(x) for x in a.levels.split(",")]
for st in a.stains.split(","):
    sel = [m for m in meta if m["stain"] == st and m["level"] in levels]
    if not sel: continue
    fig, ax = plt.subplots(len(sel), 2, figsize=(8, 3.2 * len(sel)), squeeze=False)
    for i, m in enumerate(sel):
        v = bank[m["key"]].astype(np.float32); u = display(v, brightness=a.b, contrast=a.c, gamma=a.g); o = run(u, tuple(m["spacing_um"]))
        z = u.shape[0] // 2
        ax[i, 0].imshow(u[z], cmap="gray", vmin=0, vmax=1); ax[i, 0].set_title(f"{m['key']}  u' (b{a.b} c{a.c} g{a.g})", fontsize=8)
        rgb = np.stack([u[z]] * 3, -1)
        cols = [(1, .2, .8), (.2, .9, 1), (1, .8, .1), (.4, 1, .3), (.6, .5, 1)]
        for k, name in enumerate(outs):
            mask = o[name][z] > 0.5
            rgb[mask] = 0.4 * rgb[mask] + 0.6 * np.array(cols[k % len(cols)])
        ax[i, 1].imshow(rgb); ax[i, 1].set_title(" ".join(f"{n}:{100*(o[n][z] > .5).mean():.1f}%" for n in outs), fontsize=8)
        for x in ax[i]: x.axis("off")
    fig.tight_layout(); fig.savefig(f"{a.out}/montage_{st}.png", dpi=60); plt.close(fig)
    print("wrote", f"{a.out}/montage_{st}.png")

if a.phantom:
    print("\nPHANTOM F1 (cells phantom: truth=cells; vessels phantom: truth=vessels); output channel 'seg' or --outputs[0]")
    ch = outs[0]
    for kind, kw in (("cells", dict(n_cells=25, n_vessels=0)), ("vessels", dict(n_cells=15, n_vessels=3))):
        for vn, (b, c, g) in {"base": (0, 1, 1), "c0.5": (0, .5, 1), "c2": (0, 2, 1), "g0.5": (0, 1, .5), "g2": (0, 1, 2), "b+.15": (.15, 1, 1), "b-.15": (-.15, 1, 1)}.items():
            fs = []
            for seed in range(3):
                vol, cells_, ves, sp = make_phantom(seed=seed, fg=800.0, cell_r_um=4.0, **kw)
                u = display(np.log(vol + 1), brightness=b, contrast=c, gamma=g); fs.append(f1(run(u, sp)[ch] > 0.5, cells_ if kind == "cells" else ves))
            print(f"  {kind:8s} {vn:6s} F1 {np.mean(fs):.2f}")

if a.progression:
    print("\nPROGRESSION: positive fraction of output", outs[0], "vs brightness (rows) and contrast (cols) on the first real crop of each stain (level 0) and a phantom")
    cs = (0.1, 0.25, 0.5, 1, 2, 4); bs = (-0.5, -0.3, -0.15, 0, 0.15, 0.3, 0.5)
    tests = [("phantom-cells", np.log(make_phantom(seed=1, n_cells=25, n_vessels=0, fg=800.0, cell_r_um=4.0)[0] + 1), make_phantom()[3])]
    for m in [m for m in meta if m["level"] == 0][:3]: tests.append((m["key"], bank[m["key"]].astype(np.float32), tuple(m["spacing_um"])))
    for name, v, sp in tests:
        lo, hi = window(v); print(f"  {name}   contrast: " + " ".join(f"{c:6.2f}" for c in cs))
        for b in bs:
            row = [100 * (run(display(v, lo, hi, b, c, 1.0), sp)[outs[0]] > 0.5).mean() for c in cs]
            print(f"   brightness {b:+.2f}        " + " ".join(f"{x:6.2f}" for x in row))

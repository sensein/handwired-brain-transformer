"""Whole-slab class maps: builder tissue vs hand-written tissue / cortex-wm / layers, on a +-2 plane slab around the centre
(what the viewer feeds a FOV pass), display image input."""
import json, sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hwseg.exposure import display
from hwseg.model import Program, WiredTransformer
bank = np.load("/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/slabbank.npz"); meta = [m for m in json.load(open("/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/slabbank.json")) if m["key"].endswith("_L6_full")]
W = {k: WiredTransformer(Program.from_json(open(f).read()), torch.float32) for k, f in {"builder": "demo/models/tissue.json", "f_tissue": "weights/fable/fable_tissue.json", "f_cortex_wm": "weights/fable/fable_cortex_wm.json", "f_layers": "weights/fable/fable_layers.json"}.items()}
PAL = {"bg": (0.1, 0.1, 0.15), "gray": (1, .63, .16), "white": (.24, .55, 1), "tissue": (.4, .8, .5), "cortex": (1, .63, .16), "wm": (.24, .55, 1), "sup": (1, .9, .2), "mid": (.4, .85, .4), "deep": (.2, .6, .6)}
fig, ax = plt.subplots(len(meta), 5, figsize=(22, 3.4 * len(meta)), squeeze=False)
for r, m in enumerate(meta):
    v = bank[m["key"]].astype(np.float32); zc = v.shape[0] // 2; u = display(v[max(0, zc - 2): zc + 3]); sp = tuple(m["spacing_um"]); z = u.shape[0] // 2
    ax[r, 0].imshow(u[z], cmap="gray", aspect=sp[1] / sp[2]); ax[r, 0].set_title(m["key"], fontsize=9)
    for c, (k, mod) in enumerate(W.items()):
        o = {n: t.numpy()[z] for n, t in mod.forward(torch.as_tensor(u), sp).items()}
        names = [n for n in o if n in PAL]; img = np.zeros(o[names[0]].shape + (3,)); frac = {}
        for n in names:
            msk = o[n] > 0.5; frac[n] = 100 * msk.mean()
            if n != "bg": img[msk] = PAL[n]
        img[(o["bg"] > 0.5)] = PAL["bg"] if "bg" in o else img[(o["bg"] > 0.5)]
        ax[r, c + 1].imshow(img, aspect=sp[1] / sp[2]); ax[r, c + 1].set_title(k + "  " + " ".join(f"{n}:{p:.0f}" for n, p in frac.items()), fontsize=8)
    for x in ax[r]: x.axis("off")
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=55); print("saved")

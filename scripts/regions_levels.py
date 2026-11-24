"""Do the hand-written tissue / cortex-wm programs give the same maps when the same 12 x 8 mm field of view is read at pyramid
levels 3, 4, 5? (Level 2 would be ~31 M tokens: beyond a GPU buffer, so the viewer never goes there.)"""
import json, sys
import numpy as np, torch
from scipy.ndimage import uniform_filter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hwseg.data import StreamedSlab
from hwseg.exposure import display
from hwseg.model import Program, WiredTransformer

assets = json.load(open("demo/assets.json")); out = sys.argv[1]
M = {k: WiredTransformer(Program.from_json(open(f"weights/fable/{k}.json").read()), torch.float32) for k in ("fable_tissue", "fable_cortex_wm")}
def hz(prog, sp): return sum(max([int(np.ceil(h.radius_um[0] / sp[0])) for h in L.heads if h.radius_um is not None] or [0]) for L in prog.layers)
picks = [("NN", "86"), ("YO", "90")]; LV = (3, 4, 5)
fig, ax = plt.subplots(len(picks) * 2, len(LV), figsize=(5 * len(LV), 3.6 * len(picks) * 2), squeeze=False)
for r, (stain, sample) in enumerate(picks):
    cs = sorted([a for a in assets if a["stain"] == stain and a["sample"] == sample and a["sub"] == "MITU01"], key=lambda a: a["session"] + f"{a['chunk']:03d}")
    slab = StreamedSlab([a["zarr"] for a in cs if a["session"] == cs[0]["session"]], equalize=False); top = slab.top; L6 = slab.levels[top]
    v6 = slab.crop(top, (0, L6.shape[0]), (0, L6.shape[1]), (0, L6.shape[2])); sm = uniform_filter(v6, (5, 9, 25)); z6, y6, x6 = np.unravel_index(np.argmax(sm), sm.shape)
    ref = {}
    for c, lv in enumerate(LV):
        L = slab.levels[lv]; f = 2 ** (top - lv); sp = L.spacing_um
        wy, wx = int(8000 / sp[1]), int(12000 / sp[2])
        cy, cx, cz = int(y6 * f), int(x6 * f), int(z6 * f); y0 = max(0, cy - wy // 2); x0 = max(0, cx - wx // 2)
        h = max(hz(M["fable_tissue"].p, sp), hz(M["fable_cortex_wm"].p, sp)); z0, z1 = max(0, cz - h), min(L.shape[0], cz + h + 1)
        v = slab.crop(lv, (z0, z1), (y0, min(L.shape[1], y0 + wy)), (x0, min(L.shape[2], x0 + wx))); u = display(v); zc = cz - z0
        o = {k: {n: t.numpy()[zc] for n, t in m.forward(torch.as_tensor(u), sp).items()} for k, m in M.items()}
        tis, cor, wm = o["fable_tissue"]["tissue"] > .5, o["fable_cortex_wm"]["cortex"] > .5, o["fable_cortex_wm"]["wm"] > .5
        cls = np.zeros(tis.shape + (3,)); cls[tis] = (.4, .8, .5); cls[cor] = (1, .63, .16); cls[wm] = (.24, .55, 1)
        ax[2 * r, c].imshow(u[zc], cmap="gray", aspect=sp[1] / sp[2]); ax[2 * r, c].set_title(f"{stain}_{sample} L{lv} {sp[2]:.0f}um  {u.shape} tokens {u.size/1e6:.1f}M", fontsize=9)
        ax[2 * r + 1, c].imshow(cls, aspect=sp[1] / sp[2]); ax[2 * r + 1, c].set_title(f"tissue {100*tis.mean():.0f}% cortex {100*cor.mean():.0f}% wm {100*wm.mean():.0f}%", fontsize=9)
        for x in (ax[2 * r, c], ax[2 * r + 1, c]): x.axis("off")
        print(f"{stain}_{sample} L{lv}: shape {u.shape} tissue {100*tis.mean():.1f}% cortex {100*cor.mean():.1f}% wm {100*wm.mean():.1f}%", flush=True)
fig.tight_layout(); fig.savefig(out, dpi=55)

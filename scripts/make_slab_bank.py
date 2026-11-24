"""Bank of stitched real data for model work (no bulk download: coarse whole-slab volumes + small fine-scale windows).
For 6 slab acquisitions (YO / NN / LEC): the whole slab at level 6, three level-5 windows, and three level-1 and level-0
windows at tissue-rich spots. Same npz/json format as the strip bank (key, stain, slab, level, spacing_um, shape)."""
import json, sys
import numpy as np
from scipy.ndimage import uniform_filter
from hwseg.data import StreamedSlab

out = sys.argv[1]; assets = json.load(open("demo/assets.json"))
picks = [("YO", "90"), ("YO", "117"), ("NN", "86"), ("NN", "161"), ("LEC", "150"), ("LEC", "56")]
arrs, meta = {}, []
def add(key, stain, sample, level, sp, v):
    arrs[key] = v.astype(np.float16); meta.append(dict(key=key, stain=stain, slab=sample, level=level, spacing_um=list(sp), shape=list(v.shape))); print(key, v.shape, flush=True)
for stain, sample in picks:
    cs = sorted([a for a in assets if a["stain"] == stain and a["sample"] == sample and a["sub"] == "MITU01"], key=lambda a: a["session"] + f"{a['chunk']:03d}")
    ids = [a["zarr"] for a in cs if a["session"] == cs[0]["session"]]
    slab = StreamedSlab(ids, equalize=True); top = slab.top; L6 = slab.levels[top]
    v6 = slab.crop(top, (0, L6.shape[0]), (0, L6.shape[1]), (0, L6.shape[2]))
    add(f"{stain}_{sample}_L6_full", stain, sample, top, L6.spacing_um, v6)
    sm = uniform_filter(v6, (5, 9, 25)); sm[:3] = 0; sm[-3:] = 0
    # tissue-rich spots: brightest, then the brightest at least 120 level-6 columns away, twice
    spots = []; work = sm.copy()
    for _ in range(3):
        z, y, x = np.unravel_index(np.argmax(work), work.shape); spots.append((z, y, x)); work[:, :, max(0, x - 120): x + 120] = 0
    for k, (z, y, x) in enumerate(spots):
        for lv, (dz, dy, dx) in ((5, (24, 128, 192)), (1, (24, 96, 96)), (0, (24, 96, 96))):
            L = slab.levels[lv]; f = 2 ** (top - lv)
            c = (int(z * f), int(y * f), int(x * f)); box = [(max(0, c[i] - d), min(L.shape[i], c[i] + d)) for i, d in enumerate((dz, dy, dx))]
            add(f"{stain}_{sample}_L{lv}_w{k}", stain, sample, lv, L.spacing_um, slab.crop(lv, *box))
np.savez_compressed(out + ".npz", **arrs); json.dump(meta, open(out + ".json", "w"), indent=1); print("saved", len(arrs))

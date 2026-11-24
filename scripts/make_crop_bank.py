"""Stream a fixed bank of real crops (no bulk download) so weight iteration does not re-stream: 6 slabs (2 per stain
YO/NN/LEC) x pyramid levels 0..5, 96^3 windows around the brightest tissue. Saves log(I+1) as float16 + metadata."""
import json, sys
import numpy as np
from scipy.ndimage import uniform_filter
from hwseg.data import StreamedZarr, list_zarrs, s3_url, to_logI

out = sys.argv[1]
by = {}
for aid, p in list_zarrs():
    st = p.split("stain-")[1].split("_")[0]
    if st in ("YO", "NN", "LEC"): by.setdefault(st, []).append((aid, p))
rng = np.random.default_rng(11)
arrs, meta = {}, []
for st in ("YO", "NN", "LEC"):
    for aid, p in [by[st][i] for i in rng.choice(len(by[st]), 2, replace=False)]:
        z = StreamedZarr(s3_url(aid)); Ls = z.levels; top = len(Ls) - 1
        slab = p.split("sample-")[1].split("_")[0]
        a = z.crop(top, *[(0, s) for s in Ls[top].shape[2:]]).astype(np.float32)
        idx = np.unravel_index(np.argmax(uniform_filter(a, 7)), a.shape)
        for lv in range(6):
            L = Ls[lv]; f = 2 ** (top - lv); N = 96; rg = []
            for ax, ii in enumerate(idx):
                n = L.shape[2 + ax]; c = int(np.clip(ii * f, N // 2, max(N // 2, n - N // 2))); rg.append((max(0, c - N // 2), min(n, max(0, c - N // 2) + N)))
            v = to_logI(z.crop(lv, *rg))
            key = f"{st}_{slab}_L{lv}"; arrs[key] = v.astype(np.float16)
            meta.append(dict(key=key, stain=st, slab=slab, level=lv, spacing_um=list(L.spacing_um), shape=list(v.shape)))
            print(key, v.shape, flush=True)
np.savez_compressed(out + ".npz", **arrs); json.dump(meta, open(out + ".json", "w"), indent=1)

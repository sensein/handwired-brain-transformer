"""Seam visibility of the stitched slab: vertical-gradient energy inside strip overlaps / elsewhere (1.0 = no visible seam)."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hwseg.data import StreamedSlab

assets = json.load(open("demo/assets.json")); out = sys.argv[1]
picks = [("YO", "90"), ("NN", "86"), ("LEC", "150")]
fig, ax = plt.subplots(len(picks), 2, figsize=(14, 4.2 * len(picks)))
for r, (stain, sample) in enumerate(picks):
    cs = sorted([a for a in assets if a["stain"] == stain and a["sample"] == sample and a["sub"] == "MITU01"], key=lambda a: a["session"] + f"{a['chunk']:03d}")
    ses = cs[0]["session"]; ids = [a["zarr"] for a in cs if a["session"] == ses]
    for c, eq in enumerate((False, True)):
        slab = StreamedSlab(ids, equalize=eq); l = 6; L = slab.levels[l]
        z = L.shape[0] // 2
        img = slab.crop(l, (z - 2, z + 3), (0, L.shape[1]), (0, L.shape[2])).mean(0)
        # overlap bands (rows where two strips contribute) and interior rows
        cover = np.zeros(L.shape[1], int)
        for i, m in enumerate(slab.members):
            o = L.offs[i]; cover[o[1]: o[1] + m.levels[l].shape[-2]] += 1
        tissue = img > np.log(60)
        g = np.abs(np.diff(img, axis=0)); gm = np.where(tissue[1:] & tissue[:-1], g, np.nan)
        prof = np.nanmean(gm, axis=1); ov = cover[1:] >= 2; inter = cover[1:] == 1
        score = np.nanmedian(prof[ov]) / np.nanmedian(prof[inter])
        print(f"{stain} sample {sample} strips {len(slab.members)} dropped {len(slab.dropped)} equalize={eq}: seam score {score:.3f}  seams corrected {len(slab.seam)}", flush=True)
        ax[r, c].imshow(img, cmap="gray", vmin=np.quantile(img[tissue], 0.02), vmax=np.quantile(img[tissue], 0.995), aspect=L.spacing_um[1] / L.spacing_um[2])
        ax[r, c].set_title(f"{stain} {sample} equalize={eq} seam score {score:.2f}", fontsize=9); ax[r, c].axis("off")
fig.tight_layout(); fig.savefig(out, dpi=60)

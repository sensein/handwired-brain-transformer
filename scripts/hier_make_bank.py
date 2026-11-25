"""Stream (no bulk download) a bank for the hierarchical programs: the SAME physical field of view at several pyramid levels
(scale sweep) and large mm-scale windows at 10-20 um voxels (layer views). Raw log(I+1) float16, slab seam-equalised.
  small sweep: 0.6 x 0.6 mm in-plane, +-30 um in z, levels 0..4, centred inside the cortex near the pia
  large sweep: 8 x 8 mm in-plane, +-1 level-6 plane (~+-160 um) in z, levels 2..6, same centre
  layer windows: two more 8 x 8 mm windows per slab at levels 2 and 3
usage: hier_make_bank.py OUT(prefix)"""
import json, sys
import numpy as np
from scipy.ndimage import uniform_filter, binary_erosion
from hwseg.data import StreamedSlab

out = sys.argv[1]
assets = json.load(open("demo/assets.json"))
picks = [("YO", "90"), ("YO", "117"), ("NN", "86"), ("NN", "161"), ("LEC", "150")]
arrs, meta = {}, []


def add(key, stain, sample, level, sp, v, centre_um):
    arrs[key] = v.astype(np.float16)
    meta.append(dict(key=key, stain=stain, slab=sample, level=level, spacing_um=list(sp), shape=list(v.shape), centre_um=list(centre_um)))
    print(key, v.shape, flush=True)


def crop_um(slab, lv, c_um, half_um):
    L = slab.levels[lv]
    sp = L.spacing_um
    box = []
    for a in range(3):
        c, h, n = c_um[a] / sp[a], max(half_um[a] / sp[a], 0.5), L.shape[a]
        lo, hi = int(round(c - h)), int(round(c + h)) + 1
        lo, hi = max(0, lo), min(n, max(hi, lo + 1))
        box.append((lo, hi))
    return slab.crop(lv, *box), sp


for stain, sample in picks:
    cs = sorted([a for a in assets if a["stain"] == stain and a["sample"] == sample and a["sub"] == "MITU01"], key=lambda a: a["session"] + f"{a['chunk']:03d}")
    slab = StreamedSlab([a["zarr"] for a in cs if a["session"] == cs[0]["session"]], equalize=True)
    top = slab.top
    L6 = slab.levels[top]
    sp6 = L6.spacing_um
    v6 = slab.crop(top, (0, L6.shape[0]), (0, L6.shape[1]), (0, L6.shape[2]))
    zc = int(np.argmax(uniform_filter(v6, (3, 9, 9)).reshape(v6.shape[0], -1).mean(1)))   # brightest plane = through the slab
    pl = uniform_filter(v6[max(0, zc - 1): zc + 2].mean(0), 3)
    thr = np.quantile(pl, 0.5) + 0.5 * (np.quantile(pl, 0.99) - np.quantile(pl, 0.5))
    tis = pl > thr
    inner = binary_erosion(tis, iterations=int(np.ceil(1000 / sp6[2])))         # >= ~1 mm inside the tissue edge ...
    deep = binary_erosion(tis, iterations=int(np.ceil(2500 / sp6[2])))          # ... but not deep in white matter
    cand = np.argwhere(inner & ~deep)
    rng = np.random.default_rng(int(sample))
    picks_xy = cand[rng.choice(len(cand), 3, replace=False)] if len(cand) >= 3 else np.argwhere(tis)[:3]
    for k, (y, x) in enumerate(picks_xy):
        c_um = (zc * sp6[0], y * sp6[1], x * sp6[2])
        if k == 0:
            for lv in range(0, 5):
                v, sp = crop_um(slab, lv, c_um, (30.0, 300.0, 300.0))
                add(f"{stain}_{sample}_S{lv}", stain, sample, lv, sp, v, c_um)
            for lv in range(2, top + 1):
                v, sp = crop_um(slab, lv, c_um, (160.0, 4000.0, 4000.0))
                add(f"{stain}_{sample}_B{lv}", stain, sample, lv, sp, v, c_um)
        else:
            for lv in (2, 3):
                v, sp = crop_um(slab, lv, c_um, (100.0, 4000.0, 4000.0))
                add(f"{stain}_{sample}_W{k}L{lv}", stain, sample, lv, sp, v, c_um)
    np.savez_compressed(out + ".npz", **arrs)
    json.dump(meta, open(out + ".json", "w"), indent=1)
print("saved", len(arrs))

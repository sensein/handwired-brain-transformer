"""Stream real crops from 6 slabs and run tissue (L4) and layers (L5); save montages + stats."""
import json, os, sys, time
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from hwseg.data import StreamedZarr, list_zarrs, s3_url, to_logI
from hwseg.model import WiredTransformer
from hwseg.programs import layers, tissue

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
by = {}
for aid, p in list_zarrs():
    st = p.split("stain-")[1].split("_")[0]
    if st in ("YO", "NN", "LEC"): by.setdefault(st, []).append((aid, p))
rng = np.random.default_rng(1)
TIS, LAY = WiredTransformer(tissue(), torch.float32), WiredTransformer(layers(), torch.float32)
rows = []
for st in ("YO", "NN", "LEC"):
    for aid, p in [by[st][i] for i in rng.choice(len(by[st]), 2, replace=False)]:
        z = StreamedZarr(s3_url(aid)); Ls = z.levels; top = len(Ls) - 1
        name = p.split("/")[-1].split("_SPIM")[0]; t0 = time.time()
        a = z.crop(top, (0, Ls[top].shape[2]), (0, Ls[top].shape[3]), (0, Ls[top].shape[4])).astype(np.float32)
        idx = np.unravel_index(np.argmax(uniform_filter(a, 7)), a.shape)
        # layers at level 5: whole z,y, x window of 160 voxels around the tissue peak
        l5 = Ls[5]; x5 = int(np.clip(idx[2] * 2 ** (top - 5), 80, l5.shape[4] - 80))
        v5 = to_logI(z.crop(5, (0, l5.shape[2]), (0, l5.shape[3]), (x5 - 80, x5 + 80)))
        ly = LAY.forward(v5, l5.spacing_um)
        names = ["bg", "L1", "L2", "L3", "L4", "L5", "L6", "WM"]
        pred = torch.stack([ly[n] for n in names]).argmax(0).numpy()
        ts = TIS.forward(v5, l5.spacing_um)
        tc = torch.stack([ts["bg"], ts["gray"], ts["white"]]).argmax(0).numpy()
        zc = v5.shape[0] // 2
        res = dict(name=name, t=round(time.time() - t0, 1), layer_frac=[round(float((pred == k).mean()), 3) for k in range(8)],
                   tissue_frac=[round(float((tc == k).mean()), 3) for k in range(3)])
        rows.append(res); print(json.dumps(res), flush=True)
        f, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].imshow(v5[zc], cmap="gray"); ax[0].set_title(f"{st} L5 logI")
        ax[1].imshow(tc[zc], vmin=0, vmax=2); ax[1].set_title("tissue v2 (bg/gray/white)")
        ax[2].imshow(pred[zc], vmin=0, vmax=7, cmap="nipy_spectral"); ax[2].set_title("layers (0=bg,1-6=I-VI,7=WM)")
        for x in ax: x.axis("off")
        f.savefig(f"{out}/{name}_layers.png", dpi=70, bbox_inches="tight"); plt.close(f)
json.dump(rows, open(f"{out}/summary2.json", "w"), indent=1)

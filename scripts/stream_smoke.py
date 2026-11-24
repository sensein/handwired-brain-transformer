"""Stream small crops from DANDI 000108 (no bulk download) and run each program at its natural level."""
import json, sys, time
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hwseg.data import StreamedZarr, list_zarrs, s3_url, to_logI
from hwseg.model import WiredTransformer
from hwseg.programs import cells, tissue, vessels

out = sys.argv[1]
import os; os.makedirs(out, exist_ok=True)
zs = list_zarrs()
by = {}
for aid, p in zs:
    st = p.split("stain-")[1].split("_")[0]
    if st in ("YO", "NN", "LEC"):
        by.setdefault(st, []).append((aid, p))
rng = np.random.default_rng(0)
rows = []
for st in ("YO", "NN", "LEC"):
    for aid, p in [by[st][i] for i in rng.choice(len(by[st]), 2, replace=False)]:
        t0 = time.time()
        z = StreamedZarr(s3_url(aid))
        L = z.levels
        name = p.split("/")[-1].split("_SPIM")[0]
        res = {"asset": aid, "name": name, "levels": [(l.shape[2:], l.spacing_um) for l in L[:6]]}
        # coarse tissue crop at level 4 (centered), fine crop at level 0 (centered)
        lv = min(4, len(L) - 1)
        # locate tissue: stream the coarsest level (few MB) and take the brightest smoothed location
        from scipy.ndimage import uniform_filter
        top = len(L) - 1
        a_top = z.crop(top, (0, L[top].shape[2]), (0, L[top].shape[3]), (0, L[top].shape[4])).astype(np.float32)
        idx = np.unravel_index(np.argmax(uniform_filter(a_top, 7)), a_top.shape)
        f4, f0 = 2 ** (top - lv), 2 ** top
        clamp = lambda c, n, lvl: [int(min(max(ci, n // 2), L[lvl].shape[2 + i] - n // 2)) for i, ci in enumerate(c)]
        c = clamp([i * f4 for i in idx], 96, lv)
        res["top_idx"] = [int(i) for i in idx]; res["top_mean"] = float(a_top.mean())
        crop = lambda lvl, n, c: z.crop(lvl, *[(max(0, ci - n // 2), max(0, ci - n // 2) + n) for ci in c])
        a4 = crop(lv, 96, c)
        t = WiredTransformer(tissue(), torch.float32).forward(to_logI(a4), L[lv].spacing_um)
        cls = torch.stack([t["bg"], t["gray"], t["white"]]).argmax(0).numpy()
        sh0 = L[0].shape[2:]
        c0 = clamp([i * f0 for i in idx], 64, 0)
        a0 = crop(0, 64, c0)
        tl = to_logI(a0)
        ce = WiredTransformer(cells(), torch.float32).forward(torch.as_tensor(tl), L[0].spacing_um)
        ve = WiredTransformer(vessels(), torch.float32).forward(torch.as_tensor(tl), L[0].spacing_um)
        res.update(t=round(time.time() - t0, 1), raw4_mean=float(a4.mean()), raw0_mean=float(a0.mean()),
                   tissue_frac=[float((cls == k).mean()) for k in range(3)],
                   cell_frac=float((ce["seg"] > 0.5).float().mean()), vessel_frac=float((ve["seg"] > 0.5).float().mean()))
        rows.append(res); print(json.dumps(res), flush=True)
        f, ax = plt.subplots(1, 5, figsize=(20, 4))
        ax[0].imshow(a4[48], cmap="gray"); ax[0].set_title(f"{st} L{lv} raw")
        ax[1].imshow(cls[48], vmin=0, vmax=2, cmap="viridis"); ax[1].set_title("tissue bg/gray/white")
        ax[2].imshow(tl[32], cmap="gray"); ax[2].set_title("L0 logI")
        ax[3].imshow(ce["seg"][32], cmap="magma"); ax[3].set_title("cells")
        ax[4].imshow(ve["seg"][32], cmap="magma"); ax[4].set_title("vessels")
        for x in ax: x.axis("off")
        f.savefig(f"{out}/{name}.png", dpi=70, bbox_inches="tight"); plt.close(f)
json.dump(rows, open(f"{out}/summary.json", "w"), indent=1)

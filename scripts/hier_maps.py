"""Show named stage-0 output channels of a hier1 program on the centre plane of bank windows (diagnostic only).
usage: hier_maps.py PROGRAM OUT.png CH1,CH2,... KEY ..."""
import os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts")
from hier_look import load, program
from hwseg.hier_model import HierTransformer

prog = program(sys.argv[1])
chans = sys.argv[3].split(",")
keys = sys.argv[4:]
M = HierTransformer(prog, torch.float32)
fig, ax = plt.subplots(len(keys), len(chans) + 1, figsize=(4 * (len(chans) + 1), 3.6 * len(keys)), squeeze=False)
for r, k in enumerate(keys):
    v, sp, _ = load(k)
    o = M.forward(v, sp)
    zc = int(os.environ.get("ZC", v.shape[0] // 2))
    lo, hi = np.quantile(v[zc], [0.02, 0.997])
    asp = sp[1] / sp[2]
    ax[r, 0].imshow(np.clip((v[zc] - lo) / (hi - lo + 1e-6), 0, 1), cmap="gray", aspect=asp)
    ax[r, 0].set_title(f"{k} {sp[2]:.0f}um", fontsize=8)
    for c, n in enumerate(chans):
        a = o[n][zc].numpy()
        im = ax[r, c + 1].imshow(a, cmap="viridis", aspect=asp)
        ax[r, c + 1].set_title(f"{n} [{a.min():.3f}, {a.max():.3f}]", fontsize=8)
    for a in ax[r]:
        a.axis("off")
    print(k, {n: round(float(t), 3) for n, t in o.items() if t.dim() == 0}, flush=True)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=int(os.environ.get("DPI", 60)))

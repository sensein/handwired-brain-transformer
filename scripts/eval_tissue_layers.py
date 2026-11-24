"""Montages of tissue (bg/gray/white) and layers on real crops fed the display image (for visual calibration)."""
import json, sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hwseg.exposure import display
from hwseg.model import WiredTransformer
from hwseg.programs import layers, tissue

out = sys.argv[1]; kb = float(sys.argv[2]); kg = float(sys.argv[3]); lv = int(sys.argv[4]) if len(sys.argv) > 4 else 4
bank = np.load("/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/bank.npz"); meta = [m for m in json.load(open("/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/bank.json")) if m["level"] == lv]
T = WiredTransformer(tissue(kappa_bg=kb, kappa_gm=kg), torch.float32); Ly = WiredTransformer(layers(kappa_bg=kb, kappa_gm=kg), torch.float32)
def slab(prog, u, sp):
    """Crop z to +-(sum of head z-radii in voxels) around the centre plane, as the viewer does."""
    hz = sum(max([int(np.ceil(h.radius_um[0] / sp[0])) for h in L.heads if h.radius_um is not None] or [0]) for L in prog.layers)
    zc = u.shape[0] // 2
    return u[max(0, zc - hz): zc + hz + 1], min(zc, hz)


fig, ax = plt.subplots(len(meta), 3, figsize=(10, 3 * len(meta)), squeeze=False)
for i, m in enumerate(meta):
    v = bank[m["key"]].astype(np.float32); u = display(v); sp = tuple(m["spacing_um"]); z = u.shape[0] // 2
    us, zs = slab(T.p, u, sp); t = T.forward(torch.as_tensor(us), sp); tc = torch.stack([t["bg"], t["gray"], t["white"]]).argmax(0).numpy()[zs]
    us2, zs2 = slab(Ly.p, u, sp); ly = Ly.forward(torch.as_tensor(us2), sp); names = ["bg", "L1", "L2", "L3", "L4", "L5", "L6", "WM"]; lc = torch.stack([ly[n] for n in names]).argmax(0).numpy()[zs2]
    ax[i, 0].imshow(u[z], cmap="gray"); ax[i, 0].set_title(m["key"], fontsize=8)
    ax[i, 1].imshow(tc, vmin=0, vmax=2); ax[i, 1].set_title("tissue bg/gray/white  " + " ".join(f"{100*(tc==k).mean():.0f}" for k in range(3)), fontsize=8)
    ax[i, 2].imshow(lc, vmin=0, vmax=7, cmap="nipy_spectral"); ax[i, 2].set_title("layers  " + " ".join(f"{100*(lc==k).mean():.0f}" for k in range(8)), fontsize=8)
    for x in ax[i]: x.axis("off")
fig.tight_layout(); fig.savefig(out, dpi=55)

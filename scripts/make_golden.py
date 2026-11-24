"""Golden outputs from the Python (dense attention) model for cross-checking the JS/WebGPU engines."""
import json, sys
import numpy as np, torch
from hwseg.model import WiredTransformer
from hwseg.phantom import make_phantom
from hwseg.programs import cells, layers, tissue, vessels

out = {}
vol, c, v, sp = make_phantom(shape=(14, 20, 20), seed=7, n_cells=5, n_vessels=2, fg=800.0, cell_r_um=4.0)
L = np.log(vol + 1).astype(np.float32)
for name, prog in (("cells", cells()), ("vessels_13dir", vessels(n_dirs=13))):
    r = WiredTransformer(prog, torch.float32, fast=False).forward(torch.as_tensor(L), sp)
    out[name] = dict(dims=list(L.shape), spacing=list(sp), logI=L.ravel().tolist(),
                     out={k: t.numpy().ravel().tolist() for k, t in r.items()})
rng = np.random.default_rng(1)
from scipy.ndimage import gaussian_filter
T = (5.0 + 0.1 * gaussian_filter(rng.normal(size=(12, 16, 16)), 2) * 4).astype(np.float32)
T[:, :, :5] -= 2.5                                  # dark (background-like) strip
T[:, :, 9:] += 0.9 * rng.normal(size=T[:, :, 9:].shape).astype(np.float32)   # high-texture (gray-like) region
sp2 = (41.0, 58.0, 41.0)
r = WiredTransformer(tissue(), torch.float32, fast=False).forward(torch.as_tensor(T), sp2)
out["tissue"] = dict(dims=list(T.shape), spacing=list(sp2), logI=T.ravel().tolist(),
                     out={k: t.numpy().ravel().tolist() for k, t in r.items()})
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"))
print({k: len(v["logI"]) for k, v in out.items()})

from hwseg.phantom import make_cortex_phantom
cv, lab, _ = make_cortex_phantom(shape=(8, 22, 16), spacing=(164.0, 232.0, 164.0), pia_y=3.0, cortex_vox=13.0, undulate=1.0, seed=3)
Lg = np.log(cv + 1).astype(np.float32)
sp3 = (164.0, 232.0, 164.0)
r = WiredTransformer(layers(), torch.float32, fast=False).forward(torch.as_tensor(Lg), sp3)
out["layers"] = dict(dims=list(Lg.shape), spacing=list(sp3), logI=Lg.ravel().tolist(),
                     out={k: t.numpy().ravel().tolist() for k, t in r.items()})
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"))
print({k: len(v["logI"]) for k, v in out.items()})

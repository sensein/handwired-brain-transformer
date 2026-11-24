"""Calibrate kappa/tau0 of the contrast-adaptive cells/vessels thresholds on phantoms, and check invariance
to contrast / gamma / brightness of the input (same adjustment as the demo)."""
import itertools, numpy as np, torch
from hwseg.model import WiredTransformer
from hwseg.phantom import make_phantom
from hwseg.programs import cells, vessels


from hwseg.exposure import display


def adj(v, b, c, g): return display(v, brightness=b, contrast=c, gamma=g)


def f1(p, g): return 2 * (p & g).sum() / max(p.sum() + g.sum(), 1)

VAR = {"base": (0, 1, 1), "contrast0.5": (0, .5, 1), "contrast2": (0, 2, 1), "gamma0.5": (0, 1, .5), "gamma2": (0, 1, 2), "bright+.15": (.15, 1, 1), "bright-.3": (-.3, 1, 1), "bright+.4": (.4, 1, 1), "c0.15": (0, .15, 1)}
for kind, mk, kw in [("vessels", lambda **a: vessels(n_dirs=13, **a), dict(n_cells=20, n_vessels=3, fg=800.0, cell_r_um=4.0))]:
    for kappa, tau0 in itertools.product((0.0, 0.5, 1.0), (0.04, 0.06, 0.08, 0.12)):
        m = WiredTransformer(mk(kappa=kappa, tau0=tau0), torch.float32); res = {k: [] for k in VAR}
        for seed in range(3):
            vol, c, v, sp = make_phantom(seed=seed, **kw); gt = c if kind == "cells" else v
            L = np.log(vol + 1).astype(np.float32)
            for vn, (b, cc, g) in VAR.items():
                res[vn].append(f1((m.forward(adj(L, b, cc, g), sp)["seg"] > 0.5).numpy(), gt))
        print(kind, f"kappa={kappa} tau0={tau0}", " ".join(f"{k}:{np.mean(v):.2f}" for k, v in res.items()), flush=True)

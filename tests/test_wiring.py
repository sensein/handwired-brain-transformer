import numpy as np
import torch
from scipy import ndimage

from hwseg.model import WiredTransformer
from hwseg.phantom import make_phantom
from hwseg.programs import cells, vessels
from hwseg.wiring import Builder, cov_from_axes


def test_gaussian_attention_equals_explicit_kernel():
    """A programmed attention head == normalized truncated Gaussian correlation (exact up to fp)."""
    rng = np.random.default_rng(1)
    vol = rng.normal(size=(20, 24, 24))
    sp = (2.0, 3.0, 2.0)
    P = cov_from_axes((5.0, 7.0, 4.0))
    B = Builder("t")
    o = B.alloc()
    B.add_layer([B.gaussian_head(B.d, 0, o, P)])
    B.out = {"o": o}
    prog = B.build()
    out = WiredTransformer(prog).forward(torch.as_tensor(vol), sp)["o"].numpy()
    h = prog.layers[0].heads[0]
    r = [max(1, int(np.ceil(rr / s))) for rr, s in zip(h.radius_um, sp)]
    g = np.meshgrid(*[np.arange(-ri, ri + 1) * s for ri, s in zip(r, sp)], indexing="ij")
    k = np.exp(-0.5 * sum(P[a, a] * g[a] ** 2 for a in range(3)))
    k /= k.sum()
    ref = ndimage.correlate(vol, k, mode="constant")  # interior only
    m = tuple(slice(ri, -ri) for ri in r)
    assert np.allclose(out[m], ref[m], atol=1e-9)


def _f1(pred, gt):
    tp = (pred & gt).sum()
    return 2 * tp / max(pred.sum() + gt.sum(), 1)


def test_cells_program_recovers_planted_cells():
    vol, c, v, sp = make_phantom(seed=3, n_vessels=0)
    out = WiredTransformer(cells(), dtype=torch.float32).forward(torch.log(torch.as_tensor(vol) + 1), sp)
    assert _f1(out["seg"].numpy() > 0.5, c) > 0.5


def test_vessels_program_prefers_tubes():
    vol, c, v, sp = make_phantom(seed=4, n_cells=20, n_vessels=3, fg=800.0, cell_r_um=4.0)
    out = WiredTransformer(vessels(), dtype=torch.float32).forward(torch.log(torch.as_tensor(vol) + 1), sp)
    assert _f1(out["seg"].numpy() > 0.5, v) > 0.5

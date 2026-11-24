import numpy as np
import torch
from scipy import ndimage

from hwseg.exposure import display
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
    out = WiredTransformer(cells(), dtype=torch.float32).forward(torch.as_tensor(display(np.log(vol + 1))), sp)
    assert _f1(out["seg"].numpy() > 0.5, c) > 0.5


def test_vessels_program_prefers_tubes():
    vol, c, v, sp = make_phantom(seed=4, n_cells=20, n_vessels=3, fg=800.0, cell_r_um=4.0)
    out = WiredTransformer(vessels(), dtype=torch.float32).forward(torch.as_tensor(display(np.log(vol + 1))), sp)
    assert _f1(out["seg"].numpy() > 0.5, v) > 0.5


def test_fast_path_equals_dense_attention_for_programs():
    vol, c, v, sp = make_phantom(shape=(20, 24, 24), seed=5, n_cells=6, n_vessels=2, fg=800.0)
    L = torch.log(torch.as_tensor(vol) + 1)
    for prog in (cells(), vessels(n_dirs=6)):
        a = WiredTransformer(prog, torch.float64, fast=False).forward(L, sp)["score"]
        b = WiredTransformer(prog, torch.float64, fast=True).forward(L, sp)["score"]
        assert torch.allclose(a, b, atol=1e-7), (prog.name, (a - b).abs().max().item())


def test_layers_program_orders_layers_by_depth():
    """Coarse layers: depth-ordered, within one layer for most cortex voxels (exact-layer accuracy is lower)."""
    from hwseg.phantom import make_cortex_phantom
    from hwseg.programs import layers
    vol, lab, sp = make_cortex_phantom(seed=2)
    out = WiredTransformer(layers(), dtype=torch.float32, fast=True).forward(torch.as_tensor(display(np.log(vol + 1))), sp)
    names = ["bg", "L1", "L2", "L3", "L4", "L5", "L6", "WM"]
    pred = np.stack([out[n].numpy() for n in names]).argmax(0)[lab.shape[0] // 2]
    t = lab[lab.shape[0] // 2]
    cortex = (t >= 1) & (t <= 6)
    within1 = (np.abs(pred - t) <= 1)[cortex].mean()
    bgwm = (pred == t)[(t == 0) | (t == 7)].mean()
    means = [pred[t == k].mean() for k in range(1, 7)]
    assert within1 > 0.85 and bgwm > 0.95 and all(a < b for a, b in zip(means, means[1:])), (within1, bgwm, means)


def test_scale_equivariance_of_scaled_programs():
    """program(scale=s) on spacing*s == program(scale=1) on spacing (attention is analytic in physical units)."""
    vol, c, v, sp = make_phantom(shape=(16, 20, 20), seed=9, n_cells=6, n_vessels=2, fg=800.0, cell_r_um=4.0)
    L = torch.log(torch.as_tensor(vol) + 1)
    for mk, name in ((cells, "cells"), (lambda scale: vessels(n_dirs=6, scale=scale), "vessels")):
        ref = WiredTransformer(mk(scale=1), torch.float64).forward(L, sp)["score"]
        for s in (2, 4):
            out = WiredTransformer(mk(scale=s), torch.float64).forward(L, tuple(x * s for x in sp))["score"]
            assert torch.allclose(ref, out, atol=1e-7), (name, s, (ref - out).abs().max().item())


def test_browser_python_modules_are_in_sync_and_torch_free():
    """demo/py/hwseg (what Pyodide loads) must equal src/hwseg for the torch-free modules, and import no torch."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    for f in ("program", "wiring", "programs", "exposure"):
        a = (root / "src/hwseg" / f"{f}.py").read_text()
        assert a == (root / "demo/py/hwseg" / f"{f}.py").read_text(), f"run scripts/sync_browser_py.sh ({f}.py differs)"
        assert "import torch" not in a and "scipy" not in a, f"{f}.py must stay browser-importable (numpy only)"

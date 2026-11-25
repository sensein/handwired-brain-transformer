"""Tests for the hierarchical (hier1) format and executor: properties that hold by construction."""
import json

import numpy as np
import torch

from hwseg.hier import HB, HierProgram, attn_matrix, deriv_kernel, stage_shapes
from hwseg.hier_model import HierTransformer
from hwseg.hier_programs import cells, front_end


def test_separable_attention_equals_dense_window_softmax():
    """The separable evaluation of a cross-stage Gaussian attention op equals the dense masked softmax over the 3-D box."""
    rng = np.random.default_rng(0)
    n0, sp0 = (7, 9, 11), (2.0, 3.0, 2.5)
    v = rng.normal(size=n0)
    B = HB("t", 2)
    o = B.ch(1)
    c = B.ch(0)
    B.mlp(0, [({0: 1.0}, 0.0, {c: 1.0}), ({0: -1.0}, 0.0, {c: -1.0})])   # copy logI into a scratch channel
    B.attn(0, 1, c, o, (3.0, 4.0, 5.0))
    B.out = {"o": [1, o]}
    out = HierTransformer(B.build()).forward(v, sp0)["o"].numpy()
    sh = stage_shapes(n0, 2)
    pos = lambda n, s, f: (np.arange(n) * f + (f - 1) / 2 - (s - 1) / 2)   # noqa: E731
    P0 = np.stack(np.meshgrid(*[pos(n0[a], n0[a], 1) * sp0[a] for a in range(3)], indexing="ij"), -1)
    P1 = np.stack(np.meshgrid(*[pos(sh[1][a], n0[a], 2) * sp0[a] for a in range(3)], indexing="ij"), -1)
    sig, lim = np.array([3.0, 4.0, 5.0]), np.maximum(3 * np.array([3.0, 4.0, 5.0]), np.array(sp0))
    ref = np.zeros(sh[1])
    for idx in np.ndindex(sh[1]):
        D = P0 - P1[idx]
        inside = np.all(np.abs(D) <= lim + 1e-9, -1)
        s = -0.5 * ((D / sig) ** 2).sum(-1)
        w = np.where(inside, np.exp(s - s[inside].max()), 0)
        ref[idx] = (w * v).sum() / w.sum()
    assert np.allclose(out, ref, atol=1e-10)


def test_rank_front_end_is_invariant_to_monotone_maps():
    """u = rank(logI) is unchanged by any strictly increasing intensity map (here gain, gamma and an offset)."""
    rng = np.random.default_rng(1)
    I = np.exp(rng.normal(6, 1, size=(6, 20, 24)))
    B = HB("r", 1)
    fe = front_end(B)
    B.out = {"u": [0, fe["u"]]}
    M = HierTransformer(B.build())
    u0 = M.forward(np.log(I + 1), (3, 3, 3))["u"].numpy()
    for f in (lambda x: 4 * x, lambda x: x ** 0.5 * 30, lambda x: x + 500.0, lambda x: np.log1p(x) ** 3):
        u1 = M.forward(np.log(f(I) + 1), (3, 3, 3))["u"].numpy()
        assert np.array_equal(u0, u1)


def test_derivative_kernels_are_scale_normalised():
    for h, sg in ((1.0, 2.0), (2.5, 4.0), (10.0, 30.0)):
        t = np.arange(-(len(deriv_kernel(h, sg, 0)) // 2), len(deriv_kernel(h, sg, 0)) // 2 + 1) * h
        k0, k1, k2 = (deriv_kernel(h, sg, o) for o in (0, 1, 2))
        assert abs(k0.sum() - 1) < 1e-12 and abs((k1 * t).sum() - sg) < 1e-9
        assert abs(k2.sum()) < 1e-12 and abs((k2 * t ** 2 / 2).sum() - sg ** 2) < 1e-6


def test_attn_rows_normalised_and_programs_roundtrip():
    W = attn_matrix(37, 2.5, 4, 1, 10, 37, 6.0)
    assert np.allclose(W.sum(1), 1)
    p = cells()
    q = HierProgram.from_json(json.loads(p.to_json()))
    assert q.d == p.d and len(q.ops) == len(p.ops)
    v = np.log(np.random.default_rng(2).gamma(2.0, 300.0, size=(6, 24, 24)) + 1)
    a = HierTransformer(p).forward(v, (2.5, 3.6, 2.5))["score"]
    b = HierTransformer(q).forward(v, (2.5, 3.6, 2.5))["score"]
    assert torch.allclose(a, b)


def test_unreadable_views_output_nothing():
    """Flat, dark and saturated fields of view switch every output off (front-end gates)."""
    rng = np.random.default_rng(3)
    p = cells()
    M = HierTransformer(p)
    base = rng.gamma(2.0, 400.0, size=(6, 32, 32))
    flat = np.full_like(base, 800.0) + rng.integers(0, 2, base.shape)
    dark = np.round(base * 0.002)
    sat = np.minimum(base, np.quantile(base, 0.3))
    for v in (flat, dark, sat):
        o = M.forward(np.log(v + 1), (2.5, 3.6, 2.5))
        assert float(o["ok"]) < 0.5 and float(o["seg"].max()) == 0.0

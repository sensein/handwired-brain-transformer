"""Hand-wired programs. Every constant is a declared, physical-unit design choice (um, log-ratio)."""
from __future__ import annotations

import numpy as np

from .wiring import Builder, cov_from_axes

CH = CH_LOGI = 0


def _step(x_w, b, gain=1.0, k=1.0):
    """Soft step via two ReLUs: clamp(k*(x-b),0,1). Returns two unit specs (w,b,out) to be given out chs."""
    return x_w, b, k


def cells(sigma_um=3.5, ratio=1.6, tau=0.12, sharp=8.0) -> "Program":  # noqa: F821
    """Bright-blob (nuclei / NeuN soma) detector: DoG on log intensity => illumination-invariant log-ratio."""
    B = Builder("cells")
    g1, g2, dog, seg = B.alloc(), B.alloc(), B.alloc(), B.alloc()
    h1 = B.gaussian_head(B.d, 0, g1, cov_from_axes(sigma_um))
    h2 = B.gaussian_head(B.d, 0, g2, cov_from_axes(sigma_um * ratio))
    B.add_layer([h1, h2], B.mlp(B.d, [({g1: 1.0, g2: -1.0}, 0.0, {dog: 1.0})]))
    # seg = clamp(sharp*(dog - tau), 0, 1) as relu(a) - relu(a-1)
    B.add_layer([], B.mlp(B.d, [({dog: sharp}, -sharp * tau, {seg: 1.0}),
                                ({dog: sharp}, -sharp * tau - 1.0, {seg: -1.0})]))
    B.out = {"score": dog, "seg": seg}
    B.meta = dict(kind="cells", min_um=2.0, max_um=12.0)
    return B.build()


def fib_dirs(n):
    """n quasi-uniform unit directions on the hemisphere (z,y,x), physical axes."""
    i = np.arange(n) + 0.5
    z = 1 - i / n
    r = np.sqrt(1 - z * z)
    ph = np.pi * (1 + 5**0.5) * i
    return np.stack([z, r * np.cos(ph), r * np.sin(ph)], 1)


def vessels(sigma_t=2.0, sigma_l=8.0, sigma_w=6.0, tau=0.2, n_dirs=26, sharp=6.0):
    """Tubular-structure detector (no blob-rejection term: the design sweep showed it hurts).
    score = max_d G_elong,d(logI) - G_wide(logI) - tau; max via a relu tournament."""
    B = Builder("vessels")
    gw = B.alloc()
    heads = [B.gaussian_head(B.d, 0, gw, cov_from_axes(sigma_w))]
    dch = []
    for dvec in fib_dirs(n_dirs):
        c = B.alloc()
        dch.append(c)
        heads.append(B.gaussian_head(B.d, 0, c, cov_from_axes((sigma_l, sigma_t), dvec)))
    B.add_layer(heads)
    cur = list(dch)
    while len(cur) > 1:  # max(a,b) = b + relu(a-b)
        units, nxt = [], []
        for i in range(0, len(cur) - 1, 2):
            a, b = cur[i], cur[i + 1]
            units.append(({a: 1.0, b: -1.0}, 0.0, {b: 1.0}))
            nxt.append(b)
        if len(cur) % 2:
            nxt.append(cur[-1])
        B.add_layer([], B.mlp(B.d, units))
        cur = nxt
    D = cur[0]
    score, seg = B.alloc(), B.alloc()
    B.add_layer([], B.mlp(B.d, [({D: 1.0, gw: -1.0}, -tau, {score: 1.0})]))
    B.add_layer([], B.mlp(B.d, [({score: sharp}, 0.0, {seg: 1.0}), ({score: sharp}, -1.0, {seg: -1.0})]))
    B.out = {"score": score, "seg": seg}
    B.meta = dict(kind="vessels", min_um=2.0, max_um=15.0)
    return B.build()


def tissue(sigma_fine_um=60.0, sigma_tex_um=150.0, k_bg=0.8, k_gm=0.05, sharp=6.0):
    """Coarse tissue program: background / white / gray from smoothed log-intensity and local texture
    energy (|logI - G*logI| smoothed). Thresholds are relative to tile-global statistics via
    uniform-attention (global) heads, i.e. in-context calibration with no stored per-slab constants.
    Class scores (argmax outside the net): bg, gray, white (constant 0.5 logit)."""
    B = Builder("tissue")
    m1, gm, rp, rn, tex, gt = (B.alloc() for _ in range(6))
    bg, gray, white = B.alloc(), B.alloc(), B.alloc()
    B.add_layer([B.gaussian_head(B.d, 0, m1, cov_from_axes(sigma_fine_um)), B.global_head(B.d, 0, gm)],
                B.mlp(B.d, [({0: 1.0, m1: -1.0}, 0.0, {rp: 1.0}), ({0: -1.0, m1: 1.0}, 0.0, {rn: 1.0})]))
    H = B.gaussian_head(B.d, rp, tex, cov_from_axes(sigma_tex_um))
    H.Wv[rn] = 1.0  # value = rp + rn = |res|
    B.add_layer([H])
    B.add_layer([B.global_head(B.d, tex, gt)])
    ramp = lambda ch_w, b, o: [(ch_w, b, {o: 1.0}), (ch_w, b - 1.0, {o: -1.0})]  # noqa: E731
    units = (ramp({m1: -sharp, gm: sharp}, -sharp * k_bg, bg) +
             ramp({tex: sharp, gt: -sharp}, -sharp * k_gm, gray))
    # white = 0.5 constant: ONE channel carries it
    units.append(({4: 1.0}, 0.0, {white: 0.5}))
    B.add_layer([], B.mlp(B.d, units))
    B.out = {"bg": bg, "gray": gray, "white": white}
    B.meta = dict(kind="tissue", min_um=30.0, max_um=1e9)
    return B.build()

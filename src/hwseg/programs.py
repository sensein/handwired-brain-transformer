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


def _ramp(B, d, weights, bias, out, sharp):
    """clamp(sharp*(w.x+bias),0,1) written to channel `out` (two ReLU units)."""
    w = {c: sharp * v for c, v in weights.items()}
    return [(w, sharp * bias, {out: 1.0}), (w, sharp * bias - 1.0, {out: -1.0})]


def tissue(sigma_um=60.0, k_bg=0.8, k_gm=0.10, sharp=6.0):
    """Coarse tissue program: smoothed log-intensity (a nuclear-density proxy) relative to tile-global mean
    (uniform-attention head => in-context calibration). bg: much darker than the tile mean; gray: denser than the
    tile mean; white: constant 0.5 logit. Class scores: bg, gray, white (argmax outside the net)."""
    B = Builder("tissue")
    m, gm, bg, gray, white = (B.alloc() for _ in range(5))
    B.add_layer([B.gaussian_head(B.d, 0, m, cov_from_axes(sigma_um)), B.global_head(B.d, 0, gm)],
                B.mlp(B.d, _ramp(B, B.d, {gm: 1.0, m: -1.0}, -k_bg, bg, sharp) +
                      _ramp(B, B.d, {m: 1.0, gm: -1.0}, -k_gm, gray, sharp) + [({4: 1.0}, 0.0, {white: 0.5})]))
    B.out = {"bg": bg, "gray": gray, "white": white}
    B.meta = dict(kind="tissue", min_um=30.0, max_um=1e9)
    return B.build()


PIA_DEPTHS_UM = (300.0, 600.0, 1300.0)   # I/II, II/III, III/IV boundaries, distance below the pia (declared prior)
WM_DEPTHS_UM = (1300.0, 700.0)           # IV/V, V/VI boundaries, distance above the white matter (declared prior)


def layer_edges(sigmas, pia=PIA_DEPTHS_UM, wm=WM_DEPTHS_UM):
    """Edges of s = mean_k B_k - mean_k W_k at the declared depths, from the exact Gaussian response of a flat
    boundary (Gaussian attention of a step edge = Phi(-t/sigma)). Decreasing: pia side first."""
    from math import erf, sqrt
    phi = lambda t, sg: 0.5 * (1 - erf(t / sg / sqrt(2)))           # noqa: E731
    sv = lambda t: sum(phi(t, sg) for sg in sigmas) / len(sigmas)  # noqa: E731
    return tuple(sv(t) for t in pia) + tuple(-sv(t) for t in wm)


def layers(sigma_um=60.0, depth_sigmas=(150.0, 400.0, 900.0), k_bg=0.8, k_gm=0.10, sharp=6.0, band_sharp=14.0, edges=None):
    """Cortical-layer program (coarse, ~80 um voxels). Hand-wired pipeline, no training:
      1. density proxy m = G_sigma(logI); bg / gray indicators relative to the tile-global mean (global head);
         white-matter indicator W = low density and not near the pia (so sparse layer I is not read as white matter).
      2. pia and WM proximity: bg and WM indicators smoothed by Gaussian attention at several scales;
         s = mean_k B_k - mean_k W_k is a signed depth proxy (>0 near pia, <0 near white matter).
      3. layers I..VI = soft ReLU trapezoid bands of s whose edges are computed from declared depths in um
         (PIA_DEPTHS_UM / WM_DEPTHS_UM), gated by tissue-and-not-white-matter.
    Outputs: L1..L6, WM, bg. Anatomy priors are declared constants, not fitted."""
    edges = edges or layer_edges(depth_sigmas)
    B = Builder("layers")
    m, gm, bgi, gi, W, Bs, Weff = (B.alloc() for _ in range(7))
    B.add_layer([B.gaussian_head(B.d, 0, m, cov_from_axes(sigma_um)), B.global_head(B.d, 0, gm)],
                B.mlp(B.d, _ramp(B, B.d, {gm: 1.0, m: -1.0}, -k_bg, bgi, sharp) + _ramp(B, B.d, {m: 1.0, gm: -1.0}, -k_gm, gi, sharp)))
    B.add_layer([], B.mlp(B.d, [({4: 1.0, bgi: -1.0, gi: -1.0}, 0.0, {W: 1.0})]))   # W = relu(1 - bg - gray)
    B.add_layer([B.gaussian_head(B.d, bgi, Bs, cov_from_axes(depth_sigmas[0]), extent=2.5)])   # pia proximity
    B.add_layer([], B.mlp(B.d, [({W: 1.0, Bs: -2.0}, 0.0, {Weff: 1.0})]))                      # Weff = relu(W - 2 Bs)
    Bk = [B.alloc() for _ in depth_sigmas]; Wk = [B.alloc() for _ in depth_sigmas]
    heads = [B.gaussian_head(B.d, bgi, c, cov_from_axes(sg), extent=2.5) for c, sg in zip(Bk, depth_sigmas)]
    heads += [B.gaussian_head(B.d, Weff, c, cov_from_axes(sg), extent=2.5) for c, sg in zip(Wk, depth_sigmas)]
    B.add_layer(heads)
    n = len(depth_sigmas)
    sw = {**{c: 1.0 / n for c in Bk}, **{c: -1.0 / n for c in Wk}}         # s as a linear readout
    bands = [B.alloc() for _ in range(len(edges) + 1)]
    units = []
    for k, band in enumerate(bands):
        hi = edges[k - 1] if k > 0 else None            # band k covers s in [edges[k], edges[k-1]]
        lo = edges[k] if k < len(edges) else None
        if lo is None:      # bottom band: 1 - ramp(s - hi)
            units.append(({}, 1.0, {band: 1.0}))
            units += _ramp_band(sw, -hi, band, band_sharp, -1.0)
        elif hi is None:    # top band: ramp(s - lo)
            units += _ramp_band(sw, -lo, band, band_sharp, 1.0)
        else:
            units += _ramp_band(sw, -lo, band, band_sharp, 1.0) + _ramp_band(sw, -hi, band, band_sharp, -1.0)
    B.add_layer([], B.mlp(B.d, units))
    # gate: band <- min(band, 1 - bg - Weff) = band - relu(band + bg + Weff - 1)
    B.add_layer([], B.mlp(B.d, [({band: 1.0, bgi: 1.0, Weff: 1.0}, -1.0, {band: -1.0}) for band in bands]))
    B.out = {**{f"L{k + 1}": c for k, c in enumerate(bands)}, "WM": Wk[0], "bg": Bk[0]}
    B.meta = dict(kind="layers", min_um=30.0, max_um=1e9, tile=64, edges=[round(e, 4) for e in edges])
    return B.build()


def _ramp_band(sw, shift, out, sharp, sign):
    """sign * clamp(sharp*(s + shift), 0, 1) with s = sw.x."""
    w = {c: sharp * v for c, v in sw.items()}
    return [(w, sharp * shift, {out: sign}), (w, sharp * shift - 1.0, {out: -sign})]

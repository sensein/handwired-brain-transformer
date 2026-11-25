"""Hand-wired programs. Every constant is a declared, physical-unit design choice (um, log-ratio)."""
from __future__ import annotations

import numpy as np

from .wiring import Builder, cov_from_axes

CH = CH_LOGI = 0


def _step(x_w, b, gain=1.0, k=1.0):
    """Soft step via two ReLUs: clamp(k*(x-b),0,1). Returns two unit specs (w,b,out) to be given out chs."""
    return x_w, b, k


def _abs_mean_channel(B, a_ch, out_ch):
    """Channels sp=relu(a), sn=relu(-a) (MLP units) so a global head over sp+sn gives mean|a| (in-context scale)."""
    return [({a_ch: 1.0}, 0.0, {out_ch[0]: 1.0}), ({a_ch: -1.0}, 0.0, {out_ch[1]: 1.0})]


def cells(sigma_um=3.5, ratio=1.6, kappa=2.0, tau0=0.03, sharp=400.0, scale=1.0) -> "Program":  # noqa: F821
    """Bright-blob (nuclei / NeuN soma) detector: DoG on log intensity (illumination-invariant log-ratio).
    Input is the display image u' in [0,1]. seg = step(dog - kappa*mean|dog| - tau0): kappa*mean|dog| is the FOV's own
    response spread (global head), tau0 an absolute visibility floor in display units, so detection is contrast-robust
    but vanishes in darkness, saturation and flat (zero-contrast) views.
    `scale` multiplies every physical length: scale s targets blobs s times larger (nuclear clusters when zoomed out)."""
    sigma_um = sigma_um * scale
    B = Builder("cells")
    g1, g2, dog, sp_, sn_, gs, seg = (B.alloc() for _ in range(7))
    B.add_layer([B.gaussian_head(B.d, 0, g1, cov_from_axes(sigma_um)), B.gaussian_head(B.d, 0, g2, cov_from_axes(sigma_um * ratio))],
                B.mlp(B.d, [({g1: 1.0, g2: -1.0}, 0.0, {dog: 1.0}), ({g2: 1.0, g1: -1.0}, 0.0, {dog: -1.0})]))   # dog = relu(g1-g2) - relu(g2-g1)
    B.add_layer([], B.mlp(B.d, _abs_mean_channel(B, dog, (sp_, sn_))))
    H = B.global_head(B.d, sp_, gs); H.Wv[sn_] = 1.0                 # value = |dog|
    B.add_layer([H])
    B.add_layer([], B.mlp(B.d, [({dog: sharp, gs: -sharp * kappa}, -sharp * tau0, {seg: 1.0}),
                                ({dog: sharp, gs: -sharp * kappa}, -sharp * tau0 - 1.0, {seg: -1.0})]))
    B.out = {"score": dog, "seg": seg}
    B.meta = dict(kind="cells", min_um=2.0 * scale, max_um=12.0 * scale, scale=scale, sigma0=3.5)
    return B.build()


def fib_dirs(n):
    """n quasi-uniform unit directions on the hemisphere (z,y,x), physical axes."""
    i = np.arange(n) + 0.5
    z = 1 - i / n
    r = np.sqrt(1 - z * z)
    ph = np.pi * (1 + 5**0.5) * i
    return np.stack([z, r * np.cos(ph), r * np.sin(ph)], 1)


def vessels(sigma_t=2.0, sigma_l=8.0, sigma_w=6.0, kappa=1.0, tau0=0.12, n_dirs=26, sharp=400.0, scale=1.0):
    """Tubular-structure detector. r = max_d G_elong,d(logI) - G_wide(logI) (max via a relu tournament);
    seg = clamp(sharp*(r - kappa*mean|r| - tau0)) with mean|r| from a global head (contrast-invariant threshold).
    `scale` multiplies all lengths."""
    sigma_t, sigma_l, sigma_w = sigma_t * scale, sigma_l * scale, sigma_w * scale
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
            a_, b_ = cur[i], cur[i + 1]
            units.append(({a_: 1.0, b_: -1.0}, 0.0, {b_: 1.0}))
            nxt.append(b_)
        if len(cur) % 2:
            nxt.append(cur[-1])
        B.add_layer([], B.mlp(B.d, units))
        cur = nxt
    D = cur[0]
    rp, rn, gs, score, seg = (B.alloc() for _ in range(5))
    B.add_layer([], B.mlp(B.d, [({D: 1.0, gw: -1.0}, 0.0, {rp: 1.0}), ({gw: 1.0, D: -1.0}, 0.0, {rn: 1.0})]))   # r = rp - rn
    H = B.global_head(B.d, rp, gs); H.Wv[rn] = 1.0                    # value = |r|
    B.add_layer([H])
    B.add_layer([], B.mlp(B.d, [({rp: 1.0, gs: -kappa}, -tau0, {score: 1.0})]))
    B.add_layer([], B.mlp(B.d, [({score: sharp}, 0.0, {seg: 1.0}), ({score: sharp}, -1.0, {seg: -1.0})]))
    B.out = {"score": score, "seg": seg}
    B.meta = dict(kind="vessels", min_um=2.0 * scale, max_um=15.0 * scale, scale=scale, sigma0=2.0)
    return B.build()


def _ramp(B, d, weights, bias, out, sharp):
    """clamp(sharp*(w.x+bias),0,1) written to channel `out` (two ReLU units)."""
    w = {c: sharp * v for c, v in weights.items()}
    return [(w, sharp * bias, {out: 1.0}), (w, sharp * bias - 1.0, {out: -1.0})]


def _indicators(B, m, gm, kappa_bg, kappa_gm, tau_dark, tau_g, sharp):
    """bg / gray indicators from the density proxy m (smoothed display image) vs the FOV mean gm, in display units:
    bg = dark (m < tau_dark, absolute) OR darker than the FOV mean by kappa_bg * mean|m-gm| (relative);
    gray = denser than the FOV mean by kappa_gm * mean|m-gm| + tau_g. Hard-ish steps (steep ramps)."""
    sp_, sn_, gs, bg, gray, bgr = (B.alloc() for _ in range(6))
    B.add_layer([], B.mlp(B.d, [({m: 1.0, gm: -1.0}, 0.0, {sp_: 1.0}), ({gm: 1.0, m: -1.0}, 0.0, {sn_: 1.0})]))
    H = B.global_head(B.d, sp_, gs); H.Wv[sn_] = 1.0                 # mean |m - gm|
    B.add_layer([H], B.mlp(B.d, _ramp(B, B.d, {m: -1.0}, tau_dark, bg, sharp) +                              # absolute dark floor -> bg
                           _ramp(B, B.d, {gm: 1.0, m: -1.0, gs: -kappa_bg}, 0.0, bgr, sharp) +             # relative darkness -> bgr
                           _ramp(B, B.d, {m: 1.0, gm: -1.0, gs: -kappa_gm}, -tau_g, gray, sharp)))         # relative density -> gray
    B.add_layer([], B.mlp(B.d, [({bgr: 1.0, bg: -1.0}, 0.0, {bg: 1.0})]))                                    # bg = max(bg_abs, bg_rel)
    return bg, gray


def tissue(sigma_um=60.0, kappa_bg=1.5, kappa_gm=0.3, tau_dark=0.08, tau_g=0.01, sharp=200.0, scale=1.0):
    """Coarse tissue program on the display image u' in [0,1] (density proxy = G_sigma(u')). Thresholds are relative to
    the FOV's own spread (global heads) with absolute floors (darkness => background; flat view => no gray).
    Class scores: bg, gray, white (constant 0.5 logit; argmax outside the net)."""
    sigma_um = sigma_um * scale
    B = Builder("tissue")
    m, gm, white = (B.alloc() for _ in range(3))
    B.add_layer([B.gaussian_head(B.d, 0, m, cov_from_axes(sigma_um)), B.global_head(B.d, 0, gm)])
    bg, gray = _indicators(B, m, gm, kappa_bg, kappa_gm, tau_dark, tau_g, sharp)
    B.add_layer([], B.mlp(B.d, [({4: 1.0}, 0.0, {white: 0.5})]))
    B.out = {"bg": bg, "gray": gray, "white": white}
    B.meta = dict(kind="tissue", min_um=30.0, max_um=1e9, scale=scale, sigma0=60.0, input="display")
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


def layers(sigma_um=60.0, depth_sigmas=(150.0, 400.0, 900.0), kappa_bg=1.5, kappa_gm=0.3, tau_dark=0.08, tau_g=0.01, sharp=200.0, band_sharp=14.0, edges=None):
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
    m, gm, W, Bs, Weff = (B.alloc() for _ in range(5))
    B.add_layer([B.gaussian_head(B.d, 0, m, cov_from_axes(sigma_um)), B.global_head(B.d, 0, gm)])
    bgi, gi = _indicators(B, m, gm, kappa_bg, kappa_gm, tau_dark, tau_g, sharp)
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
    B.meta = dict(kind="layers", min_um=30.0, max_um=1e9, tile=64, edges=[round(e, 4) for e in edges], input="display")
    return B.build()


def _ramp_band(sw, shift, out, sharp, sign):
    """sign * clamp(sharp*(s + shift), 0, 1) with s = sw.x."""
    w = {c: sharp * v for c, v in sw.items()}
    return [(w, sharp * shift, {out: sign}), (w, sharp * shift - 1.0, {out: -sign})]


def exposure(a=4.75, b=6.82):
    """Default-exposure program. Input ch0 = raw log(I+1) of the FOV (not the display image). Global heads give the FOV
    mean gm and the mean negative / positive deviations n, p; outputs lo = gm - a*n and hi = gm + b*p (log-domain display
    window, ~2nd..99.7th percentile on real crops; a, b are medians over a real-crop bank, scripts/fit_exposure.py) and
    mean = gm. A viewer derives gamma from (mean-lo)/(hi-lo)."""
    B = Builder("exposure")
    gm, sp_, sn_, gp, gn, lo, hi = (B.alloc() for _ in range(7))
    B.add_layer([B.global_head(B.d, 0, gm)], B.mlp(B.d, [({0: 1.0, gm: -1.0}, 0.0, {sp_: 1.0}), ({gm: 1.0, 0: -1.0}, 0.0, {sn_: 1.0})]))
    B.add_layer([B.global_head(B.d, sp_, gp), B.global_head(B.d, sn_, gn)],
                B.mlp(B.d, [({gm: 1.0, gn: -a}, 0.0, {lo: 1.0}), ({gm: 1.0, gp: b}, 0.0, {hi: 1.0})]))
    B.out = {"lo": lo, "hi": hi, "mean": gm}
    B.meta = dict(kind="exposure", input="log")
    return B.build()

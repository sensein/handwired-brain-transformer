"""Hand-programmed hierarchical transformers (hier1 format). numpy only (browser-importable).

Every constant below is a declared design choice in physical units (um), rank units ([0,1]) or log-intensity units, with
the reasoning in docs/hierarchical/SPEC.md and the iteration record in docs/hierarchical/LOG.md. No optimizer, no labels.
"""
from __future__ import annotations

import math

import numpy as np

from .hier import G_LSP, G_ONE, S0_LOGI, S0_ONE, S_ONE, HB

BIG = 50.0


# ------------------------------------------------------------------------------------------------ small MLP idioms
def _ramp(inp, bias, out, k, sign=1.0):
    """sign * clamp(k * (w.x + bias), 0, 1) -> out (two ReLU units). inp: {ch: w}."""
    w = {c: k * v for c, v in inp.items()}
    return [(w, k * bias, {out: sign}), (w, k * bias - 1.0, {out: -sign})]


def _lin(inp, bias, out, gain=1.0):
    """out += gain * (w.x + bias), exactly, with two ReLU units (x = relu(x) - relu(-x))."""
    neg = {c: -v for c, v in inp.items()}
    return [(dict(inp), bias, {out: gain}), (neg, -bias, {out: -gain})]


def _trap(lo, hi, width, off, out):
    """global units: out += trapezoid(x) with x = lsp + off: rises over [lo - width, lo], falls over [hi, hi + width]."""
    x = {G_LSP: 1.0, G_ONE: off}
    k = 1.0 / width
    return _ramp(x, -(lo - width), out, k) + _ramp(x, -hi, out, k, -1.0)


DIRS13 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1],
                   [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], float)
DIRS13 /= np.linalg.norm(DIRS13, axis=1, keepdims=True)
DIRS4_XY = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, -1]], float)
DIRS4_XY /= np.linalg.norm(DIRS4_XY, axis=1, keepdims=True)
HORD = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]   # Hzz Hyy Hxx Hzy Hzx Hyx


def _dir_weights(d, H):
    """c_d = d^T H d as {channel: weight} over the 6 Hessian channels H (zz, yy, xx, zy, zx, yx)."""
    z, y, x = d
    return {H[0]: z * z, H[1]: y * y, H[2]: x * x, H[3]: 2 * z * y, H[4]: 2 * z * x, H[5]: 2 * y * x}


# ------------------------------------------------------------------------------------------------ front end
def front_end(B: HB, kmax=4096, beta=40.0, flat_c=0.06, dark_q=3.4, dark_u=0.4, sat_u=0.9):
    """Normalisation front end (in-model). Returns dict(u=stage-0 rank channel, ok=global gate channel, ...).
    u_i = mean_j H(logI_i - logI_j) (rank / CDF within the field of view: exactly invariant to any strictly increasing
    intensity map). Global softmax heads read the FOV: Qhi / Qlo = log-intensity of the brightest / darkest ~1/beta of
    the FOV, Uhi / Ulo = the mean rank there (ties at the max/min pull Uhi down / Ulo up). The view is unreadable
    (ok = 0, every output off) when it is flat (Qhi - Qlo < flat_c log units), dark (Qhi < dark_q, i.e. the brightest
    voxels are below I ~ 30 counts, or more than ~dark_u of the FOV ties at its minimum) or saturated (more than
    ~(1 - sat_u) * 2 of the FOV ties at its maximum)."""
    u = B.ch(0)
    B.rank(0, S0_LOGI, u, kmax)
    qhi, qlo, uhi, ulo, bad, ok = (B.ch("g") for _ in range(6))
    B.gpool(0, [S0_LOGI, u], [qhi, uhi], score=u, beta=beta)
    B.gpool(0, [S0_LOGI, u], [qlo, ulo], score=u, beta=-beta)
    units = []
    units += _ramp({qhi: -1.0, qlo: 1.0}, flat_c, bad, 40.0)        # flat
    units += _ramp({qhi: -1.0}, dark_q, bad, 10.0)                   # dark: absolute floor on the bright end
    units += _ramp({ulo: 1.0}, -dark_u, bad, 40.0)                   # dark: most of the FOV at its minimum
    units += _ramp({uhi: -1.0}, sat_u, bad, 40.0)                    # saturated: a large tie at the maximum
    B.mlp("g", units)
    B.mlp("g", [({G_ONE: 1.0}, 0.0, {ok: 1.0}), ({bad: 1.0}, 0.0, {ok: -1.0}), ({bad: 1.0}, -1.0, {ok: 1.0})])  # ok = 1 - clamp(bad)
    return dict(u=u, ok=ok, qhi=qhi, qlo=qlo, uhi=uhi, ulo=ulo)


def pyramid(B: HB, u0, S, sigma_vox=1.0):
    """Overlapped patch merging: u_s = Gaussian attention pooling of u_{s-1} (stride 2, sigma = sigma_vox source voxels
    per axis, window 3 sigma), so stage s is the scale-space image at ~2^s voxels."""
    us = [u0]
    for s in range(1, S):
        c = B.ch(s)
        B.attn(s - 1, s, us[-1], c, sigma_vox, unit="vox")
        us.append(c)
    return us


def sigma_eff(s, c):
    """effective Gaussian scale (in stage-0 voxels) of a conv of sigma = c stage-s voxels on pyramid stage s."""
    return math.sqrt((c * 2 ** s) ** 2 + (4 ** s - 1) / 3.0)


def conv_vox(B: HB, s, items):
    """conv with sigma given in stage-s voxels (x spacing; isotropic in um at run time)."""
    B.conv(s, items, unit="vox")


def hess_vox(B: HB, s, src, c):
    H = B.ch(s, 6)
    conv_vox(B, s, [(src, h, od, c, 1.0) for h, od in zip(H, HORD)])
    return H


def _extreme(B: HB, s, chans, mode):
    """in-place relu tournament over channels; returns the channel holding max (mode 'max') or min ('min')."""
    cur = list(chans)
    while len(cur) > 1:
        units, nxt = [], []
        for i in range(0, len(cur) - 1, 2):
            a_, b_ = cur[i], cur[i + 1]
            if mode == "max":
                units.append(({a_: 1.0, b_: -1.0}, 0.0, {b_: 1.0}))          # b <- b + relu(a - b) = max(a, b)
            else:
                units.append(({b_: 1.0, a_: -1.0}, 0.0, {b_: -1.0}))         # b <- b - relu(b - a) = min(a, b)
            nxt.append(b_)
        if len(cur) % 2:
            nxt.append(cur[-1])
        B.mlp(s, units)
        cur = nxt
    return cur[0]


def curvature_measures(B: HB, s, H, dirs, want, tube_ab=(1.0, 1.0)):
    """From 6 Hessian channels: directional curvatures c_d = d^T H d over `dirs`; their max / min (relu tournaments) and
    the trace. Returns requested measure channels (each >= 0) on stage s and frees the scratch:
      blob  = relu(-c_max)                                (all curvatures negative: bright blob)
      tube  = relu(-l_perp - |c_max| - (l_perp - c_min))  l_perp = (trace - c_max)/2 (bright tube, axis = argmax)
      sheet = relu(-c_min - 2|c_mid|... approximated as relu(-c_min - 2 |trace - c_min - c_max| - 2 |c_max|)
      dblob = relu(c_min)                                  (all curvatures positive: dark blob / hole)
      dsheet= relu(c_max - 2|trace - c_min - c_max| - 2|c_min|) (dark sheet: sulcus, crack)"""
    n = len(dirs)
    cmx, cmn = B.ch(s, n), B.ch(s, n)
    units = []
    for d, a_, b_ in zip(dirs, cmx, cmn):
        w = _dir_weights(d, H)
        units += _lin(w, 0.0, a_) + _lin(w, 0.0, b_)
    tr = B.ch(s)
    units += _lin({H[0]: 1.0, H[1]: 1.0, H[2]: 1.0}, 0.0, tr) if len(dirs) > 4 else _lin({H[1]: 1.0, H[2]: 1.0}, 0.0, tr)
    B.mlp(s, units)
    cmax = _extreme(B, s, cmx, "max")
    cmin = _extreme(B, s, cmn, "min")
    out = {}
    units = []
    for m in want:
        o = B.ch(s)
        out[m] = o
        if m == "blob":
            units.append(({cmax: -1.0}, 0.0, {o: 1.0}))
        elif m == "dblob":
            units.append(({cmin: 1.0}, 0.0, {o: 1.0}))
        elif m == "atr":       # |trace| (scale-specific texture / noise energy for in-context calibration)
            units += [({tr: 1.0}, 0.0, {o: 1.0}), ({tr: -1.0}, 0.0, {o: 1.0})]
        elif m in ("tube", "sheet", "dsheet"):
            pass
    B.mlp(s, units) if units else None
    # |.| terms need their own channels
    if any(m in want for m in ("tube", "sheet", "dsheet")):
        acm, acn, amid = B.ch(s), B.ch(s), B.ch(s)
        mid = {tr: 1.0, cmin: -1.0, cmax: -1.0}
        B.mlp(s, [({cmax: 1.0}, 0.0, {acm: 1.0}), ({cmax: -1.0}, 0.0, {acm: 1.0}),
                  ({cmin: 1.0}, 0.0, {acn: 1.0}), ({cmin: -1.0}, 0.0, {acn: 1.0}),
                  (mid, 0.0, {amid: 1.0}), ({c: -v for c, v in mid.items()}, 0.0, {amid: 1.0})])
        units = []
        if "tube" in want:   # -l_perp - |cmax| - (l_perp - cmin) = -trace + cmax - |cmax| + cmin  (l_perp = (tr - cmax)/2)
            ta, tb = tube_ab   # -l_perp - ta |cmax| - tb (l_perp - cmin), l_perp = (tr - cmax)/2
            units.append(({tr: -0.5 - 0.5 * tb, cmax: 0.5 + 0.5 * tb, acm: -ta, cmin: tb}, 0.0, {out["tube"]: 1.0}))
        if "sheet" in want:
            units.append(({cmin: -1.0, amid: -2.0, acm: -2.0}, 0.0, {out["sheet"]: 1.0}))
        if "dsheet" in want:
            units.append(({cmax: 1.0, amid: -2.0, acn: -2.0}, 0.0, {out["dsheet"]: 1.0}))
        B.mlp(s, units)
        B.zero(s, [acm, acn, amid], "pos")
    B.zero(s, list(cmx) + list(cmn) + [tr] + list(H))
    return out


# ------------------------------------------------------------------------------------------------ detector skeleton
def _scale_detector(B: HB, fe, measure, scales, classes, dirs, kappa, tau0, sharp=200.0, kappa_s=(0.4, 0.4), floor_s=(0.03, 0.012), tube_ab=(1.0, 1.0)):
    """Shared multi-scale detector: for every (stage s, conv sigma c in stage voxels) compute the curvature `measure` on
    the pyramid image, gate it by a physical-size prior on sigma_eff (lsp-dependent: computed in-model from the voxel-size
    input), upsample to stage 0, and keep per-class maxima (selection of scale by response). seg = step(max_k C_k -
    max(kappa * FOV mean, tau0)) and ok. Returns class channels and seg on stage 0."""
    us = fe["us"]
    C = {k: B.ch(0) for k in classes}
    for s, c in scales:
        H = hess_vox(B, s, us[s], c)
        cm = curvature_measures(B, s, H, dirs, [measure, "atr"], tube_ab=tube_ab)
        R, atr = cm[measure], cm["atr"]
        ms = B.ch("g")
        B.gpool(s, atr, ms)                                                      # per-scale FOV energy (in-context calibration)
        ks = kappa_s[min(s, len(kappa_s) - 1)]
        fs = floor_s[min(s, len(floor_s) - 1)]
        th = B.ch("g")
        B.mlp("g", [({ms: ks}, 0.0, {th: 1.0}), ({ms: -ks, G_ONE: fs}, 0.0, {th: 1.0})])          # th = max(k_s m_s, floor)
        B.mlp(s, [({R: 1.0}, 0.0, {R: -1.0}), ({R: 1.0, ("g", th): -1.0}, 0.0, {R: 1.0})])    # R <- relu(R - th)
        B.zero(s, [atr], "pos")
        off = math.log2(sigma_eff(s, c))
        for k, (lo, hi) in classes.items():
            gk = B.ch("g")
            B.mlp("g", _trap(math.log2(lo), math.log2(hi), 0.25, off, gk))
            if s == 0:
                t = B.ch(0)
                B.mlp(0, [({R: 1.0, ("g", gk): BIG}, -BIG, {t: 1.0})])          # gated response
            else:
                tg = B.ch(s)
                B.mlp(s, [({R: 1.0, ("g", gk): BIG}, -BIG, {tg: 1.0})])
                t = B.ch(0)
                B.attn(s, 0, tg, t, 0.6, unit="vox")                             # upsample (Gaussian attention, fine queries)
                B.zero(s, [tg], "pos")
            B.mlp(0, [({t: 1.0, C[k]: -1.0}, 0.0, {C[k]: 1.0})])                 # C_k <- max(C_k, t)
            B.zero(0, [t], "pos")
        B.zero(s, [R], "pos")
    Cm = B.ch(0)
    B.mlp(0, [({C[k]: 1.0}, 0.0, {Cm: 1.0}) for k in list(classes)[:1]])
    for k in list(classes)[1:]:
        B.mlp(0, [({C[k]: 1.0, Cm: -1.0}, 0.0, {Cm: 1.0})])
    mean, thr = B.ch("g"), B.ch("g")
    B.gpool(0, Cm, mean)
    B.mlp("g", [({mean: kappa}, 0.0, {thr: 1.0}), ({mean: -kappa, G_ONE: tau0}, 0.0, {thr: 1.0})])   # thr = max(kappa m, tau0)
    seg = B.ch(0)
    ok = fe["ok"]
    B.mlp(0, _ramp({Cm: 1.0, ("g", thr): -1.0, ("g", ok): 1.0}, -1.0, seg, sharp))                  # step(Cm - thr) if ok
    return C, Cm, seg


def cells(S=4, kappa=0.0, tau0=0.0005, kappa_s=(0.4, 0.4), floor_s=(0.03, 0.012)):
    """Bright blobs (nuclei, somata, nuclear clusters) at every pyramid level with one weight file."""
    B = HB("hier_cells", S)
    fe = front_end(B)
    fe["us"] = pyramid(B, fe["u"], S)
    scales = [(0, 1.3), (0, 1.8)] + [(s, c) for s in range(1, S) for c in (1.0, 1.5)]
    classes = {"nucleus": (1.6, 3.6), "soma": (3.6, 8.0), "cluster": (8.0, 18.0)}   # sigma_eff in um (blob radius ~ 1.7 sigma)
    C, Cm, seg = _scale_detector(B, fe, "blob", scales, classes, DIRS13, kappa, tau0, kappa_s=kappa_s, floor_s=floor_s)
    B.out = {"seg": [0, seg], "score": [0, Cm], "u": [0, fe["u"]], **{k: [0, c] for k, c in C.items()}, "ok": [-1, fe["ok"]]}
    B.meta = dict(kind="blobs", hier=True, input="logI", classes=list(classes), sigma_classes_um=classes, zhalo_um=40.0)
    return B.build()


def vessels(S=4, kappa=0.0, tau0=0.0005, kappa_s=(0.55, 0.5), floor_s=(0.045, 0.015), tube_ab=(2.0, 1.0)):
    """Bright tubes of any calibre (LEC vessels) at every pyramid level with one weight file."""
    B = HB("hier_vessels", S)
    fe = front_end(B)
    fe["us"] = pyramid(B, fe["u"], S)
    scales = [(s, c) for s in range(S) for c in (1.0, 1.6)]
    classes = {"capillary": (1.2, 3.0), "vessel": (3.0, 7.0), "large": (7.0, 20.0)}
    C, Cm, seg = _scale_detector(B, fe, "tube", scales, classes, DIRS13, kappa, tau0, kappa_s=kappa_s, floor_s=floor_s, tube_ab=tube_ab)
    B.out = {"seg": [0, seg], "score": [0, Cm], "u": [0, fe["u"]], **{k: [0, c] for k, c in C.items()}, "ok": [-1, fe["ok"]]}
    B.meta = dict(kind="vessels_ms", hier=True, input="logI", classes=list(classes), sigma_classes_um=classes, zhalo_um=40.0)
    return B.build()

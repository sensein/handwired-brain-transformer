"""Hand-programmed hierarchical region transformer (hier1): tissue / background, cortex / white matter, cortical depth
bands. numpy only. Reasoning in docs/hierarchical/SPEC.md, iterations in docs/hierarchical/LOG.md."""
from __future__ import annotations

from .hier import G_LSP, G_ONE, S0_LOGI, HB
from .hier_programs import _lin, _ramp, front_end, pyramid

QP = (0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.36, 0.42, 0.50, 0.58, 0.66, 0.74, 0.80, 0.86, 0.92, 0.98)


def quantiles(B: HB, u, levels=QP, beta=300.0, s=0, val=S0_LOGI, mask=None):
    """Q(p) = log-intensity at rank p: global softmax heads with score -beta |u - p| (attention concentrated on the
    tokens whose rank is p; beta = 300 -> rank half-width ~0.3 %). mask (optional channel, 1 = excluded) adds a large
    negative score. Returns global channels."""
    a = B.ch(s, len(levels))
    units = []
    for c, p in zip(a, levels):
        units += [({u: 1.0}, -p, {c: 1.0}), ({u: -1.0}, p, {c: 1.0})]
        if mask is not None:
            units.append(({mask: 60.0 / beta}, 0.0, {c: 1.0}))
    B.mlp(s, units)
    Q = B.ch("g", len(levels))
    for c, q in zip(a, Q):
        B.gpool(s, val, q, score=c, beta=-beta)
    B.zero(s, a, "pos")
    return Q


def argmax_global(B: HB, vals, idx_vals, sharp=200.0):
    """relu tournament on the global token over (value, index) pairs; index = a constant >= 0 per entry. Index transport
    uses s * x = relu(x - M (1 - s)) for a binary step s (exact when the step is saturated)."""
    V = list(vals)
    Ich = []
    for iv in idx_vals:
        c = B.ch("g")
        B.mlp("g", [({G_ONE: float(iv)}, 0.0, {c: 1.0})])
        Ich.append(c)
    M = 4.0
    while len(V) > 1:
        pairs = list(range(0, len(V) - 1, 2))
        steps, units = [], []
        for k in pairs:
            s = B.ch("g")
            units += _ramp({V[k]: 1.0, V[k + 1]: -1.0}, 0.0, s, sharp)
            steps.append(s)
        B.mlp("g", units)
        units, nv, ni = [], [], []
        for s, k in zip(steps, pairs):
            a_, b_, ia, ib = V[k], V[k + 1], Ich[k], Ich[k + 1]
            units.append(({a_: 1.0, b_: -1.0}, 0.0, {b_: 1.0}))
            units.append(({ia: 1.0, ib: -1.0, s: M}, -M, {ib: 1.0}))
            units.append(({ib: 1.0, ia: -1.0, s: M}, -M, {ib: -1.0}))
            nv.append(b_)
            ni.append(ib)
        B.mlp("g", units)
        if len(V) % 2:
            nv.append(V[-1])
            ni.append(Ich[-1])
        V, Ich = nv, ni
    return V[0], Ich[0]


QP49 = tuple(round(0.02 * k, 2) for k in range(1, 50))


def tissue_front(B: HB, fe, c_nb=1.8, c_ref=2.5, sigma_t=25.0, beta=300.0):
    """Tissue vs background from the FOV's own intensity distribution (in-context, no absolute level).
    1. Q(p_k), p_k = 0.02 .. 0.98, from 49 global softmax heads over the valid voxels (I = 0 marks absent data: masked).
    2. Inverse density D_k = (Q_{k+1} - Q_k) / 0.02 (log units per rank).
    3. A histogram valley is an interval much sparser than both neighbourhoods and than the FOV average:
       W_k = min(D_k - c_nb (D_{k-2} + D_{k+2}) / 2, D_k - c_ref D_ref), D_ref = (Q_0.98 - Q_0.02) / 0.96.
       (A Gaussian or any log-concave-like mode has convex D: never a valley.)
    4. The tissue threshold is the brightest valley (dark modes below it: air, mounting medium, out-of-slab haze):
       E_k = (1 + p_k) step(W_k), argmax by relu tournament; T = p_k + 0.01 (rank), bimodal = step(max E - 0.5);
       no valley -> T = -0.1 (all tissue).
    W_k, E_k are ratios of log-intensity differences: exactly invariant to gain and power-law gamma; T is a rank.
    tissue = step(G_sigma_t(u) - T)."""
    lv = QP49
    inv = B.ch(0)
    B.mlp(0, _ramp({S0_LOGI: -1.0}, 0.01, inv, 1000.0))
    Q = quantiles(B, fe["u"], lv, beta=beta, mask=inv)
    nD = len(lv) - 1
    D, units = [], []
    for k in range(nD):
        c = B.ch("g")
        units += _lin({Q[k + 1]: 50.0, Q[k]: -50.0}, 0.0, c)
        D.append(c)
    dref = B.ch("g")
    units += _lin({Q[-1]: 1 / 0.96, Q[0]: -1 / 0.96}, 0.0, dref)
    B.mlp("g", units)
    ks = list(range(2, nD - 2))
    W = {k: B.ch("g") for k in ks}
    # W_k = min(a, b) = a - relu(a - b); a = D_k - c_nb (D_{k-2} + D_{k+2}) / 2, b = D_k - c_ref D_ref
    units = []
    for k in ks:
        a = {D[k]: 1.0, D[k - 2]: -c_nb / 2, D[k + 2]: -c_nb / 2}
        b = {D[k]: 1.0, dref: -c_ref}
        units += _lin(a, 0.0, W[k])
        amb = dict(a)
        for c_, v in b.items():
            amb[c_] = amb.get(c_, 0.0) - v
        units.append((amb, 0.0, {W[k]: -1.0}))
    B.mlp("g", units)
    E = {k: B.ch("g") for k in ks}
    units = []
    for k in ks:
        units += _ramp({W[k]: 1.0}, 0.0, E[k], 2.0, 1.0 + lv[k])    # (1 + p_k) * clamp(2 W_k): W in log/rank units (~10-50 at valleys)
    B.mlp("g", units)
    emax, pidx = argmax_global(B, [E[k] for k in ks], [lv[k] + 0.01 for k in ks])
    bim, T = B.ch("g"), B.ch("g")
    B.mlp("g", _ramp({emax: 1.0}, -0.5, bim, 20.0))
    B.mlp("g", [({pidx: 1.0, bim: 1.0}, -1.0, {T: 1.0}), ({G_ONE: 0.1}, 0.0, {T: -1.0}), ({bim: 0.1}, 0.0, {T: 1.0})])
    ubar = B.ch(0)
    B.attn(0, 0, fe["u"], ubar, sigma_t, unit="um")
    tis, bg = B.ch(0), B.ch(0)
    ok = fe["ok"]
    B.mlp(0, _ramp({ubar: 1.0, ("g", T): -1.0, ("g", ok): 1.0}, -1.0, tis, 400.0) +
          _ramp({ubar: -1.0, ("g", T): 1.0, ("g", ok): 1.0}, -1.0, bg, 400.0))
    return dict(tissue=tis, bg=bg, ubar=ubar, T=T, bim=bim, inv=inv, Q=Q)


from math import erf, sqrt


def _phi(t):
    return 0.5 * (1 + erf(t / sqrt(2)))


LAYER_FRACTIONS = (0.08, 0.15, 0.45, 0.55, 0.75)   # declared prior: I/II, II/III, III/IV, IV/V, V/VI boundaries (fraction of cortical depth)


def potential_edges(T_um=2500.0, sigma_um=1000.0, fr=LAYER_FRACTIONS):
    """x = P_wm - P_pia for a straight cortex of thickness T between a background half-space and a white-matter
    half-space, both smoothed by a Gaussian of sigma: x(d) = Phi(-(T - d)/sigma) - Phi(-d/sigma); edges at fractions."""
    return [_phi(-(T_um - f * T_um) / sigma_um) - _phi(-f * T_um / sigma_um) for f in fr]


def _band(x, lo, hi, out, k):
    """out += clamp(k (x - lo)) - clamp(k (x - hi)) (x a channel; lo/hi constants, None = open)."""
    u = []
    if lo is None:
        u.append(({}, 1.0, {out: 1.0}))   # constant 1 (bias-only unit)
    else:
        u += _ramp({x: 1.0}, -lo, out, k)
    if hi is not None:
        u += _ramp({x: 1.0}, -hi, out, k, -1.0)
    return u


def regions(S=5, sigma_c=100.0, sigma_f=200.0, sigma_clean=150.0, margin=0.05, sig_a=500.0, sig_b=1500.0, sig_p=1000.0, cortex_T=2500.0, sharp=60.0):
    """Tissue / cortex / white matter / cortical depth bands. Stage 0 holds the voxel-level decisions; stage 2 holds the
    mm-scale context (proximity to the pial surface and to white matter, in-plane Gaussian attention, sigma 0.5-1.5 mm),
    upsampled by Gaussian attention. See SPEC.md for each step."""
    B = HB("hier_regions", S, zpool=False)
    fe = front_end(B)
    ok = fe["ok"]
    tf = tissue_front(B, fe)
    tis, bg = tf["tissue"], tf["bg"]
    # --- within-tissue intensity feature f: rank (among tissue voxels) of the 100 um smoothed rank image
    u100 = B.ch(0)
    B.attn(0, 0, fe["u"], u100, sigma_c, unit="um")
    f0, f = B.ch(0), B.ch(0)
    B.rank(0, u100, f0, kmask=tis)
    B.attn(0, 0, f0, f, (1.0, sigma_f, sigma_f))                       # in-plane smoothing of the class feature (stripes, texture)
    # --- mm-scale context at stage 2 (patch merging twice), in-plane proximity of background (pia side)
    b1, b2 = B.ch(1), B.ch(2)
    B.attn(0, 1, bg, b1, 1.0, unit="vox")
    B.attn(1, 2, b1, b2, 1.0, unit="vox")
    Pa2, Pb2, Pp2 = B.ch(2), B.ch(2), B.ch(2)
    B.attn(2, 2, b2, Pa2, (1.0, sig_a, sig_a))
    B.attn(2, 2, b2, Pb2, (1.0, sig_b, sig_b))
    B.attn(2, 2, b2, Pp2, (1.0, sig_p, sig_p))
    Pa, Pb, Pp = B.ch(0), B.ch(0), B.ch(0)
    for a_, b_ in ((Pa2, Pa), (Pb2, Pb), (Pp2, Pp)):
        B.attn(2, 0, a_, b_, 0.6, unit="vox")
    # --- prototype regions (declared depths for a straight pial edge: P_sigma(d) = Phi(-d / sigma))
    O, I, W = B.ch(0), B.ch(0), B.ch(0)
    lo_o, hi_o = _phi(-800 / sig_a), _phi(-200 / sig_a)          # outer cortex 0.2-0.8 mm below the pia
    lo_i, hi_i = _phi(-2000 / sig_b), _phi(-1200 / sig_b)        # inner cortex 1.2-2.0 mm
    hi_w = _phi(-3500 / sig_b)                                   # deep tissue > 3.5 mm
    B.mlp(0, _band(Pa, lo_o, hi_o, O, 200.0) + _band(Pb, lo_i, hi_i, I, 200.0) + _band(Pb, None, hi_w, W, 2000.0))
    # core voxels: tissue in the whole +-2-plane z neighbourhood (prototype statistics avoid the slab's cut faces)
    tz, core = B.ch(0), B.ch(0)
    B.attn(0, 0, tis, tz, (60.0, 1.0, 1.0))
    B.mlp(0, _ramp({tz: 1.0}, -0.97, core, 100.0))
    B.mlp(0, [({c: 1.0}, 0.0, {c: -1.0}) for c in (O, I, W)] + [({c: 1.0, core: 1.0}, -1.0, {c: 1.0}) for c in (O, I, W)])  # c <- relu(c + core - 1)
    mO, mI, mW, eO, eW = (B.ch("g") for _ in range(5))
    B.gpool(0, f, mO, score=O, beta=30.0)
    B.gpool(0, f, mI, score=I, beta=30.0)
    # white-matter prototype: the deepest core tissue (softmax over -1000 Pb, restricted to W by +30 W)
    sW = B.ch(0)
    B.mlp(0, _lin({W: 30.0, Pb: -1000.0}, 0.0, sW))
    B.gpool(0, f, mW, score=sW, beta=1.0)
    nO, nW = B.ch("g"), B.ch("g")
    B.gpool(0, O, nO)
    B.gpool(0, W, nW)
    B.mlp("g", _ramp({nO: 1.0}, -0.003, eO, 1000.0) + _ramp({nW: 1.0}, -0.003, eW, 1000.0))   # prototypes exist (> 0.3 % of FOV)
    # --- nearest prototype: wm if |f - mW| < min(|f - mO|, |f - mI|) - margin
    dO, dI, dW = B.ch(0), B.ch(0), B.ch(0)
    units = []
    for d_, m_ in ((dO, mO), (dI, mI), (dW, mW)):
        units += [({f: 1.0, ("g", m_): -1.0}, 0.0, {d_: 1.0}), ({f: -1.0, ("g", m_): 1.0}, 0.0, {d_: 1.0})]
    B.mlp(0, units)
    pen = B.ch(0)
    B.mlp(0, [({dO: 1.0, dI: -1.0}, 0.0, {dO: -1.0}), ({Pa: 1.0}, -hi_o, {pen: 1.0})])   # dO <- min(dO, dI); pen = relu(Pa - hi_o)
    wm0, wm = B.ch(0), B.ch(0)
    # wm = step(dO - dW - margin) and tissue and both prototypes exist and ok; never within ~0.2 mm of the pia (pen)
    B.mlp(0, _ramp({dO: 1.0, dW: -1.0, tis: 1.0, ("g", eW): 1.0, ("g", ok): 1.0, ("g", eO): 1.0, pen: -10.0}, -margin - 4.0, wm0, sharp))
    wms = B.ch(0)
    B.attn(0, 0, wm0, wms, (1.0, sigma_clean, sigma_clean))           # majority vote over ~0.3 mm (removes specks / fills pinholes)
    B.mlp(0, _ramp({wms: 1.0, tis: 1.0}, -1.5, wm, 50.0))
    cortex = B.ch(0)
    B.mlp(0, [({tis: 1.0, wm: -1.0, ("g", eO): 1.0}, -1.0, {cortex: 1.0})])     # cortex = tissue - wm when the pia is in view
    # --- depth potential x = P_wm - P_pia (sigma 1 mm) and declared layer bands inside the cortex
    w1, w2, Pw2, Pw = B.ch(1), B.ch(2), B.ch(2), B.ch(0)
    B.attn(0, 1, wm, w1, 1.0, unit="vox")
    B.attn(1, 2, w1, w2, 1.0, unit="vox")
    B.attn(2, 2, w2, Pw2, (1.0, sig_p, sig_p))
    B.attn(2, 0, Pw2, Pw, 0.6, unit="vox")
    x = B.ch(0)
    B.mlp(0, _lin({Pw: 1.0, Pp: -1.0}, 0.0, x))
    edges = potential_edges(cortex_T, sig_p)
    bands = [B.ch(0) for _ in range(6)]
    units = []
    for k, b_ in enumerate(bands):
        lo = None if k == 0 else edges[k - 1]
        hi = None if k == 5 else edges[k]
        units += _band(x, lo, hi, b_, 40.0)
    B.mlp(0, units)
    B.mlp(0, [({b_: 1.0, cortex: -1.0}, 0.0, {b_: -1.0}) for b_ in bands])      # band <- band - relu(band - cortex) = min(band, cortex)
    B.out = {"depth": [0, f], "f0": [0, f0], "potential": [0, x], "tissue": [0, tis], "bg": [0, bg], "cortex": [0, cortex], "wm": [0, wm],
             **{f"L{k + 1}": [0, b_] for k, b_ in enumerate(bands)}, "WM": [0, wm],
             "u": [0, fe["u"]], "ok": [-1, ok], "T": [-1, tf["T"]], "bim": [-1, tf["bim"]], "eO": [-1, eO], "eW": [-1, eW],
             "mO": [-1, mO], "mI": [-1, mI], "mW": [-1, mW], "O": [0, O], "I": [0, I], "W": [0, W], "Pa": [0, Pa], "Pb": [0, Pb]}
    cols = {"bg": [90, 90, 100], "L1": [253, 231, 37], "L2": [170, 220, 50], "L3": [94, 201, 98], "L4": [33, 145, 140], "L5": [59, 82, 139],
            "L6": [110, 40, 150], "WM": [215, 215, 235], "tissue": [90, 200, 120], "cortex": [255, 160, 40], "wm": [60, 140, 255]}
    B.meta = dict(kind="regions", hier=True, input="logI", classes=["bg", "L1", "L2", "L3", "L4", "L5", "L6", "WM"], colors=cols,
                  layer_fractions=list(LAYER_FRACTIONS), potential_edges=[round(e, 4) for e in edges])
    return B.build()

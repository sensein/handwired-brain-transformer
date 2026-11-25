"""Hierarchical (SegFormer-style) weight-space-programmed 3D transformer: program format + geometry (numpy only, no torch).

Spec: docs/hierarchical/SPEC.md. Nothing is trained; every number is placed by a generator (hier_programs.py).

Token grids. Stage 0 is the input voxel grid (n0 = (Z, Y, X), spacing sp0 in um). Stage s has n_s = ceil(n_{s-1} / 2)
tokens per axis (stride-2 patch merging on every axis) and spacing sp0 * 2^s. The centre of token k of stage s, in
tile-centred physical coordinates, is p_s(k) = (k 2^s + (2^s - 1)/2 - (n0 - 1)/2) * sp0 (a block centre of the stage-0
lattice), so every op that maps between stages is defined by physical positions only.

Streams. Stage s carries a residual stream of d[s] channels; one extra token carries dg "global" channels (FOV-level
scalars, broadcast to every token as extra MLP inputs). Fixed channels:
  stage 0: 0 = log(I+1) (raw, no display window), 1 = one;   stage s > 0: 0 = one
  global : 0 = log2(sp0_x / 1 um) (voxel size, an input), 1 = one
Every op ADDS its result (times a gain) into destination channels (residual stream), so programs never overwrite.

Ops (all hand-set; exact semantics in SPEC.md, identical in torch / JS-CPU / WebGPU executors):
  rank   global step-kernel (sigmoid-limit) attention: u_i = mean_j H(v_i - v_j) over a strided key set -> CDF in [0,1]
  gpool  global softmax attention to a global channel: g += sum_j softmax_j(beta * x[score]_j) x[src]_j
  mlp    pointwise ReLU MLP on one stage (inputs: stage channels + global channels) or on the global token
  attn   Gaussian softmax attention between stages (pool: dst = src+1; upsample: dst < src; smooth / spatial-reduction:
         any): o_i = sum_j softmax_j(-1/2 sum_a ((p_j,a - p_i,a)/sigma_a)^2) v_j over |p_j,a - p_i,a| <= max(3 sigma_a, h_a)
  conv   depthwise separable Gaussian-derivative convolution (scale-normalised sigma^|order| d^order G_sigma), replicate
         padding: the hand-set analogue of the Mix-FFN 3x3x3 depthwise conv.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

S0_LOGI, S0_ONE, S_ONE = 0, 1, 0
G_LSP, G_ONE = 0, 1


# ------------------------------------------------------------------------------------------------ geometry
def fac(s, a, zpool=True):
    """stride of stage s relative to stage 0 along axis a (z = axis 0 keeps stride 1 when the program has zpool=False)."""
    return 1 if (a == 0 and not zpool) else 2 ** s


def stage_shapes(n0, n_stages, zpool=True):
    sh = [tuple(int(v) for v in n0)]
    for s in range(1, n_stages):
        sh.append(tuple((v + 1) // 2 if (a > 0 or zpool) else v for a, v in enumerate(sh[-1])))
    return sh


def stage_pos(n0_a, sp0_a, f, n_s_a):
    """physical centre (um, tile-centred) of each token of a stage with stride f (= 2^s) along one axis."""
    k = np.arange(n_s_a, dtype=np.float64)
    return (k * f + (f - 1) / 2 - (n0_a - 1) / 2) * sp0_a


def attn_matrix(n0_a, sp0_a, src_f, dst_f, n_src, n_dst, sigma_um):
    """Row-normalised 1-D Gaussian attention weights (n_dst x n_src) along one axis. The 3-D op is the product of the
    three axis matrices: for a diagonal precision and a box window the softmax normaliser factorises exactly."""
    ps = stage_pos(n0_a, sp0_a, src_f, n_src)
    pd = stage_pos(n0_a, sp0_a, dst_f, n_dst)
    h = sp0_a * src_f
    sig = max(float(sigma_um), 1e-6)
    D = pd[:, None] - ps[None, :]
    win = np.abs(D) <= max(3.0 * sig, h) + 1e-9
    sc = -0.5 * (D / sig) ** 2
    sc = np.where(win, sc, -np.inf)
    m = sc.max(1, keepdims=True)
    nearest = ~np.isfinite(m[:, 0])
    m[nearest] = 0
    W = np.exp(sc - m)
    if nearest.any():  # no src token inside the window (cannot happen for the lattices above; kept for safety)
        W[nearest] = 0
        W[nearest, np.abs(D[nearest]).argmin(1)] = 1
    return W / W.sum(1, keepdims=True)


def deriv_kernel(h, sigma_um, order):
    """1-D scale-normalised Gaussian-derivative correlation kernel on spacing h (um): taps t_k = k h, k = -R..R.
    order 0: sum = 1; order 1: sum t k = sigma (response to f = t is sigma f'); order 2: zero-sum and sum (t^2/2) k = sigma^2.
    sigma is clamped to >= 0.5 h (a sub-voxel kernel is a delta / the smallest centred difference)."""
    sig = max(float(sigma_um), 0.5 * h)
    R = max(1, int(math.ceil(3.0 * sig / h)))
    t = np.arange(-R, R + 1) * h
    g = np.exp(-0.5 * (t / sig) ** 2)
    g /= g.sum()
    if order == 0:
        return g
    if order == 1:
        k = t * g
        return k / (k * t).sum() * sig
    k = (t ** 2 / sig ** 2 - 1) * g
    k -= k.mean()
    return k / (k * t ** 2 / 2).sum() * sig ** 2


def conv_matrix(n, h, sigma_um, order):
    """(n x n) correlation matrix of deriv_kernel with replicate padding folded in (index clamp)."""
    k = deriv_kernel(h, sigma_um, order)
    R = (len(k) - 1) // 2
    M = np.zeros((n, n))
    for i in range(n):
        for j, w in enumerate(k):
            M[i, min(max(i + j - R, 0), n - 1)] += w
    return M


def sigma_um_of(op, s_ref, sp0, zpool=True):
    """per-axis sigma in um: unit 'um' (isotropic or per axis) or 'vox' (voxels of stage s_ref, per axis)."""
    sg = np.broadcast_to(np.asarray(op["sigma"], float), (3,))
    if op.get("unit", "um") == "vox":
        return [float(sg[a] * sp0[a] * fac(s_ref, a, zpool)) for a in range(3)]
    return [float(v) for v in sg]


def rank_keys(N, kmax):
    m = max(1, -(-N // int(kmax)))
    return m  # keys = flat indices 0, m, 2m, ...


# ------------------------------------------------------------------------------------------------ program
@dataclass
class HierProgram:
    name: str
    n_stages: int
    d: list            # channels per stage
    dg: int            # global channels
    ops: list          # list of dicts (numpy arrays for mlp weights)
    out: dict          # name -> [stage, channel]
    meta: dict = field(default_factory=dict)
    zpool: bool = True  # False: stages merge patches in-plane only (z keeps the input planes; thin slabs)

    def n_params(self):
        tot = nz = 0
        for op in self.ops:
            if op["op"] == "mlp":
                for k in ("W1", "b1", "W2"):
                    tot += op[k].size
                    nz += int(np.count_nonzero(op[k]))
            else:
                n = sum(len(np.atleast_1d(op[k])) for k in ("src", "dst", "gain") if k in op)
                n += 3 if "sigma" in op else 0
                n += sum(4 for _ in op.get("kern", []))
                tot += n
                nz += n
        return tot, nz

    def to_json(self) -> str:
        def sp(x):
            x = np.asarray(x, dtype=np.float64)
            idx = np.flatnonzero(x)
            return {"shape": list(x.shape), "i": idx.tolist(), "v": x.reshape(-1)[idx].tolist()}

        ops = []
        for op in self.ops:
            o = {}
            for k, v in op.items():
                if k in ("W1", "b1", "W2"):
                    o[k] = sp(v)
                elif isinstance(v, np.ndarray):
                    o[k] = v.tolist()
                else:
                    o[k] = v
            ops.append(o)
        return json.dumps({"format": "hier1", "name": self.name, "n_stages": self.n_stages, "d": list(self.d), "dg": self.dg,
                           "out": self.out, "meta": self.meta, "zpool": self.zpool, "ops": ops}, separators=(",", ":"))

    @classmethod
    def from_json(cls, text) -> "HierProgram":
        js = json.loads(text) if isinstance(text, (str, bytes)) else text
        assert js.get("format") == "hier1", "not a hier1 program"

        def dense(m):
            a = np.zeros(int(np.prod(m["shape"])))
            a[np.asarray(m["i"], dtype=int)] = m["v"]
            return a.reshape(m["shape"])

        ops = []
        for o in js["ops"]:
            op = dict(o)
            for k in ("W1", "b1", "W2"):
                if k in op:
                    op[k] = dense(op[k])
            ops.append(op)
        return cls(js["name"], js["n_stages"], js["d"], js["dg"], ops, js["out"], js.get("meta", {}), js.get("zpool", True))


def is_hier(js_text_or_dict) -> bool:
    js = json.loads(js_text_or_dict) if isinstance(js_text_or_dict, (str, bytes)) else js_text_or_dict
    return js.get("format") == "hier1"


# ------------------------------------------------------------------------------------------------ builder
class HB:
    """Channel allocator + op recorder. Channels are (stage, index); 'g' is the global token."""

    def __init__(self, name, n_stages, zpool=True):
        self.name, self.S, self.zpool = name, n_stages, zpool
        self.d = [2] + [1] * (n_stages - 1)
        self.dg = 2
        self.ops, self.out, self.meta = [], {}, {}
        self._free = {s: [] for s in list(range(n_stages)) + ["g"]}

    def ch(self, s, n=None):
        """allocate one (n=None) or n zero channels on stage s ('g' = global). Freed channels are reused only after
        they were explicitly zeroed by the program (see zero())."""
        def one():
            if self._free[s]:
                return self._free[s].pop()
            if s == "g":
                self.dg += 1
                return self.dg - 1
            self.d[s] += 1
            return self.d[s] - 1
        return one() if n is None else [one() for _ in range(n)]

    def zero(self, s, chans, sign_known=None):
        """free channels after subtracting them away: x <- x - x is done with two ReLU units per channel (x = relu(x) -
        relu(-x)). sign_known='pos' uses one unit (x >= 0 by construction)."""
        units = []
        for c in chans:
            units.append(({c: 1.0}, 0.0, {c: -1.0}))
            if sign_known != "pos":
                units.append(({c: -1.0}, 0.0, {c: 1.0}))
        self.mlp(s, units)
        self._free[s] += list(chans)

    # ---- ops
    def rank(self, s, src, dst, kmax=4096, kmask=None):
        """kmask: optional channel; only strided keys with kmask > 0.5 count (rank within a region, e.g. tissue)."""
        self.ops.append(dict(op="rank", s=s, src=src, dst=dst, kmax=int(kmax), kmask=kmask))

    def gpool(self, s, src, dst, score=None, beta=0.0, gain=None):
        src, dst = list(np.atleast_1d(src)), list(np.atleast_1d(dst))
        self.ops.append(dict(op="gpool", s=s, score=score, beta=float(beta), src=[int(c) for c in src], dst=[int(c) for c in dst],
                             gain=[1.0] * len(src) if gain is None else [float(v) for v in np.atleast_1d(gain)]))

    def attn(self, src_s, dst_s, src, dst, sigma, unit="um", gain=None):
        src, dst = list(np.atleast_1d(src)), list(np.atleast_1d(dst))
        self.ops.append(dict(op="attn", src_s=src_s, dst_s=dst_s, sigma=[float(v) for v in np.broadcast_to(np.asarray(sigma, float), (3,))],
                             unit=unit, src=[int(c) for c in src], dst=[int(c) for c in dst],
                             gain=[1.0] * len(src) if gain is None else [float(v) for v in np.atleast_1d(gain)]))

    def conv(self, s, kern, unit="um"):
        """kern: list of (src, dst, (oz, oy, ox), sigma, gain); sigma isotropic, in um or (unit='vox') in stage-s x-voxels."""
        self.ops.append(dict(op="conv", s=s, unit=unit, kern=[dict(src=int(a), dst=int(b), order=[int(o) for o in od], sigma=float(sg), gain=float(g))
                                                    for a, b, od, sg, g in kern]))

    def mlp(self, s, units):
        """units: list of (in{ch|('g',ch): w}, bias, out{ch: w}) on stage s (s='g': global token, all keys global)."""
        self.ops.append(dict(op="mlp", s=s, units=units))  # resolved to matrices in build()

    def build(self) -> HierProgram:
        ops = []
        for op in self.ops:
            if op["op"] != "mlp":
                ops.append(op)
                continue
            s, units = op["s"], op["units"]
            ds = 0 if s == "g" else self.d[s]
            h = len(units)
            W1 = np.zeros((h, ds + self.dg))
            b1 = np.zeros(h)
            W2 = np.zeros((self.dg if s == "g" else ds, h))
            for i, (w, b, o) in enumerate(units):
                for k, v in w.items():
                    if isinstance(k, tuple):  # ('g', ch)
                        W1[i, ds + k[1]] += v
                    elif s == "g":
                        W1[i, k] += v
                    else:
                        W1[i, k] += v
                b1[i] = b
                for k, v in o.items():
                    W2[k, i] += v
            ops.append(dict(op="mlp", s=-1 if s == "g" else s, W1=W1, b1=b1, W2=W2))
        return HierProgram(self.name, self.S, list(self.d), self.dg, ops, dict(self.out), dict(self.meta), self.zpool)


def halo_vox(prog: HierProgram, sp0):
    """Stage-0-voxel half-extent per axis that one output token depends on through local ops (attn / conv); global
    ops (rank, gpool) depend on the whole field of view by design. Used by viewers to size tiles."""
    tot = np.zeros(3)
    for op in prog.ops:
        if op["op"] == "attn":
            sg = sigma_um_of(op, op["src_s"], sp0, prog.zpool)
            hsrc = [sp0[a] * fac(op["src_s"], a, prog.zpool) for a in range(3)]
            tot += [max(3 * sg[a], hsrc[a]) / sp0[a] for a in range(3)]
        elif op["op"] == "conv":
            s = op["s"]
            r = 0.0
            for k in op["kern"]:
                h = sp0[2] * 2 ** s
                sg = k["sigma"] * (h if op.get("unit", "um") == "vox" else 1.0)
                r = max(r, max(1, math.ceil(3 * max(sg, 0.5 * h) / h)) * 2 ** s)
            tot += [r if prog.zpool else r / 2 ** s, r, r]
    return [int(math.ceil(v)) for v in tot]

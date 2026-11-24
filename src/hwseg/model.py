"""Weight-space-programmed 3D volumetric transformer.

Nothing here is trained. Every weight is written by construction (see wiring.py).
Token = voxel. Embedding channels (fixed layout, tile-local physical coordinates in um):
  0 logI | 1-3 z,y,x | 4 one | 5-7 z^2,y^2,x^2 | 8-10 zy,zx,yx | 11.. free (program-allocated)
Attention is local (radius in voxels per head); the "global" head is uniform attention
over the tile (softmax with all-zero scores) and is evaluated as a mean.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .program import (CH_CROSS, CH_LOGI, CH_ONE, CH_POS, CH_SQ, N_BASE, Head, Layer, MLP, Program)  # noqa: F401


def embed(logI: torch.Tensor, spacing, d: int, dtype) -> torch.Tensor:
    """logI (Z,Y,X) -> tokens (d,Z,Y,X) with tile-local physical coordinates centered on the tile."""
    Z, Y, X = logI.shape
    ax = [(torch.arange(n, dtype=dtype) - (n - 1) / 2) * s for n, s in zip((Z, Y, X), spacing)]
    z, y, x = torch.meshgrid(*ax, indexing="ij")
    t = torch.zeros((d, Z, Y, X), dtype=dtype)
    t[0] = logI.to(dtype)
    t[1], t[2], t[3] = z, y, x
    t[4] = 1
    t[5], t[6], t[7] = z * z, y * y, x * x
    t[8], t[9], t[10] = z * y, z * x, y * x
    return t


def _local_attn(v, q, k, r, valid):
    """v (Z,Y,X) values, q (4,Z,Y,X), k (4,Z,Y,X); local window radius r=(rz,ry,rx) voxels."""
    Z, Y, X = v.shape
    kz, ky, kx = (2 * ri + 1 for ri in r)
    pad = (r[2], r[2], r[1], r[1], r[0], r[0])
    vp = F.pad(v, pad)
    kp = F.pad(k, pad)
    mp = F.pad(valid, pad)
    out = torch.empty_like(v)
    for z in range(Z):  # chunk over z-planes to bound memory
        kw = kp[:, z:z + kz]  # (4,kz,Yp,Xp)
        kw = kw.unfold(2, ky, 1).unfold(3, kx, 1)  # (4,kz,Y,X,ky,kx)
        kw = kw.permute(2, 3, 0, 1, 4, 5).reshape(Y, X, 4, kz * ky * kx)
        vw = vp[z:z + kz].unfold(1, ky, 1).unfold(2, kx, 1).permute(1, 2, 0, 3, 4).reshape(Y, X, -1)
        mw = mp[z:z + kz].unfold(1, ky, 1).unfold(2, kx, 1).permute(1, 2, 0, 3, 4).reshape(Y, X, -1)
        s = torch.einsum("yxa,yxan->yxn", q[:, z].permute(1, 2, 0), kw)
        s = s.masked_fill(mw < 0.5, float("-inf"))
        out[z] = (torch.softmax(s, -1) * vw).sum(-1)
    return out


def _fast_attn(v, h: Head, spacing, r):
    """Exact equivalent of _local_attn for Gaussian heads: P and delta are read back from the head's own
    Q/K weights; normalized FFT correlation == masked softmax at borders. (Tested against dense path.)"""
    P = h.Wq[:3, 1:4]
    delta = np.linalg.solve(P, h.Wq[:3, CH_ONE])
    sh = [int(round(delta[a] / spacing[a])) for a in range(3)]  # shifts are voxel-aligned by construction
    g = np.meshgrid(*[np.arange(-ri, ri + 1) * s for ri, s in zip(r, spacing)], indexing="ij")
    ker = np.exp(-0.5 * sum(P[a, b] * (g[a] - delta[a]) * (g[b] - delta[b]) for a in range(3) for b in range(3)))
    ker = torch.as_tensor(ker / ker.sum(), dtype=v.dtype)
    fs = [v.shape[i] + ker.shape[i] - 1 for i in range(3)]
    fs = [int(2 ** np.ceil(np.log2(n))) for n in fs]
    def corr(a):
        A = torch.fft.rfftn(a, s=fs)
        K = torch.fft.rfftn(torch.flip(ker, (0, 1, 2)), s=fs)
        o = torch.fft.irfftn(A * K, s=fs)
        return o[tuple(slice(r[i] * 1 + r[i] * 0, r[i] + a.shape[i]) for i in range(3))]
    num = corr(v)
    den = corr(torch.ones_like(v))
    return num / den.clamp_min(1e-12)


class WiredTransformer:
    def __init__(self, prog: Program, dtype=torch.float64, fast=True):
        self.p, self.dtype, self.fast = prog, dtype, fast

    @torch.no_grad()
    def forward(self, logI: torch.Tensor, spacing) -> dict[str, torch.Tensor]:
        d = self.p.d
        logI = torch.as_tensor(logI)
        x = embed(logI, spacing, d, self.dtype)
        valid = torch.ones(logI.shape, dtype=self.dtype)
        T = lambda a: torch.as_tensor(a, dtype=self.dtype)  # noqa: E731
        for L in self.p.layers:
            delta = torch.zeros_like(x)
            for h in L.heads:
                v = torch.einsum("c,czyx->zyx", T(h.Wv), x)
                if h.radius_um is None:
                    o = v.mean().expand_as(v)
                else:
                    # keys outside the volume do not exist, so a window larger than the volume is clipped (exact)
                    r = tuple(min(max(1, int(np.ceil(rr / s))), max(1, n - 1)) for rr, s, n in zip(h.radius_um, spacing, logI.shape))
                    if self.fast:
                        o = _fast_attn(v, h, spacing, r)
                    else:
                        q = torch.einsum("ac,czyx->azyx", T(h.Wq), x)
                        k = torch.einsum("ac,czyx->azyx", T(h.Wk), x)
                        o = _local_attn(v, q, k, r, valid)
                delta += T(h.Wo)[:, None, None, None] * o
            x = x + delta
            if L.mlp is not None:
                hid = torch.relu(torch.einsum("hc,czyx->hzyx", T(L.mlp.W1), x) + T(L.mlp.b1)[:, None, None, None])
                x = x + torch.einsum("ch,hzyx->czyx", T(L.mlp.W2), hid)
        return {k: x[c] for k, c in self.p.out.items()}

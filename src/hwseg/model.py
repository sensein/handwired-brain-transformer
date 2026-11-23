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

N_BASE = 11
CH_LOGI, CH_POS, CH_ONE, CH_SQ, CH_CROSS = 0, (1, 2, 3), 4, (5, 6, 7), (8, 9, 10)


@dataclass
class Head:
    Wq: np.ndarray  # (4, d)
    Wk: np.ndarray  # (4, d)
    Wv: np.ndarray  # (d,)
    Wo: np.ndarray  # (d,)  written into residual
    radius_um: np.ndarray | None  # (3,) physical half-extent; None => global head


@dataclass
class MLP:
    W1: np.ndarray  # (h, d)
    b1: np.ndarray  # (h,)
    W2: np.ndarray  # (d, h)


@dataclass
class Layer:
    heads: list[Head] = field(default_factory=list)
    mlp: MLP | None = None


@dataclass
class Program:
    name: str
    d: int
    layers: list[Layer]
    out: dict[str, int]  # named output channels
    meta: dict = field(default_factory=dict)

    def n_params(self) -> tuple[int, int]:
        tot = nz = 0
        for L in self.layers:
            arrs = []
            for h in L.heads:
                arrs += [h.Wq, h.Wk, h.Wv, h.Wo]
            if L.mlp:
                arrs += [L.mlp.W1, L.mlp.b1, L.mlp.W2]
            for a in arrs:
                tot += a.size
                nz += int(np.count_nonzero(a))
        return tot, nz

    def to_json(self) -> str:
        def a(x):
            return np.asarray(x, dtype=np.float64).tolist()

        def sp(x):  # sparse (index,value) for compactness
            x = np.asarray(x, dtype=np.float64)
            idx = np.flatnonzero(x)
            return {"shape": list(x.shape), "i": idx.tolist(), "v": x.reshape(-1)[idx].tolist()}

        return json.dumps(
            {
                "name": self.name, "d": self.d, "out": self.out, "meta": self.meta,
                "layers": [
                    {
                        "heads": [
                            {"Wq": sp(h.Wq), "Wk": sp(h.Wk), "Wv": sp(h.Wv), "Wo": sp(h.Wo),
                             "radius_um": None if h.radius_um is None else a(h.radius_um)}
                            for h in L.heads
                        ],
                        "mlp": None if L.mlp is None else
                        {"W1": sp(L.mlp.W1), "b1": sp(L.mlp.b1), "W2": sp(L.mlp.W2)},
                    }
                    for L in self.layers
                ],
            },
            separators=(",", ":"),
        )


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


class WiredTransformer:
    def __init__(self, prog: Program, dtype=torch.float64):
        self.p, self.dtype = prog, dtype

    @torch.no_grad()
    def forward(self, logI: torch.Tensor, spacing) -> dict[str, torch.Tensor]:
        d = self.p.d
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
                    r = tuple(max(1, int(np.ceil(rr / s))) for rr, s in zip(h.radius_um, spacing))
                    q = torch.einsum("ac,czyx->azyx", T(h.Wq), x)
                    k = torch.einsum("ac,czyx->azyx", T(h.Wk), x)
                    o = _local_attn(v, q, k, r, valid)
                delta += T(h.Wo)[:, None, None, None] * o
            x = x + delta
            if L.mlp is not None:
                hid = torch.relu(torch.einsum("hc,czyx->hzyx", T(L.mlp.W1), x) + T(L.mlp.b1)[:, None, None, None])
                x = x + torch.einsum("ch,hzyx->czyx", T(L.mlp.W2), hid)
        return {k: x[c] for k, c in self.p.out.items()}

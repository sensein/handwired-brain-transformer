"""Torch reference executor for hier1 programs (src/hwseg/hier.py, docs/hierarchical/SPEC.md).

Input: raw log(I+1) volume (Z, Y, X) and the voxel spacing (um). Output: dict of named stage-0 (or any stage) channels.
Spatial ops are evaluated separably (exact: Gaussian attention with a diagonal precision over a box window factorises;
the derivative convolutions are separable by construction). rank is evaluated with a sort + binary search (exactly the
mean of the step kernel over the key set, ties counted 1/2).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .hier import (G_LSP, G_ONE, S0_LOGI, S0_ONE, S_ONE, HierProgram, attn_matrix, conv_matrix, rank_keys, sigma_um_of,
                   stage_shapes)


def _apply_axes(v, Ms):
    """v (C, Z, Y, X); Ms = (Mz, My, Mx) dense (n_dst x n_src) matrices."""
    T = lambda a: torch.as_tensor(a, dtype=v.dtype)  # noqa: E731
    v = torch.einsum("iz,czyx->ciyx", T(Ms[0]), v)
    v = torch.einsum("jy,czyx->czjx", T(Ms[1]), v)
    return torch.einsum("kx,czyx->czyk", T(Ms[2]), v)


class HierTransformer:
    def __init__(self, prog: HierProgram, dtype=torch.float64):
        self.p, self.dtype = prog, dtype
        self._mcache = {}

    @torch.no_grad()
    def forward(self, logI, spacing, return_all=False):
        p, dt = self.p, self.dtype
        self._mcache = {}
        logI = torch.as_tensor(np.asarray(logI), dtype=dt)
        n0, sp0 = tuple(logI.shape), [float(s) for s in spacing]
        shapes = stage_shapes(n0, p.n_stages)
        x = [torch.zeros((p.d[s],) + shapes[s], dtype=dt) for s in range(p.n_stages)]
        x[0][S0_LOGI] = logI
        x[0][S0_ONE] = 1
        for s in range(1, p.n_stages):
            x[s][S_ONE] = 1
        g = torch.zeros(p.dg, dtype=dt)
        g[G_LSP] = math.log2(sp0[2])
        g[G_ONE] = 1
        for op in p.ops:
            k = op["op"]
            if k == "rank":
                v = x[op["s"]][op["src"]].reshape(-1)
                if dt == torch.float32:
                    v = v.float()
                m = rank_keys(v.numel(), op["kmax"])
                keys = torch.sort(v[::m]).values
                lt = torch.searchsorted(keys, v, right=False)
                le = torch.searchsorted(keys, v, right=True)
                u = (lt.to(dt) + 0.5 * (le - lt).to(dt)) / keys.numel()
                x[op["s"]][op["dst"]] += u.reshape(shapes[op["s"]])
            elif k == "gpool":
                xs = x[op["s"]]
                N = xs[0].numel()
                if op["score"] is None or op["beta"] == 0:
                    w = torch.full((N,), 1.0 / N, dtype=dt)
                else:
                    w = torch.softmax(op["beta"] * xs[op["score"]].reshape(-1), 0)
                for c, dc, gn in zip(op["src"], op["dst"], op["gain"]):
                    g[dc] += gn * (w * xs[c].reshape(-1)).sum()
            elif k == "mlp":
                s = op["s"]
                W1, b1, W2 = (torch.as_tensor(op[n], dtype=dt) for n in ("W1", "b1", "W2"))
                if s < 0:
                    hid = torch.relu(W1 @ g + b1)
                    g = g + W2 @ hid
                else:
                    ds = p.d[s]
                    pre = torch.einsum("hc,czyx->hzyx", W1[:, :ds], x[s]) + (W1[:, ds:] @ g + b1)[:, None, None, None]
                    x[s] = x[s] + torch.einsum("ch,hzyx->czyx", W2, torch.relu(pre))
            elif k == "attn":
                a, b = op["src_s"], op["dst_s"]
                sg = sigma_um_of(op, a, sp0)
                key = ("a", a, b, tuple(sg))
                if key not in self._mcache:
                    self._mcache[key] = [attn_matrix(n0[ax], sp0[ax], a, b, shapes[a][ax], shapes[b][ax], sg[ax]) for ax in range(3)]
                o = _apply_axes(x[a][op["src"]], self._mcache[key])
                for i, (dc, gn) in enumerate(zip(op["dst"], op["gain"])):
                    x[b][dc] += gn * o[i]
            elif k == "conv":
                s = op["s"]
                hs = [sp0[ax] * 2 ** s for ax in range(3)]
                for kk in op["kern"]:
                    sg = kk["sigma"] * (hs[2] if op.get("unit", "um") == "vox" else 1.0)
                    key = ("c", s, sg, tuple(kk["order"]))
                    if key not in self._mcache:
                        self._mcache[key] = [conv_matrix(shapes[s][ax], hs[ax], sg, kk["order"][ax]) for ax in range(3)]
                    o = _apply_axes(x[s][kk["src"]][None], self._mcache[key])[0]
                    x[s][kk["dst"]] += kk["gain"] * o
            else:
                raise ValueError(k)
        out = {name: x[sc[0]][sc[1]] if sc[0] >= 0 else g[sc[1]] for name, sc in p.out.items()}
        if return_all:
            return out, x, g
        return out

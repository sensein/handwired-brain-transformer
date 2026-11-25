"""Weight-space programming primitives: write transformer weights directly (no training).

Key identity: softmax over keys j of  -1/2 (p_j-p_i-delta)^T P (p_j-p_i-delta)  with P = Sigma^-1
equals a normalized Gaussian kernel. Expanding, the i-only terms cancel in the softmax, leaving
    score_ij = (p_i + delta)^T P p_j  -  1/2 p_j^T P p_j
which is a dot product q_i . k_j with q_i = [ P (p_i+delta) ; 1 ]  (linear in pos & one channels)
and k_j = [ p_j ; -1/2 p_j^T P p_j ] (linear in pos and pos-product channels). Pure linear Q/K.
"""
from __future__ import annotations

import numpy as np

from .program import (CH_LOGI, CH_ONE, N_BASE, Head, Layer, MLP, Program)

POS = (1, 2, 3)
QUAD = {(0, 0): 5, (1, 1): 6, (2, 2): 7, (0, 1): 8, (0, 2): 9, (1, 2): 10}


def cov_from_axes(sigma_um, direction=None):
    """Precision matrix P. Isotropic/axis-aligned if direction None; else elongated along `direction`
    with sigma_um=(sigma_long, sigma_trans)."""
    if direction is None:
        s = np.broadcast_to(np.asarray(sigma_um, float), (3,))
        return np.diag(1.0 / s**2)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    sl, st = sigma_um
    return np.eye(3) / st**2 + np.outer(d, d) * (1 / sl**2 - 1 / st**2)


class Builder:
    def __init__(self, name):
        self.name, self.d, self.layers = name, N_BASE, []
        self.out, self.meta = {}, {}
        self._free: list[int] = []

    def alloc(self, n=1):
        ch = []
        for _ in range(n):
            if self._free:
                ch.append(self._free.pop())
            else:
                ch.append(self.d)
                self.d += 1
        return ch if n > 1 else ch[0]

    def free(self, *chs):
        self._free += list(chs)

    # --- heads -------------------------------------------------------------------------
    def gaussian_head(self, d, v_ch, o_ch, P, delta=(0, 0, 0), gain=1.0, extent=3.2):
        """Head writing gain * Gaussian_P-smoothed(channel v_ch, shifted by delta) into o_ch."""
        Wq, Wk = np.zeros((4, d)), np.zeros((4, d))
        delta = np.asarray(delta, float)
        for a in range(3):
            for b in range(3):
                Wq[a, POS[b]] += P[a, b]
            Wq[a, CH_ONE] += (P @ delta)[a]
            Wk[a, POS[a]] = 1.0
        Wq[3, CH_ONE] = 1.0
        for (a, b), ch in QUAD.items():
            Wk[3, ch] = -0.5 * P[a, b] * (1.0 if a == b else 2.0)
        Wv, Wo = np.zeros(d), np.zeros(d)
        Wv[v_ch] = 1.0
        Wo[o_ch] = gain
        std = np.sqrt(np.diag(np.linalg.inv(P)))
        return Head(Wq, Wk, Wv, Wo, extent * std + np.abs(delta))

    def global_head(self, d, v_ch, o_ch, gain=1.0):
        Wv, Wo = np.zeros(d), np.zeros(d)
        Wv[v_ch] = 1.0
        Wo[o_ch] = gain
        return Head(np.zeros((4, d)), np.zeros((4, d)), Wv, Wo, None)

    def add_layer(self, heads=(), mlp=None):
        self.layers.append((list(heads), mlp))

    # --- MLP ---------------------------------------------------------------------------
    @staticmethod
    def mlp(d, units):
        """units: list of (weights{ch:w}, bias, out{ch:gain}) -> relu(sum w x + b) written to out chs."""
        h = len(units)
        W1, b1, W2 = np.zeros((h, d)), np.zeros(h), np.zeros((d, h))
        for i, (w, b, o) in enumerate(units):
            for ch, val in w.items():
                W1[i, ch] = val
            b1[i] = b
            for ch, val in o.items():
                W2[ch, i] = val
        return MLP(W1, b1, W2)

    def build(self) -> Program:
        d = self.d
        pad = lambda a: np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, d - a.shape[-1])])  # noqa: E731
        layers = []
        for heads, mlp in self.layers:
            hs = [Head(pad(h.Wq), pad(h.Wk), pad(h.Wv), pad(h.Wo), h.radius_um) for h in heads]
            m = None
            if mlp is not None:
                m = MLP(pad(mlp.W1), mlp.b1, np.pad(mlp.W2, [(0, d - mlp.W2.shape[0]), (0, 0)]))
            layers.append(Layer(hs, m))
        return Program(self.name, d, layers, dict(self.out), dict(self.meta))

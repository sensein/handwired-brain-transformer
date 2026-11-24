"""Synthetic volumes with exact ground truth (the only labels available: DANDI 000108 ships none)."""
from __future__ import annotations

import numpy as np


def make_phantom(shape=(48, 64, 64), spacing=(2.56, 3.6, 2.56), n_cells=25, n_vessels=3,
                 cell_r_um=5.0, vessel_r_um=2.5, bg=300.0, fg=2500.0, gain=1.0, seed=0):
    rng = np.random.default_rng(seed)
    Z, Y, X = shape
    zz, yy, xx = np.meshgrid(*(np.arange(n) * s for n, s in zip(shape, spacing)), indexing="ij")
    ext = np.array([Z * spacing[0], Y * spacing[1], X * spacing[2]])
    cells = np.zeros(shape, bool)
    for _ in range(n_cells):
        c = rng.uniform(0.15, 0.85, 3) * ext
        cells |= (zz - c[0])**2 + (yy - c[1])**2 + (xx - c[2])**2 <= cell_r_um**2
    ves = np.zeros(shape, bool)
    for _ in range(n_vessels):
        p0, p1 = rng.uniform(0.05, 0.95, 3) * ext, rng.uniform(0.05, 0.95, 3) * ext
        d = p1 - p0
        L2 = d @ d
        pts = np.stack([zz, yy, xx], -1) - p0
        t = np.clip((pts @ d) / L2, 0, 1)
        dist2 = ((pts - t[..., None] * d)**2).sum(-1)
        ves |= dist2 <= vessel_r_um**2
    vol = bg + (fg - bg) * (cells | ves)
    vol = vol * gain * rng.lognormal(0, 0.15, shape)   # multiplicative noise
    vol = rng.poisson(np.maximum(vol, 1)).astype(np.float32)
    return vol, cells & ~ves, ves, spacing


def make_cortex_phantom(shape=(16, 96, 56), spacing=(82.0, 116.0, 82.0), pia_y=12.0, cortex_vox=26.0, undulate=3.0,
                        fracs=(0.10, 0.20, 0.45, 0.55, 0.75, 1.0), dens=(0.30, 0.90, 0.80, 1.20, 0.70, 0.60), wm=0.25, seed=0):
    """Cortical ribbon phantom (depth along y). Labels: 0 bg, 1..6 layers I..VI, 7 white matter."""
    rng = np.random.default_rng(seed)
    Z, Y, X = shape
    y = np.arange(Y)[None, :, None] * np.ones((Z, 1, X))
    xx = np.arange(X)[None, None, :] * np.ones((Z, Y, 1))
    zz = np.arange(Z)[None, :1, None] * np.ones((Z, Y, X))
    top = pia_y + undulate * np.sin(2 * np.pi * xx / 40.0 + 0.7 * zz / 4.0)
    t = (y - top) / cortex_vox                      # 0 at pia, 1 at gray/white boundary
    lab = np.zeros(shape, int)
    lab[t >= 0] = 1
    for k, f in enumerate(fracs[:-1]):
        lab[t >= f] = k + 2
    lab[t >= 1.0] = 7
    d = np.array([0.0, *dens, wm])[lab]
    vol = 20.0 + 380.0 * d
    vol = vol * rng.lognormal(0, 0.12, shape)
    return rng.poisson(np.maximum(vol, 1)).astype(np.float32), lab, spacing

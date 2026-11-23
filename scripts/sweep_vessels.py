"""Design-time sweep for the vessel program (analytic FFT path == attention path, verified in tests)."""
import itertools, sys
import numpy as np
from scipy.signal import fftconvolve
from hwseg.phantom import make_phantom
from hwseg.wiring import cov_from_axes


def fib_dirs(n):
    i = np.arange(n) + 0.5
    z = 1 - i / n  # hemisphere
    r = np.sqrt(1 - z * z)
    ph = np.pi * (1 + 5**0.5) * i
    return np.stack([z, r * np.cos(ph), r * np.sin(ph)], 1)


def kernel(P, sp, ext=3.2):
    std = np.sqrt(np.diag(np.linalg.inv(P)))
    r = [max(1, int(np.ceil(ext * s / h))) for s, h in zip(std, sp)]
    g = np.meshgrid(*[np.arange(-ri, ri + 1) * h for ri, h in zip(r, sp)], indexing="ij")
    k = np.exp(-0.5 * sum(P[a, b] * g[a] * g[b] for a in range(3) for b in range(3)))
    return k / k.sum()


def nconv(L, k):
    """Normalized convolution == masked-softmax attention at volume borders."""
    return fftconvolve(L, k, mode="same") / np.maximum(fftconvolve(np.ones_like(L), k, mode="same"), 1e-9)


def best_f1(score, gt):
    """Oracle-threshold F1 on a grid of predicted-voxel counts relative to |gt| (0.5x..3x)."""
    order = np.sort(score.ravel())[::-1]
    best, thr = 0, None
    for m in np.linspace(0.5, 3.0, 26):
        t = order[min(int(m * gt.sum()), order.size - 1)]
        p = score >= t
        f = 2 * (p & gt).sum() / max(p.sum() + gt.sum(), 1)
        if f > best:
            best, thr = f, float(t)
    return best, thr


res = []
phs = [make_phantom(seed=s, n_cells=20, n_vessels=3, fg=800.0, cell_r_um=4.0) for s in range(3)]
for n_dirs, sl, st, lam in itertools.product([13, 26, 49], [5.0, 8.0, 12.0], [2.0, 3.0], [0.0, 0.5, 1.0]):
    f = []
    for vol, c, v, sp in phs:
        L = np.log(vol + 1)
        gt = nconv(L, kernel(cov_from_axes(st), sp))
        gw = nconv(L, kernel(cov_from_axes(3 * st), sp))
        D = np.max([nconv(L, kernel(cov_from_axes((sl, st), d), sp)) for d in fib_dirs(n_dirs)], 0)
        f.append(best_f1((1 + lam) * D - lam * gt - gw, v)[0])
    res.append((np.mean(f), n_dirs, sl, st, lam))
    print(res[-1], flush=True)
print("TOP", *sorted(res, reverse=True)[:8], sep="\n")

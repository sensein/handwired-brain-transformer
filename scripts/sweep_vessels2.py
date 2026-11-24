import numpy as np
src = open('scripts/sweep_vessels.py').read().split("res = []")[0]
exec(src)
phs = [make_phantom(seed=s, n_cells=25, n_vessels=3, fg=800.0, cell_r_um=4.0) for s in range(4)]
for n in (13, 26):
    acc = {}
    for vol, c, v, sp in phs:
        L = np.log(vol + 1)
        gw = nconv(L, kernel(cov_from_axes(6.0), sp))
        Ds = np.stack([nconv(L, kernel(cov_from_axes((8.0, 2.0), d), sp)) for d in fib_dirs(n)])
        Dmax, Dmin = Ds.max(0), Ds.min(0)
        for kap in (0.0, 0.5, 1.0, 1.5):
            acc.setdefault(("max-gw-k*(min-gw)", kap), []).append(best_f1(Dmax - gw - kap * (Dmin - gw), v)[0])
        # cell-aware: penalise where isotropic blob response is high
    for k, f in acc.items():
        print(n, k, round(float(np.mean(f)), 3), flush=True)

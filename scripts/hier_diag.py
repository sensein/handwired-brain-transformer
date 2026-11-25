"""Per-scale statistics of a curvature measure on real windows (diagnostic only: prints numbers used to choose weights).
usage: hier_diag.py MEASURE(blob|tube|sheet|dsheet|dblob) KEY ..."""
import sys
import numpy as np, torch
sys.path.insert(0, "scripts")
from hier_look import load
from hwseg import hier_programs as HP
from hwseg.hier import HB
from hwseg.hier_model import HierTransformer

meas = sys.argv[1]
S = 4
B = HB("diag", S)
fe = HP.front_end(B)
us = HP.pyramid(B, fe["u"], S)
outs = {}
for s in range(S):
    for c in (1.0, 1.6):
        H = HP.hess_vox(B, s, us[s], c)
        cm = HP.curvature_measures(B, s, H, HP.DIRS13, [meas, "atr"])
        r0, a0 = B.ch(0), B.ch(0)
        if s == 0:
            B.mlp(0, [({cm[meas]: 1}, 0, {r0: 1}), ({cm["atr"]: 1}, 0, {a0: 1})])
        else:
            B.attn(s, 0, cm[meas], r0, 0.6, unit="vox")
            B.attn(s, 0, cm["atr"], a0, 0.6, unit="vox")
        outs[f"R{s}_{c}"] = [0, r0]
        outs[f"A{s}_{c}"] = [0, a0]
B.out = outs
M = HierTransformer(B.build(), torch.float32)
for k in sys.argv[2:]:
    v, sp, _ = load(k)
    o = M.forward(v, sp)
    print(k)
    for s in range(S):
        for c in (1.0, 1.6):
            R, A = o[f"R{s}_{c}"].numpy(), o[f"A{s}_{c}"].numpy()
            q = np.quantile(R, [.5, .9, .99, .999])
            print(f"  s{s} c{c} sig_eff={HP.sigma_eff(s, c) * sp[2]:6.1f}um meanA={A.mean():.4f} R p50 {q[0]:.4f} p90 {q[1]:.4f} p99 {q[2]:.4f} p999 {q[3]:.4f}  p99/meanA {q[2] / A.mean():.2f}", flush=True)

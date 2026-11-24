"""Hand-authored weights for fable_blobs.json (nuclei / NeuN somata; stains YO, NN). No fitting, no hwseg builders.

Usage: python weights/fable/make_blobs.py [OUT.json]     (also writes OUT_debug.json exposing internal channels)

Every number below is a design choice in physical units (um) or display units (u' in [0,1]); see NOTES.md.
Pipeline (7 layers):
  L1  global head: FOV mean of u'             | MLP: |u' - mean|
  L2  global head: FOV mean abs deviation MAD  | MLP: FL = FLOOR_GAIN*relu(FLOOR_MAD - MAD)  (floor rises in flat/dark views)
  L3  Gaussian heads G_k(u') (octave sigmas), G_k(r^2) (kernel-variance probes)
      | MLP: |band_k|; scale/border gate -> PEN_k
  L4  global heads: KM_k = KAPPA_G * FOV mean |band_k| + XI * FOV mean |band_{k-1}|;  Gaussian heads: PEN_k += KAPPA_L * local mean |band_k| (sigma 3 sigma_k)
      | MLP: KM_k <- max(KM_k, TAU_LO + FL) in place
  L5  MLP: excess_k = relu(band_k - KM_k - PEN_k) -> class evidence
  L6  MLP: pairwise max of class evidence     L7  MLP: near-hard steps (seg, nucleus, soma, cluster)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fable_common import gauss_head, global_head, mlp, step, write  # noqa: E402

# ---------------------------------------------------------------- chosen numbers
SIG = [2.5, 5.0, 10.0, 20.0, 40.0, 80.0]      # octave-spaced smoothing sigmas (um); band k = G[k] - G[k+1]
EXT = [3.0, 3.0, 3.0, 3.0, 2.5, 2.5]          # attention window half-extent in sigmas
KAPPA_G = [1.0, 0.0, 0.0, 0.0, 0.0]           # global relative threshold: kappa_g * FOV-mean |band_k|
XI = 1.0                                      # cross-band coupling: thr_k += XI * FOV-mean |band_{k-1}| (a small bright object leaks into the next band)
KAPPA_L = [3.0, 3.0, 3.0, 3.0, 3.0]           # local relative threshold: kappa_l * G_{LOCW*sigma_k}(|band_k|)  (edges, crowded areas)
LOCW, EXT_LOC = 3.0, 2.0                      # local-statistics window: sigma = LOCW*sigma_k, half-extent EXT_LOC sigmas
TAU_LO = 0.005                                # absolute visibility floor (display units) in a normally exposed FOV
FLOOR_MAD, FLOOR_GAIN = 0.05, 1.0             # floor += FLOOR_GAIN*relu(FLOOR_MAD - MAD): flat / dark / washed-out views
GATE_RATIO, GATE_GAIN = 0.8, 3.0              # thr_k += GATE_GAIN*relu(GATE_RATIO - kernel_variance_k/(3 sigma_k^2)):
#                                               kills sub-voxel sigmas (variance collapses) and FOV borders (variance probe goes negative)
W_CLASS = [1.0, 2.3, 5.3, 5.3, 5.3]           # scale-selection weights (band -> evidence)
CLASS_OF_BAND = [0, 1, 2, 2, 2]               # 0 nucleus, 1 soma, 2 cluster
SHARP = 400.0

for _a in filter(None, os.environ.get("FABLE_OVR", "").split(";")):      # hand-picked variants: FABLE_OVR="KAPPA_L=[3]*5;TAU_LO=0.02"
    exec(_a)
# ---------------------------------------------------------------- channels
NB = len(SIG) - 1
UB, AD, MAD = 11, 12, 13
G = [14 + k for k in range(6)]
V = [20 + k for k in range(NB)]
ABS = [25 + k for k in range(NB)]
PEN = [30 + k for k in range(NB)]              # per-voxel penalties (gate, border, dark holes)
KM = [35 + k for k in range(NB)]               # FOV-constant threshold max(kappa*m, tau0)
EV = [40, 41, 42]
MX = [43, 44, 45]                              # max of the two *other* evidences
SEG, NUC, SOMA, CLU = 46, 47, 48, 49
FL = 50
D = 51


def build(path):
    L1 = {"heads": [global_head(D, {0: 1.0}, {UB: 1.0})],
          "mlp": mlp(D, [({0: 1.0, UB: -1.0}, 0.0, {AD: 1.0}), ({0: -1.0, UB: 1.0}, 0.0, {AD: 1.0})])}
    L2 = {"heads": [global_head(D, {AD: 1.0}, {MAD: 1.0})],
          "mlp": mlp(D, [({MAD: -FLOOR_GAIN}, FLOOR_GAIN * FLOOR_MAD, {FL: 1.0})])}
    heads3 = [gauss_head(D, np.eye(3) / s**2, {0: 1.0}, {G[k]: 1.0}, [e * s] * 3) for k, (s, e) in enumerate(zip(SIG, EXT))]
    heads3 += [gauss_head(D, np.eye(3) / SIG[k]**2, {5: 1.0, 6: 1.0, 7: 1.0}, {V[k]: 1.0}, [EXT[k] * SIG[k]] * 3) for k in range(NB)]
    units3 = []
    for k in range(NB):
        units3.append(({G[k]: 1.0, G[k + 1]: -1.0}, 0.0, {ABS[k]: 1.0}))
        units3.append(({G[k]: -1.0, G[k + 1]: 1.0}, 0.0, {ABS[k]: 1.0}))
        a = GATE_GAIN / (3.0 * SIG[k] ** 2)
        units3.append(({V[k]: -a, 5: a, 6: a, 7: a}, GATE_GAIN * GATE_RATIO, {PEN[k]: 1.0}))
    L3 = {"heads": heads3, "mlp": mlp(D, units3)}
    heads4 = [global_head(D, {ABS[k]: 1.0}, {KM[j]: w for j, w in ((k, KAPPA_G[k]), (k + 1, XI)) if j < NB and w != 0}) for k in range(NB)]
    heads4 += [gauss_head(D, np.eye(3) / (LOCW * SIG[k])**2, {ABS[k]: 1.0}, {PEN[k]: KAPPA_L[k]}, [EXT_LOC * LOCW * SIG[k]] * 3) for k in range(NB)]
    L4 = {"heads": heads4, "mlp": mlp(D, [({KM[k]: -1.0, FL: 1.0}, TAU_LO, {KM[k]: 1.0}) for k in range(NB)])}   # KM <- max(KM, floor)
    units4 = [({G[k]: 1.0, G[k + 1]: -1.0, KM[k]: -1.0, PEN[k]: -1.0}, 0.0, {EV[CLASS_OF_BAND[k]]: W_CLASS[k]}) for k in range(NB)]
    L4b = {"heads": [], "mlp": mlp(D, units4)}
    n, s, c = EV
    units5 = [({s: 1.0, c: -1.0}, 0.0, {MX[0]: 1.0}), ({c: 1.0}, 0.0, {MX[0]: 1.0}),           # max(s,c) = c + relu(s-c)  (c >= 0)
              ({n: 1.0, c: -1.0}, 0.0, {MX[1]: 1.0}), ({c: 1.0}, 0.0, {MX[1]: 1.0}),           # max(n,c)
              ({n: 1.0, s: -1.0}, 0.0, {MX[2]: 1.0}), ({s: 1.0}, 0.0, {MX[2]: 1.0})]           # max(n,s)
    L5 = {"heads": [], "mlp": mlp(D, units5)}
    units6 = (step({n: 1.0, s: 1.0, c: 1.0}, 0.0, SEG, SHARP) + step({n: 1.0, MX[0]: -1.0}, 0.0, NUC, SHARP)
              + step({s: 1.0, MX[1]: -1.0}, 0.0, SOMA, SHARP) + step({c: 1.0, MX[2]: -1.0}, 0.0, CLU, SHARP))
    L6 = {"heads": [], "mlp": mlp(D, units6)}
    out = {"seg": SEG, "nucleus": NUC, "soma": SOMA, "cluster": CLU}
    meta = dict(kind="blobs", author="fable (hand-written)", sigmas_um=SIG, min_um=2.0, max_um=1e9,
                input="display image u' in [0,1]")
    layers = [L1, L2, L3, L4, L4b, L5, L6]
    write(path, "fable_blobs", D, out, meta, layers)
    dbg = dict(out, **{f"g{k}": G[k] for k in range(6)}, **{f"thr{k}": KM[k] for k in range(NB)}, **{f"pen{k}": PEN[k] for k in range(NB)}, mad=MAD, ubar=UB)
    write(path.replace(".json", "_debug.json"), "fable_blobs_debug", D, dbg, meta, layers)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "fable_blobs.json"))

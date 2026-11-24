"""Hand-authored weights for fable_vessels_v2.json: CHEAP in-plane tube detector (LEC; levels 0-1).

No fitting, no hwseg builders. One layer of local attention, z window 1 voxel (3-plane slab), in-plane windows <= 36 um,
d = 35. Oriented Gaussians lie in the image plane (NDIR directions x 2 scales), so vessels running through the plane
are seen only as dots and are rejected like other round things.
  L1  global head: FOV mean of u'             | MLP: |u' - mean|
  L2  global head: MAD                         | MLP: floor u' at mean - FLOOR*MAD (in place)
  L3  oriented heads E[s][d]; wide isotropic head W1; in-plane variance probes | MLP: direction mean Ebar_s
      (also a 1-voxel box head BX and HP = |u' - BX|, the voxel-scale roughness = noise)
  L4  MLP: A_s = sum_d relu(E_sd - Ebar_s)
  L5  global heads: THR_s = C_NU[s] * FOV mean HP + KAPPA * mean positive deviation
      | MLP: THR_s <- max(THR_s, TAU_LO + ZETA*MAD), max(THR_s, TAU_HI - FLOOR_GAIN*MAD); scale/border gate -> THR_s;
             A_s recycled as anisotropy penalty ETA*relu(Ebar_s - W_s) + GAMMA*relu(W_s - Ebar_s) (+ ETA2 cross-scale)
  L6  MLP: E_sd <- E_sd - relu(a - LAMBDA*b) - relu(a - LAMBDA_C*c), a = E_sd - W_s - THR_s (strength),
           b = E_sd - Ebar_s - penalty (anisotropy), c = BX - E_sd (voxel at least as bright as the oriented mean: trims the footprint)
  L7  MLP: excess = relu(E_sd - W_s - THR_s) -> evidence (over dead channels)
  L8  MLP: near-hard steps seg, capillary, vessel (over dead channels)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fable_common import gauss_head, global_head, mlp, step, write  # noqa: E402

SIGT = [3.0, 6.0]                 # transverse sigmas (um): capillary scale, vessel scale
RHO = 3.0                         # elongation
EXT = 2.0                         # in-plane window half-extent in per-axis standard deviations
RZ = 2.5                          # z window half-extent (um) -> 1 voxel at levels 0 and 1
NDIR = 6                          # in-plane directions, 30 degrees apart
SIGW = 18.0                       # wide isotropic surround of the coarser scale
C_NU = [0.55, 0.4]                # threshold = C_NU * FOV mean |u' - box3(u')| (about 5 sd of E_d for white voxel noise)
TAU_LO, ZETA = 0.002, 0.3
KAPPA = [4.0, 4.0]                # additive signal-relative term: kappa * FOV&direction mean of relu(E_d - Ebar)
TAU_HI, FLOOR_GAIN = 0.05, 2.0
TAU_D, DARK_GAIN = 0.06, 1.25     # floor >= TAU_D - DARK_GAIN * FOV mean of u': a nearly black view gets a high floor
ETA, ETA2, GAMMA, LAMBDA = 0.3, 0.5, 1.0, 3.0
LAMBDA_C = 3.0                    # footprint trim: response <= LAMBDA_C * (box3(u') - E_sd): the voxel itself must be as bright as the oriented mean
FLOOR = 1.5
GATE_RATIO, GATE_GAIN = 0.85, 5.0
W_CLASS = [1.0, 1.5]
SHARP = 400.0

for _a in filter(None, os.environ.get("FABLE_OVR", "").split(";")):
    exec(_a)
NS = len(SIGT)
UB, AD, MAD = 11, 12, 13
E = [[14 + NDIR * s + k for k in range(NDIR)] for s in range(NS)]
c0 = 14 + NDIR * NS
EBAR = [c0, c0 + 1]
W1 = c0 + 2
V = [c0 + 3, c0 + 4]
A = [c0 + 5, c0 + 6]
THR = [c0 + 7, c0 + 8]
BX, HP = c0 + 9, c0 + 10
D = c0 + 11
EV = [AD, MAD]                    # recycled
SEG, CAP, VES = UB, EBAR[0], EBAR[1]


def zero(chs):
    return [({c: 1.0}, 0.0, {c: -1.0}) for c in chs]


def build(path):
    L1 = {"heads": [global_head(D, {0: 1.0}, {UB: 1.0})],
          "mlp": mlp(D, [({0: 1.0, UB: -1.0}, 0.0, {AD: 1.0}), ({0: -1.0, UB: 1.0}, 0.0, {AD: 1.0})])}
    L2 = {"heads": [global_head(D, {AD: 1.0}, {MAD: 1.0})], "mlp": mlp(D, [({UB: 1.0, MAD: -FLOOR, 0: -1.0}, 0.0, {0: 1.0})])}
    heads3, units3 = [], []
    for s, st in enumerate(SIGT):
        sl = RHO * st
        for k in range(NDIR):
            th = np.pi * k / NDIR
            dv = np.array([0.0, np.sin(th), np.cos(th)])
            P = np.eye(3) / st**2 + np.outer(dv, dv) * (1.0 / sl**2 - 1.0 / st**2)
            std = np.sqrt(np.diag(np.linalg.inv(P)))
            heads3.append(gauss_head(D, P, {0: 1.0}, {E[s][k]: 1.0}, [RZ, EXT * std[1], EXT * std[2]]))
        heads3.append(gauss_head(D, np.eye(3) / st**2, {6: 1.0, 7: 1.0}, {V[s]: 1.0}, [0.5, 3.0 * st, 3.0 * st]))
        units3.append(({E[s][k]: 1.0 / NDIR for k in range(NDIR)}, 0.0, {EBAR[s]: 1.0}))
    heads3.append(gauss_head(D, np.eye(3) / SIGW**2, {0: 1.0}, {W1: 1.0}, [RZ, EXT * SIGW, EXT * SIGW]))
    heads3.append(gauss_head(D, np.eye(3) * 1e-8, {0: 1.0}, {BX: 1.0}, [0.5, 0.5, 0.5]))
    units3 += [({0: 1.0, BX: -1.0}, 0.0, {HP: 1.0}), ({0: -1.0, BX: 1.0}, 0.0, {HP: 1.0})]
    L3 = {"heads": heads3, "mlp": mlp(D, units3)}
    L4 = {"heads": [], "mlp": mlp(D, [({E[s][k]: 1.0, EBAR[s]: -1.0}, 0.0, {A[s]: 1.0}) for s in range(NS) for k in range(NDIR)])}
    W = [EBAR[1], W1]
    heads5 = [global_head(D, {HP: 1.0}, {THR[s]: C_NU[s] for s in range(NS)})] + [global_head(D, {A[s]: 1.0}, {THR[s]: KAPPA[s] / NDIR}) for s in range(NS)]
    units5 = []
    for s in range(NS):
        units5.append(({THR[s]: -1.0, MAD: ZETA}, TAU_LO, {THR[s]: 1.0}))
        units5.append(({THR[s]: -1.0, MAD: -FLOOR_GAIN}, TAU_HI, {THR[s]: 1.0}))
        units5.append(({THR[s]: -1.0, UB: -DARK_GAIN}, TAU_D, {THR[s]: 1.0}))
        a = GATE_GAIN / (2.0 * SIGT[s] ** 2)
        units5.append(({V[s]: -a, 6: a, 7: a}, GATE_GAIN * GATE_RATIO, {THR[s]: 1.0}))
        units5.append(({EBAR[s]: 1.0, W[s]: -1.0}, 0.0, {A[s]: ETA}))
        units5.append(({EBAR[s]: -1.0, W[s]: 1.0}, 0.0, {A[s]: GAMMA}))
    units5.append(({EBAR[1]: 1.0, W1: -1.0}, 0.0, {A[0]: ETA2}))
    L5 = {"heads": heads5, "mlp": mlp(D, units5 + zero(A))}
    L6 = {"heads": [], "mlp": mlp(D, [({E[s][k]: 1.0 - LAMBDA, W[s]: -1.0, THR[s]: -1.0, EBAR[s]: LAMBDA, A[s]: LAMBDA}, 0.0, {E[s][k]: -1.0})
                                      for s in range(NS) for k in range(NDIR)]
                                     + [({E[s][k]: 1.0 + LAMBDA_C, W[s]: -1.0, THR[s]: -1.0, BX: -LAMBDA_C}, 0.0, {E[s][k]: -1.0})
                                        for s in range(NS) for k in range(NDIR)])}
    units7 = [({E[s][k]: 1.0, W[s]: -1.0, THR[s]: -1.0}, 0.0, {EV[s]: W_CLASS[s]}) for s in range(NS) for k in range(NDIR)]
    L7 = {"heads": [], "mlp": mlp(D, units7 + zero(EV))}
    c, v = EV
    units8 = step({c: 1.0, v: 1.0}, 0.0, SEG, SHARP) + step({c: 1.0, v: -1.0}, 0.0, CAP, SHARP) + step({v: 1.0, c: -1.0}, 0.0, VES, SHARP)
    L8 = {"heads": [], "mlp": mlp(D, units8 + zero([SEG, CAP, VES]))}
    meta = dict(kind="vessels", author="fable (hand-written)", sigma_t_um=SIGT, rho=RHO, n_dirs=NDIR, min_um=2.0, max_um=8.5,
                input="display image u' in [0,1]")
    write(path, "fable_vessels_v2", D, {"seg": SEG, "capillary": CAP, "vessel": VES}, meta, [L1, L2, L3, L4, L5, L6, L7, L8])


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "fable_vessels_v2.json"))

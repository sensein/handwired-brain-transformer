"""Hand-authored weights for fable_blobs_v2.json: CHEAP fine-scale nuclei / soma detector (YO, NN; levels 0-1).

No fitting, no hwseg builders. One layer of local attention only, windows <= 30 um in-plane and 5 um in z, d = 23,
so a level-0 FOV pass needs a 5-plane slab (see NOTES.md, "round 2", for the buffer computation).
  L1  global head: FOV mean of u'                       | MLP: |u' - mean|
  L2  global head: MAD;  Gaussian heads G_k(u') (sigma 3, 6, 12 um; z window 5 um);  in-plane variance probes G_k(y^2+x^2);
      a 1-voxel box head BX (flat kernel, minimum window)   | MLP: HP = |u' - BX|  (voxel-scale roughness = noise)
      and |band_k| (band k = G_k - G_{k+1})
  L3  global heads: KM_k = C_NU[k] * FOV mean HP (noise term) + KAPPA[k] * FOV mean |band_k| (keeps only blob cores, separates neighbours)
      | MLP: KM_k <- max(KM_k, TAU_LO) and max(KM_k, TAU_HI - FLOOR_GAIN*MAD) (floor rises in flat / dark views)
  L4  MLP: scale / border gate added to KM_k (sub-voxel sigma or FOV border -> off)
  L5  MLP: excess_k = relu(band_k - KM_k) -> evidence (over the dead |band| channels)
  L6  MLP: near-hard steps seg, nucleus, soma (over dead channels)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fable_common import gauss_head, global_head, mlp, step, write  # noqa: E402

SIG = [3.0, 6.0, 12.0]            # smoothing sigmas (um); band 0 = nucleus (3-6), band 1 = soma (6-12)
EXT_XY = 2.5                      # in-plane window half-extent in sigmas
RZ = 5.0                          # z window half-extent (um): 2 voxels at level 0, 1 voxel at level 1
KAPPA = [1.1, 1.2]                # signal-relative term: kappa * FOV mean |band_k|
C_NU = [0.6, 0.6]                 # threshold = C_NU * FOV mean |u' - box3(u')|; 0.75 = 4 sd of band 0 for white voxel noise at level 0
TAU_LO = 0.02                     # absolute floor (display units), normally exposed FOV
TAU_HI, FLOOR_GAIN = 0.05, 2.0
TAU_D, DARK_GAIN = 0.10, 1.25     # floor >= TAU_D - DARK_GAIN * FOV mean of u': a nearly black view gets a high floor    # floor >= TAU_HI - FLOOR_GAIN*MAD: rises when the display image is flat / dark / washed out
GATE_RATIO, GATE_GAIN = 0.93, 5.0  # thr_k += GATE_GAIN*relu(GATE_RATIO - in-plane kernel variance / (2 sigma_k^2))
W_CLASS = [1.0, 2.8]              # scale selection (nucleus vs soma)
SHARP = 400.0

for _a in filter(None, os.environ.get("FABLE_OVR", "").split(";")):
    exec(_a)
NB = len(SIG) - 1
UB, AD, MAD = 11, 12, 13
G = [14 + k for k in range(NB + 1)]
V = [14 + NB + 1 + k for k in range(NB)]
BX, HP = V[-1] + 1, V[-1] + 2
ABS = [HP + 1 + k for k in range(NB)]
KM = [ABS[-1] + 1 + k for k in range(NB)]
D = KM[-1] + 1
EV = [BX, HP]                     # recycled
SEG, NUC, SOMA = AD, KM[0], KM[1]  # recycled


def zero(chs):
    return [({c: 1.0}, 0.0, {c: -1.0}) for c in chs]


def build(path):
    L1 = {"heads": [global_head(D, {0: 1.0}, {UB: 1.0})],
          "mlp": mlp(D, [({0: 1.0, UB: -1.0}, 0.0, {AD: 1.0}), ({0: -1.0, UB: 1.0}, 0.0, {AD: 1.0})])}
    heads2 = [global_head(D, {AD: 1.0}, {MAD: 1.0})]
    heads2 += [gauss_head(D, np.eye(3) / s**2, {0: 1.0}, {G[k]: 1.0}, [min(RZ, EXT_XY * s), EXT_XY * s, EXT_XY * s]) for k, s in enumerate(SIG)]
    heads2 += [gauss_head(D, np.eye(3) / SIG[k]**2, {6: 1.0, 7: 1.0}, {V[k]: 1.0}, [0.5, 3.0 * SIG[k], 3.0 * SIG[k]]) for k in range(NB)]
    heads2.append(gauss_head(D, np.eye(3) * 1e-8, {0: 1.0}, {BX: 1.0}, [0.5, 0.5, 0.5]))
    units2 = [({0: 1.0, BX: -1.0}, 0.0, {HP: 1.0}), ({0: -1.0, BX: 1.0}, 0.0, {HP: 1.0}), ({BX: 1.0}, 0.0, {BX: -1.0})]   # HP; BX zeroed
    for k in range(NB):
        units2 += [({G[k]: 1.0, G[k + 1]: -1.0}, 0.0, {ABS[k]: 1.0}), ({G[k]: -1.0, G[k + 1]: 1.0}, 0.0, {ABS[k]: 1.0})]
    L2 = {"heads": heads2, "mlp": mlp(D, units2)}
    heads3 = [global_head(D, {HP: 1.0}, {KM[k]: C_NU[k] for k in range(NB)})] + [global_head(D, {ABS[k]: 1.0}, {KM[k]: KAPPA[k]}) for k in range(NB)]
    units3 = []
    for k in range(NB):
        units3.append(({KM[k]: -1.0}, TAU_LO, {KM[k]: 1.0}))
        units3.append(({KM[k]: -1.0, MAD: -FLOOR_GAIN}, TAU_HI, {KM[k]: 1.0}))
        units3.append(({KM[k]: -1.0, UB: -DARK_GAIN}, TAU_D, {KM[k]: 1.0}))
    L3 = {"heads": heads3, "mlp": mlp(D, units3)}
    units4 = []
    for k in range(NB):
        a = GATE_GAIN / (2.0 * SIG[k] ** 2)
        units4.append(({V[k]: -a, 6: a, 7: a}, GATE_GAIN * GATE_RATIO, {KM[k]: 1.0}))
    L4 = {"heads": [], "mlp": mlp(D, units4)}
    units5 = [({G[k]: 1.0, G[k + 1]: -1.0, KM[k]: -1.0}, 0.0, {EV[k]: W_CLASS[k]}) for k in range(NB)]
    L5 = {"heads": [], "mlp": mlp(D, units5 + zero([HP]))}
    n, s = EV
    units6 = step({n: 1.0, s: 1.0}, 0.0, SEG, SHARP) + step({n: 1.0, s: -1.0}, 0.0, NUC, SHARP) + step({s: 1.0, n: -1.0}, 0.0, SOMA, SHARP)
    L6 = {"heads": [], "mlp": mlp(D, units6 + zero([SEG, NUC, SOMA]))}
    meta = dict(kind="cells", author="fable (hand-written)", sigmas_um=SIG, min_um=2.0, max_um=8.5, input="display image u' in [0,1]")
    write(path, "fable_blobs_v2", D, {"seg": SEG, "nucleus": NUC, "soma": SOMA}, meta, [L1, L2, L3, L4, L5, L6])


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "fable_blobs_v2.json"))

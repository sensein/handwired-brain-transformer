"""Hand-authored weights for fable_vessels.json (tubes; stain LEC). No fitting, no hwseg builders.

Usage: python weights/fable/make_vessels.py [OUT.json]

Every number below is a design choice in physical units (um) or display units (u' in [0,1]); see NOTES.md.
Pipeline (9 layers):
  L1  global head: FOV mean of u'             | MLP: |u' - mean|
  L2  global head: FOV mean abs deviation MAD  | MLP: floor u' at mean - FLOOR*MAD (in place); FL = FLOOR_GAIN*relu(FLOOR_MAD - MAD)
  L3  oriented Gaussian heads E[s][d] (NS scales x NDIR directions) and isotropic G_s(r^2) variance probes
      | MLP: direction mean Ebar_s  (an isotropic average; Ebar_{s+1} doubles as the wide surround W_s of scale s)
  L4  MLP: A_s = sum_d relu(E_sd - Ebar_s)  (anisotropy energy, used for the FOV statistic)
  L5  global heads: THR_s = KAPPA * FOV&direction mean of positive deviations
      | MLP: THR_s <- max(THR_s, TAU_LO + FL + ZETA*MAD) in place; scale/border gate -> THR_s;
             A_s recycled as the anisotropy penalty PENB_s = ETA*relu(Ebar_s - W_s) + GAMMA*relu(W_s - Ebar_s) + ETA2*relu(Ebar_{s+1} - W_{s+1})
  L6  MLP: tube response = min(strength, LAMBDA * anisotropy) with
             strength   a = E_sd - W_s - THR_s                 (ridge brighter than its wide surround, above threshold)
             anisotropy b = E_sd - Ebar_s - PENB_s             (this direction beats the direction mean: tubes yes, blobs no)
           written in place as E_sd <- E_sd - relu(a - LAMBDA*b)
  L7  MLP: excess = relu(E_sd - W_s - THR_s) -> class evidence (over dead AD, MAD, FL channels)
  L8  MLP: pairwise max of class evidence (over dead THR channels)
  L9  MLP: near-hard steps seg, capillary, vessel, large (over dead Ebar channels)
Dead non-negative channels are recycled by a unit that subtracts relu(channel) in the same MLP that writes the new value.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fable_common import gauss_head, global_head, mlp, step, write  # noqa: E402

# ---------------------------------------------------------------- chosen numbers
SIGT = [3.0, 6.0, 12.0, 24.0, 48.0]          # transverse sigmas (um), octave spaced
RHO = 3.0                                    # elongation sigma_long / sigma_trans
EXT = 2.0                                    # window half-extent in per-axis standard deviations
NDIR = 12                                    # directions on the hemisphere
KAPPA = [22.0, 22.0, 22.0, 22.0, 22.0]       # relative threshold: kappa * FOV&direction mean of relu(E_d - mean_d E)
TAU_LO = 0.01                                # absolute visibility floor (display units) in a normally exposed FOV
FLOOR_MAD, FLOOR_GAIN = 0.05, 0.7            # floor += FLOOR_GAIN*relu(FLOOR_MAD - MAD): flat / dark / washed-out views
ETA = 0.3                                    # blob penalty on the anisotropy test: ETA*relu(Ebar_s - W_s)   (round things: no anisotropy)
ETA2 = 0.5                                   # cross-scale blob penalty: ETA2*relu(Ebar_{s+1} - W_{s+1}) (shell of a blob that is round one octave up)
GAMMA = 1.0                                  # off-ridge penalty on the anisotropy test: GAMMA*relu(W_s - Ebar_s) (voxel darker than surround)
LAMBDA = 3.0                                 # response = min(strength, LAMBDA*anisotropy)
FLOOR = 1.5                                  # image floored at FOVmean - FLOOR*MAD
ZETA = 0.4                                   # threshold >= ZETA * MAD of the display image (holds when clipping removes the noise statistic)
PROBE = 1.0                                  # variance probe sigma = PROBE * sigma_t
GATE_RATIO, GATE_GAIN = 0.9, 5.0             # thr_s += GATE_GAIN*relu(GATE_RATIO - probe_variance/(3 sigma^2)): sub-voxel scales, FOV borders
W_CLASS = [1.0, 3.0, 7.0, 7.0, 7.0]          # scale-selection weights
CLASS_OF_SCALE = [0, 1, 2, 2, 2]             # 0 capillary, 1 vessel, 2 large
SHARP = 400.0

for _a in filter(None, os.environ.get("FABLE_OVR", "").split(";")):      # hand-picked variants: FABLE_OVR="KAPPA=[10]*5;ETA=0.5"
    exec(_a)
# ---------------------------------------------------------------- channels
NS = len(SIGT)
UB, AD, MAD, FL = 11, 12, 13, 14
E = [[15 + NDIR * s + k for k in range(NDIR)] for s in range(NS)]
c0 = 15 + NDIR * NS
V = [c0 + s for s in range(NS)]
EBAR = [c0 + NS + s for s in range(NS)]
A = [c0 + 2 * NS + s for s in range(NS)]
THR = [c0 + 3 * NS + s for s in range(NS)]
D = c0 + 4 * NS
EV = [AD, MAD, FL]                           # recycled
MX = THR[:3]                                 # recycled
SEG, CAP, VES, LARGE = EBAR[:4]              # recycled


def directions(n):
    """n quasi-uniform unit vectors (z,y,x) on the upper hemisphere: golden-angle spiral, z = 1 - (i+1/2)/n."""
    i = np.arange(n) + 0.5
    z = 1.0 - i / n
    r = np.sqrt(1.0 - z * z)
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    return np.stack([z, r * np.cos(phi), r * np.sin(phi)], 1)


def zero(chs):
    return [({c: 1.0}, 0.0, {c: -1.0}) for c in chs]      # channel >= 0  ->  channel - relu(channel) = 0


def build(path):
    dirs = directions(NDIR)
    L1 = {"heads": [global_head(D, {0: 1.0}, {UB: 1.0})],
          "mlp": mlp(D, [({0: 1.0, UB: -1.0}, 0.0, {AD: 1.0}), ({0: -1.0, UB: 1.0}, 0.0, {AD: 1.0})])}
    L2 = {"heads": [global_head(D, {AD: 1.0}, {MAD: 1.0})],
          "mlp": mlp(D, [({UB: 1.0, MAD: -FLOOR, 0: -1.0}, 0.0, {0: 1.0}),                    # floor the image (dark holes)
                         ({MAD: -FLOOR_GAIN}, FLOOR_GAIN * FLOOR_MAD, {FL: 1.0})])}             # floor rise in flat/dark views
    heads3, units3 = [], []
    for s, st in enumerate(SIGT):
        sl = RHO * st
        for k, dv in enumerate(dirs):
            P = np.eye(3) / st**2 + np.outer(dv, dv) * (1.0 / sl**2 - 1.0 / st**2)      # long axis along dv
            std = np.sqrt(np.diag(np.linalg.inv(P)))
            heads3.append(gauss_head(D, P, {0: 1.0}, {E[s][k]: 1.0}, EXT * std))
        sp = PROBE * st
        heads3.append(gauss_head(D, np.eye(3) / sp**2, {5: 1.0, 6: 1.0, 7: 1.0}, {V[s]: 1.0}, [3.0 * sp] * 3))
        units3.append(({E[s][k]: 1.0 / NDIR for k in range(NDIR)}, 0.0, {EBAR[s]: 1.0}))        # direction mean (inputs >= 0)
    L3 = {"heads": heads3, "mlp": mlp(D, units3)}
    L4 = {"heads": [], "mlp": mlp(D, [({E[s][k]: 1.0, EBAR[s]: -1.0}, 0.0, {A[s]: 1.0}) for s in range(NS) for k in range(NDIR)])}
    W = [EBAR[s + 1] if s + 1 < NS else UB for s in range(NS)]                                 # wide surround (coarsest: FOV mean)
    heads5 = [global_head(D, {A[s]: 1.0}, {THR[s]: KAPPA[s] / NDIR}) for s in range(NS)]
    units5 = []
    for s in range(NS):
        units5.append(({THR[s]: -1.0, FL: 1.0, MAD: ZETA}, TAU_LO, {THR[s]: 1.0}))                         # THR <- max(THR, floor)
        a = GATE_GAIN / (3.0 * (PROBE * SIGT[s]) ** 2)
        units5.append(({V[s]: -a, 5: a, 6: a, 7: a}, GATE_GAIN * GATE_RATIO, {THR[s]: 1.0}))    # scale / border gate
        units5.append(({EBAR[s]: 1.0, W[s]: -1.0}, 0.0, {A[s]: ETA}))
        units5.append(({EBAR[s]: -1.0, W[s]: 1.0}, 0.0, {A[s]: GAMMA}))
        if s + 1 < NS:
            units5.append(({EBAR[s + 1]: 1.0, W[s + 1]: -1.0}, 0.0, {A[s]: ETA2}))
    L5 = {"heads": heads5, "mlp": mlp(D, units5 + zero(A))}
    units6 = [({E[s][k]: 1.0 - LAMBDA, W[s]: -1.0, THR[s]: -1.0, EBAR[s]: LAMBDA, A[s]: LAMBDA}, 0.0, {E[s][k]: -1.0})
              for s in range(NS) for k in range(NDIR)]
    L6 = {"heads": [], "mlp": mlp(D, units6)}
    units7 = [({E[s][k]: 1.0, W[s]: -1.0, THR[s]: -1.0}, 0.0, {EV[CLASS_OF_SCALE[s]]: W_CLASS[s]})
              for s in range(NS) for k in range(NDIR)]
    L6b = {"heads": [], "mlp": mlp(D, units7 + zero(EV))}
    c, v, g = EV
    units7 = [({v: 1.0, g: -1.0}, 0.0, {MX[0]: 1.0}), ({g: 1.0}, 0.0, {MX[0]: 1.0}),           # max(v,g) = g + relu(v-g)
              ({c: 1.0, g: -1.0}, 0.0, {MX[1]: 1.0}), ({g: 1.0}, 0.0, {MX[1]: 1.0}),
              ({c: 1.0, v: -1.0}, 0.0, {MX[2]: 1.0}), ({v: 1.0}, 0.0, {MX[2]: 1.0})]
    L7 = {"heads": [], "mlp": mlp(D, units7 + zero(MX))}
    units8 = (step({c: 1.0, v: 1.0, g: 1.0}, 0.0, SEG, SHARP) + step({c: 1.0, MX[0]: -1.0}, 0.0, CAP, SHARP)
              + step({v: 1.0, MX[1]: -1.0}, 0.0, VES, SHARP) + step({g: 1.0, MX[2]: -1.0}, 0.0, LARGE, SHARP))
    L8 = {"heads": [], "mlp": mlp(D, units8 + zero([SEG, CAP, VES, LARGE]))}
    out = {"seg": SEG, "capillary": CAP, "vessel": VES, "large": LARGE}
    meta = dict(kind="vessels", author="fable (hand-written)", sigma_t_um=SIGT, rho=RHO, n_dirs=NDIR, min_um=2.0, max_um=1e9,
                input="display image u' in [0,1]")
    layers = [L1, L2, L3, L4, L5, L6, L6b, L7, L8]
    write(path, "fable_vessels", D, out, meta, layers)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "fable_vessels.json"))

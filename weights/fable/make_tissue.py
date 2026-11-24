"""Hand-authored weights for fable_tissue.json (tissue vs background at coarse pyramid levels; NN and YO). No fitting.

Usage: python weights/fable/make_tissue.py [OUT.json]      (also writes OUT_debug.json exposing internal channels)
       FABLE_OVR="BETA_LO=0.3;QD=0.2" python ...           hand-picked variants

Every number is a design choice in physical units (um) or display units (u' in [0,1]); see NOTES_tissue_layers.md.
Pipeline (7 layers; `tissue_layers()` is reused by make_cortex_wm.py). R = H - L.
  L1  Gaussian head: S = G_SIG_S(u')                         | MLP: 16 ramps I_j = clamp(16 (S - (j-1)/16))   (sum_j I_j/16 = S)
  L2  16 global heads: F_j = FOV mean of I_j (soft 1-CDF)    | MLP: H = sum_j 1/16 clamp((F_j - QH)/WH + .5)   (bright level:
                                                               the display level that QH of the FOV exceeds)
  L3  MLP: J = clamp(SJ (S - RHO H))                           ("as bright as tissue could be" indicator; 1-J = dark set)
  L4  global head: FJ = FOV mean of J                        | MLP: L = sum_j 1/16 clamp((F_j - 1 + QD (1 - FJ))/WL + 1)
                                                               (dark level: the QD-quantile of the dark set = background mode)
  L5  MLP: A = clamp(SJ (S - L - BETA_LO R)), B = clamp(SJ (S - L - B_HI R));  PEN = clamp((H_MIN - H)/W_VIS) + clamp((R_MIN - (H - LMIN))/W_VIS) + clamp((LMIN - L_MAX)/W_VIS)
  L6  global heads FA, FB (dark mass DM = 1 - FA, mid-band mass MB = FA - FB)
      | MLP: T = LMIN + BETA_LO (H - LMIN) + clamp(Q - C_BI (MB - KAPPA DM), 0, Q) + BIG PEN,  Q = (L + BETA R) - (LMIN + BETA_LO (H - LMIN))
             i.e. T = L + BETA (H - L) when the FOV is bimodal (few voxels between the modes compared with the dark mode)
             and T = LMIN + BETA_LO (H - LMIN) when it is not (an all-tissue FOV has no background mode; LMIN = FOV minimum
             level, from the same soft CDF in L2)
  L7  MLP: tissue = clamp(SHARP (S - T) + .5), bg = clamp(SHARP (T - S) + .5 - GATE PEN)   (both 0 when the view is not readable)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fable_common import gauss_head, global_head, mlp, step, write  # noqa: E402

# ---------------------------------------------------------------- chosen numbers
SIG_S, EXT_S = 120.0, 2.0     # smoothing of the display image (um) and window half-extent in sigmas (<= 240 um: 1-2 voxels at L6)
NLEV = 16                     # display levels t_j = (j-1)/16
QH, WH = 0.035, 0.03          # bright level H: exceeded by 2-5 % of the FOV (tissue must fill >= ~3 % of the view)
RHO, SJ = 0.75, 30.0          # dark set = voxels darker than RHO * H
QMIN = 0.01                   # FOV minimum level LMIN: the level that all but ~1 % of the FOV exceeds
QD, WL = 0.30, 0.02           # dark level L = QD-quantile of the dark set (robust to zero padding below the background mode)
BETA, BETA_LO = 0.5, 0.25     # threshold T = L + BETA (H - L) in a bimodal FOV, L + BETA_LO (H - L) otherwise
B_HI = 0.75                   # mid band = [L + BETA_LO R, L + B_HI R]
KAPPA, C_BI = 1.0, 5.0        # bimodal when mid-band mass < KAPPA * dark mass; C_BI = slope of the switch
H_MIN = 0.55                  # absolute visibility floor: the bright level must be above this display value
R_MIN = 0.20                  # the view must span at least this display range (flat / washed-out views -> nothing)
L_MAX = 0.55                  # washed-out view: even the darkest 1 % of the FOV is brighter than this -> nothing
W_VIS, BIG = 0.03, 4.0        # ramp width of the two view gates; size of the threshold push
SHARP = 400.0                 # output step (display units^-1): hard class maps for contour drawing

for _a in filter(None, os.environ.get("FABLE_OVR", "").split(";")):
    exec(_a)
# ---------------------------------------------------------------- channels
S = 11
I = [12 + j for j in range(NLEV)]          # 12..27
F = [28 + j for j in range(NLEV)]          # 28..43
H, J, FJ, L, T, TIS, BG, PEN, A, B, FA, FB, LMIN = 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56
D_TISSUE = 57


def tissue_layers(d):
    """The seven tissue layers for a residual stream of width d (>= D_TISSUE)."""
    dl = 1.0 / NLEV
    L1 = {"heads": [gauss_head(d, np.eye(3) / SIG_S**2, {0: 1.0}, {S: 1.0}, [EXT_S * SIG_S] * 3)],
          "mlp": mlp(d, sum((step({S: 1.0}, -j * dl, I[j], float(NLEV)) for j in range(NLEV)), []))}
    L2 = {"heads": [global_head(d, {I[j]: 1.0}, {F[j]: 1.0}) for j in range(NLEV)],
          "mlp": mlp(d, sum(([({F[j]: 1 / WH}, -QH / WH + .5, {H: dl}), ({F[j]: 1 / WH}, -QH / WH - .5, {H: -dl}),
                              ({F[j]: 1 / WL}, (-1 + QMIN) / WL + 1, {LMIN: dl}), ({F[j]: 1 / WL}, (-1 + QMIN) / WL, {LMIN: -dl})] for j in range(NLEV)), []))}
    L3 = {"heads": [], "mlp": mlp(d, step({S: 1.0, H: -RHO}, 0.0, J, SJ))}
    u4 = []
    for j in range(NLEV):   # clamp((F_j - 1 + QD (1 - FJ)) / WL + 1)
        w = {F[j]: 1 / WL, FJ: -QD / WL}
        b = (-1 + QD) / WL
        u4 += [(w, b + 1, {L: dl}), (w, b, {L: -dl})]
    L4 = {"heads": [global_head(d, {J: 1.0}, {FJ: 1.0})], "mlp": mlp(d, u4)}
    u5 = step({S: 1.0, L: -(1 - BETA_LO), H: -BETA_LO}, 0.0, A, SJ) + step({S: 1.0, L: -(1 - B_HI), H: -B_HI}, 0.0, B, SJ)
    u5 += [({H: -1 / W_VIS}, H_MIN / W_VIS, {PEN: 1.0}), ({H: -1 / W_VIS}, H_MIN / W_VIS - 1, {PEN: -1.0})]
    u5 += [({H: -1 / W_VIS, LMIN: 1 / W_VIS}, R_MIN / W_VIS, {PEN: 1.0}), ({H: -1 / W_VIS, LMIN: 1 / W_VIS}, R_MIN / W_VIS - 1, {PEN: -1.0})]
    u5 += [({LMIN: 1 / W_VIS}, -L_MAX / W_VIS, {PEN: 1.0}), ({LMIN: 1 / W_VIS}, -L_MAX / W_VIS - 1, {PEN: -1.0})]
    L5 = {"heads": [], "mlp": mlp(d, u5)}
    db = BETA - BETA_LO
    x = {FA: C_BI * (1 + KAPPA), FB: -C_BI}        # C_BI (MB - KAPPA DM) = C_BI ((1 + KAPPA) FA - FB - KAPPA)
    xn = {c: -v for c, v in x.items()}
    q = {L: (1 - BETA_LO) - db, LMIN: -(1 - BETA_LO), H: db}      # Q = (1 - BETA_LO)(L - LMIN) + db (H - L): bimodal bonus
    u6 = [({LMIN: 1 - BETA_LO, H: BETA_LO, PEN: BIG}, 0.0, {T: 1.0}),
          ({**xn, **q}, C_BI * KAPPA, {T: 1.0}),          # relu(Q - C x)
          (xn, C_BI * KAPPA, {T: -1.0})]                  # - relu(-C x)
    L6 = {"heads": [global_head(d, {A: 1.0}, {FA: 1.0}), global_head(d, {B: 1.0}, {FB: 1.0})], "mlp": mlp(d, u6)}
    # tissue = clamp(SHARP (S - T) + .5);  bg = clamp(SHARP (T - S) + .5 - GATE*PEN)  (PEN in [0,3] is the view-gate state, so bg is 0 too
    # when the view is not readable: GATE > SHARP*(BIG + 1))
    gate = SHARP * (BIG + 2)
    u7 = [({S: SHARP, T: -SHARP}, .5, {TIS: 1.0}), ({S: SHARP, T: -SHARP}, -.5, {TIS: -1.0}),
          ({S: -SHARP, T: SHARP, PEN: -gate}, .5, {BG: 1.0}), ({S: -SHARP, T: SHARP, PEN: -gate}, -.5, {BG: -1.0})]
    L7 = {"heads": [], "mlp": mlp(d, u7)}
    return [L1, L2, L3, L4, L5, L6, L7]


META = dict(kind="tissue", author="fable (hand-written)", min_um=40.0, max_um=1e9, input="display image u' in [0,1]",
            stains="NN, YO (stain-agnostic)", levels="4-6")


def build(path):
    layers = tissue_layers(D_TISSUE)
    write(path, "fable_tissue", D_TISSUE, {"tissue": TIS, "bg": BG}, META, layers)
    dbg = {"tissue": TIS, "bg": BG, "S": S, "H": H, "L": L, "T": T, "FJ": FJ, "FA": FA, "FB": FB, "PEN": PEN, "LMIN": LMIN}
    write(path.replace(".json", "_debug.json"), "fable_tissue_debug", D_TISSUE, dbg, META, layers)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "fable_tissue.json"))

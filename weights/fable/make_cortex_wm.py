"""Hand-authored weights for fable_cortex_wm*.json (cortex ribbon vs white matter inside tissue). No fitting.

Usage: python weights/fable/make_cortex_wm.py [OUT.json] [--stain any|nn|yo]   (also writes OUT_debug.json)
       FABLE_OVR="SIG_D=1500;THETA=0.02" python ...                            hand-picked variants

Layers 1-7 are the tissue program (make_tissue.tissue_layers: S smoothed display image, H/L bright/dark FOV levels,
threshold T, TIS, BG, view gates), with one extra in-plane Gaussian head in layer 1 (S2 = G_SIG_V(u'), local level). Then
  L8  MLP: VAL = clamp(SV (S2 - u' - DELTA (H - LMIN)) - MV (1 - TIS))     thin dark line inside tissue (closed sulcus)
  L9  in-plane Gaussian head: PB = G_SIG_D(BG + W_VAL VAL)   background within ~SIG_D: a distance-to-pia proxy
        (straight tissue edge: PB = Phi(-x/SIG_D), equal to THETA at x = 2.05 SIG_D; a slit of width w: w phi(x/SIG_D)/SIG_D)
      | MLP: cortex = clamp(K (PB - THETA) + .5 - M (1 - TIS)),  wm = clamp(-K (PB - THETA) + .5 - M (1 - TIS))
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tissue as mt  # noqa: E402
from fable_common import gauss_head, global_head, mlp, step, write  # noqa: E402

# ---------------------------------------------------------------- chosen numbers
SIG_D = 1600.0                 # um: scale of the background-proximity Gaussian
EXT_D = 2.4                    # in-plane half-extent in sigmas (3.1 mm)
SIG_Z, RZ_D = 10.0, 1.0        # the proximity head is in-plane only: the slab is ~1 mm thick and its z faces are cut surfaces, not pia,
#                                so z neighbours must not count as "background nearby" (sigma_z 10 um: zero weight one plane away)
THETA = 0.02                   # PB threshold: straight edge -> 2.05 sigma = 3.3 mm; 0.2 mm slit -> 1.35 sigma = 2.2 mm
SIG_V, EXT_V = 300.0, 2.0      # um: local level for the dark-line detector (in-plane)
DELTA, SV, MV = 0.06, 100.0, 300.0   # a line is darker than its local level by > DELTA * FOV range; ramp; tissue gate
W_VAL = 2.0                    # a 1-voxel line (82 um at L5) counts like a ~0.16 mm slit
K = 2000.0                     # step sharpness on PB (hard class maps)
M = 1.5 * K                    # tissue gate (> K so background never becomes cortex)

for _a in filter(None, os.environ.get("FABLE_OVR", "").split(";")):
    exec(_a)
# ---------------------------------------------------------------- channels
PB, CX, WM, S2, VAL = 57, 58, 59, 60, 61
D = 62


def class_units():
    g = {PB: K, mt.TIS: M}
    gn = {PB: -K, mt.TIS: M}
    return [(g, -K * THETA + .5 - M, {CX: 1.0}), (g, -K * THETA - .5 - M, {CX: -1.0}),
            (gn, K * THETA + .5 - M, {WM: 1.0}), (gn, K * THETA - .5 - M, {WM: -1.0})]


def layers_for(d):
    """The nine cortex / white-matter layers for a residual stream of width d (>= D)."""
    layers = mt.tissue_layers(d)
    layers[0]["heads"].append(gauss_head(d, np.diag([SIG_Z**-2, SIG_V**-2, SIG_V**-2]), {0: 1.0}, {S2: 1.0}, [RZ_D, EXT_V * SIG_V, EXT_V * SIG_V]))
    wv = {S2: SV, 0: -SV, mt.H: -SV * DELTA, mt.LMIN: SV * DELTA, mt.TIS: MV}
    L8 = {"heads": [], "mlp": mlp(d, [(wv, -MV, {VAL: 1.0}), (wv, -MV - 1.0, {VAL: -1.0})])}
    L9 = {"heads": [gauss_head(d, np.diag([SIG_Z**-2, SIG_D**-2, SIG_D**-2]), {mt.BG: 1.0, VAL: W_VAL}, {PB: 1.0},
                               [RZ_D, EXT_D * SIG_D, EXT_D * SIG_D])], "mlp": mlp(d, class_units())}
    return layers + [L8, L9]


def build(path, stain="any"):
    layers = layers_for(D)
    name = "fable_cortex_wm" + ("" if stain == "any" else "_" + stain)
    meta = dict(kind="cortex_wm", author="fable (hand-written)", min_um=40.0, max_um=1e9, input="display image u' in [0,1]",
                stains={"any": "NN, YO (geometry only)", "nn": "NN", "yo": "YO"}[stain], levels="4-6")
    out = {"cortex": CX, "wm": WM, "bg": mt.BG}
    write(path, name, D, out, meta, layers)
    dbg = dict(out, tissue=mt.TIS, S=mt.S, H=mt.H, L=mt.L, T=mt.T, PB=PB, VAL=VAL, LMIN=mt.LMIN)
    write(path.replace(".json", "_debug.json"), name + "_debug", D, dbg, meta, layers)


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    stain = sys.argv[sys.argv.index("--stain") + 1] if "--stain" in sys.argv else "any"
    args = [x for x in args if x != stain]
    build(args[0] if args else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "fable_cortex_wm" + ("" if stain == "any" else "_" + stain) + ".json"), stain)

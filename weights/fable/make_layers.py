"""Hand-authored weights for fable_layers.json (EXPERIMENTAL: cortical depth bands inside the cortex ribbon). No fitting.

Usage: python weights/fable/make_layers.py [OUT.json]      (also writes OUT_debug.json)

Layers 1-9 are fable_cortex_wm (make_cortex_wm.py). Layer 9 gets two more in-plane proximity heads at smaller scales,
  PS = G_SIG_SUP(BG + W_SUP VAL),  PM = G_SIG_MID(BG + W_MID VAL)
and a tenth layer cuts the cortex mask into three bands by distance from the pial surface (open edge or closed-sulcus line):
  sup  = cortex and PS > TH            (0 .. ~1 mm from pia)
  mid  = cortex and PM > TH, not sup   (~1 .. ~1.9 mm)
  deep = cortex, the rest              (~1.9 mm .. gray/white boundary)
These are distance bands, not cytoarchitectonic layers; no labels exist to validate them.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tissue as mt  # noqa: E402
import make_cortex_wm as mc  # noqa: E402
from fable_common import gauss_head, mlp, write  # noqa: E402

# ---------------------------------------------------------------- chosen numbers
SIG_SUP, W_SUP = 500.0, 4.0     # um; line weight so that an 82 um line gives the same ~1 mm reach as an open edge
SIG_MID, W_MID = 1000.0, 3.0
TH = 0.03                       # Phi(-x/sigma) = 0.03 at x = 1.88 sigma: 0.94 mm (sup), 1.88 mm (mid)
EXT = 2.4
KS = 2000.0                     # step sharpness

for _a in filter(None, os.environ.get("FABLE_OVR", "").split(";")):
    exec(_a)
PS, PM, SUP, MID, DEEP, IS, IM = mc.D, mc.D + 1, mc.D + 2, mc.D + 3, mc.D + 4, mc.D + 5, mc.D + 6
D = mc.D + 7


def build(path):
    layers = mc.layers_for(D)
    L9 = layers[-1]
    for sg, w, ch in ((SIG_SUP, W_SUP, PS), (SIG_MID, W_MID, PM)):
        L9["heads"].append(gauss_head(D, np.diag([mc.SIG_Z**-2, sg**-2, sg**-2]), {mt.BG: 1.0, mc.VAL: w}, {ch: 1.0},
                                      [mc.RZ_D, EXT * sg, EXT * sg]))
    # layer 9 MLP is rebuilt with the two extra indicators IS = step(PS - TH), IM = step(PM - TH)
    units = mc.class_units() + [({PS: KS}, -KS * TH + .5, {IS: 1.0}), ({PS: KS}, -KS * TH - .5, {IS: -1.0}),
                                ({PM: KS}, -KS * TH + .5, {IM: 1.0}), ({PM: KS}, -KS * TH - .5, {IM: -1.0})]
    L9["mlp"] = mlp(D, units)
    # layer 10: sup = relu(CX + IS - 1); mid = relu(CX + IM - IS - 1); deep = relu(CX - IM - IS)   (IS implies IM in practice)
    L10 = {"heads": [], "mlp": mlp(D, [({mc.CX: 1.0, IS: 1.0}, -1.0, {SUP: 1.0}),
                                       ({mc.CX: 1.0, IM: 1.0, IS: -1.0}, -1.0, {MID: 1.0}),
                                       ({mc.CX: 1.0, IM: -1.0, IS: -1.0}, 0.0, {DEEP: 1.0})])}
    layers.append(L10)
    meta = dict(kind="layers", author="fable (hand-written)", experimental=True, min_um=40.0, max_um=1e9, input="display image u' in [0,1]",
                stains="NN, YO (geometry only)", levels="4-6", note="distance-from-pia bands, not validated")
    out = {"sup": SUP, "mid": MID, "deep": DEEP, "wm": mc.WM, "bg": mt.BG}
    write(path, "fable_layers", D, out, meta, layers)
    write(path.replace(".json", "_debug.json"), "fable_layers_debug", D, dict(out, cortex=mc.CX, PS=PS, PM=PM, PB=mc.PB), meta, layers)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "fable_layers.json"))

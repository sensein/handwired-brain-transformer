"""Display normalization: the model input is the *display image* u' in [0, 1] (what the viewer shows).
   u = clamp((logI - lo) / (hi - lo));  u' = clamp((u - .5) * contrast + .5 + brightness) ** gamma
(identical to demo/seg.js). With the default window this is auto-contrast; with the recommended exposure from the
`exposure` program it is the default view."""
import numpy as np


def window(v, p0=0.02, p1=0.997):
    lo, hi = np.quantile(v, [p0, p1])
    return float(lo), float(max(hi, lo + 1.0))


def display(v, lo=None, hi=None, brightness=0.0, contrast=1.0, gamma=1.0):
    if lo is None:
        lo, hi = window(v)
    u = np.clip((np.asarray(v, np.float32) - lo) / (hi - lo), 0, 1)
    u = np.clip((u - 0.5) * contrast + 0.5 + brightness, 0, 1)
    return (u ** gamma).astype(np.float32)

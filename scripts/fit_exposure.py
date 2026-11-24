"""Closed-form calibration of the exposure program: lo = gm - a*n, hi = gm + b*p, where gm = FOV mean of log(I+1),
n / p = FOV mean negative / positive deviation. a, b are the medians of (gm-lo)/n and (hi-gm)/p over real crops,
against the 2nd..99.7th percentile display window. (Two scalars; no optimizer.)"""
import json, numpy as np
bank = np.load("/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/bank.npz"); meta = json.load(open("/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/bank.json"))
ra, rb, rows = [], [], []
for m in meta:
    v = bank[m["key"]].astype(np.float32); z = v.shape[0] // 2; w = v[max(0, z - 6): z + 7]
    gm = w.mean(); n = np.maximum(gm - w, 0).mean(); p = np.maximum(w - gm, 0).mean(); lo, hi = np.quantile(w, [.02, .997])
    ra.append((gm - lo) / n); rb.append((hi - gm) / p); rows.append((m["stain"], m["level"], (gm - lo) / n, (hi - gm) / p))
print("a median %.2f (IQR %.2f-%.2f)   b median %.2f (IQR %.2f-%.2f)" % (np.median(ra), *np.quantile(ra, [.25, .75]), np.median(rb), *np.quantile(rb, [.25, .75])))
a, b = float(np.median(ra)), float(np.median(rb)); err = []
for m in meta:
    v = bank[m["key"]].astype(np.float32); z = v.shape[0] // 2; w = v[max(0, z - 6): z + 7]
    gm = w.mean(); n = np.maximum(gm - w, 0).mean(); p = np.maximum(w - gm, 0).mean(); lo, hi = np.quantile(w, [.02, .997])
    err.append(((gm - a * n - lo) / (hi - lo), (gm + b * p - hi) / (hi - lo)))
print("relative window error (lo, hi): mean abs %.3f %.3f" % tuple(np.abs(err).mean(0)))
json.dump(dict(a=a, b=b), open("/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/exposure_fit.json", "w"))

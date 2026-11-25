"""Evaluation of the hier1 programs without labels (only evaluates; never changes weights).
  robust OUT.txt   : IoU of class masks under intensity transforms of the RAW intensity vs the untransformed view, and the
                     detected fraction on unreadable views (flat / dark / saturated)
  sweep  OUT.png OUT.txt : the same field of view at pyramid levels (hbank S0..S4 for cells, B2..B6 for regions)
"""
import json, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
sys.path.insert(0, "scripts")
from hier_look import load, program
from hwseg.hier_model import HierTransformer


def transforms(logI):
    I = np.expm1(logI.astype(np.float64)).clip(0)
    imax = max(I.max(), 1.0)
    m = np.median(logI)
    q = lambda a: np.log1p(np.clip(np.round(a), 0, 65535))   # noqa: E731  (uint16 quantisation)
    return {
        "gain x0.25": q(I * 0.25), "gain x4": q(I * 4), "gamma 0.5": q(imax * (I / imax) ** 0.5), "gamma 2": q(imax * (I / imax) ** 2),
        "logc 0.5": q(np.expm1(m + 0.5 * (logI - m))), "logc 2": q(np.expm1(m + 2 * (logI - m))), "offset +200": q(I + 200),
        "FLAT": q(np.expm1(m + 0.02 * (logI - m))), "DARK": q(I * 0.002), "SAT": q(np.minimum(I, np.quantile(I, 0.3))),
    }


def masks(prog, o, zc):
    kind = prog.meta["kind"]
    if kind in ("blobs", "vessels_ms"):
        return {"seg": o["seg"][zc].numpy() > 0.5}
    return {k: o[k][zc].numpy() > 0.5 for k in ("tissue", "cortex", "wm")}


def iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


SETS = {"hier_cells": ["YO_90_L0_w0:12:36:0:192:0:192", "NN_161_L0_w0:12:36:0:192:0:192", "YO_117_L1_w1:12:36:0:192:0:192", "NN_161_L1_w0:12:36:0:192:0:192"],
        "hier_vessels": ["LEC_150_L0_w0:12:36:0:192:0:192", "LEC_56_L0_w1:12:36:0:192:0:192", "LEC_56_L1_w0:12:36:0:192:0:192"],
        "hier_regions": ["YO_90_L5_w0", "NN_161_L5_w0", "NN_86_L5_w2", "YO_117_L5_w2", "YO_117_L6_full", "NN_86_L6_full"]}


def robust(out):
    lines = []
    for pname, keys in SETS.items():
        prog = program(pname.replace("hier_", "") if pname != "hier_regions" else "regions")
        M = HierTransformer(prog, torch.float32)
        rows = {}
        for k in keys:
            v, sp, _ = load(k)
            zc = v.shape[0] // 2
            m0 = masks(prog, M.forward(v, sp), zc)
            for tn, tv in transforms(v).items():
                o = M.forward(tv.astype(np.float32), sp)
                m1 = masks(prog, o, zc)
                for c in m0:
                    if tn in ("FLAT", "DARK", "SAT"):
                        val = float(m1[c].mean())          # fraction detected (should be 0)
                    else:
                        val = iou(m0[c], m1[c]) if (m0[c].any() or m1[c].any()) else np.nan
                    rows.setdefault((c, tn), []).append(val)
                print(pname, k, tn, {c: round(float(np.nanmean(rows[(c, tn)][-1:])), 3) for c in m0}, "ok", round(float(o["ok"]), 2), flush=True)
        tns = list(transforms(np.zeros((2, 2, 2))).keys())
        lines.append(f"\n{pname} ({len(keys)} FOVs; IoU vs untransformed for monotone maps, detected fraction for FLAT/DARK/SAT)")
        lines.append("class   " + " ".join(f"{t:>11s}" for t in tns))
        for c in sorted({c for c, _ in rows}):
            vals = []
            for t in tns:
                a = np.array(rows[(c, t)], float)
                vals.append(f"{np.nanmean(a):5.2f}/{np.nanmin(a):4.2f}" if t not in ("FLAT", "DARK", "SAT") else f"{np.nanmax(a):11.3f}")
            lines.append(f"{c:7s} " + " ".join(f"{s:>11s}" for s in vals))
    txt = "\n".join(lines)
    print(txt)
    open(out, "w").write(txt + "\n")


def sweep(out_png, out_txt):
    lines, panels = [], []
    for stain, slab in (("YO", "90"), ("NN", "161"), ("YO", "117"), ("NN", "86")):
        for pname, lvls, kind in (("cells", [f"S{l}" for l in range(5)], "cells"), ("regions", [f"B{l}" for l in range(2, 7)], "regions")):
            prog = program(pname)
            M = HierTransformer(prog, torch.float32)
            row = []
            for lv in lvls:
                key = f"h:{stain}_{slab}_{lv}"
                try:
                    v, sp, meta = load(key)
                except KeyError:
                    continue
                o = M.forward(v, sp)
                zc = v.shape[0] // 2
                area_mm2 = v.shape[1] * sp[1] * v.shape[2] * sp[2] * 1e-6
                if kind == "cells":
                    seg = o["seg"][zc].numpy() > 0.5
                    lab, n = ndimage.label(seg)
                    st = f"{stain}_{slab} {lv} {sp[2]:6.1f}um: ok {float(o['ok']):.0f} objects/mm2 {n / area_mm2:7.0f} area {100 * seg.mean():5.1f}%"
                    cls = np.stack([o[c][zc].numpy() for c in ("nucleus", "soma", "cluster")]).argmax(0)
                    img = np.zeros(seg.shape + (3,))
                    for i, col in enumerate(((1, .24, .78), (1, .67, .16), (.47, .78, 1))):
                        img[seg & (cls == i)] = col
                else:
                    t, c, w = (o[k][zc].numpy() > 0.5 for k in ("tissue", "cortex", "wm"))
                    st = f"{stain}_{slab} {lv} {sp[2]:6.1f}um: ok {float(o['ok']):.0f} tissue {100 * t.mean():5.1f}% cortex {100 * c.mean():5.1f}% wm {100 * w.mean():5.1f}% (of FOV)"
                    img = np.zeros(t.shape + (3,))
                    img[t] = (.35, .78, .47)
                    img[c] = (1, .63, .16)
                    img[w] = (.24, .55, 1)
                lo, hi = np.quantile(v[zc], [0.02, 0.997])
                base = np.repeat(np.clip((v[zc] - lo) / (hi - lo + 1e-6), 0, 1)[..., None], 3, -1)
                msk = img.sum(-1) > 0
                base[msk] = 0.45 * base[msk] + 0.55 * img[msk]
                row.append((base, st, sp))
                lines.append(st)
                print(st, flush=True)
            panels.append(row)
    nc = max(len(r) for r in panels)
    fig, ax = plt.subplots(len(panels), nc, figsize=(3.4 * nc, 3.2 * len(panels)), squeeze=False)
    for r, row in enumerate(panels):
        for c in range(nc):
            ax[r, c].axis("off")
            if c < len(row):
                img, st, sp = row[c]
                ax[r, c].imshow(img, aspect=sp[1] / sp[2], interpolation="nearest")
                ax[r, c].set_title(st.split(":")[0] + "\n" + st.split(":")[1].strip()[:60], fontsize=6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=70)
    open(out_txt, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "robust":
        robust(sys.argv[2])
    else:
        sweep(sys.argv[2], sys.argv[3])

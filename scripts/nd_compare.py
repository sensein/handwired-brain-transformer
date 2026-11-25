"""Label-free agreement between our hand-written programs and the Neurodesk reference models (demo/test/nd_compare.mjs output).

  python scripts/nd_compare.py OUT_DIR FIG_DIR [--md docs/neurodesk/compare_tables.md]

No ground truth exists for DANDI 000108, so every number here is *agreement between methods*, not accuracy. Per window:
counts and densities of 3-D objects whose centroid lies in the scored core (central planes, in-plane core without the context
margin); centroid matching (Hungarian assignment, distance tolerance = half the declared object diameter, in um) -> F1 for every
pair of models; mean voxel IoU of matched objects; equivalent-diameter distributions; a chance baseline (same counts placed
uniformly at random in the scored volume). Vessels: mask area fractions, Dice and IoU. Overlays: the window's central plane with
each model's instance outlines (PNG), to be looked at."""
import argparse, json, os
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELL_MODELS = ["fable_blobs_v2", "nd_cellpose", "nd_stardist", "nd_cellfinder"]
VES_MODELS = ["fable_vessels_v2", "nd_vesselboost"]
SHORT = {"fable_blobs_v2": "blobs_v2 (ours)", "nd_cellpose": "Cellpose", "nd_stardist": "StarDist", "nd_cellfinder": "cellfinder",
         "fable_vessels_v2": "vessels_v2 (ours)", "nd_vesselboost": "VesselBoost"}
COL = {"fable_blobs_v2": (1, .2, .8), "nd_cellpose": (.2, .9, 1), "nd_stardist": (1, .85, .1), "nd_cellfinder": (.4, 1, .3),
       "fable_vessels_v2": (1, .2, .8), "nd_vesselboost": (.2, .9, 1)}


def load(out, m, name, model):
    Z, (z0, z1, y0, y1, x0, x1) = None, m["box"]
    shape = (z1 - z0, y1 - y0, x1 - x0)
    if model == "raw":
        return np.fromfile(f"{out}/{name}__raw.f32", np.float32).reshape(shape)
    return np.fromfile(f"{out}/{name}__{model}.u32", np.uint32).reshape(shape)


def objects(lab, sp, sel):
    """3-D objects: id -> (centroid um, voxel count); only objects whose centroid (voxel coords) passes sel."""
    ids = lab.ravel(); nz = ids > 0
    if not nz.any(): return {}
    zz, yy, xx = np.unravel_index(np.nonzero(nz)[0], lab.shape); v = ids[nz]
    u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    c = np.stack([np.bincount(inv, w) / cnt for w in (zz, yy, xx)], 1)
    return {int(i): (c[k] * sp, int(cnt[k])) for k, i in enumerate(u) if sel(c[k])}


def match(A, B, tol):
    if not A or not B: return []
    ka, kb = list(A), list(B)
    pa = np.stack([A[k][0] for k in ka]); pb = np.stack([B[k][0] for k in kb])
    d = np.linalg.norm(pa[:, None] - pb[None], axis=2); cost = np.where(d <= tol, d, 1e6)
    r, c = linear_sum_assignment(cost)
    return [(ka[i], kb[j], d[i, j]) for i, j in zip(r, c) if d[i, j] <= tol]


def iou(la, lb, a, b):
    ma, mb = la == a, lb == b
    return (ma & mb).sum() / max((ma | mb).sum(), 1)


def outline(lab2):
    e = np.zeros(lab2.shape, bool)
    e[:-1] |= lab2[:-1] != lab2[1:]; e[1:] |= lab2[1:] != lab2[:-1]; e[:, :-1] |= lab2[:, :-1] != lab2[:, 1:]; e[:, 1:] |= lab2[:, 1:] != lab2[:, :-1]
    return e & (lab2 > 0)


def disp(raw2):
    lo, hi = np.percentile(raw2, [1, 99.7]); return np.clip((raw2 - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("figs"); ap.add_argument("--md", default=None)
    a = ap.parse_args(); os.makedirs(a.figs, exist_ok=True)
    meta = json.load(open(f"{a.out}/meta.json")); rng = np.random.default_rng(0)
    lines, summary = [], {"cells": [], "vessels": []}
    for m in meta:
        name, sp = m["name"], np.array(m["sp"]); raw = load(a.out, m, name, "raw"); Z, H, W = raw.shape
        M, C = m["margin"], m["core"]; mz = 2 if Z >= 7 else 0; zc = Z // 2
        sel = lambda c: mz - 0.5 <= c[0] < Z - mz - 0.5 and M - 0.5 <= c[1] < M + C - 0.5 and M - 0.5 <= c[2] < M + C - 0.5
        vol_mm3 = (Z - 2 * mz) * sp[0] * C * sp[1] * C * sp[2] * 1e-9
        models = CELL_MODELS if m["task"] == "cells" else VES_MODELS
        labs = {k: load(a.out, m, name, k) for k in models if os.path.exists(f"{a.out}/{name}__{k}.u32")}
        # ---- overlay PNG: central plane, one panel per model (core box drawn)
        d2 = disp(raw[zc]); fig, ax = plt.subplots(1, 1 + len(labs), figsize=(3.2 * (1 + len(labs)), 3.2 * H * sp[1] / (W * sp[2]) + 0.4))
        ax[0].imshow(d2, cmap="gray", aspect=sp[1] / sp[2]); ax[0].set_title(f"{name} z{zc} (display)", fontsize=8)
        for i, (k, lab) in enumerate(labs.items()):
            rgb = np.stack([d2] * 3, -1)
            if m["task"] == "cells": e = outline(lab[zc]); rgb[e] = COL[k]
            else: msk = lab[zc] > 0; rgb[msk] = 0.45 * rgb[msk] + 0.55 * np.array(COL[k])
            ax[i + 1].imshow(rgb, aspect=sp[1] / sp[2]); ax[i + 1].set_title(SHORT[k], fontsize=8)
        for x in ax:
            x.add_patch(plt.Rectangle((M - .5, M - .5), C, C, fill=False, ec="w", lw=0.6, ls="--")); x.axis("off")
        fig.tight_layout(); fig.savefig(f"{a.figs}/{name}.png", dpi=110); plt.close(fig)
        if m["task"] == "cells":
            tol = 0.5 * m["diamUm"]; obs = {k: objects(lab, sp, sel) for k, lab in labs.items()}
            row = {"name": name, "stain": m["stain"], "diam": m["diamUm"], "tol": tol, "vol_mm3": vol_mm3, "n": {k: len(v) for k, v in obs.items()},
                   "sec": {k: m["models"][k]["s"] for k in labs}, "f1": {}, "iou": {}, "chance": {}, "deq": {}}
            for k, o in obs.items():
                row["deq"][k] = [float((6 * c * np.prod(sp) / np.pi) ** (1 / 3)) for _, c in o.values()]
            ks = list(obs)
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    A, B = obs[ks[i]], obs[ks[j]]; mt = match(A, B, tol); key = f"{ks[i]}|{ks[j]}"
                    row["f1"][key] = 2 * len(mt) / max(len(A) + len(B), 1)
                    row["iou"][key] = float(np.mean([iou(labs[ks[i]], labs[ks[j]], p, q) for p, q, _ in mt])) if mt else float("nan")
                    # chance: same counts, uniform random centroids in the scored volume
                    lo = np.array([mz - .5, M - .5, M - .5]) * sp; hi = np.array([Z - mz - .5, M + C - .5, M + C - .5]) * sp; f = []
                    for _ in range(20):
                        RA = {t: (rng.uniform(lo, hi), 0) for t in range(len(A))}; RB = {t: (rng.uniform(lo, hi), 0) for t in range(len(B))}
                        f.append(2 * len(match(RA, RB, tol)) / max(len(A) + len(B), 1))
                    row["chance"][key] = float(np.mean(f))
            summary["cells"].append(row)
        else:
            sc = (slice(mz, Z - mz), slice(M, M + C), slice(M, M + C)); mk = {k: lab[sc] > 0 for k, lab in labs.items()}
            row = {"name": name, "frac": {k: float(v.mean()) for k, v in mk.items()}, "sec": {k: m["models"][k]["s"] for k in labs}}
            if len(mk) == 2:
                p, q = mk[VES_MODELS[0]], mk[VES_MODELS[1]]
                row["dice"] = float(2 * (p & q).sum() / max(p.sum() + q.sum(), 1)); row["iou"] = float((p & q).sum() / max((p | q).sum(), 1))
                row["ours_in_ref"] = float((p & q).sum() / max(p.sum(), 1)); row["ref_in_ours"] = float((p & q).sum() / max(q.sum(), 1))
            summary["vessels"].append(row)
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
    # ---- markdown tables
    L = []
    if summary["cells"]:
        L.append("### Cells: counts per window (objects with centroid in the scored core) and run time\n")
        L.append("| window | tol (um) | " + " | ".join(SHORT[k] for k in CELL_MODELS) + " |"); L.append("|---|---|" + "---|" * len(CELL_MODELS))
        for r in summary["cells"]:
            L.append(f"| {r['name']} | {r['tol']:.0f} | " + " | ".join(f"{r['n'].get(k, '-')} ({r['sec'].get(k, 0):.0f} s)" for k in CELL_MODELS) + " |")
        pairs = [f"{CELL_MODELS[i]}|{CELL_MODELS[j]}" for i in range(4) for j in range(i + 1, 4)]
        L.append("\n### Cells: centroid-matching F1 (chance F1 for the same counts) and mean IoU of matched objects\n")
        L.append("| window | " + " | ".join(f"{SHORT[p.split('|')[0]]} vs {SHORT[p.split('|')[1]]}" for p in pairs) + " |"); L.append("|---|" + "---|" * len(pairs))
        for r in summary["cells"]:
            L.append(f"| {r['name']} | " + " | ".join(f"{r['f1'][p]:.2f} ({r['chance'][p]:.2f}) / {r['iou'][p]:.2f}" if p in r["f1"] else "-" for p in pairs) + " |")
        L.append("| **mean** | " + " | ".join(f"**{np.mean([r['f1'][p] for r in summary['cells'] if p in r['f1']]):.2f}** ({np.mean([r['chance'][p] for r in summary['cells'] if p in r['chance']]):.2f}) / {np.nanmean([r['iou'][p] for r in summary['cells'] if p in r['iou']]):.2f}" for p in pairs) + " |")
        L.append("\n### Cells: equivalent diameter of 3-D objects (um), median [IQR], pooled per stain\n")
        L.append("| stain | " + " | ".join(SHORT[k] for k in CELL_MODELS) + " |"); L.append("|---|" + "---|" * len(CELL_MODELS))
        for st in sorted({r["stain"] for r in summary["cells"]}):
            cells = []
            for k in CELL_MODELS:
                v = np.concatenate([r["deq"].get(k, []) for r in summary["cells"] if r["stain"] == st]) if summary["cells"] else np.array([])
                cells.append(f"{np.median(v):.1f} [{np.percentile(v, 25):.1f}-{np.percentile(v, 75):.1f}] n={len(v)}" if len(v) else "-")
            L.append(f"| {st} | " + " | ".join(cells) + " |")
    if summary["vessels"]:
        L.append("\n### Vessels (LEC): mask agreement in the scored core\n")
        L.append("| window | area ours | area VesselBoost | Dice | IoU | ours inside VB | VB inside ours | time ours / VB |"); L.append("|---|---|---|---|---|---|---|---|")
        for r in summary["vessels"]:
            L.append(f"| {r['name']} | {100 * r['frac'].get(VES_MODELS[0], 0):.1f}% | {100 * r['frac'].get(VES_MODELS[1], 0):.1f}% | {r.get('dice', float('nan')):.2f} | {r.get('iou', float('nan')):.2f} | {100 * r.get('ours_in_ref', 0):.0f}% | {100 * r.get('ref_in_ours', 0):.0f}% | {r['sec'].get(VES_MODELS[0], 0):.0f} / {r['sec'].get(VES_MODELS[1], 0):.0f} s |")
    txt = "\n".join(L) + "\n"; print(txt)
    if a.md: open(a.md, "w").write(txt)


if __name__ == "__main__":
    main()

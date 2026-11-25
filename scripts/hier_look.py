"""Look at a hier1 program on real bank windows: montage of (percentile display, rank u, score, class map) on the
centre plane. Only evaluates; never changes weights.
usage: hier_look.py PROGRAM(cells|vessels|regions|path.json) OUT.png KEY[:zlo:zhi:ylo:yhi:xlo:xhi] ...
Keys come from $G/fable/slabbank.npz (default) or bank.npz (prefix 'b:')."""
import json, os, sys, time
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hwseg import hier_programs as HP
from hwseg.hier import HierProgram
from hwseg.hier_model import HierTransformer

G = os.environ.get("G", "/orcd/scratch/bcs/002/satra/hwseg-tmp")
BANKS = {"s": f"{G}/fable/slabbank", "b": f"{G}/fable/bank"}
_cache = {}


def load(key):
    b = "s"
    if key.startswith("b:"):
        b, key = "b", key[2:]
    parts = key.split(":")
    if b not in _cache:
        _cache[b] = (np.load(BANKS[b] + ".npz"), {m["key"]: m for m in json.load(open(BANKS[b] + ".json"))})
    arr, meta = _cache[b]
    v = arr[parts[0]].astype(np.float32)
    if len(parts) > 1:
        a = [int(p) for p in parts[1:]]
        v = v[a[0]:a[1], a[2]:a[3], a[4]:a[5]]
    return v, tuple(meta[parts[0]]["spacing_um"]), meta[parts[0]]


def program(name):
    if name.endswith(".json"):
        return HierProgram.from_json(open(name).read())
    return getattr(HP, name)()


PAL = {"blobs": {"nucleus": (1, .24, .78), "soma": (1, .67, .16), "cluster": (.47, .78, 1)},
       "vessels_ms": {"capillary": (.16, .86, 1), "vessel": (.31, 1, .47), "large": (1, .9, .24)}}


def classmap(prog, o, zc):
    kind = prog.meta["kind"]
    if kind in PAL:
        seg = o["seg"][zc] > 0.5
        names = list(PAL[kind])
        st = np.stack([o[n][zc] for n in names])
        am = st.argmax(0)
        img = np.zeros(seg.shape + (3,))
        for i, n in enumerate(names):
            img[seg & (am == i)] = PAL[kind][n]
        return img, f"seg {100 * seg.mean():.1f}% " + " ".join(f"{n[:3]}{100 * (seg & (am == i)).mean():.1f}" for i, n in enumerate(names))
    cls = prog.meta["classes"]
    cols = prog.meta["colors"]
    st = np.stack([o[n][zc] for n in cls])
    am = st.argmax(0)
    on = st.max(0) > 0.5
    img = np.zeros(am.shape + (3,))
    for i, n in enumerate(cls):
        img[on & (am == i)] = np.asarray(cols[n]) / 255
    return img, " ".join(f"{n}{100 * (on & (am == i)).mean():.0f}" for i, n in enumerate(cls))


def main():
    prog = program(sys.argv[1])
    out = sys.argv[2]
    keys = sys.argv[3:]
    M = HierTransformer(prog, torch.float32)
    fig, ax = plt.subplots(len(keys), 4, figsize=(20, 5.2 * len(keys)), squeeze=False)
    for r, k in enumerate(keys):
        v, sp, meta = load(k)
        t = time.time()
        o = {n: t_.numpy() if hasattr(t_, "numpy") else t_ for n, t_ in M.forward(v, sp).items()}
        dt = time.time() - t
        zc = v.shape[0] // 2
        lo, hi = np.quantile(v[zc], [0.02, 0.997])
        asp = sp[1] / sp[2]
        ax[r, 0].imshow(np.clip((v[zc] - lo) / (hi - lo + 1e-6), 0, 1), cmap="gray", aspect=asp)
        ax[r, 0].set_title(f"{k} {sp[2]:.1f}um {v.shape} ok={float(o['ok']):.0f} {dt:.0f}s", fontsize=8)
        ax[r, 1].imshow(o["u"][zc], cmap="gray", aspect=asp, vmin=0, vmax=1)
        ax[r, 1].set_title("rank u", fontsize=8)
        img, txt = classmap(prog, o, zc)
        sc = o.get("score", o.get("depth"))
        if os.environ.get("ZOOM") and prog.meta["kind"] in PAL:   # zoomed overlay (central quarter) instead of the score
            H, W = v.shape[1:]
            ys, xs = slice(H // 4, H // 4 + H // 2), slice(W // 4, W // 4 + W // 2)
            base = np.clip((v[zc] - lo) / (hi - lo + 1e-6), 0, 1)[ys, xs]
            m = img[ys, xs].sum(-1) > 0
            comp = np.repeat(base[..., None], 3, -1)
            comp[m] = 0.45 * comp[m] + 0.55 * img[ys, xs][m]
            ax[r, 2].imshow(comp, aspect=asp, interpolation="nearest")
            ax[r, 2].set_title("zoom x2 centre", fontsize=8)
        elif sc is not None:
            ax[r, 2].imshow(sc[zc], cmap="magma", aspect=asp)
            ax[r, 2].set_title(f"score max {sc[zc].max():.3f} p99 {np.quantile(sc[zc], .99):.3f}", fontsize=8)
        base = np.repeat(np.clip((v[zc] - lo) / (hi - lo + 1e-6), 0, 1)[..., None], 3, -1); m = img.sum(-1) > 0; base[m] = 0.45 * base[m] + 0.55 * img[m]
        ax[r, 3].imshow(base, aspect=asp)
        ax[r, 3].set_title(txt, fontsize=8)
        for a in ax[r]:
            a.axis("off")
        print(k, f"{dt:.1f}s", txt, flush=True)
    fig.tight_layout()
    fig.savefig(out, dpi=int(os.environ.get('DPI', 70)))


if __name__ == "__main__":
    main()

"""Tiny helpers for hand-writing Program JSON weight files (no hwseg builders are imported or called).

Channel layout of the residual stream: 0 image u' | 1-3 z,y,x (um) | 4 one | 5-7 z^2,y^2,x^2 | 8-10 zy,zx,yx | 11.. scratch.
A Gaussian attention head with precision P (um^-2):
    q_i = [P p_i ; 1]            -> Wq[a, 1+b] = P[a,b],  Wq[3, 4] = 1
    k_j = [p_j ; -1/2 p_j^T P p_j] -> Wk[a, 1+a] = 1,  Wk[3, 5+a] = -P[a,a]/2,  Wk[3, 8|9|10] = -P[zy|zx|yx]
so that softmax_j(q_i.k_j) = normalized Gaussian exp(-1/2 (p_j-p_i)^T P (p_j-p_i)).
"""
import json

import numpy as np


def sparse(a):
    a = np.asarray(a, dtype=np.float64)
    idx = np.flatnonzero(a)
    return {"shape": list(a.shape), "i": idx.tolist(), "v": [float(f"{x:.9g}") for x in a.reshape(-1)[idx]]}


def gauss_head(d, P, v, o, radius_um):
    """v, o: {channel: weight} for the value read and the residual write."""
    P = np.asarray(P, float)
    Wq, Wk, Wv, Wo = np.zeros((4, d)), np.zeros((4, d)), np.zeros(d), np.zeros(d)
    for a in range(3):
        for b in range(3):
            Wq[a, 1 + b] = P[a, b]
        Wk[a, 1 + a] = 1.0
        Wk[3, 5 + a] = -0.5 * P[a, a]
    Wq[3, 4] = 1.0
    Wk[3, 8], Wk[3, 9], Wk[3, 10] = -P[0, 1], -P[0, 2], -P[1, 2]
    for c, w in v.items():
        Wv[c] = w
    for c, w in o.items():
        Wo[c] = w
    return {"Wq": sparse(Wq), "Wk": sparse(Wk), "Wv": sparse(Wv), "Wo": sparse(Wo),
            "radius_um": [float(f"{r:.6g}") for r in radius_um]}


def global_head(d, v, o):
    Wv, Wo = np.zeros(d), np.zeros(d)
    for c, w in v.items():
        Wv[c] = w
    for c, w in o.items():
        Wo[c] = w
    z = np.zeros((4, d))
    return {"Wq": sparse(z), "Wk": sparse(z), "Wv": sparse(Wv), "Wo": sparse(Wo), "radius_um": None}


def mlp(d, units):
    """units: list of (in {ch: w}, bias, out {ch: w});  x += W2 relu(W1 x + b1)."""
    if not units:
        return None
    W1, b1, W2 = np.zeros((len(units), d)), np.zeros(len(units)), np.zeros((d, len(units)))
    for i, (w, b, o) in enumerate(units):
        for c, val in w.items():
            W1[i, c] += val
        b1[i] = b
        for c, val in o.items():
            W2[c, i] += val
    return {"W1": sparse(W1), "b1": sparse(b1), "W2": sparse(W2)}


def step(w, bias, out, sharp):
    """clamp(sharp * (w.x + bias), 0, 1) -> out, as two ReLU units."""
    ws = {c: sharp * v for c, v in w.items()}
    return [(ws, sharp * bias, {out: 1.0}), (ws, sharp * bias - 1.0, {out: -1.0})]


def write(path, name, d, out, meta, layers):
    js = {"name": name, "d": d, "out": out, "meta": meta, "layers": layers}
    with open(path, "w") as f:
        json.dump(js, f, separators=(",", ":"))
    nz = heads = units = 0
    for L in layers:
        for h in L["heads"]:
            heads += 1
            nz += sum(len(h[k]["i"]) for k in ("Wq", "Wk", "Wv", "Wo"))
        if L["mlp"]:
            units += L["mlp"]["W1"]["shape"][0]
            nz += sum(len(L["mlp"][k]["i"]) for k in ("W1", "b1", "W2"))
    print(f"{path}: d={d} layers={len(layers)} heads={heads} relu_units={units} nonzero={nz}")

"""Weight-space-programmed 3D transformer: program format (numpy only, no torch).

Importable in the browser (Pyodide) to author / edit weights; the torch executor lives in model.py, the WebGPU executor
in demo/engine.js. Embedding channel layout (tile-local physical coordinates in um):
  0 input image | 1-3 z,y,x | 4 one | 5-7 z^2,y^2,x^2 | 8-10 zy,zx,yx | 11.. free (program-allocated)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

N_BASE = 11
CH_LOGI, CH_POS, CH_ONE, CH_SQ, CH_CROSS = 0, (1, 2, 3), 4, (5, 6, 7), (8, 9, 10)


@dataclass
class Head:
    Wq: np.ndarray  # (4, d)
    Wk: np.ndarray  # (4, d)
    Wv: np.ndarray  # (d,)
    Wo: np.ndarray  # (d,)  written into residual
    radius_um: np.ndarray | None  # (3,) physical half-extent; None => global head


@dataclass
class MLP:
    W1: np.ndarray  # (h, d)
    b1: np.ndarray  # (h,)
    W2: np.ndarray  # (d, h)


@dataclass
class Layer:
    heads: list[Head] = field(default_factory=list)
    mlp: MLP | None = None


@dataclass
class Program:
    name: str
    d: int
    layers: list[Layer]
    out: dict[str, int]  # named output channels
    meta: dict = field(default_factory=dict)

    def n_params(self) -> tuple[int, int]:
        tot = nz = 0
        for L in self.layers:
            arrs = []
            for h in L.heads:
                arrs += [h.Wq, h.Wk, h.Wv, h.Wo]
            if L.mlp:
                arrs += [L.mlp.W1, L.mlp.b1, L.mlp.W2]
            for a in arrs:
                tot += a.size
                nz += int(np.count_nonzero(a))
        return tot, nz

    @classmethod
    def from_json(cls, text) -> "Program":
        """Load a weight file written by to_json (or by hand: same schema; sparse {shape,i,v} matrices)."""
        js = json.loads(text) if isinstance(text, (str, bytes)) else text

        def dense(m):
            a = np.zeros(int(np.prod(m["shape"])))
            a[np.asarray(m["i"], dtype=int)] = m["v"]
            return a.reshape(m["shape"])

        layers = []
        for L in js["layers"]:
            heads = [Head(dense(h["Wq"]), dense(h["Wk"]), dense(h["Wv"]), dense(h["Wo"]),
                          None if h["radius_um"] is None else np.asarray(h["radius_um"], float)) for h in L["heads"]]
            m = L["mlp"]
            layers.append(Layer(heads, None if m is None else MLP(dense(m["W1"]), dense(m["b1"]), dense(m["W2"]))))
        return cls(js["name"], js["d"], layers, js["out"], js.get("meta", {}))

    def to_json(self) -> str:
        def a(x):
            return np.asarray(x, dtype=np.float64).tolist()

        def sp(x):  # sparse (index,value) for compactness
            x = np.asarray(x, dtype=np.float64)
            idx = np.flatnonzero(x)
            return {"shape": list(x.shape), "i": idx.tolist(), "v": x.reshape(-1)[idx].tolist()}

        return json.dumps(
            {
                "name": self.name, "d": self.d, "out": self.out, "meta": self.meta,
                "layers": [
                    {
                        "heads": [
                            {"Wq": sp(h.Wq), "Wk": sp(h.Wk), "Wv": sp(h.Wv), "Wo": sp(h.Wo),
                             "radius_um": None if h.radius_um is None else a(h.radius_um)}
                            for h in L.heads
                        ],
                        "mlp": None if L.mlp is None else
                        {"W1": sp(L.mlp.W1), "b1": sp(L.mlp.b1), "W2": sp(L.mlp.W2)},
                    }
                    for L in self.layers
                ],
            },
            separators=(",", ":"),
        )



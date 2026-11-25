"""Export the hier1 programs to demo/models and write golden outputs of the Python reference (float64) on small real crops,
for the JS-CPU (demo/test/hier_cpu_test.mjs) and WebGPU (demo/test/hier_gputest.mjs) parity tests.
usage: make_golden_hier.py OUT.json [MODELDIR]"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from hier_look import load
from hwseg.hier_model import HierTransformer
from hwseg.hier_programs import cells, vessels
from hwseg.hier_regions import regions

out_path = sys.argv[1]
mdir = sys.argv[2] if len(sys.argv) > 2 else "demo/models"
progs = {"hier_cells": cells(), "hier_vessels": vessels(), "hier_regions": regions()}
for k, p in progs.items():
    open(f"{mdir}/{k}.json", "w").write(p.to_json())
crops = {"hier_cells": "NN_161_L0_w0:20:28:60:100:60:100", "hier_vessels": "LEC_56_L0_w1:20:28:60:100:60:100",
         "hier_regions": "NN_161_L5_w0:21:27:60:160:150:300"}
gold = {}
for k, p in progs.items():
    v, sp, _ = load(crops[k])
    v = v.astype(np.float32)
    o = HierTransformer(p, torch.float64).forward(v.astype(np.float64), sp)
    gold[k] = dict(crop=crops[k], dims=list(v.shape), spacing=list(sp), logI=v.ravel().tolist(),
                   out={n: (float(t) if t.dim() == 0 else t.numpy().ravel().tolist()) for n, t in o.items()})
    print(k, v.shape, {n: (round(float(t), 4) if t.dim() == 0 else round(float(t.mean()), 4)) for n, t in o.items()})
json.dump(gold, open(out_path, "w"), separators=(",", ":"))

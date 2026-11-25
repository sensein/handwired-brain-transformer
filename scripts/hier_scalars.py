"""Print a hier1 program's global (FOV-level) outputs and class fractions on bank windows (diagnostic only).
usage: hier_scalars.py PROGRAM KEY ..."""
import sys
import numpy as np, torch
sys.path.insert(0, "scripts")
from hier_look import load, program
from hwseg.hier_model import HierTransformer

prog = program(sys.argv[1])
M = HierTransformer(prog, torch.float32)
for k in sys.argv[2:]:
    v, sp, _ = load(k)
    o = M.forward(v, sp)
    zc = v.shape[0] // 2
    sc = {a: round(float(t), 3) for a, t in o.items() if t.dim() == 0}
    fr = {a: round(float(t[zc].mean()), 3) for a, t in o.items() if t.dim() == 3 and a in prog.meta.get("classes", [])}
    print(k, sc, fr, flush=True)

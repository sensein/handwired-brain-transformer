import json, numpy as np, torch
from hwseg.exposure import display
from hwseg.model import WiredTransformer
from hwseg.programs import tissue
bank = np.load("/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/bank.npz")
v = bank["YO_122_L4"].astype(np.float32); u = display(v)
prog = tissue(); print("out", prog.out, "d", prog.d, "layers", len(prog.layers))
for li, L in enumerate(prog.layers): print(li, "heads", len(L.heads), "mlp units", None if L.mlp is None else L.mlp.W1.shape[0])
m = WiredTransformer(prog, torch.float32)
r = m.forward(torch.as_tensor(u), (41.0, 58.0, 41.0))
for k, t in r.items(): print(k, float(t.min()), float(t.mean()), float(t.max()))
print("u mean", u.mean())
tc = torch.stack([r["bg"], r["gray"], r["white"]]).argmax(0).numpy()
print("class fractions whole volume", [float((tc == k).mean()) for k in range(3)], "center plane", [float((tc[48] == k).mean()) for k in range(3)])
print("ch values center plane: bg", float(r["bg"][48].mean()), "gray", float(r["gray"][48].mean()))
import json
meta = [m for m in json.load(open("/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/bank.json")) if m["key"] == "YO_122_L4"][0]
print("meta spacing", meta["spacing_um"], "shape", meta["shape"], "u.shape", u.shape)

import numpy as np, torch
from hwseg.model import WiredTransformer
from hwseg.phantom import make_cortex_phantom
from hwseg.programs import layers
vol, lab, sp = make_cortex_phantom(seed=2)
prog = layers()
out = WiredTransformer(prog, torch.float32, fast=True).forward(torch.log(torch.as_tensor(vol) + 1), sp)
names = ["bg", "L1", "L2", "L3", "L4", "L5", "L6", "WM"]
pred = np.stack([out[n].numpy() for n in names]).argmax(0)
zc = lab.shape[0] // 2
cm = np.zeros((8, 8), int)
for t, p in zip(lab[zc].ravel(), pred[zc].ravel()): cm[t, p] += 1
print("rows=true (bg,L1..L6,WM) cols=pred"); print(cm)
col = 28
print("column x=%d profile (true | scores bg L1..L6 WM):" % col)
for y in range(0, 90, 6):
    print(y, lab[zc, y, col], " ".join(f"{float(out[n][zc, y, col]):.2f}" for n in names))

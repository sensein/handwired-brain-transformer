import numpy as np, sys
sys.argv=['x']
src=open('scripts/sweep_vessels.py').read().split("res = []")[0]
exec(src)
for fg in (600.0, 800.0, 2500.0):
    thr=[]
    for s in range(4):
        vol,c,v,sp=make_phantom(seed=s,n_cells=20,n_vessels=3,fg=fg,cell_r_um=4.0)
        L=np.log(vol+1)
        gw=nconv(L,kernel(cov_from_axes(6.0),sp))
        D=np.max([nconv(L,kernel(cov_from_axes((8.0,2.0),d),sp)) for d in fib_dirs(26)],0)
        f,t=best_f1(D-gw,v); thr.append((round(f,3),round(t,3)))
    print(fg, thr, flush=True)

#!/bin/bash
# run_r2.sh TAG KIND(blobs|vessels) PART "OVERRIDES"  : round-2 iteration (weights to scratch)
set -eo pipefail
R=/orcd/scratch/bcs/002/satra/hwseg-tmp/run_py.sh; W=weights/fable; O=/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/out/r2_$1; mkdir -p $O
SB=/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/slabbank
J=$O/fable_$2_v2.json
FABLE_OVR="$4" $R $W/make_$2_v2.py $J
if [ $2 = blobs ]; then ST=YO,NN; K=cells; SK=balls; SZ=6,8,10,12,16,20,25; else ST=LEC; K=vessels; SK=tubes; SZ=4,6,8,10,12,16,20; fi
case $3 in
a) $R $W/fable_check.py cost $J
   $R $W/fable_check.py phantom $J --kind $K
   $R $W/fable_check.py sizes $J --kind $SK --sizes $SZ
   [ $2 = vessels ] && $R $W/fable_check.py sizes $J --kind balls --sizes 8,12,16,24
   $R scripts/fable_eval.py $J $O/old --stains $ST --levels 0,1
   $R scripts/fable_eval.py $J $O/slab --bank $SB --stains $ST --levels 0,1;;
b) $R $W/fable_check.py robust $J --stains $ST --levels 0,1
   $R $W/fable_check.py robust $J --bank $SB --stains $ST --levels 0,1
   $R $W/fable_check.py dark $J --stains $ST --levels 0;;
esac

#!/bin/bash
# run_r2final.sh PART : round-2 final tables and montages for weights/fable/fable_{blobs,vessels}_v2.json
set -eo pipefail
R=/orcd/scratch/bcs/002/satra/hwseg-tmp/run_py.sh; W=weights/fable; O=/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/out/r2_final; mkdir -p $O
SB=/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/slabbank; B=$W/fable_blobs_v2.json; V=$W/fable_vessels_v2.json
case $1 in
gen) $R $W/make_blobs_v2.py $B; $R $W/make_vessels_v2.py $V; $R $W/fable_check.py cost $B; $R $W/fable_check.py cost $V;;
b1) $R scripts/fable_eval.py $B $O/blobs_old --stains YO,NN --levels 0,1 --phantom --progression
    $R scripts/fable_eval.py $B $O/blobs_slab --bank $SB --stains YO,NN --levels 0,1
    $R $W/fable_check.py sizes $B --kind balls --sizes 6,8,10,12,16,20,25;;
b2) $R $W/fable_check.py robust $B --stains YO,NN --levels 0,1; $R $W/fable_check.py robust $B --bank $SB --stains YO,NN --levels 0,1;;
v1) $R scripts/fable_eval.py $V $O/vessels_old --stains LEC --levels 0,1 --phantom
    $R scripts/fable_eval.py $V $O/vessels_slab --bank $SB --stains LEC --levels 0,1
    $R scripts/fable_eval.py $V $O/vessels_on_cells --bank $SB --stains YO,NN --levels 0
    $R $W/fable_check.py phantom $V --kind vessels
    $R $W/fable_check.py sizes $V --kind tubes --sizes 4,6,8,10,12,16,20
    $R $W/fable_check.py sizes $V --kind balls --sizes 8,12,16,24;;
v2) $R $W/fable_check.py robust $V --stains LEC --levels 0,1; $R $W/fable_check.py robust $V --bank $SB --stains LEC --levels 0,1
    $R $W/fable_check.py dark $V --stains LEC --levels 0;;
bv) $W/run_r2views.sh $B $O/blobs_views YO,NN;;
vv) $W/run_r2views.sh $V $O/vessels_views LEC;;
esac

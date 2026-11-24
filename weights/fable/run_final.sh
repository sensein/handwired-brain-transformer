#!/bin/bash
# run_final.sh PART : final evaluation jobs (outputs under $O)
set -eo pipefail
R=/orcd/scratch/bcs/002/satra/hwseg-tmp/run_py.sh
W=weights/fable
O=/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/out/final
mkdir -p $O
B=$W/fable_blobs.json; V=$W/fable_vessels.json
case $1 in
gen) $R $W/make_blobs.py $B; $R $W/make_vessels.py $V; rm -f $W/*_debug.json;;
b1) $R scripts/fable_eval.py $B $O/blobs_default --stains YO,NN --levels 0,2,4 --phantom --progression;;
b2) $R $W/fable_check.py robust $B --stains YO,NN --levels 0,2,4
    $R scripts/fable_eval.py $B $O/blobs_c2 --stains YO,NN --levels 0,2 --c 2
    $R scripts/fable_eval.py $B $O/blobs_c05 --stains YO,NN --levels 0,2 --c 0.5
    $R scripts/fable_eval.py $B $O/blobs_g2 --stains YO,NN --levels 0,2 --g 2
    $R scripts/fable_eval.py $B $O/blobs_dark --stains YO,NN --levels 0,2 --b -0.5 --c 0.5
    $R scripts/fable_eval.py $B $O/blobs_L13 --stains YO,NN --levels 1,3
    $R $W/fable_check.py phantom $B --kind cells;;
v1) $R scripts/fable_eval.py $V $O/vessels_default --stains LEC --levels 0,2,4 --phantom
    $R scripts/fable_eval.py $V $O/vessels_L13 --stains LEC --levels 1,3;;
v2) $R scripts/fable_eval.py $V $O/vessels_c2 --stains LEC --levels 0,2 --c 2
    $R scripts/fable_eval.py $V $O/vessels_c05 --stains LEC --levels 0,2 --c 0.5
    $R scripts/fable_eval.py $V $O/vessels_g2 --stains LEC --levels 0,2 --g 2
    $R scripts/fable_eval.py $V $O/vessels_on_cells --stains YO,NN --levels 0
    $R $W/fable_check.py phantom $V --kind vessels;;
v3) $R $W/fable_check.py robust $V --stains LEC --levels 0,2,4;;
v4) $R $W/fable_check.py dark $V --stains LEC --levels 0;;
esac

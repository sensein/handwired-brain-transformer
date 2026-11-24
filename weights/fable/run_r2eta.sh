#!/bin/bash
set -eo pipefail
R=/orcd/scratch/bcs/002/satra/hwseg-tmp/run_py.sh; W=weights/fable; O=/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/out/r2_10; mkdir -p $O; SB=/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/slabbank
FABLE_OVR="$1" $R $W/make_vessels_v2.py $O/fable_vessels_v2.json
$R scripts/fable_eval.py $O/fable_vessels_v2.json $O/on_cells --bank $SB --stains NN --levels 0
$R scripts/fable_eval.py $O/fable_vessels_v2.json $O/slab --bank $SB --stains LEC --levels 0,1
$R $W/fable_check.py phantom $O/fable_vessels_v2.json --kind vessels
$R $W/fable_check.py sizes $O/fable_vessels_v2.json --kind tubes --sizes 6,10,16
$R $W/fable_check.py sizes $O/fable_vessels_v2.json --kind balls --sizes 12,16,24

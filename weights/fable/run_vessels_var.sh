#!/bin/bash
# usage: run_vessels_var.sh TAG "OVERRIDES"   -> quick variant check (phantom P/R + sizes + montage), weights written to scratch only
set -eo pipefail
R=/orcd/scratch/bcs/002/satra/hwseg-tmp/run_py.sh
W=weights/fable
O=/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/out/$1
mkdir -p $O
FABLE_OVR="$2" $R $W/make_vessels.py $O/fable_vessels.json
$R $W/fable_check.py phantom $O/fable_vessels.json --kind vessels
$R $W/fable_check.py sizes $O/fable_vessels.json --kind tubes --sizes 6,8,10,12,16,20,25,32
$R $W/fable_check.py sizes $O/fable_vessels.json --kind balls --sizes 8,16,32,50
$R scripts/fable_eval.py $O/fable_vessels.json $O/d --stains LEC --levels 0,2,4

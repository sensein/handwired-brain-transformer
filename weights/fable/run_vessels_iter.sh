#!/bin/bash
# usage: run_vessels_iter.sh TAG [quick]
set -eo pipefail
R=/orcd/scratch/bcs/002/satra/hwseg-tmp/run_py.sh
W=weights/fable
O=/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/out/$1
mkdir -p $O
$R $W/make_vessels.py $W/fable_vessels.json
cp $W/fable_vessels.json $O/fable_vessels_$1.json
date
$R $W/fable_check.py stats $W/fable_vessels_debug.json --stains LEC --levels 0,2,4
date
$R $W/fable_check.py sizes $W/fable_vessels.json --kind tubes
$R $W/fable_check.py sizes $W/fable_vessels.json --kind balls --sizes 8,16,32,64
date
$R scripts/fable_eval.py $W/fable_vessels.json $O/d --stains LEC --levels 0,2,4 --phantom
date
if [ "$2" != quick ]; then
$R $W/fable_check.py robust $W/fable_vessels.json --stains LEC --levels 0,2,4
fi
date

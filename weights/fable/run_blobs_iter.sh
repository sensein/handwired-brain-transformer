#!/bin/bash
# usage: run_blobs_iter.sh TAG PART   (PART = a: stats/sizes/phantom/montages, b: robustness + progression)
set -eo pipefail
R=/orcd/scratch/bcs/002/satra/hwseg-tmp/run_py.sh
W=weights/fable
O=/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/out/$1
mkdir -p $O
if [ "$2" = a ]; then
FABLE_OVR="$3" $R $W/make_blobs.py $O/fable_blobs.json
$R $W/fable_check.py stats $O/fable_blobs_debug.json --stains YO,NN --levels 0,2,4 | grep -v "band[0-9]"
$R $W/fable_check.py sizes $O/fable_blobs.json --kind balls --sizes 6,10,12,16,20,25,40,64
$R $W/fable_check.py phantom $O/fable_blobs.json --kind cells
$R scripts/fable_eval.py $O/fable_blobs.json $O/d --stains YO,NN --levels 0,2,4
$R scripts/fable_eval.py $O/fable_blobs.json $O/c2 --stains YO,NN --levels 0,2 --c 2.0
else
$R $W/fable_check.py robust $O/fable_blobs.json --stains YO,NN --levels 0,2,4
$R $W/fable_check.py dark $O/fable_blobs.json --stains YO,NN --levels 0
fi

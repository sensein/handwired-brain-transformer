#!/bin/bash
set -eo pipefail
R=/orcd/scratch/bcs/002/satra/hwseg-tmp/run_py.sh
FABLE_OVR="$1" $R weights/fable/make_exposure.py weights/fable/fable_exposure.json
$R weights/fable/fable_check.py exposure weights/fable/fable_exposure.json --levels 0,1,2,3,4,5

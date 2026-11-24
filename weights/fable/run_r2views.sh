#!/bin/bash
# run_r2views.sh JSON OUTDIR STAINS : adjusted-view montages on both banks
set -eo pipefail
R=/orcd/scratch/bcs/002/satra/hwseg-tmp/run_py.sh; SB=/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/slabbank; J=$1; O=$2; ST=$3
$R scripts/fable_eval.py $J $O/slab_c2 --bank $SB --stains $ST --levels 0 --c 2
$R scripts/fable_eval.py $J $O/slab_g2 --bank $SB --stains $ST --levels 0 --g 2
$R scripts/fable_eval.py $J $O/slab_c05 --bank $SB --stains $ST --levels 0 --c 0.5
$R scripts/fable_eval.py $J $O/slab_dark --bank $SB --stains $ST --levels 0 --b -0.5 --c 0.5
$R scripts/fable_eval.py $J $O/slab_sat --bank $SB --stains $ST --levels 0 --b 0.5
$R scripts/fable_eval.py $J $O/old_L2 --stains $ST --levels 2,3

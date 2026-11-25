#!/bin/bash
# hier_sb.sh NAME MINUTES CMD...  -> submit a Slurm job from this checkout (quicktest <= 15 min, else preemptable), wait, tail the log
G=${G:-/orcd/scratch/bcs/002/satra/hwseg-tmp}
L=$G/hier/logs; mkdir -p $L
ROOT=$(cd "$(dirname "$0")/.." && pwd)
name=$1; mins=$2; shift 2
part=mit_quicktest; [ "$mins" -gt 15 ] && part=mit_preemptable
env -i HOME=$HOME PATH=$PATH USER=$USER sbatch --wait -p $part -t $mins -c ${CPUS:-16} --mem ${MEM:-64G} -J hier-$name -o $L/$name.log --export=NONE \
  --wrap "cd $ROOT && export PYTHONPATH=$ROOT/src OMP_NUM_THREADS=${CPUS:-16} G=$G && $*" > $L/$name.sub 2>&1
echo "exit $? $(cat $L/$name.sub)"; tail -n ${TAILN:-60} $L/$name.log

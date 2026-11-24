#!/bin/bash
# sb.sh LOGNAME MINUTES CMD...   -> submit a fresh Slurm job (not a step of the current one), wait, print the log tail
L=/orcd/scratch/bcs/002/satra/hwseg-tmp/fable/out/logs; mkdir -p $L
name=$1; mins=$2; shift 2
part=mit_quicktest; [ "$mins" -gt 15 ] && part=mit_preemptable
env -i HOME=$HOME PATH=$PATH USER=$USER sbatch --wait -p $part -t $mins -c 8 --mem 24G -J fable-$name -o $L/$name.log --export=NONE --wrap "cd /orcd/scratch/orcd/013/satra/handwired-brain-transformer && export OMP_NUM_THREADS=8 && $*" > $L/$name.sub 2>&1
echo "exit $? $(cat $L/$name.sub)"; tail -n 200 $L/$name.log

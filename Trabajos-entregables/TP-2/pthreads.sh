#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --partition=XeonPHI
#SBATCH -o ./output-pthreads.txt
#SBATCH -e ./errors-pthreads.txt
source $ONEAPI_PATH/setvars.sh > /dev/null 2>&1
icc -O2 -pthread matrices-pthreads.c -o mp

sizes=(512 1024 2048 4096)
threads=(2 4 8)
block_size=64

for t in "${threads[@]}"; do
    for n in "${sizes[@]}"; do
        ./mp -n $n -b $block_size -t $t
    done
done

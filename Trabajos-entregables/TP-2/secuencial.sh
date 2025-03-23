#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --partition=XeonPHI
#SBATCH -o ./output-secuencial.txt
#SBATCH -e ./errors.txt
source $ONEAPI_PATH/setvars.sh > /dev/null 2>&1
icc -O3 -o msecuencial matrices-secuencial.c

sizes=(512 1024 2048 4096)
block_sizes=(64)

for b in "${block_sizes[@]}"; do
    for n in "${sizes[@]}"; do
        ./msecuencial -n $n -b $b
    done
done

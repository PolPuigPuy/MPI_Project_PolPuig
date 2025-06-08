#!/bin/bash
#SBATCH -N 1
##SBATCH -n 4
#SBATCH --exclusive
#SBATCH --distribution=cyclic
##SBATCH --partition=cuda-ext.q # Aolin
#SBATCH --partition=nodo.q # Wilma
#SBATCH --output=out_sbatch.txt
#SBATCH --error=err_sbatch.txt

module add gcc
module add openmpi

mpicc -Ofast laplace.c -o exec -lm
#mpirun -n 4 ./exec

## UTILITIES:
#Run multiple times to check scalability:
for i in {2,4,6,8,10}; do
    mpirun -n $i ./exec
done

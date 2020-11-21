#!/bin/sh
#SBATCH --job-name param2
#SBATCH --ntasks-per-node 24
#SBATCH --partition shas
#SBATCH --mem-per-cpu 4GB
#SBATCH --nodes 7
#SBATCH --time 24:00:00

module purge
module load intel
module load impi
module load loadbalance
mpirun lb cmd_lines

#!/bin/sh
#SBATCH --job-name param2
#SBATCH --partition shas
#SBATCH --nodes 7
#SBATCH --time 24:00:00

module purge
module load intel
module load impi
module load loadbalance
mpirun lb cmd_lines

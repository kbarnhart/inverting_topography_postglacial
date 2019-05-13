#!/bin/sh
#SBATCH --job-name queso_stops
#SBATCH --ntasks-per-node 24
#SBATCH --partition shas
#SBATCH --mem-per-cpu 4GB
#SBATCH --nodes 1
#SBATCH --time 12:00:00
#SBATCH --account ucb19_summit1

module purge
module load intel
module load impi
module load loadbalance

# make sure environment variables are set correctly
source ~/.bash_profile

mpirun lb cmnd_lines



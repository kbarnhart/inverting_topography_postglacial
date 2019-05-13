#!/bin/sh
#SBATCH --job-name dream_pce
#SBATCH --ntasks-per-node 24
#SBATCH --partition shas
#SBATCH --mem-per-cpu 4GB
#SBATCH --nodes 1
#SBATCH --time 24:00:00
#SBATCH --account ucb19_summit1

# load environment modules
module load intel/16.0.3
module load openmpi/1.10.2
module load cmake/3.5.2
#module load perl
module load mkl
module load gsl

# make sure environment variables are set correctly
source ~/.bash_profile
## run dakota using a restart file if it exists.
if [ -e dakota.rst ]
then
dakota -i dakota_mcmc.in -o dakota_mcmc.out --read_restart dakota.rst --write_restart dakota.rst
else
dakota -i dakota_mcmc.in -o dakota_mcmc.out --write_restart dakota.rst
fi

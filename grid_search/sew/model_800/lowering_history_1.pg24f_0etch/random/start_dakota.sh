#!/bin/sh
#SBATCH --job-name sew_WV_GRID
#SBATCH --ntasks-per-node 24
#SBATCH --partition shas
#SBATCH --nodes 2
#SBATCH --time 24:00:00
#SBATCH --account ucb19_summit1

# load environment modules
module purge
module load intel/16.0.3
module load openmpi/1.10.2
module load mkl/11.3.3
module load cmake/3.5.2
module load gsl/2.1

# make sure environment variables are set correctly
source ~/.bash_profile

## run dakota using a restart file if it exists.
#if [ -e dakota.rst ]
#then
#dakota -i dakota_grid.in -o dakota_grid.out --read_restart dakota.rst &> grid.log
#else
dakota -i dakota_random.in -o dakota_random.out &> random.log
#fi


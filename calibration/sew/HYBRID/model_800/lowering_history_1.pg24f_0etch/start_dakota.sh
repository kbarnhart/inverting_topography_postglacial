#!/bin/sh
#SBATCH --job-name WV_CALIB_TEST
#SBATCH --ntasks-per-node 24
#SBATCH --partition shas
#SBATCH --nodes 1
#SBATCH --time 24:00:00
#SBATCH --account ucb19_summit1

# load environment modules
module purge
module load intel
module load openmpi
module load mkl
module load cmake
module load gsl

# make sure environment variables are set correctly
source ~/.bash_profile

# run dakota using a restart file if it exists.
if [ -e dakota.rst ]
then
dakota -i dakota_hybrid_calibration.in -o dakota_hybrid_calibration.out --read_restart dakota.rst &> optim.log
else
dakota -i dakota_hybrid_calibration.in -o dakota_hybrid_calibration.out &> optim.log
fi


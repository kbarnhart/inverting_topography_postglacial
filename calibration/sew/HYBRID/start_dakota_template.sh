#!/bin/sh
#SBATCH --job-name sew_hybrid
#SBATCH --ntasks-per-node 24
#SBATCH --partition shas
#SBATCH --mem-per-cpu 4GB
#SBATCH --nodes 1
#SBATCH --time 24:00:00
#SBATCH --account ucb19_summit1

# load environment modules
module load intel/16.0.3
module load openmpi/1.10.2
module load mkl/11.3.3
module load cmake/3.5.2
module load gsl/2.1

# make sure environment variables are set correctly
source ~/.bash_profile
## run dakota using a restart file if it exists.
if [ -e dakota.rst ]
then
dakota -i dakota_hybrid_calibration.in -o dakota_hybrid_calibration.out --read_restart dakota.rst &> dakota.log
else
dakota -i dakota_hybrid_calibration.in -o dakota_hybrid_calibration.out &> dakota.log
fi

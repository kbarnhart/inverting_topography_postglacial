#!/bin/sh
#SBATCH --job-name WV_MOAT_LAUNCH
#SBATCH --ntasks-per-node 1
#SBATCH --partition shas
#SBATCH --nodes 1
#SBATCH --time 24:00:00
#SBATCH --account ucb19_summit1

# load environment modules
module purge
module load intel/16.0.3
module load openmpi/1.10.2
module load cmake/3.5.2
#module load perl
module load mkl
module load gsl

# make sure environment variables are set correctly
source ~/.bash_profile

# run the job creation script
python create_and_run_dakota_files.py

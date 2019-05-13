#!/bin/sh
#SBATCH --array 0-9
#SBATCH --ntasks 1
#SBATCH --job-name synthesize
#SBATCH --ntasks-per-node 1
#SBATCH --partition shas
#SBATCH --mem-per-cpu 50GB
#SBATCH --nodes 10
#SBATCH --time 24:00:00
#SBATCH --account ucb19_summit1

# make sure environment variables are set correctly
source ~/.bash_profile

# calculate syntheis
python make_combined_prediction_estimates.py

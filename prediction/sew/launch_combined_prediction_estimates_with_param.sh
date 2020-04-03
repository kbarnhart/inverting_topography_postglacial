#!/bin/sh
#SBATCH --array 0-10
#SBATCH --ntasks 1
#SBATCH --job-name synthesize
#SBATCH --ntasks-per-node 1
#SBATCH --qos preemptable
#SBATCH --mem-per-cpu 180GB
#SBATCH --nodes 11
#SBATCH --time 24:00:00

# make sure environment variables are set correctly
source ~/.bash_profile

# calculate syntheis
python make_combined_prediction_estimates_with_param.py

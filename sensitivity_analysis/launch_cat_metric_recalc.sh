#!/bin/sh
#SBATCH --job-name WV_MOAT_LAUNCH
#SBATCH --ntasks-per-node 24
#SBATCH --partition shas
#SBATCH --nodes 2
#SBATCH --time 24:00:00
#SBATCH --account ucb19_summit1

# run the job creation script
python recalculate_cat_metrics.py

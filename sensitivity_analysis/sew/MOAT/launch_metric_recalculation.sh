#!/bin/sh
#SBATCH --job-name WV_MOAT_RECALC
#SBATCH --ntasks-per-node 1
#SBATCH --partition shas
#SBATCH --nodes 1
#SBATCH --time 16:00:00

# run the job creation script
python recalculate_metrics.py

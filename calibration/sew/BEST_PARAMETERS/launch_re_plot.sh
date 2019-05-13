#!/bin/sh
#SBATCH --job-name valid_replot
#SBATCH --ntasks-per-node 24
#SBATCH --partition shas
#SBATCH --nodes 1
#SBATCH --time 2:00:00
#SBATCH --account ucb19_summit1

# run the job creation script
python re_plot_figures.py

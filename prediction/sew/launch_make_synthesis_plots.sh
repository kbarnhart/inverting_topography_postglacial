#!/bin/sh
#SBATCH --job-name plot_syn
#SBATCH --ntasks-per-node 24
#SBATCH --partition shas
#SBATCH --mem-per-cpu 4GB
#SBATCH --nodes 1
#SBATCH --time 24:00:00
#SBATCH --account ucb19_summit1

# make sure environment variables are set correctly
source ~/.bash_profile

# calculate syntheis
python make_synthesis_plots.py

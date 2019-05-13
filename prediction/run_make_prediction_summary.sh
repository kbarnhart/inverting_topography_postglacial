#!/bin/sh
#SBATCH --job-name plot_prediction
#SBATCH --ntasks-per-node 24
#SBATCH --partition shas
#SBATCH --mem-per-cpu 4GB
#SBATCH --nodes 1
#SBATCH --time 24:00:00
#SBATCH --account ucb19_summit1

module purge
module load R

Rscript make_prediction_summary_plots.R


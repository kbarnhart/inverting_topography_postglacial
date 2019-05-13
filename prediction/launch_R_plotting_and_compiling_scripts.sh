#!/bin/sh
#SBATCH --ntasks 1
#SBATCH --job-name rplots
#SBATCH --ntasks-per-node 1
#SBATCH --partition shas
#SBATCH --mem-per-cpu 50GB
#SBATCH --nodes 1
#SBATCH --time 24:00:00
#SBATCH --account ucb19_summit1

# make sure environment variables are set correctly
source ~/.bash_profile

# load R
ml R

# run each of the R plot scripts

# param uncertainty
Rscript prediction_param_uncert_plots_and_tables.R

# IC uncertainty
Rscript prediction_IC_uncert_plots_and_tables.R

# capture uncertainty
Rscript prediction_breach_uncert_plots_and_tables.R

# summary plots
Rscript make_prediction_summary_plots.R

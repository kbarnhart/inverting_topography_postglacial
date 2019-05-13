#!/bin/sh
cd /work/WVDP_EWG_STUDY3/study3py/calibration/sew/MCMC_testing/dream-gp/model_000/lowering_history_0.pg24f_ic5etch/
sbatch start_dakota.sh

cd /work/WVDP_EWG_STUDY3/study3py/calibration/sew/MCMC_testing/queso-gp/model_000/lowering_history_0.pg24f_ic5etch/
sbatch start_dakota.sh

cd /work/WVDP_EWG_STUDY3/study3py/calibration/sew/MCMC_testing/queso-pce/model_000/lowering_history_0.pg24f_ic5etch/
sbatch start_dakota.sh

cd /work/WVDP_EWG_STUDY3/study3py/calibration/sew/MCMC_testing/dream-pce/model_000/lowering_history_0.pg24f_ic5etch/
sbatch start_dakota.sh

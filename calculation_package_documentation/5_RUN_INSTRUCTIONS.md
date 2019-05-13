# RUN INSTRUCTIONS
Katherine Barnhart -- katherine.barnhart@colorado.edu -- March 2018

_Note_: This is a Markdown file, it is recommended that you preview the file as Markdown in order to benefit from formatting.  

## 1. Introduction
This document describes the steps taken to create and evaluate this calculation package. It has 4 major numerical experiments.

  A. Sensitivity Analysis

  B. Calibration

  C. Validation

  D. Prediction

The remainder of this document describes how to re-evaluate the calculation package. If full recalculation is desired, it is recommended to re-calculate in the order presented.

Note that this document does not attempt to provide description of results. Its only purpose is to provide the set of function evaluations needed to create (or re-create) the calculation package. Refer to the associated report for analysis and discussion.

Note also that this calculation package was designed to run on a Red Hat Enterprise Linux 7 supercomputer running the slurm submission protocol. It was run on the Summit heterogeneous supercomputing cluster.

(https://www.rc.colorado.edu/resources/compute/summit).

Undertaking the development, testing, final model evaluations, and post-processing analyses took 1.36 Million Core Hours. Re-running the entire modeling and analysis suite will create approximately 5 TB of files.

## 2. Sensitivity analysis

All sensitivity analysis was done at two locations, the South East Watershed domain of Upper Franks Creek (called 'sew') and and the gully domain (called 'gully'). Each of the following steps needs to be done for both of these locations. For clarity, here we've replaced the string `'sew'` and `'gully'` with `$LOCATION` so as to not present instructions in duplicate.

Sensitivity analysis used one method, the Morris One At a Time Method (MOAT). We attempted using the Distributed Analysis of Local Sensitivity Analysis method (DELSA) but this was not completed.

### a. Pre-computation steps
From `/work/WVDP_EWG_STUDY3/study3py/sensitivity_analysis/$LOCATION` evaluate:
```
python make_input_templates_and_drivers.py
```
This creates the input file and model driver templates used in this numerical experiment.

### b. Computation steps

1. From `/work/WVDP_EWG_STUDY3/study3py/sensitivity_analysis/$LOCATION/MOAT`
evaluate
```
source launch_job_creation.py
```
This will create a series of files with the name ``submit_jobs_to_summit_**.sh`.

2. Evaluate each of these scripts:
```
source submit_jobs_to_summit_**.sh
```
The sensitivity analysis was done as we were learning how to make Dakota run large numbers of jobs that may or may not finish within the 24 hour job duration. We also we learning about the typical range of time it would take a job to finish. For this reason, getting the sensitivity analysis jobs to complete is not the most streamlined process.

3. Once the first set of jobs finished, run:
```
python job_cleanup.py
```
to identify which jobs need to be re-submitted. This will create a set of files with the name `cleanup_submit_jobs_to_summit_XX.sh`. Evaluate each of these scripts:
```
source cleanup_submit_jobs_to_summit_XX.sh
```

4. Repeat step 3 until all jobs are done.

### c. Post-processing and analysis

We changed the comparison metrics after starting the sensitivity analysis model integrations. Thus it is necessary to recalculate the metrics.

From `/work/WVDP_EWG_STUDY3/study3py/sensitivity_analysis` evaluate:
```source launch_cat_metric_recalc.sh```

Then from `/work/WVDP_EWG_STUDY3/study3py/sensitivity_analysis/$LOCATION/MOAT` evaluate:

```
python compilation_of_output.py
python analysis.py
```

Finally, after after both locations have finished their runs, evaluate:

```
Rscript make_sensitivity_analysis_R_plots.R
```

from `/work/WVDP_EWG_STUDY3/study3py/sensitivity_analysis' to make the summary plots.

## 3. Calibration

Calibration only successfully completed for the `sew` location but not the `gully` location. Two methods were used, first EGO2, a hybrid between the EGO method and the NL2SOL method, and the Bayesian calibration method QUESO-DRAM. We attempted EGO2 at the `gully` location, but did not successfully complete it. Finally, we made a series of model evaluations at the BEST_PARAMETERS of the EGO2 method.

### a. Pre-computation steps
From `/work/WVDP_EWG_STUDY3/study3py/calibration/sew` run the following scripts to create the input files and driver templates used in the numerical experiment:
```
python make_input_templates_and_drivers.py
python make_input_templates_and_drivers_best_param.py
python make_input_templates_and_drivers_cat.py
```

### b. Computation, post-processing, and analysis

#### i. EGO2

1. From  `/work/WVDP_EWG_STUDY3/study3py/calibration/sew/EGO2` run:
```
python create_and_run_dakota_files.py
```
This will create `launch_dakota_calibration.sh`.

2. Execute `launch_dakota_calibration.sh` and wait for jobs to complete. Some models will finish calibrating with one submission, others will require many. To re-start jobs, re-execute:
```
source launch_dakota_calibration.sh
```

3. After jobs are complete run:
```
python analysis.py
```
This will do two things: first it will assess which jobs are done, and create a new file `restart_calibration.sh`; second, it will do postprocessing analysis including making result tables. Restart calibration by executing:
```
source restart_calibration.sh
```

4. Repeat Step 3 until all models have calibrated.


#### ii. QUESO_DRAM

EGO2 must be complete to move onto this step because the confidence intervals of EGO2 Dakota output files are used to create the prior distribution of parameter values used in this method.

The QUESO_DRAM method has two phases. First the method completes ten surrogate updating iterations. Second, ten separate jobs are created to stop the iterations short after each sequential iteration. These runs are called 'short stops'. This is done to compare the posterior parameter distributions over the course of the method.

1. From  `/work/WVDP_EWG_STUDY3/study3py/calibration/sew/QUESO_DRAM` run:
```
python create_and_run_dakota_files.py
```
This will create `launch_dakota_calibration.sh`.

2. Execute `launch_dakota_calibration.sh` and wait for jobs to complete. If you
do not want to do the QUESO_DRAM experiment on all models, edit this file to
remove models you don't want to run.
```
source launch_dakota_calibration.sh
```

3. Repeat Step 2 until all Dakota runs have completed. This will result in all
models being run for 10 surrogate updating iterations.

4. Create all short-stop input files by evaluating
```
python make_stop_short_input_files_and_cmnd_lines.py
```
from  `/work/WVDP_EWG_STUDY3/study3py/calibration/sew/QUESO_DRAM`.

5. This will create the file `start_all_short_stops.sh`
Run all short stop runs by evaluating:
```
source start_all_short_stops.sh
```

6. Some of these short stops may need to be restarted in order to complete. If
this is the case, navigate to the relevant folder:
 `/work/WVDP_EWG_STUDY3/study3py/calibration/sew/QUESO_DRAM/model_MMM/LL.III/short_stop/SSS`
 and execute:
```
source start_posterior_stop_SSS.sh
```

7. When all short stops are complete, some diagnostic and result plots can be created by running:
```
Rscript analyze_short_stops.R
```

#### iii. BEST_PARAMETERS
This must be done after EGO2 is complete as 'best parameters' are the best parameters from the EGO2 method.

1. From  `/work/WVDP_EWG_STUDY3/study3py/calibration/sew/BEST_PARAMETERS` run:
```
python create_and_run_dakota_files.py
```
This will create `submit_best_parameter_runs.sh`.

2. Execute `submit_best_parameter_runs.sh` and wait for jobs to complete.
```
source submit_best_parameter_runs.sh
```
These runs should all complete within one job submission.

3. Then run:
```
source launch_re_plot.sh
```
to create additional plots.

### c. Final post-processing and analysis
From `/work/WVDP_EWG_STUDY3/study3py/calibration` execute:
```
Rscript make_calibration_R_plots.R
```
For this script to fully execute, the validation step described below must also be complete.

## 4. Validation
### a. Pre-computation steps
From `/work/WVDP_EWG_STUDY3/study3py/validation` execute:

```
python make_input_templates_and_drivers_best_param.py
```

### b. Computation steps

1. From `/work/WVDP_EWG_STUDY3/study3py/validation/BEST_PARAMETERS_SMALL_DOMAIN` execute
```
python create_and_run_best_parameters.py
```
This will create a file `submit_best_parameter_runs.sh`

2. Execute
```
source submit_best_parameter_runs.sh
```
This should only take one set of jobs to complete.

### c. Post-processing and analysis
Execute
```
python analysis.py
```

## 5. Prediction

The prediction effort has four parts:

A. Predictions at EGO2 BEST_PARAMETERS

B. A suite of runs designed to evaluate initial condition uncertainty (IC_UNCERTAINTY)

C. A suite of runs designed to evaluate PARAMETER_UNCERTAINTY at specific points.

D. A suite of runs designed to evaluate uncertainty from BREACHING scenarios.

After all of these components have completed, there are scripts that synthesize the results.

### a. Pre-computation steps
From `/work/WVDP_EWG_STUDY3/study3py/prediction/sew` execute the following commands:
```
python make_input_templates_and_drivers_param_uncertain.py
python make_input_templates_and_drivers_ic_uncert.py
python make_input_templates_and_drivers_breaching.py
python make_input_templates_and_drivers_best_param.py
```

### b. Computation steps
#### i. BEST_PARAMETERS
1. Create run files by executing:
```
python create_and_run_best_parameters.py
```
from `/work/WVDP_EWG_STUDY3/study3py/prediction/sew/BEST_PARAMETERS`.
This will create the file  `submit_best_parameter_runs.sh`.

2. Execute
```
source submit_best_parameter_runs.sh
```
and wait for jobs to finish.

3. Execute the following two commands to compile output and make figures.
```
python re_plot_figures.py
python compile_output.py
```

#### ii. BREACHING
1. Create run files by executing:
```
python create_and_run_breaching.py
```
from `/work/WVDP_EWG_STUDY3/study3py/prediction/sew/BREACHING`.
This will create the file  `submit_breaching_runs.sh`.

2. Execute
```
source submit_breaching_runs.sh
```
and wait for jobs to finish.

3. Execute the following two commands to compile output and make figures.
```
python re_plot_figures.py
python compile_output.py
```

#### iii. IC_UNCERTAINTY
1. Create run files by executing:
```
python create_and_run_initial_condition.py
```
from `/work/WVDP_EWG_STUDY3/study3py/prediction/sew/IC_UNCERTAINTY`.
This will create the file  `submit_best_parameter_runs.sh`.

2. Execute:
```
source submit_best_parameter_runs.sh
```
and wait for jobs to finish.

3. Execute the following two commands to compile output and make figures.
```
python re_plot_figures.py
python compile_output.py
```

#### iv. PARAMETER_UNCERTAINTY
1. From the directory `/work/WVDP_EWG_STUDY3/study3py/prediction/sew/PARAMETER_UNCERTAINTY` evaluate
```
python create_and_run_param_uncertainty.py
```
to construct the experiment.
This will create the file `start_complex_sampling_for_surrogate.sh`.

2. Evaluate:
```
source start_complex_sampling_for_surrogate.sh
```
in order to make complex model evaluations.

3. If any complex sampling runs are not complete, relaunch them until they are complete by re-evaluating:
```
source start_complex_sampling_for_surrogate.sh
```

4. Once all complex evaluations are complete, run:
```
python create_surrogate_dakota_files.py
```
from `/work/WVDP_EWG_STUDY3/study3py/prediction/sew/PARAMETER_UNCERTAINTY`.
This will create the file `submit_all_surrogate_evals.sh`.

5. Create and evaluate the surrogates by running:
```
source submit_all_surrogate_evals.sh
```

6. Finally, run
```
python compile_output_figures.py
```
to compile figures.

### c. Post-processing and analysis
1. After all four prediction experiments are complete, from the directory `/work/WVDP_EWG_STUDY3/study3py/prediction/sew` evaluate:
```
Rscript make_inset_plots.R
```
to create inset plots used for orientation.

2. Next, evaluate
```
source launch_combined_prediction_estimates.sh
```
to create the synthesis netcdfs.

3. Then evaluate
```
source launch_make_synthesis_plots.sh
```
to make the synthesis plots.

4. Four R scripts make a number of relevant plots and summary tables. They can be run individually:
```
Rscript make_prediction_summary_plots.R
Rscript prediction_breach_uncert_plots_and_tables.
Rscript prediction_IC_uncert_plots_and_tables.R
Rscript prediction_param_uncert_plots_and_tables.R
```
Or, all four can be run with the following command:
```
source launch_R_plotting_and_compiling_scripts.sh
```

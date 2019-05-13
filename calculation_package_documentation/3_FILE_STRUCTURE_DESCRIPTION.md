# FILE STRUCTURE
Katherine Barnhart -- katherine.barnhart@colorado.edu -- March 2018

_Note_: This is a Markdown file, it is recommended that you preview the file as Markdown in order to benefit from formatting.  

## Introduction
This file provides a description of the files and file structure provide in this
calculation package.

Note that not every file and folder is documented. The entire calculation package contains ~5.5 million files and ~550,000 directories and takes up about 5 TB (4.6 TiB). For this reason, every attempt was made to script the creation of file structure, Dakota and Erosion Model Suite input files, to systematically name output files, and to provided redundant information about model runs in the form of the file paths.

Below is a file tree describing the calculation package structure (Section marked **File Structure**). Next to relevant files and folders comments have been written to provide information about what files do. For example, in each of the prediction experiments are folders for each model, labeled with the three digit model ID key. Inside of each of those folders are folders named with the lowering future, initial condition DEM, and climate future key. In this file structure, systematically repetitive folders and files structures have been edited to show the structure in terms of simplifying codes described in the next section. For example, in almost all of the numerical experiments, there is a folder for each of the 37 alternative models with the name `model_MMM` in which `MMM` is the model code. In the file structure below all 37 of those folders have been replaced with one folder with the name `model_MMM`. Python scripts named `create_and_run---.py` were used to create the file structure. The actual value of the keys used was determined by file names, `.csv` input files, or other input values. The correspondence between specific input files and key codes can be determined by examining the `create_and_run---.py` script.  

Adjacent to `/work/WVDP_EWG_STUDY3/study3py` is a folder called `/work/WVDP_EWG_STUDY3/results`. This folder contains most, but not all, of the model output files. The Dakota `.in` files used to run model experiments will include the file path of the work directory used. By referring to the `.in` file used to launch the Dakota computations it will be possible to determine where model output is located.

Additionally, many filenames will occur repeatedly (e.g. LHS_2.out, fort.13). These files will have comments the first time they appear, but not in subsequent appearances. Further, many files are automatically created by Dakota or are created by python scripts designed to set up calculations. See the `RUN_INSTRUCTIONS.md` document on details of run instructions.

Finally, most of the file structure was either automatically generated, or are the result of model output. We recognize that navigating this file structure is likely challenging. There are a few sets of files that we suspect are of greater interest than others (either in terms of their use in constructing the calculation package) or in terms of their role in synthesizing the results. They have been marked with `### IMPORTANT FILES ###` in the file structure below. Before the file structure, we have also provided an outline of the basic file structure of most of the computational experiments (Section marked **Basic File Structure of Most of the Computational Experiments**).

## Location of the calculation package
This calculation package expects that it is located at `/work/WVDP_EWG_STUDY3/study3py` and lives next to a folder called `/work/WVDP_EWG_STUDY3/results`.
This calculation package expects that it is located at `/work/WVDP_EWG_STUDY3/study3py` and lives next to a folder called `/work/WVDP_EWG_STUDY3/results`. If this is not the case, some scripts will not successfully run.


## Simplifying codes

- `BBB` - Breaching key which indicated which breaching location was used.
- `BYY` - Year in which breaching began.
- `CCC` - Key indicating which alternative climate future was used.
- `CHE` - Chi-Elevation metric code.
- `CTS` - Calibration test step size.
- `LLL` - Key indicating which watershed outlet lowering history was used.
- `OOO` - Numeric value of the objective function of a particular model integration.
- `III` - Key indicating the name of initial condition DEM.
- `MMM` - Model name key.
- `RRR` -  Run number.
- `SSS` - Number of QUESO-DRAM surrogate iterations.
- `SUR` - Location of surrogate sampling site.
- `TTT` - Model output time, typically in units of 100 years (e.g. 0001 = 100 model years).
- `XXX` - Slurm output run ID.

## Basic File Structure of Most of the Computational Experiments
```
/work/WVDP_EWG_STUDY3/study3py/
`-- NAME_OF_EXPERIMENT # e.g. sensitivity analysis, calibration, validation, prediction
    `-- LOCATION_NAME # e.g. sew (South East Watershed of Franks Creek) or gully
        |-- make_input_templates_and_drivers.py # python script used to make input templates and drivers
        |-- # some other scripts, perhaps more than one input template was necessary, or model re-runs necessitated scripts to cleanup files before re-running
         `-- NAME_OF_METHOD # Dakota method used, e.g. MOAT or GAUSSNEWTON
            |-- create_and_run_dakota_files.py # script used to create the file structure below this level, and provide each Dakota run its necessary files
            |-- DAKOTA_TEMPLATE.in # template of the Dakota .in file used for each Dakota run
            |-- analysis.py # File used to do post-Dakota run analysis and synthesis
            |-- launch_dakota_calibration.sh # file automatically created by `create_and_run_dakota_files.py` to launch all model runs
            |-- FIGURE_FOLDER # often output figures were compiled into a folder at this level of the file structure for ease of comparison
            |   
            |   # Below this level, all files and folders are either created by running `create_and_run_dakota_files.py` or by the Dakota runs themselves
            |
            `-- model_MMM # model name, using model code.
                `-- LLL.III.OTHER_CODES # folder name that provides some, but not all information about this model run
                    |-- start_dakota.sh # shell script to launch this MMM/III/LLL/CCC's Dakota run
                    |-- driver.py # model driver
                    |-- inputs_template.txt # model input template
                    |-- DAKOTA.IN # Dakota input file created from the template
                    |
                    | # Below this level, all files are output created by the
                    | # Dakota run.
                    |
                    |-- DAKOTA.OUT # Dakota .out file
                    |-- DAKOTA.LOG # Dakota .log file
                    |-- DAKOTA.RST # Dakota binary restart file
                    |-- DAKOTA.dat # Dakota tabular data file
                    |-- various dakota output files # depending on the method
                    |                               # used, many other files may
                    |                               # be created
                    |
                    | # The slurm submission system creates log files for each
                    | # slurm job
                    |
                    `-- slurm-XXX.out # SLURM job log
```

The starting characters of most important python scripts are the same.

- `make_input_templates_and_driver----.py`: A file that makes all of the input templates and drivers used for the experiment.
- `create_and_run----.py`: A file that makes input files for Dakota, creates the experiment file structure, and the shell scripts used to submit jobs to Summit.
- `analysis---.py`: A file that does most of the postprocessing and/or figure and table construction associated with the experiment.

## File Structure
Below is an edited tree of the `/work/WVDP_EWG_STUDY3/study3py/` file structure. It was constructed by calling

`tree --dirsfirst --charset=ascii /work/WVDP_EWG_STUDY3/study3py/ &>study3py.txt`

using the [tree](https://linux.die.net/man/1/tree) command and then editing the resulting tree structure for clarity.

```
/work/WVDP_EWG_STUDY3/study3py/
|-- allocation_requests # files associated with making computation allocation requests to CU research computing.
|-- auxillary_inputs # folder containing all of the auxiliary information to set up and run the model computations
|   |-- chi_elev_categories # files that identify which chi-elevation category each model grid cell is located in.
|   |   |-- gully.chi_elev_cat.20.txt
|   |   |-- sew.chi_elev_cat.0.txt
|   |   |-- sew.chi_elev_cat.20.txt
|   |   `-- validation.chi_elev_cat.20.txt
|   |-- chi_mask # chi-domain masks for each domain.
|   |   |-- sew
|   |   |   |-- chi\ mask.qgs
|   |   |   |-- chi_mask.asc.aux.xml
|   |   |   |-- chi_mask.dbf
|   |   |   |-- chi_mask.prj
|   |   |   |-- chi_mask.qpj
|   |   |   |-- chi_mask.shp
|   |   |   |-- chi_mask.shx
|   |   |   |-- chi_mask.tiff
|   |   |   |-- chi_mask.txt
|   |   |   `-- make_chi_mask.sh
|   |   `-- validation
|   |       |-- chi_mask.txt
|   |       `-- make_chi_mask.py
|   |-- climate_futures # documentation of the construction of the climate futures used in prediction.  
|   |   |-- ClimateFuture_mean_storm__intensity.png
|   |   |-- climate_future_1.constant_climate.txt
|   |   |-- climate_future_2.RCP45.txt
|   |   |-- climate_future_3.RCP85.txt
|   |   |-- maca_future_parameters.csv
|   |   |-- maca_futures.docx
|   |   |-- make_climate_future_files.py # file used to construct the climate future text files.
|   |   |-- stat_est_CMIP5_30yr_3pixels.png
|   |   `-- stat_est_CMIP5_30yr_scenarios.png
|   |-- dems # input DEMS in esrii ascii and netcdf format used for each of the domains, initial_conditions are for the postglacial initial condition and modern is the current topography.
|   |   |-- gully
|   |   |   |-- initial_conditions
|   |   |   |   |-- gcomb_0pctF.nc
|   |   |   |   |-- gcomb_0pctF.txt
|   |   |   |   |-- gcomb_14F.nc
|   |   |   |   |-- gcomb_14F.txt
|   |   |   |   |-- gcomb_3p5F.nc
|   |   |   |   |-- gcomb_3p5F.txt
|   |   |   |   |-- gcomb_7F.nc
|   |   |   |   |-- gcomb_7F.txt
|   |   |   |   |-- gcomb_7randF.nc
|   |   |   |   `-- gcomb_7randF.txt
|   |   |   `-- modern
|   |   |       |-- gdem3r1f.nc
|   |   |       `-- gdem3r1f.txt
|   |   |-- sew
|   |   |   |-- initial_conditions
|   |   |   |   |-- pg24f_0etch.nc
|   |   |   |   |-- pg24f_0etch.txt
|   |   |   |   |-- pg24f_14etch.nc
|   |   |   |   |-- pg24f_14etch.txt
|   |   |   |   |-- pg24f_3pt5etch.nc
|   |   |   |   |-- pg24f_3pt5etch.txt
|   |   |   |   |-- pg24f_7etch.nc
|   |   |   |   |-- pg24f_7etch.txt
|   |   |   |   |-- pg24f_ic5etch.nc
|   |   |   |   |-- pg24f_ic5etch.txt
|   |   |   |   |-- pg24f_randetch.nc
|   |   |   |   `-- pg24f_randetch.txt
|   |   |   `-- modern
|   |   |       |-- dem24fil_ext.nc
|   |   |       `-- dem24fil_ext.txt
|   |   `-- validation
|   |       |-- initial_conditions
|   |       |   |-- vpg_0fr3.nc
|   |       |   |-- vpg_0fr3.txt
|   |       |   |-- vpg_14fr3.nc
|   |       |   |-- vpg_14fr3.txt
|   |       |   |-- vpg_3p5fr3.nc
|   |       |   |-- vpg_3p5fr3.txt
|   |       |   |-- vpg_7fr3.nc
|   |       |   |-- vpg_7fr3.txt
|   |       |   |-- vpg_randfr3.nc
|   |       |   `-- vpg_randfr3.txt
|   |       `-- modern
|   |           |-- vmodern24fr3.nc
|   |           `-- vmodern24fr3.txt
|   |-- lowering_histories # Files and documentation associated with the construction of the lowering histories (postglacial to present) and lowering futures (present to +13 ka)
|   |   |-- EWG\ Study\ 1\ Draft\ Report\ -\ Vol\ I_3.27.17\ -\ Figure\ 4.6-4.pdf
|   |   |-- Lowering\ History\ Recommendations.docx
|   |   |-- alternative_incision_histories.pdf
|   |   |-- alternative_incision_histories_and_futures.pdf
|   |   |-- creating_lowering_futures.py # script used to create the lowering futures.
|   |   |-- creating_lowering_histories.py # script used to create the lowering histories.
|   |   |-- downcutting_table_20170504.xlsx
|   |   |-- lowering_future_1.txt
|   |   |-- lowering_future_2.txt
|   |   |-- lowering_future_3.txt
|   |   |-- lowering_history_0.txt
|   |   |-- lowering_history_1.txt
|   |   `-- lowering_history_2.txt
|   |-- modern_metric_files # files of the modern values of metrics.
|   |   |-- create_modern_metric_files.py
|   |   |-- dem24fil_ext.metrics.chi.txt
|   |   |-- dem24fil_ext.metrics.txt
|   |   |-- gdem3r1f.metrics.chi.txt
|   |   `-- gdem3r1f.metrics.txt
|   |-- roads # files used to create the road mask used for weighting.
|   |   `-- sew
|   |       |-- make_road_mask.sh
|   |       |-- sew_road_mask.txt
|   |       |-- sew_roads.dbf
|   |       |-- sew_roads.prj
|   |       |-- sew_roads.qpj
|   |       |-- sew_roads.shp
|   |       |-- sew_roads.shx
|   |       |-- sew_roads_50_buffer.dbf
|   |       |-- sew_roads_50_buffer.prj
|   |       |-- sew_roads_50_buffer.qpj
|   |       |-- sew_roads_50_buffer.shp
|   |       `-- sew_roads_50_buffer.shx
|   |-- rock_till # esri ascii files of the bedrock elevation in all three domains.
|   |   |-- gully
|   |   |   `-- gully_bdrx.txt
|   |   |-- sew
|   |   |   `-- bdrx_24.txt
|   |   `-- validation
|   |       `-- valbdx24bnd2.txt
|   |-- weights # weights
|   |   |-- gully.chi_elev_effective_weight.20.txt
|   |   |-- gully.chi_elev_weight.20.txt
|   |   |-- gully_variance.txt
|   |   |-- sew.chi_elev_effective_weight.0.txt
|   |   |-- sew.chi_elev_effective_weight.20.txt
|   |   |-- sew.chi_elev_weight.0.txt
|   |   |-- sew.chi_elev_weight.20.txt
|   |   |-- sew_variance.txt
|   |   |-- sew_variance_with_model800.txt
|   |   |-- validation.chi_elev_effective_weight.20.txt
|   |   `-- validation.chi_elev_weight.20.txt
|   |-- West_Valley_Param_Ranges.aux
|   |-- West_Valley_Param_Ranges.log
|   |-- West_Valley_Param_Ranges.pdf
|   |-- West_Valley_Param_Ranges.synctex.gz
|   |-- West_Valley_Param_Ranges.tex # Latex file (and associated other files) associated with early work on defining the parameter ranges.
|   |           ### IMPORTANT FILES ###
|   |           # these files control which model uses which parameters in each experiment, what the parameter ranges are, estimates of model duration, and names used in results tables. Depending on the experiment, the `create_and_run---.py` file will point to one or more of these files.
|   |
|   |-- metric_names.csv
|   |-- model_name_element_match.csv
|   |-- model_parameter_calibration_start_values_sew.csv
|   |-- model_parameter_match.csv
|   |-- model_parameter_match_calibration_gully.csv
|   |-- model_parameter_match_calibration_sew.csv
|   |-- model_parameter_order.csv
|   |-- model_time.csv
|   `-- parameter_ranges.csv
|              ### END IMPORTANT FILES ###
|-- calculation_package_documentation
|   |-- compilation\ shell\ scripts
|   |   |-- dakota_cmake_openmpi_intel_boost1.53
|   |   |-- install_boost.sh
|   |   `-- install_dakota.sh
|   |-- 1_CONTRIBUTORS.md
|   |-- 2_ACKNOWLEDGEMENTS.md
|   |-- 3_FILE_STRUCTURE_DESCRIPTION.md # This file.
|   |-- 4_INSTALL_INSTRUCTIONS.md
|   |-- 5_RUN_INSTRUCTIONS.md
|   `-- EWG_Study3_Modeling_Long_Term_Erosion_20180226.pdf
|-- calibration
|   |-- gully # files associated with gully calibration, this was not successful. This file structure parallels that of the adjacent `sew` directory.
|   |-- sew
|   |   |-- BEST_PARAMETERS # Evaluation of models at best parameter value.
|   |   |   |-- model_MMM # Experiment file structure. Below this level is autogenerated
|   |   |   |   `-- LLL.III
|   |   |   |       |-- driver.py # model driver
|   |   |   |       |-- inputs.txt # model input
|   |   |   |       |-- model_MMM_TTT.nc # model output netcdf file
|   |   |   |       |-- outputs_for_analysis.txt # summary of output
|   |   |   |       `-- usage.txt # usage file
|   |   |   |-- cmd_lines # used in submission
|   |   |   |-- create_and_run_best_parameters.py # job construction file
|   |   |   |-- launch_re_plot.sh # submission file
|   |   |   |-- re_plot_figures.py # re create plots
|   |   |   |-- slurm-XXX.out
|   |   |   `-- submit_best_parameter_runs.sh # submission file.
|   |   |-- EGO # Initial calibration attempt with EGO+GAUSSNEWTON
|   |   |-- EGO2 # Sucessfully calibration attempt with EGO+NL2SOL
|   |   |   |-- best_runs # Figures showing the best run topography
|   |   |   |-- model_MMM # Experiment file structure. Below this level is autogenerated
|   |   |   |   `-- LLL.III
|   |   |   |       |-- LHS_2.out # Dakota output
|   |   |   |       |-- LHS_3.out # Dakota output
|   |   |   |       |-- LHS_7.out # Dakota output
|   |   |   |       |-- LHS_8.out # Dakota output
|   |   |   |       |-- LHS_9.out # Dakota output
|   |   |   |       |-- LHS_distributions.out # Dakota output
|   |   |   |       |-- LHS_samples.out # Dakota output
|   |   |   |       |-- dakota.log # Dakota output
|   |   |   |       |-- dakota_beale.in  # input file for using beales method, not used
|   |   |   |       |-- dakota_calib.rst #  # Dakota output
|   |   |   |       |-- dakota_calib.txt # Dakota output
|   |   |   |       |-- dakota_hybrid_calibration.in # Dakota input file, autogenerated
|   |   |   |       |-- dakota_hybrid_calibration.out  # Dakota output
|   |   |   |       |-- driver.py # model driver
|   |   |   |       |-- fort.13 # Dakota output
|   |   |   |       |-- inputs_template.txt # model input template
|   |   |   |       |-- slurm-XXX.out # Slurm output
|   |   |   |       |-- start_beale.sh # shell script to start the beales method effort, not used.
|   |   |   |       |-- start_dakota.sh # shell script to start Dakota experiment.
|   |   |   |       `-- wv_model_MMM_calib.dat  # Dakota output
|   |   |   |-- analysis.py # Postprocessing
|   |   |   |-- calibration_parameters.txt # summary table
|   |   |   |-- calibration_summary_table.txt # summary table
|   |   |   |-- create_and_run_dakota_files.py
|   |   |   |-- dakota_beale_template.in # Dakota input file template
|   |   |   |-- dakota_hybrid_template.in # Dakota input file template
|   |   |   |-- launch_dakota_calibration.sh # Shell script to launch all Dakota files.
|   |   |   |-- restart_calibration.sh
|   |   |   |-- sew.EGO_EGONL2_comparison.pdf
|   |   |   |-- sew.calibration_of_figure.build.0.pdf
|   |   |   |-- sew.calibration_of_figure.build.1.pdf
|   |   |   |-- sew.calibration_of_figure.build.2.pdf
|   |   |   |-- sew.calibration_of_figure.build.3.pdf
|   |   |   |-- sew.calibration_of_figure.build.4.pdf
|   |   |   |-- sew.calibration_of_figure.pdf
|   |   |   |-- sew.calibration_of_figure.withValidation.pdf
|   |   |   |-- sew.calibration_of_figure.with_error.pdf
|   |   |   |-- sew.calibration_summary_figure.pdf
|   |   |   `-- sew.stochastic_comparison.pdf
|   |   |-- GAUSSNEWTON # Files associated with the failed GAUSSNEWTON method attempt
|   |   |-- HYBRID # Files associated with the failed HYBRID method attempt
|   |   |-- HYBRID-CAT # Files associated with the failed HYBRID-CAT method attempt
|   |   |-- MCMC_testing # Files associated with testing MCMC methods
|   |   |-- QUESO_DRAM # Files associated with MCMC calibration
|   |   |   |-- figures # summary figures
|   |   |   |-- model_MMM # Experiment file structure. Below this level is autogenerated
|   |   |   |   `-- LLL.III # Lowering and Initial DEM
|   |   |   |       |-- QuesoDiagnostics # A folder of Queso diagnostics provided by Dakota
|   |   |   |       |   |-- display_sub0.txt
|   |   |   |       |   |-- mh_output_sub0.m
|   |   |   |       |   |-- raw_chain.m
|   |   |   |       |   |-- raw_chain_loglikelihood.m
|   |   |   |       |   |-- raw_chain_loglikelihood_sub0.m
|   |   |   |       |   |-- raw_chain_logtarget.m
|   |   |   |       |   |-- raw_chain_logtarget_sub0.m
|   |   |   |       |   `-- raw_chain_sub0.m
|   |   |   |       |-- short_stop # We stopped the method after each surrogate update for 10 `short stops`. Each one gets its own folder
|   |   |   |       |   `-- SSS # Short stop folder
|   |   |   |       |       |-- QuesoDiagnostics # A folder of Queso diagnostics provided by Dakota
|   |   |   |       |       |   |-- display_sub0.txt
|   |   |   |       |       |   |-- mh_output_sub0.m
|   |   |   |       |       |   |-- raw_chain.m
|   |   |   |       |       |   |-- raw_chain_loglikelihood.m
|   |   |   |       |       |   |-- raw_chain_loglikelihood_sub0.m
|   |   |   |       |       |   |-- raw_chain_logtarget.m
|   |   |   |       |       |   |-- raw_chain_logtarget_sub0.m
|   |   |   |       |       |   `-- raw_chain_sub0.m
|   |   |   |       |       |-- LHS_2.out # Dakota output
|   |   |   |       |       |-- LHS_3.out # Dakota output
|   |   |   |       |       |-- LHS_7.out # Dakota output
|   |   |   |       |       |-- LHS_8.out # Dakota output
|   |   |   |       |       |-- LHS_9.out # Dakota output
|   |   |   |       |       |-- LHS_distributions.out # Dakota output
|   |   |   |       |       |-- LHS_samples.out # Dakota output
|   |   |   |       |       |-- OPT_DEFAULT.out # Dakota output
|   |   |   |       |       |-- dakota.rst # Dakota output
|   |   |   |       |       |-- dakotaSSS.log # Dakota output
|   |   |   |       |       |-- dakota_mcmc.rst # Dakota output
|   |   |   |       |       |-- dakota_mcmc_CredPredIntervals.dat # Dakota output
|   |   |   |       |       |-- dakota_queso_dram_short_SSS.in # Dakota input file
|   |   |   |       |       |-- dakota_queso_dram_short_SSS.out # Dakota output file
|   |   |   |       |       |-- data.dat # calibration data required by Dakota
|   |   |   |       |       |-- driver.py # model driver
|   |   |   |       |       |-- fort.13 # Dakota output
|   |   |   |       |       |-- inputs_template.txt # model input template
|   |   |   |       |       |  ### IMPORTANT FILES ###
|   |   |   |       |       |   # This is the posterior parameter estimate, it is
|   |   |   |       |       |   # in Dakota annotated format.
|   |   |   |       |       |-- posterior_SSS.dat # posterior file
|   |   |   |       |       |  ### END IMPORTANT FILES ###
|   |   |   |       |       |-- slurm-XXX.out # Slurm file
|   |   |   |       |       |-- start_posterior_stop_SSS.sh # submission file
|   |   |   |       |       `-- wv_model_MMM_mcmc_SSS.dat # Dakota output
|   |   |   |       |-- LHS_2.out # Dakota output
|   |   |   |       |-- LHS_3.out # Dakota output
|   |   |   |       |-- LHS_7.out # Dakota output
|   |   |   |       |-- LHS_8.out # Dakota output
|   |   |   |       |-- LHS_9.out # Dakota output
|   |   |   |       |-- LHS_distributions.out # Dakota output
|   |   |   |       |-- LHS_samples.out # Dakota output
|   |   |   |       |-- OPT_DEFAULT.out # Dakota output
|   |   |   |       |-- cmnd_lines # files used for submission, autogenerated
|   |   |   |       |-- dakota.log # Dakota output
|   |   |   |       |-- dakota.rst # Dakota output
|   |   |   |       |-- dakota_calib.rst # Dakota output
|   |   |   |       |-- dakota_calib.txt # Dakota output
|   |   |   |       |-- dakota_mcmc.dat # Dakota output
|   |   |   |       |-- dakota_mcmc.rst # Dakota output
|   |   |   |       |-- dakota_mcmc.txt # Dakota output
|   |   |   |       |-- dakota_mcmc_CredPredIntervals.dat # Dakota output
|   |   |   |       |-- dakota_queso_dram.in # Dakota input
|   |   |   |       |-- dakota_queso_dram.out # Dakota output
|   |   |   |       |-- dakota_recreate_surrogate.in # Dakota input for testing the surrotate
|   |   |   |       |-- dakota_recreate_surrogate.out # Dakota output
|   |   |   |       |-- data.dat # calibration file
|   |   |   |       |-- driver.py # model driver
|   |   |   |       |-- exported_surrogate.CHE.alg  # Exported surrogate .alg file
|   |   |   |       |-- fort.13 # Dakota output
|   |   |   |       |-- inputs_template.txt # model input template
|   |   |   |       |-- posterior.dat  # Dakota output
|   |   |   |       |-- recreate_surrogate.log  # Dakota output
|   |   |   |       |-- slurm-XXX.out # Dakota output
|   |   |   |       |-- start_all_individually.sh # Submission file
|   |   |   |       |-- start_dakota.sh # Submission file
|   |   |   |       |-- start_surrogate_sampling.sh # Submission file
|   |   |   |       |-- wv_model_MMM_mcmc.dat # dakota output
|   |   |   |       |-- wv_model_MMM_surrogate_samples.dat # dakota output
|   |   |   |       `-- wv_model_MMM_surrogate_samples_01.dat # dakota output
|   |   |   |-- analyze_short_stops.R # postprocessing analysis
|   |   |   |-- create_and_run_dakota_files.py # launch python script
|   |   |   |-- dakota_queso_dram_template.in # Dakota input template
|   |   |   |-- data.dat # Dakota calibration data
|   |   |   |-- launch_dakota_calibration.sh # submission script
|   |   |   |-- launch_evaluate_best_points.sh # submission script
|   |   |   |-- make_stop_short_input_files_and_cmnd_lines.py
|   |   |   |-- plot_000_surrogate.py # file for testing surrogate
|   |   |   |-- plot_points.py # file for testing surrogate
|   |   |   `-- start_all_short_stops.sh # submission script
|   |   |-- make_input_templates_and_drivers.py
|   |   |-- make_input_templates_and_drivers_best_param.py
|   |   `-- make_input_templates_and_drivers_cat.py
|   |-- test # Test folder
|   |-- test2 # Test folder
|   |-- clean_up_EG02_rst_files.py # helper script for cleaning up restart for fresh calibration attempts
|   |-- make_and_place_failure_logs.py # helper script for cleaning up restart for fresh calibration attempts
|   `-- make_calibration_R_plots.R # R script used to make calibration summary plots
|-- drivers # this folder contains driver files created for specific models.
|   |     # except the dakota drivers, these are autogenerated from templates
|   |-- dakota # dakota parallel drivers
|   |   |-- analysis_moat_driver.py
|   |   |-- parallel_lhc_driver.py
|   |   `-- parallel_model_run_driver.py
|   `-- models
|       `-- model_MMM---_driver.py # all of the model driver templates used.
|-- erosion_modeling_suite # python package used for modeling.
|   |-- erosion_model
|   |   |        ### Important Files ###
|   |   |-- baselevel_handler # baselevel handlers
|   |   |   |-- capture_node_baselevel_handler.py
|   |   |   `-- single_node_baselevel_handler.py
|   |   |-- basic_combination # this is where each of the models is located
|   |   |   |-- model_MMM_name
|   |   |   |   |-- __init__.py
|   |   |   |   `-- model_000_name.py
|   |   |-- single_component # single component models used for testing
|   |   |-- erosion_model.py # primary model base class
|   |   |-- precip_changer.py # component used for making time variable precipitation
|   |   `-- stochastic_erosion_model.py # base class for all stochastic models.
|   |         ### End Important Files ###
|   |-- examples_and_tests
|   |   |-- basic_combination_models
|   |   |   |-- model_MMM_basic
|   |   |   |   |-- model_MMM_basic.ipynb # Ipython notebook showing basic usage of each model.
|   |   |   |   `-- model_MMM_basic_inputs.txt
|   |   |-- example_gully_full_length_runs
|   |   |   |-- create_all_example_model_runs.py
|   |   |   |-- launch_simple_test.sh
|   |   |   `-- simple_test.py
|   |   |-- example_west_valley_full_length_runs
|   |   |   |-- create_all_example_model_runs.py
|   |   |   |-- launch_simple_test.sh
|   |   |   `-- simple_test.py
|   |   |-- grid_boundary_conditions
|   |   |   `-- test_grid_boundary_conditions.ipynb
|   |   |-- single_component_models
|   |   `-- topo_data_for_testing
|   |       |-- dem24_fillclip.txt
|   |       |-- dem48_MASK_again.tif.aux.xml
|   |       |-- dem48_MASK_ascii.asc
|   |       |-- dem48_MASK_ascii.asc.aux.xml
|   |       |-- dem48_fillclip_ascii.txt
|   |       |-- dem48_fillclip_ascii.txt.aux.xml
|   |       |-- sitec2dem24_padded.txt
|   |       `-- sitec2dem48padded.txt
|   `-- setup.py # file used to compile the package.
|-- exploratory_analysis # various exploratory analysis folders
|-- grid_search # Grid search tests to verify location of the objective function minima. Additional tests done in the `testing` folder
|   |-- gully
|   |   |-- model_800
|   |   |   `-- lowering_history_1.gcomb_0pctF # three gully grid searches
|   |   |       |-- coarse
|   |   |       |   |-- dakota_grid.in
|   |   |       |   |-- driver.py
|   |   |       |   |-- inputs_template.txt
|   |   |       |   `-- start_dakota.sh
|   |   |       |-- fine
|   |   |       |   |-- dakota_grid.in
|   |   |       |   |-- driver.py
|   |   |       |   |-- inputs_template.txt
|   |   |       |   `-- start_dakota.sh
|   |   |       `-- random
|   |   |           |-- dakota_random.in
|   |   |           |-- driver.py
|   |   |           |-- inputs_template.txt
|   |   |           `-- start_dakota.sh
|   |   |-- analysis_grid_coarse_800_gully.py
|   |   |-- analysis_grid_fine_800_gully.py
|   |   `-- reprocess_gully_grid_figures.py
|   `-- sew
|       |-- model_800 # three model 800 grid searches for SEW
|       |   `-- lowering_history_1.pg24f_0etch
|       |       |-- coarse
|       |       |   |-- dakota_grid.in
|       |       |   |-- driver.py
|       |       |   |-- inputs_template.txt
|       |       |   `-- start_dakota.sh
|       |       |-- fine
|       |       |   |-- dakota_grid.in
|       |       |   |-- driver.py
|       |       |   |-- inputs_template.txt
|       |       |   `-- start_dakota.sh
|       |       `-- random
|       |           |-- dakota_random.in
|       |           |-- driver.py
|       |           |-- inputs_template.txt
|       |           `-- start_dakota.sh
|       |-- Cumulative\ Distibution\ of\ Elevation.pdf
|       |-- analysis_grid_coarse_800_sew.py # python analysis files of grid searches
|       |-- analysis_grid_fine_800_sew.py
|       |-- analysis_random_800_sew.py
|       `-- analysis_sew_elev_area_distribution.py
|-- metric_and_objective_function_calculation
|   |-- develop_chi_metric # exploratory analysis
|   |   |-- chi_saver_test
|   |   |-- exploratory_analysis.py
|   |   `-- testdensitysave.csv
|   |-- identifying_chi_cat_catagories
|   |   |    ### IMPORTANT FILES ###
|   |   |-- chi-elev-catagories.py # file used to create the chi-elevation categories.
|   |   | ### END IMPORTANT FILES ###
|   |   `-- *.png # many diagnostic plots
|   |-- metric_calculator # calculation package for making metrics
|   |   |-- __init__.py
|   |   |    ### IMPORTANT FILES ###
|   |   |-- grouped_differences.py # python submodules for calculating metrics
|   |   |-- metric_calculator.py
|   |   |-- metric_difference.py
|   |   `-- ncextractor.py
|   |-- weigh_metrics
|   |   |-- metric_weighing_scratch.py
|   |   |-- metrics.chi.txt
|   |   `-- weight_metrics.py
|   |    ### END IMPORTANT FILES ###
|   |-- metrics.chi.txt
|   |-- metrics.txt
|   `-- setup.py # compilation file.
|-- misc_utilities
|   `-- remove_model_files.py
|-- multi_model_analysis # folder for doing multi_model_analysis, this was done as part of prediction.
|-- prediction
|   |-- gully # placeholder folder made in expectation of running gully scale predictions. This was not done.
|   |-- sew
|   |   |-- BEST_PARAMETERS # predictions at BEST_PARAMETER value
|   |   |   |-- model_MMM # Experiment file structure. Below this level is autogenerated
|   |   |   |   `-- LLL.III.CCC
|   |   |   |       |-- BEST_PARAMETERS.model_MMM.LLL.III.CCC.elev_change.png # example figures
|   |   |   |       |-- BEST_PARAMETERS.model_MMM.LLL.III.CCC.png
|   |   |   |       |-- driver.py # model driver
|   |   |   |       |-- inputs.txt # input file
|   |   |   |       |-- model_MMM_TTT.nc # output netcdfs
|   |   |   |       `-- usage.txt
|   |   |   |-- topo_figures # folder for output topography.
|   |   |   |-- cleanup_before_RCPrestart.py # cleanup files for restarting runs
|   |   |   |-- cmd_lines # submission file
|   |   |   |-- compile_output.py # python script to compile output.
|   |   |   |-- create_and_run_best_parameters.py # job creation file
|   |   |   |-- replot_figures.py # script to re plot figures
|   |   |   |-- slurm-XXX.out # slurm output
|   |   |   |-- submit_best_parameter_runs.sh # submission file.
|   |   |   `-- testing_100_variant_climate_parameters.py # exploratory analysis script
|   |   |-- BREACHING
|   |   |   |-- gis_files_and_profiles # GIS files showing the location of profiles used in breaching geometry calcuation.
|   |   |   |   |-- Gully\ Breaching\ Profile.csv
|   |   |   |   |-- Gully\ Breaching\ Profile.png
|   |   |   |   |-- Heinz\ Fan\ Profile.csv
|   |   |   |   |-- Heinz\ Fan\ Profile.png
|   |   |   |   |-- breaching_paths.dbf
|   |   |   |   |-- breaching_paths.prj
|   |   |   |   |-- breaching_paths.qpj
|   |   |   |   |-- breaching_paths.shp
|   |   |   |   `-- breaching_paths.shx
|   |   |   |-- model_MMM # Experiment file structure. Below this level is autogenerated
|   |   |   |   `-- LLL.III.CCC
|   |   |   |       `-- BBB.BYY
|   |   |   |           |-- BREACHING.model_MMM.LLL.III.CCC.BBB.BYY.elev_change.png
|   |   |   |           |-- BREACHING.model_MMM.LLL.III.CCC.BBB.BYY.png
|   |   |   |           |-- driver.py # model driver
|   |   |   |           |-- elevation_at_points_df.csv
|   |   |   |           |-- inputs.txt
|   |   |   |           |-- model_MMM_TTT.nc
|   |   |   |           `-- usage.txt
|   |   |   |-- topography_figures # compilation of topography figures
|   |   |   |-- breaching_table.txt # summary table
|   |   |   |-- cmd_lines # submission file
|   |   |   |-- compilation_of_sew_breaching_uncert_output.csv # summary table
|   |   |   |-- compile_output.py # python script to compile output
|   |   |   |-- create_and_run_breaching.py # python script to create experiment
|   |   |   |-- re_plot_figures.py # python script to replot figures
|   |   |   |-- submit_best_parameter_runs.sh # not used
|   |   |   `-- submit_breaching_runs.sh # submission script.
|   |   |-- IC_UNCERTAINTY
|   |   |   |-- model_MMM model_MMM # Experiment file structure. Below this level is autogenerated
|   |   |   |   `-- LLL.III.CCC
|   |   |   |       `-- run.RR
|   |   |   |           |-- IC_UNCERTAINTY.model_MMM.LLL.III.CCC.run.RRR.elev_change.png
|   |   |   |           |-- IC_UNCERTAINTY.model_MMM.LLL.III.CCC.run.RRR.png
|   |   |   |           |-- driver.py
|   |   |   |           |-- elevation_at_points_df.csv
|   |   |   |           |-- inputs.txt
|   |   |   |           `-- model_MMM_TTT.nc
|   |   |   |-- topography_figures #synthesis topography figures
|   |   |   |-- cmd_lines
|   |   |   |-- compilation_of_sew_IC_uncert_output.csv
|   |   |   |-- compile_output.py # python script to compile output
|   |   |   |-- create_and_run_initial_condition.py # python script to create experiment
|   |   |   |-- re_plot_842_figures.py # replotting script
|   |   |   |-- re_plot_figures.py
|   |   |   `-- submit_best_parameter_runs.sh # submission script
|   |   |-- PARAMETER_UNCERTAINTY
|   |   |   |-- model_MMM model_MMM # Experiment file structure. Below this level is autogenerated
|   |   |   |   |-- LLL.III.CCC
|   |   |   |   |   |-- surrogates
|   |   |   |   |   |   `-- SUR # Folder for sampling of location SUR
|   |   |   |   |   |       |-- complex_samples.dat # complex model samples used to construct the surrogate
|   |   |   |   |   |       |-- dakota_create_and_sample_surrogate.in # dakota input file
|   |   |   |   |   |       |-- dakota_create_and_sample_surrogate.out # dakota output file
|   |   |   |   |   |       |-- dakota_surrogate.log # dakota output file
|   |   |   |   |   |       |-- dakota_surrogate_pred.rst # dakota output file
|   |   |   |   |   |       |-- fort.13 # dakota output file
|   |   |   |   |   |       |   ### IMPORTANT FILES ###
|   |   |   |   |   |       |   #  This is the location of the surrogate samples of predictions.
|   |   |   |   |   |       `-- model_MMM.LLL.III.CCC.surrogate_samples.surrogate_samples.dat # Surrogate evaluations
|   |   |   |   |   |          ### END IMPORTANT FILES ###
|   |   |   |   |   |-- LHS_2.out # Dakota output
|   |   |   |   |   |-- LHS_3.out # Dakota output
|   |   |   |   |   |-- LHS_7.out # Dakota output
|   |   |   |   |   |-- LHS_8.out # Dakota output
|   |   |   |   |   |-- LHS_9.out # Dakota output
|   |   |   |   |   |-- LHS_distributions.out # Dakota output
|   |   |   |   |   |-- LHS_samples.out # Dakota output
|   |   |   |   |   |-- cmnd_lines # submission file
|   |   |   |   |   |-- cmnd_lines.sh # submission file
|   |   |   |   |   |-- dakota.log # Dakota output
|   |   |   |   |   |-- dakota_lhc_pred.in # Dakota LHC input
|   |   |   |   |   |-- dakota_lhc_pred.out # Dakota output
|   |   |   |   |   |-- dakota_pred.rst # Dakota output
|   |   |   |   |   |-- driver.py # model driver
|   |   |   |   |   |-- inputs.txt # input file
|   |   |   |   |   |-- inputs_template.txt # model input template
|   |   |   |   |   |-- slurm-XXX.out # slurm output
|   |   |   |   |   |-- start_dakota.sh # submission file
|   |   |   |   |   |-- submit_surrogate_evaluations.sh # submission script
|   |   |   |   |   `-- wv_model_MMM_prediction_sampling.dat  # Dakota output
|   |   |   |   `-- posterior.dat # posterior parameter distribution for this model from calibration  
|   |   |   |-- topography_figures # compilation of parameter uncertainty figures, grouped by model.
|   |   |   |   `-- model_MMM
|   |   |   |-- cmd_lines # submission file
|   |   |   |-- compile_output_figures.py # script to compile output figures
|   |   |   |-- create_and_run_param_uncertainty.py # script to construct the complex model sampling part of the experiment
|   |   |   |-- create_surrogate_dakota_files.py # script to construct the surrogate sampling part of the experiment
|   |   |   |-- dakota_create_and_sample_surrogate_template.in # input template for sampling the surrogate
|   |   |   |-- dakota_lhc_template.in # input template for making complex model evaluations
|   |   |   |-- start_complex_sampling_for_surrogate.sh # submission of complex model evaluation.
|   |   |   |-- submit_all_surrogate_evals.sh # submission of surrogate sampling
|   |   |   `-- submit_best_parameter_runs.sh # not used
|   |   |-- synthesis_netcdfs
|   |   |   |-- all800s_synthesis_TTT.nc
|   |   |   `-- only842_synthesis_TTT.nc
|   |   |-- synthesis_plots # many .png files synthesizing the results
|   |   |-- testing # exploratory analysis
|   |   |   ### IMPORTANT FILES ###
|   |   |     # this is the file that identifies which row/column pairs in the grid are used for surrogate sampled sites.
|   |   |-- PredictionPoints_ShortList.csv # file of the points used
|   |   |   ### END IMPORTANT FILES ###
|   |   |-- cross_model.csv # helper file used to make finding and indexing output files faster.
|   |   |-- expected_value_topography.nc # not used
|   |   |-- initial_condition.csv # helper file used to make finding and indexing output files faster.
|   |   |-- launch_combined_prediction_estimates.sh # submission script for prediction estimates
|   |   |-- launch_make_synthesis_plots.sh # submission script for synthesis plots
|   |   |-- make_combined_prediction_estimates.py # script that synthesizes topography into synthesis netcdfs
|   |   |-- make_input_templates_and_drivers_best_param.py # files to make input templates
|   |   |-- make_input_templates_and_drivers_breaching.py
|   |   |-- make_input_templates_and_drivers_ic_uncert.py
|   |   |-- make_input_templates_and_drivers_param_uncertain.py
|   |   |-- make_synthesis_plots.py # script to make synthesis plots once synthesis netcdfs are made.
|   |   |-- output_summary_file.csv # summary file of erosion values through time.
|   |   |-- slurm-XXX.out
|   |   `-- synthesis_log_ file.txt
|   |-- clean_up_rst_and_fail_logs.py # helper script for full restart of runs
|   |-- launch_R_plotting_and_compiling_scripts.sh # submission script for all R plotting
|   |-- make_inset_plots.R # R script to make inset plots
|   |-- make_prediction_summary_plots.R # R scripts to analyze all parts of prediction
|   |-- prediction_IC_uncert_plots_and_tables.R
|   |-- prediction_breach_uncert_plots_and_tables.R
|   |-- prediction_param_uncert_plots_and_tables.R
|   |-- run_make_prediction_summary.sh # submission of part of anaylsis
|   |-- run_param_uncert.sh # submission of part of prediction analysis
|   `-- slurm-XXX.out
|-- result_figures # folder with figures produced by different experiments' analysis scripts
|   |              # figures are located in a parallel structure to the experiments themselves
|   |-- calibration
|   |   `-- sew
|   |       `-- BEST_PARAMETERS
|   |-- edited_for_report # folder with all figures that were hand edited for the report.
|   |   |-- BumpyObjectiveFunction.pdf
|   |   |-- CapturedNodes.pdf
|   |   |-- capture_scenario_explanation.pdf
|   |   |-- cat.morris_df_short.model_802.lowering.bar.pdf
|   |   |-- cat.morris_df_short.model_802.lowering.bar.withannotations.pdf
|   |   |-- create_capture_node_figure.py
|   |   |-- method_of_morris_interpretation.pdf
|   |   |-- model_000_cat_misfit.pdf
|   |   |-- model_000_cat_misfit_simple.pdf
|   |   |-- model_analysis_flowchart.pdf
|   |   |-- modelapproachflowchart.pdf
|   |   |-- scatterplotmatrix_sew_802_lowering_history_0_pg24f_ic5etch.pdf
|   |   |-- sew.calibration_of_figure.withValidation.pdf
|   |   |-- sew.calibration_of_figure.with_error.pdf
|   |   |-- sew_EGO_EGONL2_comparison.pdf
|   |   |-- sew_stochastic_comparison.pdf
|   |   `-- trace21ka_mean_over_time.pdf
|   |-- prediction
|   |   `-- sew
|   |       |-- BEST_PARAMETERS
|   |       |-- BREACHING
|   |       |-- IC_UNCERTAINTY
|   |       |-- INSET_PLOTS
|   |       `-- PARAMETER_UNCERTAINTY\
|   |-- sensitivity_analysis
|   |   |-- gully
|   |   `-- sew
|   |-- site_shapefile # shapefile of the site plotted on many figures.
|   |   |-- WVSITEFIG_SP_ft.CPG
|   |   |-- WVSITEFIG_SP_ft.dbf
|   |   |-- WVSITEFIG_SP_ft.sbn
|   |   |-- WVSITEFIG_SP_ft.sbx
|   |   |-- WVSITEFIG_SP_ft.shp
|   |   `-- WVSITEFIG_SP_ft.shx
|   |-- calibration_appendix_figures.tex
|   |-- create_SA_figure_appendix.py # python scripts to create the LaTeX .tex files for the report appendix
|   |-- create_calib_figure_appendix.py # python scripts to create the LaTeX .tex files for the report appendix
|   |-- create_pred_figure_appendix.py # python scripts to create the LaTeX .tex files for the report appendix
|   |-- prediction_appendix_figures.tex
|   |-- sensitivity_analysis_appendix_figures.tex
|   `-- unique_field_names.csv # list of unique field names
|-- result_tables  # folder with tables produced by different experiments' analysis scripts
|   |              # figures are located in a parallel structure to the experiments themselves
|   |-- calibration
|   |   |-- gully
|   |   `-- sew
|   |-- misc_tables
|   |-- prediction
|   |   `-- sew
|   |       |-- BEST_PARAMETERS
|   |       |-- BREACHING
|   |       |-- IC_UNCERTAINTY
|   |       `-- PARAMETER_UNCERTAINTY
|   |-- sensitivity_analysis
|   |   |-- gully
|   |   `-- sew
|   |-- calibration_ego2_tables.tex
|   |-- calibration_probabilities.csv
|   |-- calibration_summary_table.txt
|   |-- create_calibration_ego_param_appendix.py # script to create the EGO2 result appendix tables.
|   |-- create_calibration_fixed_value_table.py # script to create the table identifying which calibration parameters were fixed.
|   |-- create_posterior_tables.py # script to create posterior parameter estimate tables
|   |-- create_weight_and_observed_value_tables.py # script to make weight and observed value tables
|   |-- field_names_used_by_models.csv
|   |-- make_final_summary_table.py # script to make final summary table
|   |-- param_values_fixed_calibration_table.tex
|   |-- posterior_latex_tables.tex
|   |-- summary_table_latex.tex
|   |-- unique_field_names.csv
|   `-- which_fields_used_by_models.py # script to identify which field names are used by which models.
|-- sensitivity_analysis
|   |-- sew (or gully, the folders have a are parallel structure for MOAT)
|   |   |-- DELSA # this method was started, but not finished.  
|   |   |-- MOAT
|   |   |   |-- model_MMM # Experiment file structure. Below this level is autogenerated
|   |   |   |   |-- LLL.III # lowering and initial condition folder
|   |   |   |   |   |-- cmd_lines_all # submission script
|   |   |   |   |   |-- dakota.rst # dakota restart file
|   |   |   |   |   |-- dakota_moat.in # dakota input file
|   |   |   |   |   |-- dakota_moat.out # dakota output file
|   |   |   |   |   |-- driver.py # model driver
|   |   |   |   |   |-- inputs_template.txt # model input template
|   |   |   |   |   |-- run.log
|   |   |   |   |   `-- stderr.log
|   |   |   |   |-- all_cmd_lines
|   |   |   |   |-- cmd_lines_0
|   |   |   |   `-- submit_script_0.sh # submission script for this model.
|   |   |   |-- MOAT_SEW_latex_combined.log
|   |   |   |-- MOAT_SEW_latex_combined.tex # summary of MOAT in LaTeX tables
|   |   |   |-- analysis.py # analysis script
|   |   |   |-- cleanup_submit_jobs_to_summit_XX.sh # script to resubmit jobs
|   |   |   |-- compilation_of_output.py # script to compile output
|   |   |   |-- create_and_run_dakota_files.py # script to create experiment
|   |   |   |-- how_many_remain.py # helper script
|   |   |   |-- job_cleanup.py # script to cleanup and create resubmission files
|   |   |   |-- launch_compilation.sh # submission of compilation script
|   |   |   |-- launch_job_creation.sh # submission of job creation
|   |   |   |-- launch_metric_recalculation.sh # submission of metric recalculation
|   |   |   |-- make_R_plots.R # this script not used
|   |   |   |-- number_remaining.csv # summary of the number of jobs remaining
|   |   |   |-- sensitivity_analysis_parameters.txt
|   |   |   |-- status_of_runs.py # script used to monitor the status of runs.
|   |   |   `-- submit_jobs_to_summit_0.sh # script used to submit initial jobs to summit
|   |   |-- make_input_templates_and_drivers.py
|   |   `-- monitor_queue.py
|   |-- launch_cat_metric_recalc.sh
|   |-- launch_metric_recalc.sh
|   |-- launch_sensitivity_analysis_jobs.sh
|   |-- make_sensitivity_analysis_R_plots.R
|   |-- recalculate_cat_metrics.py
|   `-- recalculate_metrics.py
|-- stream_profiles # exploratory analysis
|-- task32a # files associated with task 3.2 A
|-- task32b # files associated with task 3.2 B
|-- task33 # files associated with task 3.3
|-- templates # templates used for input file and driver construction
|-- testing # a variety of tests and exploratory analysis.
|   |-- test
|   |   `-- model_842 # a test of model 842
|   |       `-- LLL.III.CCC
|   |           |-- model_842.LLL.III.CCC.png
|   |           |-- model_842_TTT.nc
|   |           |-- usage.txt
|   |           |-- driver.py
|   |           `-- inputs.txt
|   |-- test_runs
|   |   |-- calibration_tests # tests of the calibration method, mostly to determine the correct step size to use.
|   |   |   |-- model_MMM_CTS
|   |   |   |   |-- OPTIM
|   |   |   |   |   `-- run.RRR
|   |   |   |   |-- OPT_DEFAULT.out
|   |   |   |   |-- dakota.rst
|   |   |   |   |-- dakota_calib.in
|   |   |   |   |-- dakota_calib.out
|   |   |   |   |-- driver.py
|   |   |   |   |-- inputs_template.txt
|   |   |   |   |-- optim.log
|   |   |   |   |-- start_dakota.sh
|   |   |   |   `-- wv_model_MMM_test_calib.dat
|   |   |   |-- cmd_lines
|   |   |   |-- compilation_of_output.py
|   |   |   |-- launch_both.sh
|   |   |   `-- slurm-XXX.out
|   |   |-- grid_search # grid searches used to determine how the optimal parameter set found by the calibration method matches up with a grid search result.
|   |   |   |-- model_MMM
|   |   |   |   |-- GRID
|   |   |   |   |   `-- run.RRR
|   |   |   |   |-- dakota.rst
|   |   |   |   |-- dakota_grid.in
|   |   |   |   |-- dakota_grid.out
|   |   |   |   |-- driver.py
|   |   |   |   |-- inputs_template.txt
|   |   |   |   |-- optim.log
|   |   |   |   |-- slurm-XXX.out
|   |   |   |   |-- start_dakota.sh
|   |   |   |   `-- wv_model_MMM_test_grid.dat
|   |   |   |-- compilation_of_output.py
|   |   |   `-- recalculate_metrics.py
|   |   |-- line_search # line searches done to identify the character of the objective function surface.
|   |   |   `-- model_MMM
|   |   |       |-- var_D
|   |   |       |   |-- GRID
|   |   |       |   |-- dakota.log
|   |   |       |   |-- dakota.rst
|   |   |       |   |-- dakota_line_search.in
|   |   |       |   |-- dakota_line_search.out
|   |   |       |   |-- driver.py
|   |   |       |   |-- inputs_template.txt
|   |   |       |   |-- slurm-XXX.out
|   |   |       |   |-- start_dakota.sh
|   |   |       |   `-- wv_model_000_line_search_var_D.dat
|   |   |       `-- var_K
|   |   |           |-- GRID
|   |   |           |-- dakota.log
|   |   |           |-- dakota.rst
|   |   |           |-- dakota_line_search.in
|   |   |           |-- dakota_line_search.out
|   |   |           |-- driver.py
|   |   |           |-- inputs_template.txt
|   |   |           |-- slurm-XXX.out
|   |   |           |-- start_dakota.sh
|   |   |           `-- wv_model_000_line_search_var_K.dat
|   |   |-- SEW_cat.png             # a variety of post processing scripts and associated figures based on the calibration, grid, and line tests.
|   |   |-- Vary_D.pdf
|   |   |-- Vary_D.png
|   |   |-- Vary_D_2.pdf
|   |   |-- Vary_D_2.png
|   |   |-- Vary_K.pdf
|   |   |-- Vary_K.png
|   |   |-- analysis_000.py
|   |   |-- analysis_000_correlation.py
|   |   |-- analysis_000_line_search.py
|   |   |-- analysis_000_with_chi_cat_resid.py
|   |   |-- analysis_000_with_model800_based_weights.py
|   |   |-- analysis_000_with_topo_resid.py
|   |   |-- analysis_040_line_search.py
|   |   |-- analysis_800.py
|   |   |-- analysis_800_with_chi_cat_resid.py
|   |   |-- df_grid_000.csv
|   |   |-- df_grid_800.csv
|   |   |-- model_000_cat_misfit.pdf
|   |   |-- model_000_cat_misfit.png
|   |   |-- model_000_cat_pits.png
|   |   |-- model_000_damisfit.png
|   |   |-- model_000_er_misfit.png
|   |   |-- model_000_topo_misfit.png
|   |   |-- model_100.pdf
|   |   |-- model_800_3d.png
|   |   |-- model_800_cat_misfit_Kr_D.png
|   |   |-- model_800_cat_misfit_Kt_D.png
|   |   |-- model_800_cat_misfit_Kt_Kr.png
|   |   |-- recalculate_metrics
|   |   |-- slopecrit_relationship.png
|   |   |-- start_dakota_orig.sh
|   |   `-- test_grid_boundaries.nc0001.nc
|   |-- check_moat_numbers_same.py # utility script
|   |-- double_check_moat_input_values.py #
|   |-- move_lowering3.py # utility script
|   |-- seedcheck.py # utility script
|   `-- verifying_moat_analysis_against_dakota_moat # utility script
|-- time_test # exploratory analyses for time estimates
|-- validation
|   |-- gully # placeholder folder for doing gully validation (which was not done).
|   `-- sew
|       |-- BEST_PARAMETERS # an initial validation attempt that we determined used a too-large domain.
|       |-- BEST_PARAMETERS_SMALL_DOMAIN # folder of the validation experiment
|       |   |-- figures # compiled topography figures
|       |   |-- model_MMM # Experiment file structure. Below this level is autogenerated
|       |   |   `-- LLL.III # lowering and initial condition folder
|       |   |       |-- driver.py # model driver
|       |   |       |-- inputs.txt # model input
|       |   |       |-- model_MMM_TTT.nc # model output netcdf file
|       |   |       |-- outputs_for_analysis.txt # summary of output
|       |   |       `-- usage.txt # usage file
|       |   |-- analysis.py # script to do postprocessing analysis
|       |   |-- cmd_lines # submission file
|       |   |-- create_and_run_best_parameters.py # experiment construction script
|       |   |-- make_latex_summary.py # script to make LaTeX summary file.
|       |   |-- slurm_XXX.out
|       |   |-- submit_best_parameter_runs.sh # submission file
|       |   |-- validation_summary.csv
|       |   `-- validation_summary.txt
|       `-- make_input_templates_and_drivers_best_param.py # imput template construction script.
|-- README.md # The readme for the calc package
```

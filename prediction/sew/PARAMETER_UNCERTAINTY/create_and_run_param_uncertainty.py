#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:30:21 2017

@author: barnhark
"""

# Run one evaluation from each model, using the best parameters

import os
import numpy as np
import pandas as pd
import shutil
import datetime
import glob
from landlab.io import read_esri_ascii
import yaml
import codecs

from joblib import Parallel, delayed

#from dakotathon.utils import add_dyld_library_path
#add_dyld_library_path()

# get the current files filepath
dir_path = os.path.dirname(os.path.realpath(__file__))
loc = dir_path.split(os.sep)[-2]

###############
dakota_analysis_driver = 'driver.py'
###############

ncore = 24

def create_model_jobs(model_name=None,
                      model_driver_name=None,
                      seed=None,
                      input_template_lines=None,
                      inital_dems=None,
                      lowering_and_climate=None,
                      model_dictionary=None,
                      parameter_dictionary=None,
                      dir_path=None,
                      input_template_folderpath=None,
                      model_time=None,
                      order_dict=None,
                      initial_parameter_values=None,
                      mean_parameter_values=None,
                      std_parameter_values=None,
                      metric_names=None,
                      mcmc_bounds=None):

    """Create PARAMETER UNCERTAINTY prediction model jobs."""
#    total_number_of_jobs = 0
#    # initialize container for all command lines and submission scripts.
#    all_cmd_lines = []
#    this_models_submission_scripts = []

    # use one seed per model for consistency across catagorical variables.

    seed = int(seed)
    cmnd_line = []

    # for initial condition in initial conditions
    for dem_filepath in inital_dems:
        dem_name = os.path.split(dem_filepath)[-1].split('.')[0]

        # for lowering history in lowering future and climate in climate futures.
        for cli in range(len(lowering_and_climate)):
            lowering_filepath = lowering_and_climate[cli][0]
            climate_future = lowering_and_climate[cli][1]

            # Determine Name and Create folder

            lowering_name = os.path.split(lowering_filepath)[-1].split('.')[0]

            climate_name = os.path.split(climate_future)[-1].split('.')[1]
            folder_name = '.'.join([lowering_name, dem_name, climate_name])
            new_folder_path = os.path.join(dir_path, model_name, folder_name)

            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)

            # get climate vars from parameter file.
            with open(climate_future, 'r') as f:
                climate_vars = yaml.load(f)

            # set the variable names, ranges, and descriptors
            # its important that these are always in the same order.\
            order_info = order_dict[model_name]
            variables = sorted(order_info, key=order_info.get)


            # parameter bounds:
            upper_bounds = []
            lower_bounds = []

            for var in variables:
                prior_min = parameter_dictionary[var]['min']
                prior_max = parameter_dictionary[var]['max']

                mcmc_min = mcmc_bounds[model_name][var]['min']
                mcmc_max = mcmc_bounds[model_name][var]['max']

                upper_bounds.append(str(min(mcmc_max, prior_max)))
                lower_bounds.append(str(max(mcmc_min, prior_min)))


            # Modify input template
            input_template = []
            for line in input_template_lines:
                line = line.replace('{inital_DEM_file}', dem_filepath)
                line = line.replace('{lowering_history_file}', lowering_filepath)
                line = line.replace('{output_filename}', model_name)
                if line.startswith('outlet_id'):
                    outlet_id = int(line.split(':')[-1].strip())

                # put in best parameter values

                line = line.strip(' \t\n\r')
                input_template.append(line+'\n')

            # find outlet end_value
            # read modern grid to get current outlet elevation
            (temp_grid, temp_z) = read_esri_ascii(dem_filepath,
                                                  name='topographic__elevation',
                                                  halo=1)

            current_outlet_elevation = temp_z[outlet_id]

            with open(lowering_filepath, 'r') as lhfp:
                llines = lhfp.readlines()
            end_change = float(llines[-1].split(',')[-1])
            modern_outlet_elevation = current_outlet_elevation + end_change

            input_template.append('modern_outlet_elevation: ' + str(modern_outlet_elevation) + '\n')
            input_template.append('points_file: ' + plot_locations + '\n')
            # append climate variables:
            for cv in climate_vars:
                input_template.append(cv + ': ' + str(climate_vars[cv]) +'\n')

            # Write input template
            with open(os.path.join(new_folder_path, 'inputs_template.txt'), 'w') as itfp:
                itfp.writelines(input_template)

            # Copy correct model driver
            model_driver = os.path.join(new_folder_path, 'driver.py')
            shutil.copy(model_driver_name, model_driver)

            # Create Dakota files
            # analysis driver is driver name with name of working directory
            with open(os.path.join(dir_path, 'dakota_lhc_template.in'), 'r') as f:
                dakota_lines = f.readlines()

            dakota_file = []
            str_var = [repr(v) for v in variables]
            str_met = [repr(m) for m in metric_names]

            evaluation_concurrency = (5*24) - 1
            recovery_string = str(len(metric_names))+'*'+str(1e6)

            for line in dakota_lines:
                line = line.replace('{model_name}', model_name)

                line = line.replace('{lowering_history}', os.path.split(lowering_filepath)[-1].split('.')[0])
                line = line.replace('{initial_condition}', os.path.split(dem_filepath)[-1].split('.')[0])
                line = line.replace('{climate_future}', climate_name)
                line = line.replace('{loc}', loc)
                line = line.replace('{evaluation_concurrency}', str(evaluation_concurrency))

                line = line.replace('{num_variables}', str(len(variables)))
                line = line.replace('{variable_names}', ' '.join(str_var))

                line = line.replace('{upper_bounds}', ' '.join(upper_bounds))
                line = line.replace('{lower_bounds}', ' '.join(lower_bounds))

                line = line.replace('{num_responses}', str(len(metric_names)))
                line = line.replace('{responses_names}', ' '.join(str_met))

                line = line.replace('{recovery_values}', recovery_string)

                line = line.replace('{seed}', str(seed))

                dakota_file.append(line)

            # Write dakota input file
            with open(os.path.join(new_folder_path, 'dakota_lhc_pred.in'), 'w') as dakota_f:
                dakota_f.writelines(dakota_file)


            # to actually submit the jobs to summit, create a unique submit script
            # for each cmd_lines chunk.
            script_contents = ['#!/bin/sh',
                               '#SBATCH --job-name ' + model_name + '_sursample',
                               '#SBATCH --ntasks-per-node 24',
                               '#SBATCH --partition shas',
                               '#SBATCH --mem-per-cpu 4GB',
                               '#SBATCH --nodes 5',
                               '#SBATCH --time 24:00:00',
                               '#SBATCH --account ucb19_summit1',
                               '',
                               '# load environment modules',
                               'module load intel/16.0.3',
                               'module load openmpi/1.10.2',
                               'module load cmake/3.5.2',
                               '#module load perl',
                               'module load mkl',
                               'module load gsl',
                               '',
                               '# make sure environment variables are set correctly',
                               'source ~/.bash_profile',
                               '## run dakota using a restart file if it exists.',
                               'if [ -e dakota_pred.rst ]',
                               'then',
                               'dakota -i dakota_lhc_pred.in -o dakota_lhc_pred.out --read_restart dakota_pred.rst --write_restart dakota_pred.rst &> dakota.log',
                               'else',
                               'dakota -i dakota_lhc_pred.in -o dakota_lhc_pred.out --write_restart dakota_pred.rst &> dakota.log',
                               'fi']

            script_path = os.path.join(new_folder_path,'start_dakota.sh')

            with open(script_path, 'w') as f:
                for line in script_contents:
                    f.write(line+"\n")

            cmnd_line.append('cd ' + os.path.abspath(new_folder_path) + '; sbatch start_dakota.sh')

    return (len(variables)+1, cmnd_line)

# Define filepaths. Here these are given as lists, for cross platform
# compatability

input_template_folderpath = ['..', '..', '..', 'templates']

parameter_dict_folderpath = ['..', '..', '..', 'auxillary_inputs']

dem_folderpath = ['..', '..', '..', 'auxillary_inputs', 'dems', 'sew', 'modern']

lowering_history_folderpath = ['..', '..', '..', 'auxillary_inputs', 'lowering_histories']
climate_future_folderpath = ['..', '..', '..', 'auxillary_inputs', 'climate_futures']
models_driver_folderpath = ['..', '..', '..', 'drivers', 'models']

dakota_driver_folderpath = ['..', '..', '..', 'drivers', 'dakota']

# loop within models.

# Get model space and parameter ranges (these will be loaded in from a file)
metric_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['metric_names.csv'])))
model_time_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_time.csv'])))
parameter_range_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['parameter_ranges.csv'])))
model_parameter_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_parameter_match_calibration_sew.csv'])))
model_parameter_order_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_parameter_order.csv'])))
model_parameter_start_value_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_parameter_calibration_start_values_sew.csv'])))

#metric_df = pd.read_csv(metric_input_file)
time_df = pd.read_csv(model_time_input_file)

param_range_df = pd.read_csv(parameter_range_input_file)

model_param_df = pd.read_csv(model_parameter_input_file)
model_param_df = model_param_df.dropna() # Remove those rows with models that are not yet complete

order_df = pd.read_csv(model_parameter_order_file)
order_df = order_df.dropna() # Remove those rows with models that are not yet complete
order_dict = {}

start_val_df = pd.read_csv(model_parameter_start_value_file)
start_val_df = start_val_df.dropna() # Remove those rows with models that are not yet complete
start_dict = {}

## these weights are provided as the variance, we want to weight by 1/variance,
##weights_df = 1./pd.read_csv(os.path.join(*weights_filepath), header=None, names=['weight'])
#
#variance_df = pd.read_csv(os.path.join(*weights_filepath),index_col=0)
#weights_df = pd.DataFrame(1.0/(variance_df.Model_800_Based_Variance + variance_df.Error_Based_Variance))
#weights_df.columns = ['weight']
#temp = {}
#for w in list(weights_df.index):
#    temp[w] = float(weights_df.weight[w])
#weights_dict ={'weights':temp}

# construct model time and model dictionary:
# first get all model ID numbers and same them to a dict with the equivalent
# lenght three padded strings
mids = {i: str(i).rjust(3, '0') for i in model_param_df['ID'].values}

mids = ['800', '802', '804', '808', '810', '840', '842', 'A00', 'C00'] # only do prediction parameter uncertainty for a few models
# ,

# initialize data structures
model_time = {}
model_dictionary = {}
seed_dict = {}

# get parameter bounds based on the MCMC results
mcmc_bounds = {}

for mid in mids:
    dakota_out_folderpath = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration',  'sew' , 'QUESO_DRAM', 'model_'+mid, 'lowering_history_0.pg24f_ic5etch', 'dakota_queso_dram.out']
    dakota_out_file = os.path.join(os.path.abspath(os.sep), *dakota_out_folderpath)

    with codecs.open(dakota_out_file, "r",encoding='utf-8', errors='ignore') as f:
        dakota_out_lines = f.readlines()

    dakota_out_text = ''.join(dakota_out_lines)

    parameter_distribution_lines = dakota_out_text.split('Sample moment statistics for each posterior variable:')[-1].split('Sample moment statistics for each response function:')[0].strip().split('\n')

    stat_names = ['Mean', 'Std Dev', 'Skewness', 'Kurtosis']

    temp_dict = {}
    for line in parameter_distribution_lines[1:]:
        vals = line.strip().split()
        param_name = vals[0]
        mean = float(vals[1])
        std = float(vals[2])

        temp_dict[param_name] = {'min': mean - (3.0*std),
                                 'max': mean + (3.0*std)}
    mcmc_bounds['model_'+mid] = temp_dict

dt_ind = np.where(model_param_df.columns == 'dt')[0][0]
parameter_names = list(model_param_df.columns[dt_ind:].values)

for mid in mids:

    #check if the model is ready
    if model_param_df.loc[model_param_df['ID']==mid]['ready'].values[0]:

        # make the model key
        key = 'model_'+mid

        # copy the estimated model run time into the dictionary
        model_time[key] = float(time_df.loc[time_df['ID']==mid]['sew'].values[0])

        # get_seed
        seed_dict[key] = int(model_param_df.loc[model_param_df['ID']==mid]['dakota_ga_seed'])

        # construct model_dictionary
        param_info = {}
        for param in parameter_names:
            if model_param_df.loc[model_param_df['ID']==mid][param].values[0] == 'variable':
                param_info[param]='var'
        model_dictionary[key] = param_info

        # construct order information
        order_info = {}
        for param in list(param_info.keys()):
            order_number = order_df.loc[order_df['ID']==mid][param].values[0]
            order_info[param] = int(order_number)
        order_dict[key] = order_info

        # start location information
        start_info = {}
        for param in list(param_info.keys()):
            start_value = start_val_df.loc[start_val_df['ID']==mid][param].values[0]
            start_info[param] = float(start_value)
        start_dict[key] = start_info

# construct parameter range dictionary
parameter_dictionary = {}
for param in parameter_names:
    range_dict ={}
    range_dict['min']=float(param_range_df.loc[param_range_df['Short Name']==param]['Minimum Value'].values[0])
    range_dict['max']=float(param_range_df.loc[param_range_df['Short Name']==param]['Maximum Value'].values[0])
    if param.startswith('linear_diffusivity'):
        range_dict['min'] = -4.0
    parameter_dictionary[param]=range_dict

# get initial condition DEMS
inital_dems = [os.path.abspath(os.path.join(*(dem_folderpath+[pth]))) for pth in ['dem24fil_ext.txt']]

# get lowering histories and get climage futures

lowering_and_climate=[]
lowering_and_climate.append([os.path.abspath(os.path.join(*(lowering_history_folderpath+['lowering_future_3.txt']))),
                            os.path.abspath(os.path.join(*(climate_future_folderpath+['climate_future_3.RCP85.txt'])))])

lowering_and_climate.append([os.path.abspath(os.path.join(*(lowering_history_folderpath+['lowering_future_1.txt']))),
                            os.path.abspath(os.path.join(*(climate_future_folderpath+['climate_future_1.constant_climate.txt'])))])
# create container for each job submission
all_submission_scripts = {}
total_number_of_jobs = 0

# for model in models
parallel_inputs = []

# get the names of the output locations
plot_locations = os.path.join(os.path.abspath(os.sep), *['work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', 'sew', 'PredictionPoints_ShortList.csv'])
plot_location_df = pd.read_csv(plot_locations)

timesteps = np.arange(0, 101)

metric_names = []
loc_names = np.sort(plot_location_df.Point_Name)
for loc_name in loc_names:
    mets = [loc_name+'.'+str(time) for time in timesteps]
    metric_names.extend(mets)

# for model in models
for model_name in  sorted(list(model_dictionary.keys())):

    # get means and stds from results_tables
    table_folderpath = ['work', 'WVDP_EWG_STUDY3', 'study3py','result_tables', 'calibration',  'sew' , 'ego2.sew.parameters.full.' + model_name + '.csv']
    table_file = os.path.join(os.path.abspath(os.sep), *table_folderpath)

    # only run if EGO2 has completed and best parameters are known.
    if os.path.exists(table_file):
        best_params = pd.read_csv(table_file, index_col=0)

        # use one seed per model for consistency across catagorical variables.
        seed = seed_dict[model_name]

        # Get input template
        input_template_filepath = os.path.abspath(os.path.join(dir_path, *(input_template_folderpath+['sew_prediction_inputs_template_'+model_name+'.txt'])))
        with open(input_template_filepath,'r') as itfp:
            input_template_lines = itfp.readlines()

        # get model driver name
        model_driver_name = os.path.abspath(os.path.join(dir_path, *(models_driver_folderpath+['sew_prediction_param_uncert_'+model_name+'_driver.py'])))

        # for model 000 we;ve done a grid search and know the "best" start value
        # for other models we try and start at the equivalent of model 000.

        pvars = list(model_dictionary[model_name].keys())
        mean_parameter_values = {}
        std_parameter_values = {}
        initial_parameter_values = {}
        for pv in pvars:
            initial_parameter_values[pv] = start_dict[model_name][pv]
            mean_parameter_values[pv]=best_params.loc[pv, 'best_parameters']
            std_parameter_values[pv]=best_params.loc[pv, 'parameter_standard_deviation']

        inputs = {'model_name': model_name,
                  'model_driver_name': model_driver_name,
                  'seed': seed,
                  'input_template_lines': input_template_lines,
                  'inital_dems': inital_dems,
                  'lowering_and_climate': lowering_and_climate,
                  'model_dictionary': model_dictionary,
                  'parameter_dictionary': parameter_dictionary,
                  'dir_path': dir_path,
                  'input_template_folderpath': input_template_folderpath,
                  'model_time': model_time,
                  'order_dict': order_dict,
                  'initial_parameter_values': initial_parameter_values,
                  'mean_parameter_values': mean_parameter_values,
                  'std_parameter_values': std_parameter_values,
                  'metric_names': metric_names,
                  'mcmc_bounds': mcmc_bounds}

        parallel_inputs.append(inputs)

print('Starting Job Creation')
output = Parallel(n_jobs=1)(delayed(create_model_jobs)(**inputs) for inputs in parallel_inputs)

os.chdir(dir_path)

cmnd_lines = []
for out in output:
    cmnd_lines.extend(out[1])

with open('start_complex_sampling_for_surrogate.sh', 'w') as f:
    for line in cmnd_lines:
        f.write(line + '\n')

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
                      lowering_histories=None, 
                      model_dictionary=None, 
                      parameter_dictionary=None, 
                      dir_path=None, 
                      input_template_folderpath=None,
                      work_directory_folderpath=None,
                      metric_names=None,
                      model_time=None,
                      order_dict=None,
                      initial_parameter_values=None, 
                      mean_parameter_values=None,
                      std_parameter_values=None, 
                      weights=None):
    
    """Create SEW QUESO DRAM MCMC model jobs."""
#    total_number_of_jobs = 0
#    # initialize container for all command lines and submission scripts. 
#    all_cmd_lines = []
#    this_models_submission_scripts = []
    
    # use one seed per model for consistency across catagorical variables. 

    seed = int(seed)
    cls = []
    # for initial condition in initial conditions
    for dem_filepath in inital_dems:
        dem_name = os.path.split(dem_filepath)[-1].split('.')[0] 
    
        # for lowering history in lowering histories
        for lowering_filepath in lowering_histories:
            lowering_name = os.path.split(lowering_filepath)[-1].split('.')[0] 
            
            # Create folder
            folder_name = '.'.join([lowering_name, dem_name])
            new_folder_path = os.path.join(dir_path, model_name, folder_name)
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            
            # set the variable names, ranges, and descriptors
            # its important that these are always in the same order.\
            order_info = order_dict[model_name]
            variables = sorted(order_info, key=order_info.get)

            # Modify input template
            input_template = []
            for line in input_template_lines:
                line = line.replace('{inital_DEM_file}', dem_filepath)
                line = line.replace('{lowering_history_file}', lowering_filepath)
                line = line.replace('{output_filename}', model_name)
                
                for var in variables:
                    vark = '{' + var + '}'
                    line = line.replace(vark, str(mean_parameter_values[var]))
                # put in best parameter values
                
                line = line.strip(' \t\n\r')
                input_template.append(line+'\n')
                
            # Write input template
            with open(os.path.join(new_folder_path, 'inputs.txt'), 'w') as itfp:
                itfp.writelines(input_template)
                
            # Copy correct model driver
            model_driver = os.path.join(new_folder_path, 'driver.py')
            shutil.copy(model_driver_name, model_driver)
            
            cmnd_line = 'cd ' + new_folder_path + '; python driver.py'
            cls.append(cmnd_line)
        
    return (len(variables)+1, cls)

# Define filepaths. Here these are given as lists, for cross platform
# compatability

input_template_folderpath = ['..', '..', '..', 'templates']

parameter_dict_folderpath = ['..', '..', '..', 'auxillary_inputs']

dem_folderpath = ['..', '..', '..', 'auxillary_inputs', 'dems', 'validation', 'initial_conditions']

lowering_history_folderpath = ['..', '..', '..', 'auxillary_inputs', 'lowering_histories']


models_driver_folderpath = ['..', '..', '..', 'drivers', 'models']

dakota_driver_folderpath = ['..', '..', '..', 'drivers', 'dakota']

work_directory_folderpath = ['work', 'WVDP_EWG_STUDY3', 'results','calibration', 'validation', 'Evaluate_Best_Parameters'] # path to scratch where model will be run.

# loop within models. 
catagory_file = os.path.join(os.path.abspath(os.sep), *['work', 'WVDP_EWG_STUDY3', 'study3py','auxillary_inputs', 'chi_elev_categories', 'validation.chi_elev_cat.20.txt'])
cat = np.loadtxt(catagory_file)
cat_vals = np.sort(np.unique(cat[cat>0]))

catagory_weight_file = os.path.join(os.path.abspath(os.sep), *['work', 'WVDP_EWG_STUDY3', 'study3py','auxillary_inputs', 'weights', 'validation.chi_elev_weight.20.txt'])

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
             
# initialize data structures
model_time = {}
model_dictionary = {}
seed_dict = {}
  
dt_ind = np.where(model_param_df.columns == 'dt')[0][0]
parameter_names = list(model_param_df.columns[dt_ind:].values)

for mid in mids.keys():
    
    #check if the model is ready
    if model_param_df.loc[model_param_df['ID']==mid]['ready'].values[0]:
        
        # make the model key
        key = 'model_'+mids[mid]
        
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

# get metric names       
#metric_names =  list(metric_df.values[:,0])
metric_names = ['chi_elev_'+str(int(cv)) for cv in cat_vals ]
temp = {}
for mn in metric_names:
    temp[mn] = 1
weights_dict = {'weights' : temp}

# get initial condition DEMS
inital_dems = glob.glob(os.path.abspath(os.path.join(*(dem_folderpath + ['*.txt']))))

# get lowering histories
lowering_histories = [os.path.abspath(os.path.join(*(lowering_history_folderpath+[pth]))) for pth in ['lowering_history_0.txt']]

# create container for each job submission 
all_submission_scripts = {}
total_number_of_jobs = 0

# for model in models
parallel_inputs = []

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
        input_template_filepath = os.path.abspath(os.path.join(dir_path, *(input_template_folderpath+['validation_best_param_inputs_template_'+model_name+'.txt'])))
        with open(input_template_filepath,'r') as itfp:
            input_template_lines = itfp.readlines()
        
        # get model driver name
        model_driver_name = os.path.abspath(os.path.join(dir_path, *(models_driver_folderpath+['validation_best_param_'+model_name+'_driver.py'])))
        
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
                  'lowering_histories': lowering_histories, 
                  'model_dictionary': model_dictionary, 
                  'parameter_dictionary': parameter_dictionary, 
                  'dir_path': dir_path, 
                  'input_template_folderpath': input_template_folderpath,
                  'work_directory_folderpath': work_directory_folderpath,
                  'model_time': model_time,
                  'metric_names': metric_names,
                  'order_dict': order_dict, 
                  'initial_parameter_values':initial_parameter_values, 
                  'mean_parameter_values':mean_parameter_values, 
                  'std_parameter_values':std_parameter_values,                   
                  'weights':weights_dict}
    
        parallel_inputs.append(inputs)
   
print('Starting Job Creation')
output = Parallel(n_jobs=1)(delayed(create_model_jobs)(**inputs) for inputs in parallel_inputs)

os.chdir(dir_path)

cmnd_lines = []
total_number_of_cores = 0
for out in output:
    total_number_of_cores += out[0]
    cmnd_lines.extend(out[1])
num_nodes = int(np.ceil(total_number_of_cores/24))

with open('cmd_lines', 'w') as f:
    for line in cmnd_lines:
        f.write(line + '\n')

with open('submit_best_parameter_runs.sh', 'w') as f:
    script = ['#!/bin/sh',
              '#SBATCH --job-name valid_sew',
              '#SBATCH --ntasks-per-node 24',
              '#SBATCH --partition shas',
              '#SBATCH --mem-per-cpu 4GB',
              '#SBATCH --nodes 1',
              '#SBATCH --time 24:00:00',
              '#SBATCH --account ucb19_summit1',
              '',
              'module purge',
              'module load intel',
              'module load impi',
              'module load loadbalance',
              'mpirun lb cmd_lines']
    for line in script:
        f.write(line+'\n')
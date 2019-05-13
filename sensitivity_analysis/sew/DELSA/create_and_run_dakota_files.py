#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os
from numpy.random import RandomState
import numpy as np
import pandas as pd
import shutil

from dakotathon import Dakota

from joblib import Parallel, delayed

#from dakotathon.utils import add_dyld_library_path
#add_dyld_library_path()

# get the current files filepath
dir_path = os.path.dirname(os.path.realpath(__file__))

###############
dakota_analysis_driver = 'parallel_lhc_driver.py'
dakota_modeling_driver = 'parallel_model_run_driver.py'

seed = 5439
###############

#%%
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
                      dakota_analysis_driver_filepath=None,
                      metric_names=None):
    """
    """
    total_number_of_jobs = 0
    
    # initialize container for all command lines and submission scripts.
    all_cmd_lines = []
    this_models_submission_scripts = []
    
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
            variables = list(model_dictionary[model_name].keys())
            lower_bounds = [parameter_dictionary[var]['min'] for var in variables]
            upper_bounds = [parameter_dictionary[var]['max'] for var in variables]
            variable_range = np.asarray(upper_bounds)-np.asarray(lower_bounds)
            step_size = variable_range/100.
            
            # Modify input template
            input_template = []
            for line in input_template_lines:
                line = line.replace('{inital_DEM_file}', dem_filepath)
                line = line.replace('{lowering_history_file}', lowering_filepath)
                line = line.replace('{output_filename}', model_name)
                line = line.strip(' \t\n\r')
                input_template.append(line+'\n')
                
            # Write input template
            with open(os.path.join(new_folder_path, 'inputs_template.txt'), 'w') as itfp:
                itfp.writelines(input_template)
            
            # Copy correct model driver
            model_driver = os.path.join(new_folder_path, 'driver.py')
            shutil.copy(model_driver_name, model_driver)
            
            # Modify centered_template
            centered_template = []
            str_var = [repr(v) for v in variables]
            str_met = [repr(m) for m in metric_names]
            for line in centered_template_lines:
                line = line.replace('{steps_per_parameter}', ('1 '*len(variables))[:-1])
                line = line.replace('{step_vector}', ' '.join(step_size.astype(str)))
                line = line.replace('{number_of_parameters}', str(len(variables)))
                initial_point = ['{'+v+'}' for v in variables]
                line = line.replace('{initial_point}', ' '.join(initial_point))
                line = line.replace('{parameter_descriptors}', ' '.join(str_var))
                line = line.replace('{analysis_driver}', dakota_model_driver_filepath)
                line = line.replace('{run_directory}', new_folder_path)
                line = line.replace('{number_of_metrics}', str(len(metric_names)))
                line = line.replace('{metrics}', ' '.join(str_met))

                centered_template.append(line)
                
            # Write centered_template
            with open(os.path.join(new_folder_path, 'dakota_centered_template.in'), 'w') as ctfp:
                ctfp.writelines(centered_template)

            # Create Dakota files 
            # analysis driver is driver name with name of working directory
            analysis_driver = 'python '+ dakota_analysis_driver_filepath + ' ' + new_folder_path
            work_directory = os.path.join(os.path.join(os.path.abspath(os.sep), *(work_directory_folderpath+[model_name, folder_name])))
            work_folder = 'center_points'
            d = Dakota(method='sampling', 
                       variables='uniform_uncertain',
                       input_file='dakota_LHC.in',
                       output_file='dakota_LHC.out',
                       interface='fork',
                       analysis_driver=analysis_driver,
                       work_directory=work_directory,
                       work_folder=work_folder,
                       run_directory=new_folder_path)
            
            # what is the analysis_components yaml file?
            #d.interface._configuration_file

            d.variables.descriptors = variables  
            d.variables.lower_bounds = lower_bounds
            d.variables.upper_bounds = upper_bounds
            
            # set remaining method properties
            d.method.sample_type = 'lhs'
            
            d.method.seed = seed
            d.method.samples = 10*len(variables)
            
            # Dakota is setting # probability levels, don't want this
            d.method.probability_levels=() # this works
        
            # set response descriptors
            d.responses.response_descriptors = metric_names
            
            # set name of template file 
            d.template_file = 'input_template.txt'
            
            # remove the environment block
            d.blocks = ('method', 'variables', 'interface', 'responses') # remove 'environment' block
           
            # name of results file and name of parameters file are the default
            # no need to set. 
            # these are located at: 
                # d.interface.parameters_file
                # d.interface.results_file
            
            # Run Dakota files
            d.write_input_file()
            d.run()

            # This will create the cmd_lines_all file with a line for each of the job scripts
            # CD to the run folder and execute the submit_jobs_to_summit.sh script
            
            # open and add these to a complete list. 
            cmd_lines_path = os.path.join(new_folder_path, 'cmd_lines_all')
            with open(cmd_lines_path, 'r') as f:
                cmd_lines = f.readlines()
            cmd_lines_clean = [c_l.strip() for c_l in cmd_lines]
            all_cmd_lines.extend(cmd_lines_clean)
                
    # once all initial conditions and boundary conditions are done initializing
    # write out all_cmd_lines, and job sized chuncks of all cmd lines. 
    
    # first write out a file with all the command lines
    model_folder_path = os.path.join(dir_path, model_name)
    with open(os.path.join(model_folder_path, 'all_cmd_lines'), 'w') as f:
        for line in all_cmd_lines:
            f.write(line+"\n")

    # next, cacluate the number of chunks.
    estimated_time_per_task = model_time[model_name]
    wall_time = 20.
    num_processors = 23.
    number_tasks_per_processor = np.floor(wall_time/float(estimated_time_per_task))
    num_task_per_job = int(num_processors*number_tasks_per_processor)
    num_jobs = np.int(np.ceil(float(len(all_cmd_lines))/float(num_task_per_job)))

    # then split up all_cmd_lines into ~20 hour sized chunks. 
    for nj in range(num_jobs):
        sel_cmd_lines = []
        for nt in range(num_task_per_job):
            try:
                sel_cmd_lines.append(all_cmd_lines.pop())
            except IndexError:
                pass
            
        cdl_file = os.path.join(model_folder_path, '_'.join(['cmd_lines',str(nj)]))
        with open(cdl_file, 'w') as f:
            for line in sel_cmd_lines:
                f.write(line+"\n")
                            
        # to actually submit the jobs to summit, create a unique submit script
        # for each cmd_lines chunk.
        script_contents = ['#!/bin/sh',
                           '#SBATCH --job-name WV_DELSA',
                           '#SBATCH --ntasks-per-node 24',
                           '#SBATCH --partition shas',
                           '#SBATCH --mem 96GB',
                           '#SBATCH --mem-per-cpu 4GB',
                           '#SBATCH --nodes 1',
                           '#SBATCH --time 24:00:00',
                           '',
                           'module purge',
                           'module load intel',
                           'module load impi',
                           'module load loadbalance',
                           'mpirun lb cmd_lines_'+str(nj)]
            
        script_name = '.'.join(['_'.join(['submit_script',str(nj)]), 'sh'])
        script_path = os.path.join(model_folder_path,script_name)
        with open(script_path, 'w') as f:
            for line in script_contents:
                f.write(line+"\n")
                    
        this_models_submission_scripts.append(script_path)
        total_number_of_jobs += 1

    #all_submission_scripts[model_folder_path]=this_models_submission_scripts
    return (total_number_of_jobs, {model_folder_path: this_models_submission_scripts})


#%%
random_process = RandomState(seed)

# Define filepaths. Here these are given as lists, for cross platform
# compatability

input_template_folderpath = ['..', '..', '..', 'templates']

parameter_dict_folderpath = ['..', '..', '..', 'auxillary_inputs']


dem_folderpath = ['..', '..',  '..','auxillary_inputs', 'dems', 'sew', 'initial_conditions']

lowering_history_folderpath = ['..', '..', '..', 'auxillary_inputs', 'lowering_histories']

models_driver_folderpath = ['..', '..',  '..', 'drivers', 'models']

dakota_driver_folderpath = ['..', '..',  '..', 'drivers', 'dakota']

work_directory_folderpath = ['work', 'WVDP_EWG_STUDY3', 'results','sensitivity_analysis', 'sew', 'DELSA'] # path to scratch where model will be run.

# Get model space and parameter ranges (these will be loaded in from a file)

metric_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['metric_names.csv'])))
model_time_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_time.csv'])))
parameter_range_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['parameter_ranges.csv'])))
model_parameter_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_parameter_match.csv'])))

metric_df = pd.read_csv(metric_input_file)
time_df = pd.read_csv(model_time_input_file)
param_range_df = pd.read_csv(parameter_range_input_file)
model_param_df = pd.read_csv(model_parameter_input_file)
model_param_df = model_param_df.dropna() # Remove those rows with models that are not yet complete

# construct model time and model dictionary:
# first get all model ID numbers and same them to a dict with the equivalent
# lenght three padded strings
mids = {i: str(i).rjust(3, '0') for i in model_param_df['ID'].values}
             
# initialize data structures
model_time = {}
model_dictionary = {}
  
dt_ind = np.where(model_param_df.columns == 'dt')[0][0]
parameter_names = list(model_param_df.columns[dt_ind:].values)

for mid in mids.keys():
    #check if the model is ready
    if model_param_df.loc[model_param_df['ID']==mid]['ready'].values[0]:

        # make the model key
        key = 'model_'+mids[mid]
    
        # copy the estimated model run time into the dictionary
        model_time[key] = float(time_df.loc[time_df['ID']==mid]['Estimated Time'].values[0])
    
        # construct model_dictionary
        param_info = {}
        for param in parameter_names:
            if model_param_df.loc[model_param_df['ID']==mid][param].values[0] == 'variable':
                param_info[param]='var'
        model_dictionary[key] = param_info
    
# construct parameter range dictionary
parameter_dictionary = {}
for param in parameter_names:
    range_dict ={}
    range_dict['min']=float(param_range_df.loc[param_range_df['Short Name']==param]['Minimum Value'].values[0])
    range_dict['max']=float(param_range_df.loc[param_range_df['Short Name']==param]['Maximum Value'].values[0])
    
    parameter_dictionary[param]=range_dict

# get metric names       
metric_names =  list(metric_df.values[:,0])
# get initial condition DEMS
inital_dems = [os.path.abspath(os.path.join(*(dem_folderpath+[pth]))) for pth in os.listdir(os.path.join(*dem_folderpath)) if (pth.endswith('.txt') and (pth.startswith('dem24fil_ext.txt')==False))]

# get lowering histories
lowering_histories = [os.path.abspath(os.path.join(*(lowering_history_folderpath+[pth]))) for pth in os.listdir(os.path.join(*lowering_history_folderpath)) if pth.endswith('.txt')]

# get the full path to the Dakota analysis and model run driver
dakota_analysis_driver_filepath = os.path.abspath(os.path.join(*(dakota_driver_folderpath+ [dakota_analysis_driver])))
dakota_model_driver_filepath = os.path.abspath(os.path.join(*(dakota_driver_folderpath+[dakota_modeling_driver])))
    
# Get the centered dakota run template, this is necessary for the present implementation of the
# two layer dakota run. 
# soon replace this with dakotathon. 
centered_template_filepath = os.path.abspath(os.path.join(*(input_template_folderpath+['dakota_centered_template.in'])))
with open(centered_template_filepath, 'r') as ctfp:
    centered_template_lines = ctfp.readlines()

# generate seeds
seeds = list(random_process.randint(500, 5000, size=(len(model_dictionary.keys()))))

# create container for each job submission 
all_submission_scripts = {}
total_number_of_jobs = 0

# for model in models
parallel_inputs = []

for model_name in  sorted(list(model_dictionary.keys())):
    # use one seed per model for consistency across catagorical variables. 
    seed = int(seeds.pop())
    
    # Get input template
    input_template_filepath = os.path.abspath(os.path.join(dir_path, *(input_template_folderpath+['inputs_template_'+model_name+'.txt'])))
    with open(input_template_filepath,'r') as itfp:
        input_template_lines = itfp.readlines()
    
    # get model driver name
    model_driver_name = os.path.abspath(os.path.join(dir_path, *(models_driver_folderpath+[model_name+'_driver.py'])))
    
    
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
              'work_directory_folderpath': work_directory_folderpath}
    
    parallel_inputs.append(inputs)
    #output = create_model_jobs(**inputs)

# run the Dakota submission in parallel, then compile 
output = Parallel(n_jobs=23)(delayed(create_model_jobs)(**inputs) for inputs in parallel_inputs)

for out in output:
    total_number_of_jobs += out[0]
    all_submission_scripts.update(out[1])

# create a set of final submission scripts that are ~100 jobs each.
final_submission_scripts = []

number_of_submission_scripts = np.ceil(float(total_number_of_jobs)/100.)
all_folder_paths = list(all_submission_scripts.keys())

for i in range(int(number_of_submission_scripts)):
    remaining_jobs = 100
    submission_contents = ['#!/bin/sh',
                           '#',
                           '#  submit_jobs_to_summit.sh',
                           '#  ',
                           '#  Created by Katherine Barnhart on 5/9/17.',
                           '']
    
    while remaining_jobs > 0:
        model_folder_path = all_folder_paths[-1]
        submission_contents.append('cd '+ model_folder_path)
        
        this_models_submission_scripts = all_submission_scripts[model_folder_path]
        
        for j in range(remaining_jobs):    
            try:
                script = this_models_submission_scripts.pop()
                submission_contents.append('sbatch '+ script)
                remaining_jobs -= 1
            
            except IndexError:
                pass
            
        if len(this_models_submission_scripts)>0:
            pass
        else:
            all_folder_paths.pop()
            
        if len(all_folder_paths) == 0:
            # all jobs are done. 
            remaining_jobs = 0
            
    final_submission_script = os.path.join(dir_path, 'submit_jobs_to_summit_' + str(i) + '.sh')
    with open(final_submission_script, 'w') as f:
        for line in submission_contents:
            f.write(line+"\n")
            
    final_submission_scripts.append(final_submission_script)

# run the sbatch submission script
#for final_submission_script in final_submission_scripts
#   os.system('source '+final_submission_script)

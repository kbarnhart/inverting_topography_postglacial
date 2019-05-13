    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os
import numpy as np
import pandas as pd
import shutil
import glob
pd.set_option('display.max_colwidth',1000)

from dakotathon import Dakota

from joblib import Parallel, delayed

#from dakotathon.utils import add_dyld_library_path
#add_dyld_library_path()

# get the current files filepath
dir_path = os.path.dirname(os.path.realpath(__file__))

###############
dakota_analysis_driver = 'parallel_model_run_driver.py'

###############

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
                      metric_names=None,
                      model_time=None,
                      order_dict=None):
    """Create MOAT model jobs."""
    total_number_of_jobs = 0
    # initialize container for all command lines and submission scripts. 
    all_cmd_lines = []
    this_models_submission_scripts = []
    
    # use one seed per model for consistency across catagorical variables. 
    seed = int(seed)
    
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
            lower_bounds = [parameter_dictionary[var]['min'] for var in variables]
            upper_bounds = [parameter_dictionary[var]['max'] for var in variables]
            
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

            # Create Dakota files 
            # analysis driver is driver name with name of working directory
            analysis_driver = 'python '+ dakota_analysis_driver_filepath + ' ' + new_folder_path
            work_directory = os.path.join(os.path.abspath(os.sep), *(work_directory_folderpath+[model_name, folder_name]))
            work_folder = 'run'
            d = Dakota(method='psuade_moat', 
                       variables='continuous_design',
                       input_file='dakota_moat.in',
                       output_file='dakota_moat.out',
                       interface='fork',
                       analysis_driver=analysis_driver,
                       work_directory=work_directory,
                       work_folder=work_folder,
                       run_directory=new_folder_path)

            d.variables.descriptors = variables  
            d.variables.lower_bounds = lower_bounds
            d.variables.upper_bounds = upper_bounds
            d.variables._initial_point = None
            
            # set remaining method properties
            d.method.seed = seed
            d.method.samples = 10*(len(variables)+1)
            d.method.partitions = 10*(len(variables)+1)
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
            
            # the command lines are appended, so if any exist already, they
            # should be removed before running dakota. 
            cmd_lines_path = os.path.join(new_folder_path, 'cmd_lines_all')
            if os.path.exists(cmd_lines_path):
                os.remove(cmd_lines_path)
                
            # Run Dakota files
            d.write_input_file()
            d.run()
            
            # This will create the cmd_lines_all file with a line for each of the job scripts
            # CD to the run folder and execute the submit_jobs_to_summit.sh script
            
            # open and add these to a complete list. 
            with open(cmd_lines_path, 'r') as f:
                cmd_lines = f.readlines()
            cmd_lines_clean = [c_l.strip() for c_l in cmd_lines]
            all_cmd_lines.extend(cmd_lines_clean)
            
    # once all initial conditions and boundary conditions are done initializing
    # write out all_cmd_lines, and job-sized chuncks of all cmd lines. 
    
    # first write out a file with all the command lines
    model_folder_path = os.path.join(dir_path, model_name)
    with open(os.path.join(model_folder_path, 'all_cmd_lines'), 'w') as f:
        for line in all_cmd_lines:
                f.write(line+"\n")
    
    # next, cacluate the number of chunks. 
    estimated_time_per_task = model_time[model_name]
    wall_time = 20.
    num_processors = 23. + 4.*24.
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
                           '#SBATCH --job-name WV_MOAT',
                           '#SBATCH --ntasks-per-node 24',
                           '#SBATCH --partition shas',
                           '#SBATCH --mem-per-cpu 4GB',
                           '#SBATCH --nodes 5',
                           '#SBATCH --time 24:00:00',
                           '#SBATCH --account ucb19_summit1',
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
        
    all_submission_scripts[model_folder_path]=this_models_submission_scripts
    
    #all_submission_scripts[model_folder_path]=this_models_submission_scripts
    return (total_number_of_jobs, {model_folder_path: this_models_submission_scripts})

# Define filepaths. Here these are given as lists, for cross platform
# compatability

input_template_folderpath = ['..', '..', '..', 'templates']

parameter_dict_folderpath = ['..', '..', '..', 'auxillary_inputs']

dem_folderpath = ['..', '..', '..', 'auxillary_inputs', 'dems', 'sew', 'initial_conditions']

lowering_history_folderpath = ['..', '..', '..', 'auxillary_inputs', 'lowering_histories']

models_driver_folderpath = ['..', '..', '..', 'drivers', 'models']

dakota_driver_folderpath = ['..', '..', '..', 'drivers', 'dakota']

work_directory_folderpath = ['work', 'WVDP_EWG_STUDY3', 'results','sensitivity_analysis', 'sew', 'MOAT'] # path to scratch where model will be run.

# Get model space and parameter ranges (these will be loaded in from a file)

metric_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['metric_names.csv'])))
model_time_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_time.csv'])))
parameter_range_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['parameter_ranges.csv'])))
model_parameter_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_parameter_match.csv'])))
model_parameter_order_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_parameter_order.csv'])))

metric_df = pd.read_csv(metric_input_file)
time_df = pd.read_csv(model_time_input_file)
param_range_df = pd.read_csv(parameter_range_input_file)
model_param_df = pd.read_csv(model_parameter_input_file)
model_param_df = model_param_df.dropna() # Remove those rows with models that are not yet complete
order_df = pd.read_csv(model_parameter_order_file)
order_df = order_df.dropna() # Remove those rows with models that are not yet complete
order_dict = {}

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
        seed_dict[key] = int(model_param_df.loc[model_param_df['ID']==mid]['dakota_moat_seed'])
        
        # construct model_dictionary
        param_info = {}
        for param in parameter_names:
            if model_param_df.loc[model_param_df['ID']==mid][param].values[0] == 'variable':
                param_info[param]='var'
        model_dictionary[key] = param_info

        # construct order information
        order_info = {}
        for param in list(param_info.keys()):
            order_number = order_df.loc[model_param_df['ID']==mid][param].values[0]
            order_info[param] = int(order_number)
        order_dict[key] = order_info

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
lowering_histories = [os.path.abspath(os.path.join(*(lowering_history_folderpath+[pth]))) for pth in os.listdir(os.path.join(*lowering_history_folderpath)) if (pth.endswith('.txt') and pth!='lowering_history_0.txt')]

# get the full path to the Dakota analysis and model run driver
dakota_analysis_driver_filepath = os.path.abspath(os.path.join(*(dakota_driver_folderpath+[dakota_analysis_driver])))

# create container for each job submission 
all_submission_scripts = {}
total_number_of_jobs = 0

# for model in models
parallel_inputs = []

# for model in models
for model_name in  sorted(list(model_dictionary.keys())):
    # use one seed per model for consistency across catagorical variables. 
    seed = seed_dict[model_name]
    
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
              'work_directory_folderpath': work_directory_folderpath,
              'dakota_analysis_driver_filepath': dakota_analysis_driver_filepath,
              'model_time': model_time,
              'metric_names': metric_names,
              'order_dict': order_dict}

    parallel_inputs.append(inputs)
    #output = create_model_jobs(**inputs)
#%%
    # create summary latex table of parameters in sensitivity analyis set constant. 
param_replace_dict = {}
for index, row in param_range_df.iterrows():
    if str(row['Latex_Symbol']).startswith('$'):
        symbol = row['Latex_Symbol']
        param_replace_dict[row['Short Name']] = symbol
    
df_list = []
for model in model_dictionary.keys():
    model_key = model.split('_')[1]
    if model_key != '842':
        
        param_list = []
        md = model_dictionary[model]
        param_names = md.keys() 
        nparams = len(param_names)
        param_list = ', '.join([param_replace_dict[p] for p in param_names]) 
        
        out = {'number_of_parameters': nparams,
               'parameters_varied': param_list}
        df_list.append(pd.DataFrame({model_key: out}))
        
df = pd.concat(df_list, axis=1).T
df.sort_index(level=-1, inplace=True)
df.to_latex('sensitivity_analysis_parameters.txt', 
                        escape=False,
                        multirow=True)
#%%


# run the Dakota submission in parallel, then compile
print('Starting Job Creation')
output = Parallel(n_jobs=23)(delayed(create_model_jobs)(**inputs) for inputs in parallel_inputs)

for out in output:
    total_number_of_jobs += out[0]
    all_submission_scripts.update(out[1])

# create a set of final submission scripts that are ~100 jobs each.
final_submission_scripts = []

number_of_submission_scripts = np.ceil(float(total_number_of_jobs)/100.)
all_folder_paths = list(all_submission_scripts.keys())

# look for and delete prior submission scripts, if any.
old_submission_scripts = glob.glob(os.path.join(dir_path, 'submit_jobs_to_summit_*.sh'))
for o_script in old_submission_scripts:
    os.remove(o_script)

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

print('done')
# run the sbatch submission script
#for final_submission_script in final_submission_scripts
#   os.system('source '+final_submission_script)

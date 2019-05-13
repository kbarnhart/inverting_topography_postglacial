    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os
import numpy as np
import pandas as pd
import shutil
import datetime
import glob

from dakotathon import Dakota

from joblib import Parallel, delayed

#from dakotathon.utils import add_dyld_library_path
#add_dyld_library_path()

# get the current files filepath
dir_path = os.path.dirname(os.path.realpath(__file__))

###############
dakota_analysis_driver = 'driver.py'
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
                      metric_names=None,
                      model_time=None,
                      order_dict=None,
                      initial_parameter_values=None, 
                      weights=None):
    
    """Create SEW HYBRID CALIBRATION model jobs."""
#    total_number_of_jobs = 0
#    # initialize container for all command lines and submission scripts. 
#    all_cmd_lines = []
#    this_models_submission_scripts = []
    
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
            lower_bounds = [str(parameter_dictionary[var]['min']) for var in variables]
            upper_bounds = [str(parameter_dictionary[var]['max']) for var in variables]
            initial_point = [str(initial_parameter_values[var]) for var in variables]
            weights = [str(weights['weights'][metric]) for metric in metric_names]
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
            with open(os.path.join(dir_path, 'dakota_hybrid_template.in'), 'r') as f:
                dakota_lines = f.readlines()
            
            dakota_file = []
            str_var = [repr(v) for v in variables]
            str_met = [repr(m) for m in metric_names]
            
            for line in dakota_lines:
                line = line.replace('{model_name}', model_name)
                
                line = line.replace('{lowering_history}', os.path.split(lowering_filepath)[-1].split('.')[0])
                line = line.replace('{initial_condition}', os.path.split(dem_filepath)[-1].split('.')[0])

                line = line.replace('{num_variables}', str(len(variables)))
                line = line.replace('{variable_names}', ' '.join(str_var))

                line = line.replace('{upper_bounds}', ' '.join(upper_bounds))
                line = line.replace('{lower_bounds}', ' '.join(lower_bounds))
                line = line.replace('{initial_point}', ' '.join(initial_point))

                line = line.replace('{num_responses}', str(len(metric_names)))
                line = line.replace('{responses_names}', ' '.join(str_met))
                line = line.replace('{responses_weights}', ' '.join(weights))
 
                dakota_file.append(line)
                
            # Write dakota input file 
            with open(os.path.join(new_folder_path, 'dakota_hybrid_calibration.in'), 'w') as dakota_f:
                dakota_f.writelines(dakota_file)

            
        # to actually submit the jobs to summit, create a unique submit script
        # for each cmd_lines chunk.
        script_contents = ['#!/bin/sh',
                           '#SBATCH --job-name sew_hybrid'+model_name,
                           '#SBATCH --ntasks-per-node 24',
                           '#SBATCH --partition shas',
                           '#SBATCH --mem-per-cpu 4GB',
                           '#SBATCH --nodes 1',
                           '#SBATCH --time 24:00:00',
                           '#SBATCH --account ucb19_summit1',
                           '',
                           '# load environment modules',
                           'module load intel/16.0.3',
                           'module load openmpi/1.10.2',
                           'module load mkl/11.3.3',
                           'module load cmake/3.5.2',
                           'module load gsl/2.1',
                           '',
                           '# make sure environment variables are set correctly',
                           'source ~/.bash_profile',
                           '## run dakota using a restart file if it exists.',
                           'if [ -e dakota.rst ]',
                           'then',
                           'dakota -i dakota_hybrid_calibration.in -o dakota_hybrid_calibration.out --read_restart dakota.rst &> dakota.log',
                           'else',
                           'dakota -i dakota_hybrid_calibration.in -o dakota_hybrid_calibration.out &> dakota.log',
                           'fi']
            
        script_path = os.path.join(new_folder_path,'start_dakota.sh')
        with open(script_path, 'w') as f:
            for line in script_contents:
                f.write(line+"\n") 
        cmnd_line = 'cd ' + new_folder_path + '; sh start_dakota.sh'
    return (len(variables)+1, cmnd_line)

# Define filepaths. Here these are given as lists, for cross platform
# compatability

input_template_folderpath = ['..', '..', '..', 'templates']

parameter_dict_folderpath = ['..', '..', '..', 'auxillary_inputs']

dem_folderpath = ['..', '..', '..', 'auxillary_inputs', 'dems', 'sew', 'initial_conditions']

lowering_history_folderpath = ['..', '..', '..', 'auxillary_inputs', 'lowering_histories']

weights_filepath = ['..', '..', '..', 'auxillary_inputs', 'weights', 'sew_variance_with_model800.txt']

models_driver_folderpath = ['..', '..', '..', 'drivers', 'models']

dakota_driver_folderpath = ['..', '..', '..', 'drivers', 'dakota']

work_directory_folderpath = ['work', 'WVDP_EWG_STUDY3', 'results','calibration', 'sew', 'GAUSSNEWTON'] # path to scratch where model will be run.

moat_work_dir_list = ['work', 'WVDP_EWG_STUDY3', 'results','sensitivity_analysis', 'sew', 'MOAT']
moat_work_dir = os.path.join(os.path.abspath(os.sep), *moat_work_dir_list)

moat_output_filename = 'moat_combined_output.csv'
    
# loop within models. 
moat_dat_file = os.path.join(moat_work_dir, moat_output_filename)
moat_df = pd.read_csv(moat_dat_file)

# Get model space and parameter ranges (these will be loaded in from a file)
metric_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['metric_names.csv'])))
model_time_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_time.csv'])))
parameter_range_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['parameter_ranges.csv'])))
model_parameter_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_parameter_match_calibration_sew.csv'])))
model_parameter_order_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_parameter_order.csv'])))
model_parameter_start_value_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_parameter_calibration_start_values_sew.csv'])))

metric_df = pd.read_csv(metric_input_file)
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

# these weights are provided as the variance, we want to weight by 1/variance, 
#weights_df = 1./pd.read_csv(os.path.join(*weights_filepath), header=None, names=['weight'])

variance_df = pd.read_csv(os.path.join(*weights_filepath),index_col=0)
weights_df = pd.DataFrame(1.0/(variance_df.Model_800_Based_Variance + variance_df.Error_Based_Variance))
weights_df.columns = ['weight']
temp = {}
for w in list(weights_df.index):
    temp[w] = float(weights_df.weight[w])
weights_dict ={'weights':temp}

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
    
    parameter_dictionary[param]=range_dict

# get metric names       
metric_names =  list(metric_df.values[:,0])

# get initial condition DEMS
inital_dems = [os.path.abspath(os.path.join(*(dem_folderpath+[pth]))) for pth in ['pg24f_7etch.txt']]

# get lowering histories
lowering_histories = [os.path.abspath(os.path.join(*(lowering_history_folderpath+[pth]))) for pth in ['lowering_history_0.txt']]

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
    input_template_filepath = os.path.abspath(os.path.join(dir_path, *(input_template_folderpath+['sew_calibration_inputs_template_'+model_name+'.txt'])))
    with open(input_template_filepath,'r') as itfp:
        input_template_lines = itfp.readlines()
    
    # get model driver name
    model_driver_name = os.path.abspath(os.path.join(dir_path, *(models_driver_folderpath+['sew_calibration_'+model_name+'_driver.py'])))
    
    # for model 000 we;ve done a grid search and know the "best" start value
    # for other models we try and start at the equivalent of model 000. 
    
    pvars = list(model_dictionary[model_name].keys())
    initial_parameter_values = {}
    for pv in pvars:
        initial_parameter_values[pv] = start_dict[model_name][pv]
    
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
              'weights':weights_dict}

    parallel_inputs.append(inputs)
    #output = create_model_jobs(**inputs)
    
## do cleanup. 
## in work directory, and study3py directory, make a copy and move
#work_dir = os.path.join(os.sep, *work_directory_folderpath)
#results_dir = os.path.join(*(work_dir, model_name))
#start_dir = os.path.join(*(dir_path, model_name))
#now = datetime.datetime.now()
#now_str = '.'.join([str(now.year), str(now.month), str(now.day), str(now.hour)])
#    
## if work directory exists, then models have been run and we want to save
#if os.path.exists(work_dir):
#    dest_dir = work_dir+'_'+now_str
#    
#    # if nothing has been copied before this time, 
#    if os.path.exists(dest_dir) == False:
#        # copy results to the dated filder
#        shutil.copytree(work_dir, dest_dir)
#        
#        # remove work dir (this will be re-made by running dakota)
#        shutil.rmtree(work_dir)
#        
#    # make a copy of the study3py files
#    if os.path.exists(dir_path):
#        dest_dir = dir_path+'_'+now_str
#    
#        if os.path.exists(dest_dir) == False:
#            
#            # copy results to the dated filder
#            shutil.copytree(dir_path, dest_dir)
#            
#            # remove model folders. 
#            mod_folders = glob.glob(dir_path+os.sep+'model_*')
#            for mod in mod_folders:
#                shutil.rmtree(mod)
            
# run the Dakota submission in parallel, then compile
print('Starting Job Creation')
output = Parallel(n_jobs=1)(delayed(create_model_jobs)(**inputs) for inputs in parallel_inputs)

os.chdir(dir_path)
cmnd_lines = []
total_number_of_cores = 0
for out in output:
    total_number_of_cores += out[0]
    cmnd_lines.append(out[1])
num_nodes = int(np.ceil(total_number_of_cores/24))

# write out command lines
with open('cmd_lines', 'w') as f:
    for line in cmnd_lines:
        f.write(line + '\n')

# create launch script
script_contents = ['#!/bin/sh',
                   '#SBATCH --job-name sew_calib',
                   '#SBATCH --ntasks-per-node 24',
                   '#SBATCH --partition shas',
                   '#SBATCH --mem-per-cpu 4GB',
                   '#SBATCH --nodes ' + str(num_nodes),
                   '#SBATCH --time 24:00:00',
                   '#SBATCH --account ucb19_summit1',
                   '',
                   'module purge',
                   'module load intel',
                   'module load impi',
                   'module load loadbalance',
                   'mpirun lb cmd_lines']

with open('launch_dakota_calibration.sh', 'w') as f:
    for line in script_contents:
        f.write(line + '\n')
        
## run the sbatch submission script
##os.system('source '+'launch_dakota_calibration.sh')

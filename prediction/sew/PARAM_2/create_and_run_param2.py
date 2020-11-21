# Run one evaluation from each model, using the best parameters

import os
import numpy as np
import glob
import yaml
from itertools import product
import pandas as pd

np.random.seed(42)

boilerplate = """
# Inputs for Model:
# {modelName}
run_duration: 10000.0
output_interval: 1000.0
save_first_timestep: True
meters_to_feet: True
DEM_filename: {initial_path}
outlet_id: 178576
outlet_lowering_file_path: {lowering_path}
output_filename: model_{mid}_
rock_till_file__name: /work/WVDP_EWG_STUDY3/study3py/auxillary_inputs/rock_till/sew/bdrx_24.txt
dt: 10
modern_outlet_elevation: {modern_outlet}
points_file: /work/WVDP_EWG_STUDY3/study3py/prediction/sew/PredictionPoints_ShortList.csv
"""

with open("driver.py", "r") as f:
    driver_boilerplate = f.read()

# get the current files filepath
dir_path = os.path.dirname(os.path.realpath(__file__))
loc = dir_path.split(os.sep)[-2]

input_template_folderpath = ['..', '..', '..', 'templates']
parameter_dict_folderpath = ['..', '..', '..', 'auxillary_inputs']
dem_folderpath = ['..', '..', '..', 'auxillary_inputs', 'dems', 'sew', 'modern']
lowering_history_folderpath = ['..', '..', '..', 'auxillary_inputs', 'lowering_histories']
climate_future_folderpath = ['..', '..', '..', 'auxillary_inputs', 'climate_futures']
models_driver_folderpath = ['..', '..', '..', 'drivers', 'models']
dakota_driver_folderpath = ['..', '..', '..', 'drivers', 'dakota']

points_folderpath = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction',  'sew' , 'PredictionPoints_ShortList.csv']
points_file = os.path.join(os.path.abspath(os.sep), *points_folderpath)

# loop within models.

# Get model space and parameter ranges (these will be loaded in from a file)
parameter_range_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['parameter_ranges.csv'])))
model_parameter_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_parameter_match_calibration_sew.csv'])))

param_range_df = pd.read_csv(parameter_range_input_file)

model_param_df = pd.read_csv(model_parameter_input_file)
model_param_df = model_param_df.dropna() # Remove those rows with models that are not yet complete


mids = {i: str(i).rjust(3, '0') for i in model_param_df['ID'].values}

mids = ['800', '802','804', '808', '810', '840', '842', 'A00', 'C00'] # only do prediction parameter uncertainty for a few models

nruns = 1000

# initialize data structures
model_dictionary = {}
seed_dict = {}

dt_ind = np.where(model_param_df.columns == 'dt')[0][0]
parameter_names = list(model_param_df.columns[dt_ind:].values)

# deal with getting parameter values from file:
parameter_dict_folderpath = ['..', '..', '..', 'auxillary_inputs']
parameter_range_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['parameter_ranges.csv'])))

models_used = {}
models_name = {}

for mid in mids:

    # make the model key
    key = 'model_'+mid
    models_used[mid] = model_param_df.loc[model_param_df['ID']==mid]["Model Used"].values[0]
    models_name[mid] = model_param_df.loc[model_param_df['ID']==mid]["Model Name"].values[0]

    # construct model_dictionary
    param_info = {}
    for param in parameter_names:
        if model_param_df.loc[model_param_df['ID']==mid][param].values[0] == 'variable':
            param_info[param]='var'
        else:
            if model_param_df.loc[model_param_df['ID']==mid][param].values[0] != "na":
                param_info[param]= model_param_df.loc[model_param_df['ID']==mid][param].values[0]
    model_dictionary[key] = param_info

#%%
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

# get lowering histories
lowering_histories = np.sort(glob.glob(os.path.abspath(os.path.join(*(lowering_history_folderpath+['lowering_future*.txt'])))))[:1]

# get climage futures =
climate_futures = np.sort(glob.glob(os.path.abspath(os.path.join(*(climate_future_folderpath+['climate_future*.txt'])))))[:1]

modern_elevations = {"lowering_future_1": 1171.202,
                     "lowering_future_2": 1101.202,
                     "lowering_future_3": 971.202,}

# for model in models
cmnd_lines = []

# for model in models
for model_name in  sorted(list(model_dictionary.keys())):

    mid = model_name.split("_")[-1]
    modelUsed = models_used[mid]
    modelName = models_name[mid]

    # posterior file
    posterior_folderpath = ['..', '..', '..', 'calibration','sew','QUESO_DRAM','model_{mid}'.format(mid=mid),'lowering_history_0.pg24f_ic5etch','posterior.dat']
    posterior_file = os.path.abspath(os.path.join(*(posterior_folderpath)))

    model_vars = model_dictionary[model_name]
    # only run if EGO2 has completed and best parameters are known.
    if os.path.exists(posterior_file):
        df = pd.read_csv(posterior_file, sep='\s+', engine='python')

        # drop what starts with chi_elev and the first two cols
        df = df.loc[:, ~df.columns.str.startswith('chi_elev')]
        df = df.drop(columns =['%mcmc_id', 'interface'])
        nsamples = df.shape[0]

        sample_ids = np.random.choice(nsamples, size=nruns, replace=False)

        for (initial_path, lowering_path, climate_path) in product(
            inital_dems,
            lowering_histories,
            climate_futures):

            lowering = lowering_path.split(os.path.sep)[-1].split('.')[0]
            climate = climate_path.split(os.path.sep)[-1].split('.')[1]
            ic = initial_path.split(os.path.sep)[-1].split('.')[0]

            modern_outlet = modern_elevations[lowering.split('.')[0]]

            # get climate vars from parameter file.
            with open(climate_path, 'r') as f:
                climate_vars = yaml.safe_load(f)

            for nr in range(nruns):

                # make dir
                dir_name = os.path.abspath(os.path.join(*[model_name, "{lowering}.{ic}.{climate}".format(lowering=lowering, ic=ic, climate=climate), "run.{nr}".format(nr=nr)]))

                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)

                # create input file:
                input_text = boilerplate.format(
                    modelName=modelName,
                    mid=mid,
                    lowering_path=lowering_path,
                    modern_outlet = modern_outlet,
                    initial_path=initial_path)


                # write input file, append additional values.
                with open(dir_name + os.path.sep +"inputs.txt", "w") as f:
                    f.write(input_text)

                    # append climate variables.
                    for key, value in climate_vars.items():
                        f.write("{key}: {value}\n".format(key=key, value=value))

                    # append input variables.
                    for key, value in model_vars.items():
                        if value == "var":
                            value =  df[key][sample_ids[nr]]
                        f.write("{key}: {value}\n".format(key=key, value=value))

                # copy driver, replace model name,
                with open(dir_name + os.path.sep +"driver.py", "w") as f:
                    text = driver_boilerplate.format(modelName=modelName, modelUsed=modelUsed)
                    f.write(text)

                # append run to cmndlines.
                cmnd_lines.append('cd ' + dir_name + '; python driver.py;')

with open('cmd_lines', 'w') as f:
    for line in cmnd_lines:
        f.write(line + '\n')

with open('submit_PARAM2_runs_1000.sh', 'w') as f:
    script = ['#!/bin/sh',
              '#SBATCH --job-name param2',
              '#SBATCH --partition shas',
              '#SBATCH --nodes 7',
              '#SBATCH --time 24:00:00',
              '',
              'module purge',
              'module load intel',
              'module load impi',
              'module load loadbalance',
              'mpirun lb cmd_lines']
    for line in script:
        f.write(line+'\n')

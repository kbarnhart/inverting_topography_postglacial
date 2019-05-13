#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os
import numpy as np
import pandas as pd

from landlab.io import read_esri_ascii

dir_path = os.path.dirname(os.path.realpath(__file__))

###############
# Define filepaths. Here these are given as lists, for cross platform
# compatability
outlet_id = 178576
input_template_folderpath = ['..', '..', 'templates']
driver_folderpath = ['..', '..', 'drivers', 'models']

parameter_dict_folderpath = ['..', '..', 'auxillary_inputs']
initial_dem_filepath = ['..', '..', 'auxillary_inputs', 'dems', 'sew', 'modern', 'dem24fil_ext.txt']
chi_mask_filepath = ['..', '..', 'auxillary_inputs', 'chi_mask', 'sew' , 'chi_mask.txt']
rock_till_filepath = ['..', '..', 'auxillary_inputs', 'rock_till', 'sew' , 'bdrx_24.txt']

metric_folder_path = ['..', '..', 'auxillary_inputs', 'modern_metric_files']

modern_dem_name = os.path.join(*initial_dem_filepath)

modern_dem_metric_file = os.path.abspath(os.path.join(*(metric_folder_path+['.'.join([os.path.split(modern_dem_name)[1].split('.')[0], 'metrics', 'txt'])])))
modern_dem_chi_file = os.path.abspath(os.path.join(*(metric_folder_path+['.'.join([os.path.split(modern_dem_name)[1].split('.')[0], 'metrics', 'chi', 'txt'])])))

# get model driver template
driver_template_filepath = os.path.abspath(os.path.join(*(input_template_folderpath+['model_driver_template.txt'])))
with open(driver_template_filepath, 'r') as mdfp:
    model_driver_lines = mdfp.readlines()

# Get model space information
model_parameter_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_parameter_match.csv'])))
model_param_df = pd.read_csv(model_parameter_input_file)

# construct model time and model dictionary:
# first get all model ID numbers and same them to a dict with the equivalent
# lenght three padded strings
mids = {i: str(i).rjust(3, '0') for i in model_param_df['ID'].values}
mname = {i: model_param_df.loc[model_param_df['ID']==i]['Model Name'].values[0] for i in mids.keys()}
mused = {i: model_param_df.loc[model_param_df['ID']==i]['Model Used'].values[0] for i in mids.keys()}
                  
# initialize data structures
model_dictionary = {}
num_variable_dictionary = {}
  

dt_ind = np.where(model_param_df.columns == 'dt')[0][0]
parameter_names = list(model_param_df.columns[dt_ind:].values)

mid_names = {}
mid_used = {}
for mid in mids.keys():
    
    # make the model key
    key = 'model_'+mids[mid]
    mid_names[key] = mname[mid]
    mid_used[key] = mused[mid]
    if type(mid_used[key]) == str:
        # construct model_dictionary
        param_info = {}
        num_var = 0
        for param in parameter_names:
            if model_param_df.loc[model_param_df['ID']==mid][param].values[0]!='na':
                
                if model_param_df.loc[model_param_df['ID']==mid][param].values[0]=='variable':
                    param_info[param] = '{'+param+'}'
                    num_var += 1
                else:
                    param_info[param] = model_param_df.loc[model_param_df['ID']==mid][param].values[0]
        num_variable_dictionary[key] = num_var                   
        model_dictionary[key] = param_info
        

    
# modern DEM filepath, chi mask filepath, and rock-till filepath
modern_dem = os.path.abspath(os.path.join(*initial_dem_filepath))
chi_mask = os.path.abspath(os.path.join(*chi_mask_filepath))
rock_till = os.path.abspath(os.path.join(*rock_till_filepath))


# read modern grid to get modern outlet elevation
(temp_grid, temp_z) = read_esri_ascii(modern_dem,
                                      name='topographic__elevation',
                                      halo=1)   

modern_outlet_elevation = temp_z[outlet_id]

# identify a few special models that need extra arguments in the input file. 
rt_models = ['model_'+ mid for mid in ['800', '802', '804', '808', '810', '840', '842', 'A00', 'C00']]
st_models = ['model_'+ mid for mid in ['100', '102', '104', '108', '110', '180', '300', '380']]
hy_models = ['model_'+ mid for mid in ['010', '012', '014', '018', '030', '110', '210', '410', '810']]

# for model in models
for model_name in model_dictionary.keys():
    
    # only write template and driver if model exists. 
    if type(mid_used[model_name]) == str:
        # First, make the input template
        lines = ['# Inputs for Model: ' + model_name, 
                 '# ' + mid_names[model_name],
                 '# use ' + mid_used[model_name],
                 'run_duration: 13000.0',
                 'output_interval: 13000.0',    
                 'meters_to_feet: True', 
                 'DEM_filename: {inital_DEM_file}',
                 'outlet_id: ' + str(outlet_id),
                 'outlet_lowering_file_path: {lowering_history_file}',	
                 'output_filename: {output_filename}',
                 'chi_mask_dem_name: '+ chi_mask, 
                 'modern_dem_name: ' + modern_dem, 
                 'modern_outlet_elevation: ' + str(modern_outlet_elevation),
                 'modern_dem_metric_file: ' + modern_dem_metric_file, 
                 'modern_dem_chi_file: ' + modern_dem_chi_file]
        
        if model_name in rt_models:
            lines.append('rock_till_file__name: '+ rock_till)
        if model_name in st_models:
            lines.append('opt_stochastic_duration: False')
        if model_name in hy_models:
            lines.append('solver: adaptive')
            
        model_params = model_dictionary[model_name]
        
        for param in model_params.keys():
            lines.append(param+': '+str(model_params[param]))
    
        # Second, make the model driver
        input_template_filepath = os.path.abspath(os.path.join(*(input_template_folderpath+['inputs_template_'+model_name+'.txt'])))
        with open(input_template_filepath,'w') as f:
            f.write("\n".join(lines))
            
        # Modify model driver template
        model_lines = []
        for line in model_driver_lines:
            line = line.replace('{ModelID}', model_name.split('_')[1])
            line = line.replace('{ModelName}', mid_names[model_name])
            line = line.replace('{ModelUsed}', mid_used[model_name])
            line = line.strip('\n\r')
            model_lines.append(line+'\n')
        
        # CREATE FOLDER IF IT DOESN'T EXIST:
        if os.path.exists(os.path.join(*driver_folderpath)):
            pass
        else:
            os.makedirs(os.path.join(*driver_folderpath))
        
        # Write model driver
        model_driver_filepath = os.path.abspath(os.path.join(*(driver_folderpath+[model_name+'_driver.py'])))
    
        with open(model_driver_filepath, 'w') as mdfp:
            mdfp.writelines(model_lines)

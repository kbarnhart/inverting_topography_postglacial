#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:52:28 2017

@author: katybarnhart

This script compiles output from Dakota runs and saves them into a csv file. It
should produce results similar in format to the .dat file created when running
Dakota with a tabular environment. 
"""

# Start by importing the necessary python libraries
import os
import glob
import pandas as pd
import numpy as np
from yaml import load
import time
from joblib import Parallel, delayed

##############################################################################
#                                                                            #
#        Part 0: Name of results directory, params and results files         #
#                                                                            #
##############################################################################

results_dir = ['work', 'WVDP_EWG_STUDY3', 'results','sensitivity_analysis', 'sew', 'MOAT']
parameter_dict_folderpath = ['..', '..', '..', 'auxillary_inputs']
metric_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['metric_names.csv'])))
output_file = 'metric_diff.txt'
cat_file = 'cat_outputs_for_analysis.txt'
params_file = 'params.in'
resource_file = 'usage.txt'

# Use glob to find all folders with 'model' in the name.
model_dir_search = results_dir
model_dir_search.append('model_***/')
model_dirs = glob.glob(os.path.join(os.path.abspath(os.sep), *model_dir_search))

# get metric names
metric_df = pd.read_csv(metric_input_file)
metrics = list(metric_df['metric names'].values)

# cat_outputs 
cat_metrics = np.arange(1, 21, dtype=float)
#%%
all_models = []
start_time = time.time()

def compile_metrics(m_dir):  
    # Use glob to find all folders with 'run' in the name.
    run_dirs = glob.glob(os.path.join(m_dir, '**/run*/'))
    # Intialize an empty list to collect information about each run. 
    params = []
    for r_d in run_dirs:
        # open the results file. This is where the outputs were written by the
        # driver script.
        if os.path.exists(os.path.join(r_d, output_file)):
            with open(os.path.join(r_d, output_file), 'r') as f:
                outputs = load(f)
                del outputs['Modern topography file'] 
                del outputs['Model topography file']
        else:
            # here we'll put NANs in place for the ~40 runs that didn't complete. 
            outputs = {}
            for m in metrics:
                outputs[m] = np.nan
        # open the cat results file. This is where the outputs were written.
        if os.path.exists(os.path.join(r_d, cat_file)):
            with open(os.path.join(r_d, cat_file), 'r') as f:
                cat_outputs = load(f)
                del cat_outputs['Modern topography file'] 
                del cat_outputs['Model topography file']
        else:
            # here we'll put NANs in place for the ~40 runs that didn't complete. 
            cat_outputs = {}
            for m in cat_metrics:
                cat_outputs[m] = np.nan
        # open the parameters file. This is where Dakota wrote the values of the
        # parameters that it varies. 
        with open(os.path.join(r_d, params_file)) as f:
            params_text=[]
            lines = f.readlines()
            for line in lines:
                params_text.append(line.split())
        # open the resource usage file
        if os.path.exists(os.path.join(r_d, resource_file)):
            with open(os.path.join(r_d, resource_file)) as f:
                usage = f.readlines()
        else:
            usage = []
        # clean some content of the params file  removing extranous lines
        remove_lines = ['variables', 'functions', 'derivative_variables', 'analysis_components']
        for i in range(len(params_text)-1, -1, -1):
            if params_text[i][1] in remove_lines:
                params_text.pop(i)
            if params_text[i][1][:3]=='DVV':
                params_text.pop(i)
            if params_text[i][0][0]=='/':
                params_text.pop(i)
            if params_text[i][1][:2]=='AC':
                params_text.pop(i)
            if params_text[i][1][:2]=='AS':
                params_text.pop(i)
        # Convert values from string to float. 
        params_dict = {}
        for p in params_text:
            params_dict[p[1]] = float(p[0])
        # assign the value from output into the params datastructure
        for key in list(outputs.keys()):
            if key.startswith('elev'):
                number = int(key[4:])
                new_number = '{:03d}'.format(int(number))
                new_key ='ASV_elev_' + new_number
            elif key.startswith('cumarea'):
                number = int(key[7:])
                new_number = '{:03d}'.format(int(number))
                new_key ='ASV_cumarea_' + new_number
            elif key.endswith('nodes'):
                #four_cell_nodes
                if key == 'four_cell_nodes':
                    new_key = 'ASV_cumarea_000_04_cell_nodes'
                #three_cell_nodes
                if key == 'three_cell_nodes':
                    new_key = 'ASV_cumarea_000_03_cell_nodes'
                #two_cell_nodes
                if key == 'two_cell_nodes':
                    new_key = 'ASV_cumarea_000_02_cell_nodes'
                #one_cell_nodes
                if key == 'one_cell_nodes':
                    new_key = 'ASV_cumarea_000_01_cell_nodes'
            else:
                new_key = 'ASV_' + key
            params_dict[new_key] = outputs[key]
        # assign the values from cat into the params datastructure
        for key in list(cat_outputs.keys()):
            key2 = '{:02d}'.format(int(key))
            new_key = 'chi_elev_'+key2
            params_dict[new_key] = cat_outputs[key]
        # Convert the dictionary to a pandas series.            
        p = pd.Series(params_dict)
        # add information about the model run. 
        fp_info = r_d.split(os.sep)
        p['model'] = fp_info[-4]
        p['lowering'] = fp_info[-3].split('.')[0]
        p['initial_condition'] = fp_info[-3].split('.')[1]
        p['bc_ic'] = fp_info[-3]
        p['run'] = int(fp_info[-2].split('.')[-1])
        p['elapsed_time'] = np.nan
        p['memory_used'] = np.nan
        for line in usage:
            if line[:7]=='Elapsed':
                p['elapsed_time'] = np.abs(float(line.strip().split(':')[-1]))
            if line[:22]=='Max. Resident Set Size':
                p['memory_used'] = np.abs(float(line.strip().split('=')[-1]))
        params.append(p)
##############################################################################
#                                                                            #
#        Part 2: Convert output to a dataframe and save to a csv             #
#                                                                            #
##############################################################################
    # Convert the list of pandas series to a data frame. 
    params = pd.DataFrame(params)
    # Sort the data frame by eval_ID number and reset the index.
    params = params.sort_values(by='eval_id')
    params = params.reset_index(drop=True)
    # Save the data frame to a CSV file. 
    params.to_csv(os.path.join(m_dir, 'moat_combined_output.csv'), na_rep='nan')
    return params

#params = compile_metrics(model_dirs[0])

ncores = 23
all_models = Parallel(n_jobs=ncores)(delayed(compile_metrics)(m_dir) for m_dir in model_dirs)    

all_models_df = pd.concat(all_models) 
all_models_file_out = os.path.join(os.path.abspath(os.sep), *results_dir[:-1])
all_models_df.to_csv(os.path.join(all_models_file_out, 'moat_combined_output.csv'), na_rep='nan')

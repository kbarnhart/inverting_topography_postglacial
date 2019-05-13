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

##############################################################################
#                                                                            #
#        Part 0: Name of results directory, params and results files         #
#                                                                            #
##############################################################################

results_dir = ['work', 'WVDP_EWG_STUDY3', 'results','sensitivity_analysis', 'sew', 'DELSA']
output_file = 'outputs_for_analysis.txt'
params_file = 'params.in'
resource_file = 'usage.txt'

# Use glob to find all folders with 'model' in the name.
model_dir_search = results_dir
model_dir_search.append('model**/')
model_dirs = glob.glob(os.path.join(os.path.abspath(os.sep), *model_dir_search))

for m_dir in model_dirs:
    
    # Use glob to find all folders with 'run' in the name.
    run_dirs = glob.glob(os.path.join(m_dir, '**/center_points*/run*/'))
    
    # Intialize an empty list to collect information about each run.
    params = []
    for r_d in run_dirs:
        
        # open the results file. This is where the outputs were written by the
        # driver script.
        with open(os.path.join(r_d, output_file)) as f:
            outputs = f.readlines()
        
        # open the parameters file. This is where Dakota wrote the values of the
        # parameters that it varies.
        with open(os.path.join(r_d, params_file)) as f:
            params_text=[]
            lines = f.readlines()
            for line in lines:
             params_text.append(line.split())
    
        # open the resource usage file
        with open(os.path.join(r_d, resource_file)) as f:
            usage = f.readlines()
        
        # clean some content of the params file  removing extranous lines
        remove_lines = ['variables', 'functions', 'derivative_variables', 'analysis_components']
        for i in range(len(params_text)-1, -1, -1):
            if params_text[i][1] in remove_lines:
                params_text.pop(i)
            if params_text[i][1][:3]=='DVV':
                params_text.pop(i)
            if params_text[i][1][:2]=='AC':
                    params_text.pop(i)
                
        # assign the value from output into the params datastructure
        for i in range(len(params_text)):
            if params_text[i][1][:3]=='ASV':
                ind = int(params_text[i][1].split(':')[0].split('_')[-1])-1
                params_text[i][0] = outputs[ind].strip()
        
        params_dict = {}
        for p in params_text:
            params_dict[p[1]] = float(p[0])
        

         # Convert the dictionary to a pandas series.
        p = pd.Series(params_dict)

        # add information about the model run.
        fp_info = r_d.split(os.sep)

        p['model'] = fp_info[-5]
        p['lowering'] = fp_info[-4].split('.')[0]
        p['initial_condition'] = fp_info[-4].split('.')[1]
        p['bc_ic'] = fp_info[-4]
        p['center_point'] = fp_info[-3].split('.')[1]
        p['run'] = int(fp_info[-2].split('.')[-1])
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

    # Sort the data frame by center point and run number and reset the index.
    params = params.sort_values(by=['center_point', 'run'])
    params = params.reset_index(drop=True)
    
    # Save the data frame to a CSV file.
    params.to_csv(os.path.join(m_dir, 'combined_output.csv'), na_rep='nan')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:33:57 2017

@author: barnhark
"""

# Start by importing the necessary python libraries
import os
import glob
from joblib import Parallel, delayed
from yaml import load
import time
import numpy as np
import pandas as pd

from metric_calculator import  GroupedDifferences

##############################################################################
#                                                                            #
#        Part 0: Name of results directory, params and results files         #
#                                                                            #
##############################################################################
# get the current files filepath

# additional file paths
results_dir = ['work', 'WVDP_EWG_STUDY3', 'results','sensitivity_analysis', '**', 'MOAT']
models_driver_folderpath = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'drivers', 'models']
parameter_dict_folderpath = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'auxillary_inputs']

# file names
input_file_name = 'inputs.txt'

# Use glob to find all folders with a 'params.in' file.
results_dir.extend(['model_**', '**', 'run*', input_file_name])
input_files = sorted(glob.glob(os.path.join(os.path.abspath(os.sep), *results_dir)))

def recalculate_metrics_from_topography(input_file):
    # load the params file to get the correct file names
    work_folder = os.path.split(input_file)[0]
    loc = work_folder.split(os.sep)[-5]  
    with open(input_file, 'r') as f:
        # load params file
        params = load(f)
    model_name = params['output_filename']
    topo_filename = model_name + '0001.nc'
    model_dem_name = os.path.join(work_folder, topo_filename)
    if os.path.exists(model_dem_name):
        modern_dem_name = params['modern_dem_name']
        outlet_id = params['outlet_id']  
        # these were made before category files
        category_file = os.path.join(os.path.abspath(os.sep), *['work', 'WVDP_EWG_STUDY3', 'study3py','auxillary_inputs', 'chi_elev_categories', loc + '.chi_elev_cat.20.txt'])
        category_weight_file = os.path.join(os.path.abspath(os.sep), *['work', 'WVDP_EWG_STUDY3', 'study3py','auxillary_inputs', 'weights', loc + '.chi_elev_weight.20.txt'])
        category_values = np.loadtxt(category_file)
        weight_values = np.loadtxt(category_weight_file)
        # calculate metrics
        try:
            # calculate metrics
            gd = GroupedDifferences(model_dem_name, modern_dem_name,  
                                    outlet_id=outlet_id, 
                                    category_values=category_values,
                                    weight_values=weight_values)
            gd.calculate_metrics()
            output_bundle = gd.dakota_bundle()
            # write out metrics as Dakota expects
            dakota_bundle_filepath = os.path.join(work_folder, 'cat_results.out')
            with open(dakota_bundle_filepath, 'w') as fp:
                for metric in output_bundle:
                    fp.write(str(metric)+'\n') 
            # and as the compile script expects:
            compile_filepath = os.path.join(work_folder, 'cat_outputs_for_analysis.txt')
            gd.save_metrics(filename=compile_filepath)
            out = {'file': input_file, 'outcome': 'Sucess'}
        except:
            out = {'file': input_file, 'outcome': 'Failed'}
    else:
        out = {'file': input_file, 'outcome' :'False'}
    return out
 
# this would run all as one paralell job, but parellel seems to hang up with more than ~2000 tasks.      
# output = Parallel(n_jobs=23)(delayed(recalculate_metrics_from_topography)(input_file) for input_file in input_files)

#%%
start_time = time.time()
output = recalculate_metrics_from_topography(input_files[0])
end_time = time.time()

elapsed_time = end_time-start_time
print(elapsed_time)
#%%

not_done = True
start = 0
ncores = 47

output_list = []
while not_done:
    end = min(start + 10*ncores, len(input_files))
    start_time = time.time()
    output = Parallel(n_jobs=ncores)(delayed(recalculate_metrics_from_topography)(input_file) for input_file in input_files[start:end])
    end_time = time.time()
    pace = (start-end)/(start_time-end_time)
    total_estimate = len(input_files)/pace/(60*60)
    print(time.ctime())
    print('Start: '+str(start)+' End: '+str(end))
    print('Rate: '+ str(pace)+ ' files per second')
    print('Expected Total Time : '+str(total_estimate)+ ' hours\n' )
    output_list.extend(output)
    
    if np.remainder(start, 3000) == 0:
        df = pd.DataFrame(output_list)
        df.to_csv('recalc_metric_log.csv')
        

    start = end - 3
    if end==len(input_files):
        not_done=False

df = pd.DataFrame(output_list)
df.to_csv('recalc_metric_log.csv')

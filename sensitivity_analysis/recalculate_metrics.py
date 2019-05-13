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

from metric_calculator import  MetricDifference

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
    with open(input_file, 'r') as f:
        # load params file
        params = load(f)
    modern_dem_name = params['modern_dem_name']
    outlet_id = params['outlet_id']
    model_name = params['output_filename']
    topo_filename = model_name + '0001.nc'
    model_dem_name = os.path.join(work_folder, topo_filename)
    output_file_name = os.path.join(work_folder, 'metric_diff.txt')
    metric_folder = [os.sep, 'work', 'WVDP_EWG_STUDY3', 'study3py', 'auxillary_inputs', 'modern_metric_files']
    metric_folder_path = os.path.join(*metric_folder)
    modern_dem_metric_file = os.path.join(metric_folder_path, '.'.join([os.path.split(modern_dem_name)[1].split('.')[0], 'metrics', 'txt']))
    modern_dem_chi_file = os.path.join(metric_folder_path, '.'.join([os.path.split(modern_dem_name)[1].split('.')[0], 'metrics', 'chi', 'txt']))
    if os.path.exists(model_dem_name):
        # calculate metrics
        try:
            md = MetricDifference(model_dem_name=model_dem_name,
                                  modern_dem_metric_file = modern_dem_metric_file, 
                                  modern_dem_chi_file = modern_dem_chi_file, 
                                  outlet_id = outlet_id,
                                  output_file_name = output_file_name)
            md.run()
            # write out metric file
            # write out metric diffs
            output_bundle = md.dakota_bundle()
            output_file = os.path.join(work_folder, 'outputs_for_analysis.txt')
            with open(output_file, 'w') as fp:
                for metric in output_bundle:
                    fp.write(str(metric)+'\n')
            out = {'file': input_file, 'outcome': 'True'}
        except:
            out = {'file': input_file, 'outcome': 'Failed'}
    else:
        out = {'file': input_file, 'outcome' :'False'}
    return out
 
# this would run all as one paralell job, but parellel seems to hang up with more than ~2000 tasks.      
# output = Parallel(n_jobs=23)(delayed(recalculate_metrics_from_topography)(input_file) for input_file in input_files)

not_done = True
start = 0
ncores = 23

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

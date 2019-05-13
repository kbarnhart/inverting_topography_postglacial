#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 09:30:10 2017

@author: barnhark
"""

import os
import glob
import pandas as pd
##############################################################################
#                                                                            #
#        Part 0: Name of results directory, params and results files         #
#                                                                            #
##############################################################################
# get the current files filepath
dir_path = os.getcwd() #os.path.dirname(os.path.realpath(__file__))

# additional file paths
results_dir = ['work', 'WVDP_EWG_STUDY3', 'results','sensitivity_analysis', 'sew', 'MOAT']

# file names
output_file = 'outputs_for_analysis.txt'
params_file = 'params.in'
topo_file = 'model_4100001.nc'
usage_file = 'usage.txt'

# Use glob to find all folders with a 'params.in' file.
results_dir.extend(['model_410', '**', 'run*', params_file])
input_files = glob.glob(os.path.join(os.path.abspath(os.sep), *results_dir))
# for each run file determine which 
status = {}
for in_file in input_files:
    path = os.path.split(in_file)[0]
    out_file = os.path.join(path, output_file)
    t_file = os.path.join(path, topo_file)
    u_file = os.path.join(path, usage_file)
    status[path] = 'not_started' 
    if os.path.exists(u_file):
        status[path] = 'started'   
    if os.path.exists(t_file):
        status[path] = 'failed'    
    if os.path.exists(out_file):
        # model integraton ran sucessfully and created output
        status[path] = 'completed'
df = pd.DataFrame({'status':status})

df.to_csv('status_sew_410.csv')
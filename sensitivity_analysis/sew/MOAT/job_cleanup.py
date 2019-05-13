#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:52:28 2017

@author: katybarnhart

"""

# Start by importing the necessary python libraries
import os
import glob
import numpy as np

##############################################################################
#                                                                            #
#        Part 0: Name of results directory, params and results files         #
#                                                                            #
##############################################################################
# get the current files filepath
dir_path = os.path.dirname(os.path.realpath(__file__))

# additional file paths
results_dir = ['work', 'WVDP_EWG_STUDY3', 'results','sensitivity_analysis', 'sew', 'MOAT']
models_driver_folderpath = ['..', '..', '..', 'drivers', 'models']
parameter_dict_folderpath = ['..', '..', '..', 'auxillary_inputs']

# file names
output_file = 'outputs_for_analysis.txt'
params_file = 'params.in'

# Use glob to find all folders with a 'params.in' file.
results_dir.extend(['model**', '**', 'run*', params_file])
input_files = glob.glob(os.path.join(os.path.abspath(os.sep), *results_dir))

# create a container for un-run input files
model_dont_run = []
to_run = {}
for in_file in input_files:
    path = os.path.split(in_file)[0]
    out_file = os.path.join(path, output_file)
    if os.path.exists(out_file):
        # model integraton ran sucessfully and created output
        pass
    else:
        # model integration did not finish and needs to be re-run
        model_name = path.split(os.sep)[-3] 
        if model_name not in model_dont_run:
            model_driver_name = os.path.abspath(os.path.join(dir_path, *(models_driver_folderpath+[model_name+'_driver.py'])))
            to_run[path] = model_driver_name

models_remaining = {}
cmd_line_sets = {}
for trk in to_run.keys():    
    md = trk.split(os.path.sep)[-3]
    if md in models_remaining:
        models_remaining[md] += 1
        
        path = trk
        driver = to_run[path]
        line = 'cd ' + path + '; python '+ driver
        cmd_line_sets[md].append(line)
    else:
        models_remaining[md] = 1
        cmd_line_sets[md] = []
        path = trk
        driver = to_run[path]
        line = 'cd ' + path + '; python '+ driver
        cmd_line_sets[md].append(line)
        
for md in models_remaining:
    print(md, models_remaining[md])

# for the moment, assume that these jobs will each take 20 hours, so put 23 each on a job
## next, cacluate the number of chunks. 
all_submission_scripts = []
total_number_of_jobs = 0
for md in cmd_line_sets.keys():
    sel_cmd_lines = cmd_line_sets[md]
    
    estimated_time_per_task = 19
    wall_time = 20.
    num_tasks = len(sel_cmd_lines)
    num_processors = min(5, int(num_tasks/np.floor(wall_time/estimated_time_per_task)))
    num_cores = int(np.ceil((num_processors+1)/24))

# then split up all_cmd_lines into ~20 hour sized chunks. 

    job_name = '_'.join(['cleanup_cmd_lines',str(md)]) 
    cdl_file = os.path.join(dir_path, job_name)
    with open(cdl_file, 'w') as f:
        for line in sel_cmd_lines:
            f.write(line+"\n")       
    # to actually submit the jobs to summit, create a unique submit script
    # for each cmd_lines chunk.
    script_contents = ['#!/bin/sh',
                       '#SBATCH --job-name sew_'+md+'_MOAT_cleanup',
                       '#SBATCH --ntasks-per-node 24',
                       '#SBATCH --partition shas',
                       '#SBATCH --mem-per-cpu 2GB',
                       '#SBATCH --nodes '+str(num_processors),
                       '#SBATCH --time 24:00:00',
                       '#SBATCH --account ucb19_summit1',
                       '',	
                       'module purge',
                       'module load intel',
                       'module load impi',
                       'module load loadbalance',
                       'mpirun lb '+ job_name]     
    script_name = '.'.join(['_'.join(['cleanup_submit_script',str(md)]), 'sh'])
    script_path = os.path.join(dir_path, script_name)
    with open(script_path, 'w') as f:
        for line in script_contents:
            f.write(line+"\n")
    all_submission_scripts.append(script_path)
    total_number_of_jobs += 1
    
# create a set of final submission scripts that are ~100 jobs each.
final_submission_scripts = []
number_of_submission_scripts = np.ceil(float(total_number_of_jobs)/100.)

for i in range(int(number_of_submission_scripts)):
    remaining_jobs = 100
    submission_contents = ['#!/bin/sh',
                           '#',
                           '#  cleanup_submit_jobs_to_summit.sh',
                           '#  ',
                           '#  Created by Katherine Barnhart on 5/9/17.',
                           '']
    while remaining_jobs > 0:
        for j in range(remaining_jobs):    
            try:
                script = all_submission_scripts.pop()
                submission_contents.append('sbatch '+ script)
                remaining_jobs -= 1  
            except IndexError:
                remaining_jobs = 0
    final_submission_script = os.path.join(dir_path, 'cleanup_submit_jobs_to_summit_' + str(i) + '.sh')
    with open(final_submission_script, 'w') as f:
        for line in submission_contents:
            f.write(line+"\n") 
    final_submission_scripts.append(final_submission_script)

# run the sbatch submission script
#for final_submission_script in final_submission_scripts
#   os.system('source '+final_submission_script)  

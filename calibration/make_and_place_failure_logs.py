#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:08:39 2017

@author: barnhark
"""

# make and place a file 

import os
import glob

loc = 'gully'
models = {'gully': ['010' ,'012', '014', '030', '110', '210', '810'],
          'sew': ['440']}

failed_dict = {}
log = []
for loc in models.keys():
    for model in models[loc]:
        # look at all run folders, 
        path = os.path.join(os.sep, *['work', 'WVDP_EWG_STUDY3', 'results', 'calibration', loc, 'EGO2', 'model_'+model, '*', 'run.*'])
        # get all run directories
        run_dirs = glob.glob(path)
        # loop through directories
        failed_runs = []
        for r_d in run_dirs:
            # find directories in which usage.txt exists, but results.out does not
            usage = r_d + os.sep + 'usage.txt'
            params = r_d + os.sep + 'params.in'
            results = r_d + os.sep + 'results.out'
            if os.path.exists(usage) and os.path.exists(params):
                if os.path.exists(results) is False:
                    failed_runs.append(r_d)  
                    # if failure occured and no file exists yet, write out a fail log. 
                    fail_log = r_d + os.sep + 'fail_log.txt'
                    if os.path.exists(fail_log):
                        pass
                    else:
                        with open(fail_log, 'w') as fp:
                            fp.write('fail')
                        log.append(fail_log + '\n')
        # print out diagnostics and save some info            
        print(loc, model, str(len(failed_runs)))
        failed_dict[model] = failed_runs

with open('fail_log_placement.txt', 'a') as f:
    for line in log:
        f.write(line)
   
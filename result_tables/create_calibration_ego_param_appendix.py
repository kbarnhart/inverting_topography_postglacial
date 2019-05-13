#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:10:21 2018

@author: barnhark
"""

import glob
import os
import numpy as np

fin = np.sort(glob.glob(os.path.join(os.path.sep, *['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', 'sew' , 'ego2.sew.parameters.latex.model_*.txt'])))

fOut=r'calibration_ego2_tables.tex'
#%%
f = open(fOut, 'w')
#%%
for fname in fin:
    
    with open(fname, 'r') as t:
        lines=t.readlines()
            
    model_name = fname.split('_')[-1].split('.')[0]
    if len(model_name)<4:
        print(model_name)
    
    
        f.write('% EGO calibration for model number ' + model_name +'\n')
        f.write('\\begin{table}% \n')
        f.write('\\caption{Calibrated parameters from hybrid calibration method (EGO and NL2SOL) for model ' + model_name + ' in Upper Franks Creek Watershed (SEW domain)}% \n')
        f.write('   \label{tab:ego2_param_' + model_name + '}% \n')
        f.write('\centering\n')
        for line in lines:
            f.write(line)
        f.write('\end{table}\n')
        f.write('\n')
        f.write('\n')
f.close()

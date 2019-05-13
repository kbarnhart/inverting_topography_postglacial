#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:41:56 2017

@author: barnhark
"""

import os
import shutil
import glob
import pandas as pd
import numpy as np

pngs = glob.glob(os.path.join(*['*','*', '*.png']))

if os.path.exists('figures') is False:
    os.mkdir('figures')

summary = []

for png in pngs:
    
    outputs = os.path.split(png)[0] + os.sep + 'outputs_for_analysis.txt'
    
    model = os.path.split(os.path.split(png)[0])[0]
    lowering = os.path.split(os.path.split(png)[0])[-1].split('.')[0]
    ic = os.path.split(os.path.split(png)[0])[-1].split('.')[-1]
    
    if os.path.exists(outputs):
        metrics = []
        with open(outputs, 'r') as f:
            for line in f:
                metrics.append(float(line))
    
        objective_function = np.sum(np.square(metrics))
    else:
        objective_function = np.nan
    
    temp_dict = {'model': model, 
                 'lowering': lowering,
                 'initial_condition': ic, 
                 'objective_function': objective_function}
    try:
        new_fig_name = 'of_' + str(int(objective_function)) + '.' + os.path.split(png)[-1]
    except ValueError:
        new_fig_name = 'of_' + str(objective_function) + '.' + os.path.split(png)[-1]

    shutil.copy(png, os.path.join('figures', new_fig_name))

    
    summary.append(temp_dict)
    
df = pd.DataFrame(summary)
df.to_csv('validation_summary.csv')
    
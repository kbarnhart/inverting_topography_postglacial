#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:14:24 2017

@author: katybarnhart
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from scipy.spatial.distance import pdist, squareform


import os
import glob


results_dir_list = ['work', 'WVDP_EWG_STUDY3', 'results','sensitivity_analysis', 'sew', 'DELSA']
results_dir = os.path.join(os.path.abspath(os.sep), *results_dir_list)

output_filename = 'combined_output.csv'
resource_file = 'usage.txt'
    

# loop within models. 
model_dirs = glob.glob(os.path.join(results_dir, 'model**/'))

for m_d in model_dirs:
    plt.close('all')
    dat = pd.read_csv(os.path.join(m_d, output_filename))
    dat = dat.sort_values(by='eval_id').reset_index()
    
    # use the model-parameter dictionary to identify which parameters are
    # being varied and what their range is. 
    
    # for the moment this is hard coded: 
    inputs = {'K_sp_exp':{'range':3,'min':-4, 'max':-1},
              'linear_diffusivity_exp':{'range':3,'min':-4, 'max':-1}}
        
    outputs = [col for col in dat.columns if col.startswith('ASV')]

    for output in outputs:   
        
        v = np.max(np.abs(dat[output]))
        
        plt.figure()
        plt.scatter(dat[list(inputs.keys())[0]], 
                    dat[list(inputs.keys())[1]],
                    c=dat[output],
                    cmap='seismic',
                    vmin=-v,
                    vmax=v)
        plt.ylabel(list(inputs.keys())[1])
        plt.xlabel(list(inputs.keys())[0])
        plt.title(output)
        plt.colorbar()
        plt.show()
   

#%%
outputs.pop(10)
just_outcomes = dat[outputs]

cov = just_outcomes.cov()
of = []
for row in range(dat.shape[0]):
    vals = just_outcomes.values[row,:]
    of.append(np.dot(np.dot(vals.T, cov), vals))
    
dat['objective_function'] = of



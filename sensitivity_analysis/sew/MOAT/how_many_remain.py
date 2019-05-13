#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:56:52 2017

@author: barnhark
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import ggplot
from ggplot import *

import os
import glob

##############################################################################
#                                                                            #
#        Part 0: Name of compiled outputs and the initial Dakota .in file    #
#                                                                            #
##############################################################################
run_dir_list = ['work', 'WVDP_EWG_STUDY3', 'study3py','sensitivity_analysis', 'sew', 'MOAT']
run_dir = os.path.join(os.path.abspath(os.sep), *run_dir_list)

results_dir_list = ['work', 'WVDP_EWG_STUDY3', 'results','sensitivity_analysis', 'sew', 'MOAT']
results_dir = os.path.join(os.path.abspath(os.sep), *results_dir_list)

output_filename = 'moat_combined_output.csv'
dakota_in = 'dakota_moat.in'

# loop within models.
all_models_datfile = os.path.join(results_dir, output_filename)
all_models_df = pd.read_csv(all_models_datfile)

color_list = ['#1f78b4','#33a02c','#e31a1c', '#ff7f00','#6a3d9a', '#a6cee3','#b2df8a','#fb9a99','#fdbf6f','#cab2d6', '#ffff99']
across_color_list = ['#e5e5e5', '#1f78b4','#33a02c','#e31a1c', '#ff7f00','#6a3d9a', '#a6cee3','#b2df8a','#fb9a99','#fdbf6f','#cab2d6', '#ffff99']

models = all_models_df.model.unique()


finished = {}
num_remain = {}
num_created_m = {}
for model in models:
    
    dat_all = all_models_df.loc[all_models_df['model'] == model].dropna(axis=1, how='all')
    
    num_created = dat_all.shape[0]
    num_finished = num_created-np.sum(np.isnan(dat_all.ASV_chi_density_sum_squares))
    
    finished[model] = num_finished/num_created
    num_remain[model] = num_created - num_finished
    num_created_m[model] = num_created
df = pd.DataFrame({'finished':finished, 'remain':num_remain, 'total':num_created_m})

df.to_csv('number_remaining.csv')

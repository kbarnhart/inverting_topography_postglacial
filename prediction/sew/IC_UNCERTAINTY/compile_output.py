#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 07:42:36 2018

@author: barnhark
"""

import pandas as pd
import os
import glob
from yaml import load
import shutil

search_path = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction',  'sew' , 'IC_UNCERTAINTY', 'model_*', '*', 'run*', 'elevation_at_points_df.csv']
results_files = glob.glob(os.path.join(os.path.abspath(os.sep), *search_path))

fig_path = os.path.join(os.path.abspath(os.sep), *['work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction',  'sew' , 'IC_UNCERTAINTY', 'topography_figures'])

if not os.path.exists(fig_path):
    os.mkdir(fig_path)

# initialize df list
df_list = []

for rf in results_files:
    print(rf)
    # open df and transpose
    df = pd.read_csv(rf, sep=',', header=None, index_col=0).T
    data_columns = df.columns
    # open associated params.in file and get seed value
    params = os.path.join(os.path.split(rf)[0], 'inputs.txt')
    with open(params, 'r') as f:
        params = load(f)
        seed = params['seed']
    # get lowering, ic, climate, model
    df['model_name'] = rf.split(os.path.sep)[7].split('_')[-1]
    df['lowering_future'] = rf.split(os.path.sep)[8].split('.')[0]
    df['initial_condition'] = rf.split(os.path.sep)[8].split('.')[1]
    df['climate_future'] = rf.split(os.path.sep)[8].split('.')[-1]
    # set column order
    column_order = ['model_name', 'lowering_future', 'initial_condition', 'climate_future']
    column_order.extend(data_columns)
    # append re-ordered dataframe
    df_list.append(df[column_order])
    # copy figures
    figs = glob.glob(os.path.join(os.path.split(rf)[0], '*.png'))
    for fig in figs:
        shutil.copy(fig, fig_path)

# combine dataframes and save
df_full = pd.concat(df_list)
df_full.to_csv('compilation_of_sew_IC_uncert_output.csv')

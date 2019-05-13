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

search_path = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction',  'sew' , 'BREACHING', 'model_*', '*', '*', 'elevation_at_points_df.csv']
results_files = glob.glob(os.path.join(os.path.abspath(os.sep), *search_path))

fig_path = os.path.join(os.path.abspath(os.sep), *['work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction',  'sew' , 'BREACHING', 'topography_figures'])

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
        df['post_stabilization_incision_rate'] = params['post_stabilization_incision_rate']
        df['capture_start_time'] = params['capture_start_time']
        df['capture_stabilize_time'] = params['capture_stabilize_time']
        df['capture_incision_rate'] = params['capture_incision_rate']
        df['capture_node'] = params['capture_node']

    # get lowering, ic, climate, model
    df['model_name'] = rf.split(os.path.sep)[7].split('_')[-1]
    df['lowering_future'] = rf.split(os.path.sep)[8].split('.')[0]
    df['initial_condition'] = rf.split(os.path.sep)[8].split('.')[1]
    df['climate_future'] = rf.split(os.path.sep)[8].split('.')[-1]
    df['breach_location'] = rf.split(os.path.sep)[9].split('.')[0]

    # set column order
    column_order = ['model_name', 'lowering_future', 'initial_condition', 'climate_future', 'capture_node',
                    'breach_location', 'capture_start_time', 'capture_stabilize_time',
                    'capture_incision_rate', 'post_stabilization_incision_rate']
    column_order.extend(data_columns)
    # append re-ordered dataframe
    df_list.append(df[column_order])
    # copy figures
    figs = glob.glob(os.path.join(os.path.split(rf)[0], '*.png'))
    for fig in figs:
        shutil.copy(fig, fig_path)

# combine dataframes and save
df_full = pd.concat(df_list)
df_full.to_csv('compilation_of_sew_breaching_uncert_output.csv')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 21:24:52 2017

@author: barnhark
"""

# import modules
import pandas as pd
import os
import yaml
import numpy as np

#%%
# get weights

weights_filepath_sew = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'auxillary_inputs', 'weights', 'sew_variance.txt']
new_weights_filepath_sew = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'auxillary_inputs', 'weights', 'sew_variance_with_model800.txt']
variance_df = pd.read_csv(os.path.join(os.sep, *new_weights_filepath_sew),index_col=0)
df_sew = pd.DataFrame(data = dict(Weights=1.0/(variance_df.Model_800_Based_Variance + variance_df.Error_Based_Variance)))

weights_filepath_gully = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'auxillary_inputs', 'weights', 'gully_variance.txt']
df_gully = 1./pd.read_csv(os.path.join(os.sep, *weights_filepath_sew), header=None, names=['Weights'])

#%%
# make stds
df_sew['Error Based Standard Deviation'] = (variance_df.Error_Based_Variance)**0.5
df_sew['Model 800 Based Standard Deviation'] = (variance_df.Model_800_Based_Variance)**0.5
df_sew['Combined Standard Deviation'] = (variance_df.Model_800_Based_Variance + variance_df.Error_Based_Variance)**0.5              
                    
df_gully['Error Based Standard Deviation'] = (1./df_gully.Weights)**0.5
df_gully['Model 800 Based Standard Deviation'] =np.nan
df_gully['Combined Standard Deviation'] = np.nan#(variance_df.Model_800_Based_Variance + variance_df.Error_Based_Variance)**0.5              
#%%
# get modern values
modern_values_filepath_sew = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'auxillary_inputs', 'modern_metric_files', 'dem24fil_ext.metrics.txt']
with open(os.path.join(os.sep, *modern_values_filepath_sew), 'r') as f:
    modern_sew = yaml.load(f)
    modern_sew.pop('Topo file')
    # the chi_density sum of squares is not stored in this file since it is
    # a surface misfit. It has an effective modern value of 0.0
    modern_sew['chi_density_sum_squares'] = 0.0
    
df_sew['Observed Values'] = pd.Series(modern_sew)  
df_sew['Coefficient of Variation'] = df_sew['Observed Values']/df_sew['Combined Standard Deviation']

modern_values_filepath_gully = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'auxillary_inputs', 'modern_metric_files', 'gdem3r1f.metrics.txt']
with open(os.path.join(os.sep, *modern_values_filepath_gully), 'r') as f:
    modern_gully = yaml.load(f)
    modern_gully.pop('Topo file')
    # the chi_density sum of squares is not stored in this file since it is
    # a surface misfit. It has an effective modern value of 0.0
    modern_sew['chi_density_sum_squares'] = 0.0
    
df_gully['Observed Values']  = pd.Series(modern_gully)  
df_gully['Coefficient of Variation'] = df_gully['Observed Values']/df_gully['Error Based Standard Deviation']

#%%
# re-arrange orders
column_order = ['Observed Values', 'Error Based Standard Deviation', 'Model 800 Based Standard Deviation', 'Combined Standard Deviation', 'Coefficient of Variation', 'Weights']
index_order = ['chi_density_sum_squares', 'chi_gradient', 'chi_intercept', 
               'one_cell_nodes', 'two_cell_nodes', 'three_cell_nodes', 'four_cell_nodes',
               'cumarea95', 'cumarea96', 'cumarea97', 'cumarea98', 'cumarea99', 
               'elev02', 'elev08', 'elev23', 'elev30', 'elev36', 
               'elev50', 'elev75', 'elev85',
               'elev90', 'elev96', 'elev100',
               'hypsometric_integral',
               'mean_elevation', 'var_elevation',
               'mean_elevation_chi_area', 'var_elevation_chi_area',
               'mean_gradient', 'var_gradient',
               'mean_gradient_chi_area','var_gradient_chi_area']

df_sew = df_sew.reindex(index=index_order, columns=column_order)
df_gully = df_gully.reindex(index=index_order, columns=column_order)

weights_filepath_gully = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'metrics', 'gully_observations_and_weights.csv']
df_gully.to_csv(os.path.join(os.sep, *weights_filepath_gully), float_format='%.3e')

weights_filepath_sew = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'metrics', 'sew_observations_and_weights.csv']
df_sew.to_csv(os.path.join(os.sep, *weights_filepath_sew), float_format='%.3e')

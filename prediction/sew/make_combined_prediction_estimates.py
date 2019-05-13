#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 07:46:29 2018

@author: barnhark
"""

import os
import glob

import numpy as np
import pandas as pd

import xarray as xr

from joblib import Parallel, delayed
from landlab.io import read_esri_ascii

# set paths to result .nc files
best_param_path = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', 'sew', 'BEST_PARAMETERS', 'model_*', '*.*.*', 'model_*_*.nc']
ic_path = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', 'sew', 'IC_UNCERTAINTY', 'model_*', '*.*.*', 'run.*', 'model_*_*.nc']


cm_fn = 'cross_model.csv'
if os.path.exists(cm_fn):
    print('loading cm file')
    cross_model = pd.read_csv(cm_fn, index_col=0, dtype={'model_name': str})
else:
    print('creating cm file')
    # get all resulting nc files
    cross_model = pd.DataFrame({'file_name':np.sort(glob.glob(os.path.join(os.path.sep, *best_param_path)))})
    # create variables for easy subsetting by  model and climate or lowering scenario
    cross_model['model_name'] = cross_model.file_name.str.split(pat=os.path.sep).apply(lambda x: x[-1]).str.split('_').apply(lambda x: x[1])
    cross_model['model_time'] = cross_model.file_name.str.split(pat=os.path.sep).apply(lambda x: x[-1]).str.split('_').apply(lambda x: x[-1][:4]).astype(int)*100
    cross_model['lowering_future'] = cross_model.file_name.str.split(pat=os.path.sep).apply(lambda x: x[-2]).str.split('.').apply(lambda x: x[0])
    cross_model['climate_future'] = cross_model.file_name.str.split(pat=os.path.sep).apply(lambda x: x[-2]).str.split('.').apply(lambda x: x[2])
    cross_model.to_csv(cm_fn)
    
ic_fn = 'initial_condition.csv'
if os.path.exists(ic_fn):
    print('loading ic file')
    initial_condition = pd.read_csv(ic_fn, index_col=0, dtype={'model_name': str})
else:
    print('creating ic file')
    initial_condition = pd.DataFrame({'file_name':np.sort(glob.glob(os.path.join(os.path.sep, *ic_path)))})  
    initial_condition['model_name'] = initial_condition.file_name.str.split(pat=os.path.sep).apply(lambda x: x[-1]).str.split('_').apply(lambda x: x[1])
    initial_condition['model_time'] = initial_condition.file_name.str.split(pat=os.path.sep).apply(lambda x: x[-1]).str.split('_').apply(lambda x: x[-1][:4]).astype(int)*100
    initial_condition['lowering_future'] = initial_condition.file_name.str.split(pat=os.path.sep).apply(lambda x: x[-3]).str.split('.').apply(lambda x: x[0])
    initial_condition['climate_future'] = initial_condition.file_name.str.split(pat=os.path.sep).apply(lambda x: x[-3]).str.split('.').apply(lambda x: x[2])
    initial_condition['model_climate_lowering'] = initial_condition[['model_name', 'climate_future', 'lowering_future']].apply(lambda x: '.'.join(x), axis=1)
    initial_condition.to_csv(ic_fn)

# save these dataframes since they take quite a long time to make.

# seems like these take about 20 minutes to make,  give it more power.

# construct model set dictionary
model_sets = {'only842': ['842'],
              'all800s': ['800', '802', '804', '808', '810', '840', '842', 'A00', 'C00']}

# identify all times evaluated
times = cross_model.model_time.unique()

# set varibles to ignore for faster opening of nc files
ignore_vars = ['K_br',
                'bedrock__elevation',
                'cumulative_erosion__depth',
                'depression__depth',
                'drainage_area',
                'effective_drainage_area',
                'erosion__threshold',
                'flow__sink_flag',
                'initial_topographic__elevation',
                'is_pit',
                'rock_till_contact__elevation',
                'sediment__flux',
                'sediment_fill__depth',
                'soil__depth',
                'soil_production__rate',
                'sp_crit_br',
                'substrate__erodibility',
                'subsurface_water__discharge',
                'surface_water__discharge',
                'topographic__steepest_slope',
                'water__unit_flux_in']

# create an output file if it doesn't yet exist.
out_folder = 'synthesis_netcdfs'
if os.path.exists(out_folder) is False:
    os.mkdir(out_folder)

path =  ['work','WVDP_EWG_STUDY3','study3py','auxillary_inputs','dems','sew', 'modern', 'dem24fil_ext.txt']

modern_dem = os.path.join(os.path.sep, *path)

grd, zzm = read_esri_ascii(modern_dem, name='topographic__elevation', halo=1)

array_ID = os.environ['SLURM_ARRAY_TASK_ID']

# main analysis loop
parallel_inputs = []
for set_key in  np.sort(list(model_sets.keys())): # for model set
    for t in times: # for recorded times
        
        if int(np.remainder(t, 1000)/100) == int(array_ID):
            parallel_inputs.append([set_key, t, cross_model, initial_condition, ignore_vars, model_sets, zzm.reshape(grd.shape)])

        #out = average_results(set_key, t, cross_model, initial_condition, ignore_vars, model_sets)


def average_results(set_key, t, cross_model, initial_condition, ignore_vars, used_models, zzm):

    print(set_key, t)

    out_name = os.path.join(out_folder, set_key+'_synthesis_' + str(t).zfill(5) + '.nc')
    
    # if a file exists, append it and don't run
    if os.path.exists(out_name):
        run = False
    
    # otherwise, check if it is in the log file list
    else:
        run=True
            
        
    if run:
        used_models = model_sets[set_key] # get used models
        
        # select the correct parts of the PD dataframe for cm and ic
        cm_sel = cross_model[cross_model.model_name.isin(used_models)&(cross_model.model_time == t)]
        ic_sel = initial_condition[initial_condition.model_name.isin(used_models)&(initial_condition.model_time == t)]
        
        # 1) expected value is the mean of the IC values since it is now a balanced experiment
        # however this involves opening 900 or 8100 15 mb files at the same time= bad, 
        # open them in chunks of 100 and average
        
        # d) IC uncertainty,
        mcls = ic_sel.model_climate_lowering.unique()
        ic_list = []
        ic_mean_list = []
        for mcl in mcls:
            ic_sel_mcl = ic_sel[ic_sel.model_climate_lowering == mcl]
        
            ds = xr.open_mfdataset(ic_sel_mcl.file_name,
                   concat_dim='nr',
                   engine='netcdf4',
                   data_vars=['topographic__elevation'],
                   drop_variables=ignore_vars)
        
            ic_list.append(ds.std(dim='nr').squeeze()**2)
            ic_mean_list.append(ds.mean(dim='nr').squeeze())
            
        out_data_array = xr.concat(ic_mean_list, dim='nic').topographic__elevation.mean(dim='nic').squeeze()
        out_dataset = xr.Dataset({'expected_topographic__elevation': out_data_array})
        
        expected_cumulative_erosion = out_dataset.expected_topographic__elevation - zzm
        out_dataset.__setitem__('expected_cumulative_erosion__depth', (expected_cumulative_erosion.dims, expected_cumulative_erosion.values))
        
        # 2) uncertanties
        
        std_ic_topo = xr.concat(ic_list, dim='nic').topographic__elevation.mean(dim='nic').squeeze()**0.5
        out_dataset.__setitem__('std_topo_ic', (std_ic_topo.dims, std_ic_topo.values))
        
        
        # a) across model uncertainty
        if set_key == 'all800s':
            models = cm_sel.model_name.unique()
        
            model_list = []
            for model in models:
                cm_sel_mod = cm_sel[cm_sel.model_name == model]
        
                ds = xr.open_mfdataset(cm_sel_mod.file_name,
                       concat_dim='nr',
                       engine='netcdf4',
                       data_vars=['topographic__elevation'],
                       drop_variables=ignore_vars)
        
                model_list.append(ds.mean(dim='nr').squeeze())
        
            topo_model_std = xr.concat(model_list, dim='nm').topographic__elevation.std(dim='nm').squeeze()
        
            out_dataset.__setitem__('std_topo_model', (topo_model_std.dims, topo_model_std.values))
        
        # b) across lowering uncertainty
        lowerings = cm_sel.lowering_future.unique()
        
        lowering_list = []
        for lowering in lowerings:
            cm_sel_low = cm_sel[cm_sel.lowering_future == lowering]
        
            ds = xr.open_mfdataset(cm_sel_low.file_name,
                   concat_dim='nr',
                   engine='netcdf4',
                   data_vars=['topographic__elevation'],
                   drop_variables=ignore_vars)
        
            lowering_list.append(ds.mean(dim='nr').squeeze())
        
        topo_lower_std = xr.concat(lowering_list, dim='nl').topographic__elevation.std(dim='nl').squeeze()
        
        out_dataset.__setitem__('std_topo_lower', (topo_lower_std.dims, topo_lower_std.values))
        
        # c) across climate
        climates = cm_sel.climate_future.unique()
        
        climate_list = []
        for climate in climates:
            cm_sel_cli = cm_sel[cm_sel.climate_future == climate]
        
            ds = xr.open_mfdataset(cm_sel_cli.file_name,
                   concat_dim='nr',
                   engine='netcdf4',
                   data_vars=['topographic__elevation'],
                   drop_variables=ignore_vars)
        
            climate_list.append(ds.mean(dim='nr').squeeze())
        
        topo_cli_std = xr.concat(climate_list, dim='nc').topographic__elevation.std(dim='nc').squeeze()
        
        out_dataset.__setitem__('std_topo_clim', (topo_cli_std.dims, topo_cli_std.values))
        
        
        if set_key == 'all800s':
            std_total_topo = (topo_cli_std**2 + topo_lower_std**2 + topo_model_std**2 + std_ic_topo**2)**0.5
        
        else:
            std_total_topo = (topo_cli_std**2 + topo_lower_std**2 + std_ic_topo**2)**0.5
        
        out_dataset.__setitem__('std_total_topo', (std_total_topo.dims, std_total_topo.values))
        out_dataset.to_netcdf(out_name, engine='netcdf4', format='NETCDF4')

output = Parallel(n_jobs=1)(delayed(average_results)(*inputs) for inputs in parallel_inputs)

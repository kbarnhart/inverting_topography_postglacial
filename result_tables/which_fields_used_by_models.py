#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:30:00 2018

@author: barnhark
"""

# get model field names
import xarray as xr
import glob
import pandas as pd
import os

path = '/work/WVDP_EWG_STUDY3/study3py/prediction/sew/BEST_PARAMETERS/model_*/lowering_future_1.dem24fil_ext.constant_climate/*_0100.nc'
files = glob.glob(path)

df_list = []
for file in files:
    ds = xr.open_mfdataset(file, engine='netcdf4')
    model_key = os.path.split(file)[-1].split('_')[1]
    variables = []
    for var in ds.variables:
        variables.append(var)
    model_df = pd.DataFrame({'field_names' : variables})
    model_df['model'] = model_key
    df_list.append(model_df)
    

df = pd.concat(df_list, axis=0)

df.set_index(['field_names'], inplace=True)
df.sort_index(level=-1, inplace=True)

df.to_csv('field_names_used_by_models.csv')

unique_names = pd.DataFrame(df.index.unique())
unique_names.to_csv('unique_field_names.csv', index=False, header=False)
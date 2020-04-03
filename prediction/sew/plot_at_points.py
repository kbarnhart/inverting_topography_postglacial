#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 07:42:38 2020

@author: barnhark
"""

import os
import glob

import numpy as np
import pandas as pd
 
import xarray as xr

import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
ft2m = 0.3048


#%%
# points
pdf = pd.read_csv("PredictionPoints_ShortList.csv")
pdf.set_index("Point_Name", inplace=True)

#%%
colors = {"lower": "#e6ab02",
          "climate":"#1b9e77",
          "model": "#e7298a",
          "param":"#66a61e",
          "interactions": "#d95f02",
          "ic": "#7570b3",
          "total": "k"}

order = [ "interactions", "lower", "climate", "ic", "param", "model"]

for mod in ["only842", "all800s"]:
    
    files = np.sort(glob.glob("synthesis_netcdfs_with_param_test/{key}*.nc".format(key=mod)))
    
    time = np.asarray([int(file.split('.')[-2].split('_')[-1]) for file in files])/1000

    ds = xr.open_mfdataset(files, concat_dim='nt', engine='netcdf4')
    
    spec_shape = (5,5)
    
    fig = plt.figure(figsize=(5, 5), dpi=300)  # convert mm to inches
    spec = gridspec.GridSpec(
        ncols=spec_shape[0],
        nrows=spec_shape[0],
        left=0, right=1, top=1, bottom=0,
        figure=fig,
    )

    fig2 = plt.figure(figsize=(5, 5), dpi=300)  # convert mm to inches
    spec2 = gridspec.GridSpec(
        ncols=spec_shape[0],
        nrows=spec_shape[0],
        left=0, right=1, top=1, bottom=0,
        figure=fig2,
    )
    

    fig3 = plt.figure(figsize=(5, 5), dpi=300)  # convert mm to inches
    spec3 = gridspec.GridSpec(
        ncols=spec_shape[0],
        nrows=spec_shape[0],
        left=0, right=1, top=1, bottom=0,
        figure=fig3,
    )
    
    for i, point in enumerate(pdf.index):
        
        row = pdf.Row_number.loc[point]
        col = pdf.Column_number.loc[point]
        
        vals = {}
        vals["ic"] = ds.topo_ic_std.values[:, row, col]*ft2m
        vals["lower"] = ds.topo_lower_std.values[:, row, col]*ft2m
        vals["climate"] = ds.topo_cli_std.values[:, row, col]*ft2m
        vals["param"] = ds.topo_exp2_param_independent_std.values[:, row, col]*ft2m
        
        
        vals["interactions"] = ds.topo_interactions_std.values[:, row, col]*ft2m
        
        if mod =="all800s":
            vals["model"] = ds.topo_model_std.values[:, row, col]*ft2m
            vals["model2"] = ds.topo_exp2_model_std.values[:, row, col]*ft2m
            
            vals["exp1_mean"] = ds.expected_topographic__elevation.values[:, row, col]*ft2m
            vals["exp2_mean"] = ds.topo_exp2_mean.values[:, row, col]*ft2m    
            
            vals["param_cor"] = ds.topo_exp2_model_param_correlated_std.values[:, row, col]*ft2m
            vals["param_global_std"] = ds.topo_exp2_std.values[:, row, col]*ft2m
            
            
        
        vals["total"] = ds.topo_total_star_std.values[:, row, col]*ft2m
        
        ax = fig.add_subplot(spec[np.unravel_index(i, spec_shape)])
        
        for key in order:
            if key in vals:
                ax.plot(time, vals[key], colors[key])
        ax.plot(time, vals["total"], colors["total"])

        ax.patch.set_alpha(0. )
        ax.text(0.2,
                 0.9, point,
                   transform=ax.transAxes,
                   color="k",
                   ha="left",
                   va="center"
               )
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        ax = fig2.add_subplot(spec2[np.unravel_index(i, spec_shape)])
        v = []
        c = []
        for key in order:
            if key in vals:
                v.append(vals[key]**2 / vals["total"]**2) 
                c.append(colors[key])
        
        ax.stackplot(time, np.vstack(v), colors=c)
    
        ax.patch.set_alpha(0. )
        ax.text(0.1,
                  0.9, point,
                    transform=ax.transAxes,
                    color="k",
                    ha="left",
                    va="center"
                )
        
        if mod == "all800s":
            ax = fig3.add_subplot(spec3[np.unravel_index(i, spec_shape)])
            
            ax.plot(time, vals["param_global_std"], "k")
            ax.plot(time, vals["model2"], "g")            
            ax.plot(time, vals["param"], "r")
            ax.plot(time, vals["param_cor"], "orange")
            
            ax.patch.set_alpha(0. )
            ax.text(0.1,
                      0.9, point,
                        transform=ax.transAxes,
                        color="k",
                        ha="left",
                        va="center"
                    )
        
           # (vals["model2"]**2 + vals["param"]**2) / vals["param_global_std"] **2

  
    fig.savefig("proportion_plots/{key}_test.png".format(key=mod))
    fig2.savefig("proportion_plots/{key}_proportion.png".format(key=mod))
    fig3.savefig("proportion_plots/{key}_modcomparison.png".format(key=mod))
    
# #%%
    
# mu_cor = np.zeros((ds.dims['ni'], ds.dims['nj']))
# sigma_cor = np.zeros((ds.dims['ni'], ds.dims['nj']))

# from itertools import product
# from scipy.stats import linregress

# for (ni, nj) in product(
#         range(ds.dims['ni']), 
#         range(ds.dims['nj'])):
#     test = ds.expected_topographic__elevation.values[0, ni, nj] > 0
#     if test:
#         print(ni)
#         X = ds.expected_topographic__elevation.values[1:, ni, nj]
#         y = ds.topo_exp2_mean.values[1:, ni, nj]
#         reg = linregress(X, y)
#         mu_cor[ni, nj]  = reg.rvalue
        
#         X = ds.topo_model_std.values[1:, ni, nj]
#         y = ds.topo_exp2_model_std.values[1:, ni, nj]
#         reg = linregress(X, y)
#         sigma_cor[ni, nj]  = reg.rvalue
        

# #%%             
# plt.plot(ds.topo_model_std.values.flatten()*ft2m, ds.topo_exp2_model_std.values.flatten()*ft2m, ".", ms=0.1, alpha=0.2)
# plt.show()

# #%%
# plt.imshow(sigma_cor)

# #%%
# plt.imshow(mu_cor)
# #%%
# plt.plot(ds.expected_topographic__elevation.values.flatten()*ft2m, ds.topo_exp2_mean.values.flatten()*ft2m, ".", ms=0.1, alpha=0.2)
# plt.xlim([0, 1000])
# plt.ylim([0, 1000])
# plt.show()



#%%

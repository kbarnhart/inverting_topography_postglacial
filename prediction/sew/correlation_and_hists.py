#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:01:50 2020

@author: barnhark
"""

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

from itertools import product
from scipy.stats import linregress

from landlab.io import read_esri_ascii

#%%
path = [
    "..",
    "..",
    "auxillary_inputs",
    "dems",
    "sew",
    "modern",
    "dem24fil_ext.txt",
]

modern_dem = os.path.join(*path)

grd, zzm = read_esri_ascii(modern_dem, name="topographic__elevation", halo=1)

#%%
ft2m = 0.3048

bins = np.arange(-140, 40, 0.1)

colors = ["#a6611a", "#dfc27d", "#018571", "#80cdc1"]
out = {}

plot_text = {"only842": "BasicChRtTh", "all800s": "Eight Rt variants"}
figure, ax = plt.subplots(2, 1, figsize=(5,4), sharex=True)
for mod in ["only842", "all800s"]:    
    files = np.sort(glob.glob("synthesis_netcdfs_with_param/{key}*.nc".format(key=mod)))    
    time = np.asarray([int(file.split('.')[-2].split('_')[-1]) for file in files])/1000
    
    with xr.open_mfdataset(files, concat_dim='nt', engine='netcdf4') as ds:
        core = ds.expected_topographic__elevation.values[10, :, :]>0
        not_core = ds.expected_topographic__elevation.values[10, :, :]<0
    
        five = ds.expected_cumulative_erosion__depth.values[5, :, :][core]*ft2m
        ten = ds.expected_cumulative_erosion__depth.values[10, :, :][core]*ft2m
        print(np.median(five))
        print(np.median(ten))
        
        c = colors.pop()
        ax[0].hist(five, bins=bins, density=True,   histtype="step", label=plot_text[mod]+", +5 ka", color=c)
        ax[1].hist(five, bins=bins, density=True,   histtype="step", cumulative=True, color=c)
        c = colors.pop()
        
        ax[0].hist(ten, bins=bins, density=True,  histtype="step", label=plot_text[mod]+", +10 ka", color=c)
        ax[1].hist(ten, bins=bins, density=True,  histtype="step", cumulative=True, color=c)
        
        out[(mod, "five")] = ds.expected_cumulative_erosion__depth.values[5, :, :]*ft2m
        out[(mod, "ten")] = ds.expected_cumulative_erosion__depth.values[10, :, :]*ft2m
        
plt.xlim(2, -10)
ax[1].set_ylim(1, 0)
ax[1].set_yticks(ticks=(1, 0.5, 0))
ax[1].set_yticklabels(labels=(0, 0.5, 1))

ax[0].annotate("a. PDF", xy=(0.03, 0.85), xycoords='axes fraction')
ax[1].annotate("b. CDF", xy=(0.03, 0.85), xycoords='axes fraction')

# both have about 17 percent exeeding 5 m erosion. 
#ax[1].hlines(y=0.17, xmin=-10, xmax=2)
#ax[1].vlines(x=-5, ymin=0, ymax=1)

ax[0].set_yticks([])

ax[0].legend(loc="upper right")
plt.xlabel(r"${Deposition}$                  Elevation Change [m]                  $Erosion$")
plt.savefig("hists.png", dpi=500)
plt.show()  
    #%%
colors = ["#a6611a", "#dfc27d", "#018571", "#80cdc1"]

figure = plt.figure(figsize=(5,4))
for mod in ["only842", "all800s"]:    
    files = np.sort(glob.glob("synthesis_netcdfs_with_param/{key}*.nc".format(key=mod)))    
    time = np.asarray([int(file.split('.')[-2].split('_')[-1]) for file in files])/1000
    
    with xr.open_mfdataset(files, concat_dim='nt', engine='netcdf4') as ds:
        core = ds.expected_topographic__elevation.values[10, :, :]>0
        not_core = ds.expected_topographic__elevation.values[10, :, :]<0
    
        five = ds.expected_cumulative_erosion__depth.values[5, :, :][core]*ft2m
        ten = ds.expected_cumulative_erosion__depth.values[10, :, :][core]*ft2m
        print(np.median(five))
        print(np.median(ten))
        plt.hist(five, bins=bins, density=True,   histtype="step", cumulative=True, label=plot_text[mod]+", +5 ka", color=colors.pop())
        plt.hist(ten, bins=bins, density=True,  histtype="step", cumulative=True, label=plot_text[mod]+", +10 ka", color=colors.pop())
        
        out[(mod, "five")] = ds.expected_cumulative_erosion__depth.values[5, :, :]*ft2m
        out[(mod, "ten")] = ds.expected_cumulative_erosion__depth.values[10, :, :]*ft2m
        
plt.xlim(0, -40)
#plt.ylim(0,0.5)
plt.legend(loc="upper right")
plt.xlabel(r"${Deposition}$                  Elevation Change [m]                  $Erosion$")
plt.savefig("hists2.png", dpi=500)
plt.show()  
    
#%%


files = np.sort(glob.glob("synthesis_netcdfs_with_param/all800s*.nc"))    
time = np.asarray([int(file.split('.')[-2].split('_')[-1]) for file in files])/1000

with xr.open_mfdataset(files, concat_dim='nt', engine='netcdf4') as ds:
    core = ds.expected_topographic__elevation.values[10, :, :]>0
    not_core = ds.expected_topographic__elevation.values[10, :, :]<0
    
    std_lim = 33 * ft2m
    ero_lim = 100 *ft2m
        
    mc = ds.topo_model_climate_interaction_std.values[10, :, :]*ft2m
    cl = ds.topo_climate_lowering_interaction_std.values[10, :, :]*ft2m
    ml = ds.topo_model_lowering_interaction_std.values[10, :, :]*ft2m
    mcl = ds.topo_model_climate_lowering_interaction_std.values[10, :, :]*ft2m
    
    mc[not_core] = np.nan
    cl[not_core] = np.nan
    ml[not_core] = np.nan
    mcl[not_core] = np.nan
    
    
    fig, ax = plt.subplots(3, 2, figsize=(6,6), dpi=300, gridspec_kw={"height_ratios":[1,1,0.2]})
    
    ax[0,0].imshow(mc, cmap="viridis_r", vmin=0, vmax=std_lim)
    ax[0,0].set_title("a. Model-Climate")
    ax[0,0].axis("off")
    ax[0,0].invert_yaxis()
    
    
    ax[0,1].imshow(cl, cmap="viridis_r", vmin=0, vmax=std_lim)
    ax[0,1].set_title("b. Climate-Downcutting")
    ax[0,1].axis("off")
    ax[0,1].invert_yaxis()
    
    ax[1,0].imshow(ml, cmap="viridis_r", vmin=0, vmax=std_lim)
    ax[1,0].set_title("c. Model-Downcutting")
    ax[1,0].axis("off")
    ax[1,0].invert_yaxis()
    
    im =  ax[1,1].imshow(mcl, cmap="viridis_r", vmin=0, vmax=std_lim)
    ax[1,1].set_title("d. Model-Climate-Downcutting")
    ax[1,1].axis("off")
    ax[1,1].invert_yaxis()
    
    #cax = plt.axes( [0.2, 0.2, 0.7, 0.0.05])
    ax[2,0].axis("off")
    ax[2,1].axis("off")
    plt.colorbar(im, ax=ax[2,:], orientation="horizontal", label="Standard Deviation[m]", fraction=0.8)
    
    plt.savefig("interactions.png")
    
    

    # mean vs mean
    exp1mu = ds.expected_cumulative_erosion__depth.values[10, :, :]*ft2m
    exp2mu = (ds.topo_exp2_mean.values[10, :, :]-zzm.reshape(grd.shape))*ft2m 
    
    
    # model vs model 
    exp1sigma = ds.topo_model_std.values[10, :, :]*ft2m
    exp2sigma = ds.topo_exp2_model_std.values[10, :, :]*ft2m
    
    exp1mu[not_core] = np.nan
    exp2mu[not_core] = np.nan
    exp1sigma[not_core] = np.nan
    exp2sigma[not_core] = np.nan
    
    figure = plt.figure(figsize=(4,3.5), dpi=300)
    
    ax3= plt.subplot(121)
    plt.imshow(exp1sigma, cmap="viridis_r", vmin=0, vmax=std_lim)
    plt.title("a. Experiment 1")
    plt.axis("off")
    plt.gca().invert_yaxis()
    
    ax4 = plt.subplot(122)
    plt.imshow(exp2sigma, cmap="viridis_r", vmin=0, vmax=std_lim)
    plt.title("b. Experiment 2")
    plt.axis("off")
    plt.gca().invert_yaxis()
    
    plt.colorbar(ax=[ax3, ax4], location='bottom', shrink=0.6, label="Standard Deviation[m]")
    
    plt.savefig("mu_sigma.png")
    
    # cor vs not. 
    
    exp2inde = ds.topo_exp2_std.values[10, :, :]*ft2m
    exp2_corr = ds.topo_exp2_model_param_correlated_std.values[10, :, :]*ft2m
    
    
    exp2inde[not_core] = np.nan
    exp2_corr[not_core] = np.nan
    
    figure = plt.figure(figsize=(2.5,6), dpi=300)

    ax1= plt.subplot(311)
    plt.imshow(exp2inde, cmap="viridis_r", vmin=0, vmax=std_lim)
    plt.title("a. Independent")
    plt.axis("off")
    plt.gca().invert_yaxis()
    
    ax2 = plt.subplot(312)
    plt.imshow(exp2_corr, cmap="viridis_r", vmin=0, vmax=std_lim)
    plt.title("b. Correlated")
    plt.axis("off")
    plt.gca().invert_yaxis()
    plt.colorbar(ax=[ax1, ax2], location='right', shrink=0.6, label="Standard Deviation [m]")

    ax3 = plt.subplot(313)
    plt.imshow(exp2inde - exp2_corr, cmap="PuBu", vmin=0, vmax=2)
    plt.title("c. Difference (a-b)")
    plt.axis("off")
    plt.gca().invert_yaxis()
    plt.colorbar(ax=[ax3], location='right', shrink=0.8,label="Difference [m]")

    
    plt.savefig("cor_compare.png")

mean = (exp2inde + exp2_corr)/2
out = (exp2inde - exp2_corr)/(mean)
plt.hist(out.flatten(), bins=100, cumulative=True, density=True)
    
# #%%
# comp = (out[("all800s", "ten")] - out[("only842", "ten")])/(out[("all800s", "ten")] + out[("only842", "ten")])

# im = plt.imshow(np.log10(np.abs(comp)), cmap="PuOr", vmin=-2, vmax=2)
# plt.colorbar(im)
# #%%
    
# for mod in ["all800s"]:    
#     files = np.sort(glob.glob("synthesis_netcdfs_with_param/{key}*.nc".format(key=mod)))    
#     time = np.asarray([int(file.split('.')[-2].split('_')[-1]) for file in files])/1000
    
#     ds = xr.open_mfdataset(files, concat_dim='nt', engine='netcdf4')
    
#     mu_cor = np.zeros((ds.dims['nj'], ds.dims['ni']))
#     sigma_cor = np.zeros((ds.dims['nj'], ds.dims['ni']))

    
#     for (ni, nj) in product(
#             range(ds.dims['ni']), 
#             range(ds.dims['nj'])):
#         test = ds.expected_topographic__elevation.values[0, nj, ni] > 0
#         if test:
#             if ni%10==0:
#                 print(ni)
#             X = ds.expected_topographic__elevation.values[1:, nj, ni]
#             y = ds.topo_exp2_mean.values[1:, nj, ni]
#             reg = linregress(X, y)
#             mu_cor[nj, ni]  = reg.rvalue
            
#             X = ds.topo_model_std.values[1:, nj, ni]
#             y = ds.topo_exp2_model_std.values[1:, nj, ni]
#             reg = linregress(X, y)
#             sigma_cor[nj, ni]  = reg.rvalue
        
        
# #%%
# plt.subplot(121)

# plt.imshow(mu_cor)
# plt.title("mu ")
# plt.subplot(122)

# plt.imshow(sigma_cor)
# plt.title("sigma ")

# plt.colorbar()

# plt.savefig("musigma.png", dpi=500)
# plt.show

#%%

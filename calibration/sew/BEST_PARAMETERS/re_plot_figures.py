#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:33:57 2017

@author: barnhark
"""

# Start by importing the necessary python libraries
import os
import glob
import yaml
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pylab as plt

from landlab.io.netcdf import read_netcdf
from landlab.io import read_esri_ascii

from landlab.plot import imshow_grid

fig_out = os.path.join(os.path.sep, *['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_figures', 'calibration', 'sew' , 'BEST_PARAMETERS'])

if os.path.exists(fig_out) is False:
    os.makedirs(fig_out)
    
SMALL_SIZE = 7
MEDIUM_SIZE = 9
BIGGER_SIZE = 11

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

ncs = glob.glob(os.path.join(*['*','*', 'model_**_0130.nc']))

def replot(nc):
    nc_full = os.path.abspath(nc)
    print(nc_full)
    grid = read_netcdf(nc)
    
    # z = modeled modern
    z = grid.at_node['topographic__elevation']

    fig_name = fig_out + os.path.sep + '.'.join(nc_full.split(os.path.sep)[6:9])+'.png'
    inputs = glob.glob(os.path.split(nc)[0] + os.sep + 'inputs.txt')[0]

    with open(inputs, 'r') as f:
        params = yaml.load(f)
        outlet_id = params['outlet_id']
        initial_dem = params['DEM_filename']
        modern_dem = params['modern_dem_name']
        weight_file = params['category_weight_file']


    grid.set_watershed_boundary_condition_outlet_id(outlet_id, z, nodata_value=-9999)

    # read modern and initial
    
    # iz = initial 
    (igrid, iz) = read_esri_ascii(initial_dem,
                                 name='topographic__elevation',
                                 halo=1)
    igrid.set_watershed_boundary_condition_outlet_id(outlet_id, z, nodata_value=-9999)

    # mz= actual modern
    (mgrid, mz) = read_esri_ascii(modern_dem,
                                  name='topographic__elevation',
                                  halo=1)
    mgrid.set_watershed_boundary_condition_outlet_id(outlet_id, z, nodata_value=-9999)


    #topographic change from start
    elevation_change = z - iz
    ec_lim = np.max(np.abs(elevation_change[grid.core_nodes]))

    # topographic difference from modern
    difference_from_modern = z - mz
    dfm_lim = np.max(np.abs(difference_from_modern[grid.core_nodes]))

    # weighted difference.
    effective_weight_file = weight_file.replace('chi_elev_weight', 'chi_elev_effective_weight')
    effective_weights = np.loadtxt(effective_weight_file)
    squared_weighted_residual = (difference_from_modern**2) * effective_weights
    swr_lim = np.max(np.abs(squared_weighted_residual[grid.core_nodes]))

    fs = (3, 2.4)
    plt.figure(figsize=fs, dpi=300)
    imshow_grid(grid, z, vmin=1230, vmax=1940, cmap='viridis', plot_name='End of Model Run Topography [ft]')
    plt.savefig(fig_name[:-4]+'.png')

    plt.figure(figsize=fs, dpi=300)
    imshow_grid(grid, elevation_change, vmin=-160, vmax=160, cmap='RdBu', plot_name='Change Since Start [ft]')
    plt.savefig(fig_name[:-4]+'.elevation_change_since_start.png')

    plt.figure(figsize=fs, dpi=300)
    imshow_grid(grid, difference_from_modern, vmin=-160, vmax=160, cmap='PuOr', plot_name='Change From Modern [ft]')
    plt.savefig(fig_name[:-4]+'.diff_modern.png')

    plt.figure(figsize=fs, dpi=300)
    imshow_grid(grid, squared_weighted_residual, vmin=0, vmax=0.4, cmap='plasma', color_for_closed = 'w', plot_name='Effective Residual [-]')
    plt.savefig(fig_name[:-4]+'.eff_resid.png')

    plt.close('all')
    return True

#ncores = 23
#output = Parallel(n_jobs=ncores)(delayed(replot)(nc) for nc in ncs)
#%%
for nc in ncs:
    output = replot(nc)

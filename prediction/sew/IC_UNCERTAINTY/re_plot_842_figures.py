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
import numpy as np

from matplotlib.collections import LineCollection

from landlab.io.netcdf import read_netcdf
from landlab.io import read_esri_ascii

from landlab.plot import imshow_grid

import shapefile as ps

plt.switch_backend('agg')

fig_out = os.path.join(os.path.sep, *['work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction', 'sew' , 'IC_UNCERTAINTY', 'topography_figures'])


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

ncs = np.sort(glob.glob(os.path.join(*['*','lowering_future_3.dem24fil_ext.RCP85', '*', 'model_842_0100.nc'])))


site_fp = '/work/WVDP_EWG_STUDY3/study3py/result_figures/site_shapefile/WVSITEFIG_SP_ft.shp'

x_offs = 1121548.2005005
y_offs = 886503.65051269

sf = ps.Reader(site_fp)
shapes = sf.shapes()
segments = []
for shape in shapes:
    points = shape.points
    x, y = zip(*points)
    x = np.array(x)-x_offs
    y = np.array(y)-y_offs
    segment = np.array((x, y)).T
    segments.append(segment)
    #%%

def replot(nc):
    nc_full = os.path.abspath(nc)
    print(nc_full)
    grid = read_netcdf(nc)

    # z = modeled modern
    z = grid.at_node['topographic__elevation']

    fig_name = fig_out + os.path.sep + '.'.join(nc_full.split(os.path.sep)[6:10])+'.png'
    inputs = glob.glob(os.path.split(nc)[0] + os.sep + 'inputs.txt')[0]

    with open(inputs, 'r') as f:
        params = yaml.load(f)
        outlet_id = params['outlet_id']
        initial_dem = params['DEM_filename']

    file_parts = nc_full.replace('IC_UNCERTAINTY', 'BEST_PARAMETERS').split(os.path.sep)

    grid.set_watershed_boundary_condition_outlet_id(outlet_id, z, nodata_value=-9999)

    # read modern and initial

    # iz = initial
    (igrid, iz) = read_esri_ascii(initial_dem,
                                 name='topographic__elevation',
                                 halo=1)
    igrid.set_watershed_boundary_condition_outlet_id(outlet_id, z, nodata_value=-9999)

    # equivalent without breaching
    nc_wo_br = os.path.join(os.path.sep, *(file_parts[:-2] + [file_parts[-1]]))
    base_grid = read_netcdf(nc_wo_br)

    base_grid.set_watershed_boundary_condition_outlet_id(outlet_id, z, nodata_value=-9999)
    bz = base_grid.at_node['topographic__elevation']

    #topographic change from start
    elevation_change = z - iz
    ec_lim = np.max(np.abs(elevation_change[grid.core_nodes]))

    # topographic difference from equivalent without breaching
    difference_from_base = z - bz
    dfm_lim = np.max(np.abs(difference_from_base[grid.core_nodes]))


    fs = (3, 2.4)
    fig, ax = plt.subplots(figsize=fs, dpi=300)
    imshow_grid(grid, z, vmin=1230, vmax=1940, cmap='viridis', plot_name='End of Model Run Topography [ft]')
    line_segments = LineCollection(segments, colors='k', linewidth=0.1)
    plt.axis('off')
    ax.add_collection(line_segments)
    plt.savefig(fig_name[:-4]+'.png')

    fig, ax = plt.subplots(figsize=fs, dpi=300)
    imshow_grid(grid, elevation_change, vmin=-160, vmax=160, cmap='RdBu', plot_name='Change Since Start [ft]')
    line_segments = LineCollection(segments, colors='k', linewidth=0.1)
    plt.axis('off')
    ax.add_collection(line_segments)
    plt.savefig(fig_name[:-4]+'.elevation_change_since_start.png')

    fig, ax = plt.subplots(figsize=fs, dpi=300)
    imshow_grid(grid, difference_from_base, vmin=-50, vmax=50, cmap='PuOr', plot_name='Change From Reference [ft]')
    line_segments = LineCollection(segments, colors='k', linewidth=0.1)
    plt.axis('off')
    ax.add_collection(line_segments)
    plt.savefig(fig_name[:-4]+'.diff_base.png')


    plt.close('all')
    return True

#ncores = 23
#output = Parallel(n_jobs=ncores)(delayed(replot)(nc) for nc in ncs)
#%%
for nc in ncs:
    output = replot(nc)

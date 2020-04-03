#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 07:41:02 2018

@author: barnhark
"""

#

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import shapefile as ps

plt.switch_backend('agg')

import xarray as xr

from landlab import RasterModelGrid
from landlab.plot import imshow_grid

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


in_folder = 'synthesis_netcdfs'

out_folder = 'synthesis_plots_meters'
if os.path.exists(out_folder) is False:
    os.mkdir(out_folder)


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

mg = RasterModelGrid((415, 438), dx=24)


set_keys = ['only842', 'all800s']
fs = (3, 2.4)

std_lim = 80
ero_lim = 150
ero_col = 'RdBu'

set_keys = ['only842', 'all800s']
plot_text = {'only842': 'only model 842', 'all800s': 'all nine 800 variants'}

text_x = 0.03*(mg.x_of_node.max() - mg.x_of_node.min())
text_y = 0.93  *(mg.y_of_node.max() - mg.y_of_node.min())

ft2m = 0.3048
for set_key in set_keys:

    files = np.sort(glob.glob(os.path.abspath(os.path.join(in_folder, set_key+'_synthesis_*.nc'))))

    for file in files:

        print(file)

        time = file.split(os.path.sep)[-1].split('_')[-1].split('.')[0]

        ds = xr.open_dataset(file, engine='netcdf4')

        mg.set_nodata_nodes_to_closed(ds.expected_topographic__elevation.values.flatten(), -9999)


        basic_fig_name = os.path.join(out_folder, set_key+'_'+time)

        # expected topography
        fig, ax = plt.subplots(figsize=fs, dpi=300)
        imshow_grid(mg, ds.expected_topographic__elevation.values.flatten()*ft2m, vmin=1130*ft2m, vmax=1940*ft2m, cmap='viridis', plot_name='Expected Elevation at +' + time + ' yr [m]')
        plt.axis('off')
        line_segments = LineCollection(segments, colors='k', linewidth=0.1)
        ax.add_collection(line_segments)
        plt.text(text_x, text_y, plot_text[set_key], color='w')
        plt.savefig(basic_fig_name +'_expected_topography.png')

        # expected erosion
        fig, ax = plt.subplots(figsize=fs, dpi=300)
        imshow_grid(mg, ds.expected_cumulative_erosion__depth.values.flatten()*ft2m, vmin=-ero_lim*ft2m, vmax=ero_lim*ft2m, cmap=ero_col, plot_name='Expected Erosion at +' + time + ' yr [m]')
        plt.axis('off')
        line_segments = LineCollection(segments, colors='k', linewidth=0.1)
        ax.add_collection(line_segments)
        plt.text(text_x, text_y, plot_text[set_key], color='w')
        plt.savefig(basic_fig_name +'_expected_erosion.png')

        # total std
        fig, ax = plt.subplots(figsize=fs, dpi=300)
        imshow_grid(mg, ds.std_total_topo.values.flatten()*ft2m, vmin=0, vmax=std_lim, cmap='gist_ncar_r', plot_name='Total$^{*}$ Uncertainty (1$\sigma$) at +' + time + ' yr [m]')
        plt.axis('off')
        line_segments = LineCollection(segments, colors='k', linewidth=0.1)
        ax.add_collection(line_segments)
        plt.text(text_x, text_y, plot_text[set_key], color='w')
        plt.savefig(basic_fig_name +'_topography_uncertainty_total.png')

        # model
        if set_key == 'all800s':
            fig, ax = plt.subplots(figsize=fs, dpi=300)
            imshow_grid(mg, ds.std_topo_model.values.flatten()*ft2m, vmin=0, vmax=std_lim*ft2m, cmap='gist_ncar_r', plot_name='Uncertainty from Model (1$\sigma$) at +' + time + ' yr [m]')
            plt.axis('off')
            line_segments = LineCollection(segments, colors='k', linewidth=0.1)
            ax.add_collection(line_segments)
            plt.text(text_x, text_y, plot_text[set_key], color='w')
            plt.savefig(basic_fig_name +'_topography_uncertainty_model.png')

        # lowering
        fig, ax = plt.subplots(figsize=fs, dpi=300)
        imshow_grid(mg, ds.std_topo_lower.values.flatten()*ft2m, vmin=0, vmax=std_lim*ft2m, cmap='gist_ncar_r', plot_name='Uncertainty from Lowering (1$\sigma$) at +' + time + ' yr [m]')
        plt.axis('off')
        line_segments = LineCollection(segments, colors='k', linewidth=0.1)
        ax.add_collection(line_segments)
        plt.text(text_x, text_y, plot_text[set_key], color='w')
        plt.savefig(basic_fig_name +'_topography_uncertainty_lower.png')

        # climate
        fig, ax = plt.subplots(figsize=fs, dpi=300)
        imshow_grid(mg, ds.std_topo_clim.values.flatten()*ft2m, vmin=0, vmax=std_lim*ft2m, cmap='gist_ncar_r', plot_name='Uncertainty from Climate(1$\sigma$) at +' + time + ' yr [m]')
        plt.axis('off')
        line_segments = LineCollection(segments, colors='k', linewidth=0.1)
        ax.add_collection(line_segments)
        plt.text(text_x, text_y, plot_text[set_key], color='w')
        plt.savefig(basic_fig_name +'_topography_uncertainty_clim.png')

        # ic
        fig, ax = plt.subplots(figsize=fs, dpi=300)
        imshow_grid(mg, ds.std_topo_ic.values.flatten()*ft2m, vmin=0, vmax=std_lim*ft2m, cmap='gist_ncar_r', plot_name='Uncertainty from IC (1$\sigma$) at +' + time + ' yr [m]')
        plt.axis('off')
        line_segments = LineCollection(segments, colors='k', linewidth=0.1)
        ax.add_collection(line_segments)
        plt.text(text_x, text_y, plot_text[set_key], color='w')
        plt.savefig(basic_fig_name +'_topography_uncertainty_ic.png')


        # exp - 1 sigma
        fig, ax = plt.subplots(figsize=fs, dpi=300)
        imshow_grid(mg, ds.expected_cumulative_erosion__depth.values.flatten()*ft2m - ds.std_total_topo.values.flatten()*ft2m, vmin=-ero_lim*ft2m, vmax=ero_lim*ft2m, cmap=ero_col, plot_name='Expected Erosion - 1$\sigma$ at +' + time + ' yr [m]')
        plt.axis('off')
        line_segments = LineCollection(segments, colors='k', linewidth=0.1)
        ax.add_collection(line_segments)
        plt.text(text_x, text_y, plot_text[set_key], color='w')
        plt.savefig(basic_fig_name +'_expected_erosion_m1sigma.png')

        # exp + 1 sigma
        fig, ax = plt.subplots(figsize=fs, dpi=300)
        imshow_grid(mg, ds.expected_cumulative_erosion__depth.values.flatten()*ft2m + ds.std_total_topo.values.flatten()*ft2m, vmin=-ero_lim*ft2m, vmax=ero_lim*ft2m, cmap=ero_col, plot_name='Expected Erosion + 1$\sigma$ at +' + time + ' yr [m]')
        plt.axis('off')
        line_segments = LineCollection(segments, colors='k', linewidth=0.1)
        ax.add_collection(line_segments)
        plt.text(text_x, text_y, plot_text[set_key], color='w')
        plt.savefig(basic_fig_name +'_expected_erosion_p1sigma.png')

        fig, ax = plt.subplots(figsize=fs, dpi=300)
        imshow_grid(mg, ds.expected_cumulative_erosion__depth.values.flatten() - 2.*ds.std_total_topo.values.flatten()*ft2m, vmin=-ero_lim*ft2m, vmax=ero_lim*ft2m, cmap=ero_col, plot_name='Expected Erosion - 2$\sigma$ at +' + time + ' yr [m]')
        plt.axis('off')
        line_segments = LineCollection(segments, colors='k', linewidth=0.1)
        ax.add_collection(line_segments)
        plt.text(text_x, text_y, plot_text[set_key], color='w')
        plt.savefig(basic_fig_name +'_expected_erosion_m2sigma.png')

        # exp + 2 sigma
        fig, ax = plt.subplots(figsize=fs, dpi=300)
        imshow_grid(mg, ds.expected_cumulative_erosion__depth.values.flatten()*ft2m + 2.*ds.std_total_topo.values.flatten()*ft2m, vmin=-ero_lim*ft2m, vmax=ero_lim*ft2m, cmap=ero_col, plot_name='Expected Erosion + 2$\sigma$ at +' + time + ' yr [m]')
        plt.axis('off')
        line_segments = LineCollection(segments, colors='k', linewidth=0.1)
        ax.add_collection(line_segments)
        plt.text(text_x, text_y, plot_text[set_key], color='w')
        plt.savefig(basic_fig_name +'_expected_erosion_p2sigma.png')

        # exp - 2 sigma > 20 ft
        fig, ax = plt.subplots(figsize=fs, dpi=300)
        vals = (ds.expected_cumulative_erosion__depth.values.flatten()*ft2m - 2.*ds.std_total_topo.values.flatten()*ft2m)<-6.0
        imshow_grid(mg, vals, vmin=0, vmax=1, cmap='Wistia', plot_name='Expected Erosion - 2$\sigma$ exceeds 6 m at +' + time + ' yr')
        plt.axis('off')
        line_segments = LineCollection(segments, colors='k', linewidth=0.1)
        ax.add_collection(line_segments)
        #cbar = fig.colorbar(ax, ticks=[0, 1])
        #cbar.ax.set_yticklabels(['False', 'True'])  # vertically oriented colorbar
        plt.text(text_x, text_y, plot_text[set_key], color='w')
        plt.savefig(basic_fig_name +'_expected_erosion_m2sigma_exceed20.png')

        # exp  > 20 ft
        fig, ax = plt.subplots(figsize=fs, dpi=300)
        vals = (ds.expected_cumulative_erosion__depth.values.flatten()*ft2m)<-6.0
        imshow_grid(mg, vals, vmin=0, vmax=1, cmap='Wistia', plot_name='Expected Erosion exceeds 6 m at +' + time + ' yr')
        plt.axis('off')
        line_segments = LineCollection(segments, colors='k', linewidth=0.1)
        ax.add_collection(line_segments)
        plt.text(text_x, text_y, plot_text[set_key], color='w')
        plt.savefig(basic_fig_name +'_expected_erosion_exceed20.png')


        plt.close('all')
        ds.close()

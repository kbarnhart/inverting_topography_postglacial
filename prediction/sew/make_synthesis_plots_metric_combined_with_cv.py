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
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
 
import shapefile as ps

plt.switch_backend("agg")

import xarray as xr

from landlab import RasterModelGrid
from landlab.plot import imshow_grid

SMALL_SIZE = 7
MEDIUM_SIZE = 9
BIGGER_SIZE = 11

plt.rc("text", usetex=True)


plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


in_folder = "synthesis_netcdfs_with_param"

out_folder = "synthesis_plots_meters_combined_with_param"
if os.path.exists(out_folder) is False:
    os.mkdir(out_folder)


site_fp = "site_shapefile/WVSITEFIG_SP_ft.shp"

x_offs = 1121548.2005005
y_offs = 886503.65051269

sf = ps.Reader(site_fp)
shapes = sf.shapes()
segments = []
for shape in shapes:
    points = shape.points
    x, y = zip(*points)
    x = np.array(x) - x_offs
    y = np.array(y) - y_offs
    segment = np.array((x, y)).T
    segments.append(segment)

#%%
outline_fp = "/Users/barnhark/Google Drive (barnhark@colorado.edu)/MANUSCRIPTS/WestValleyPubs/python_and_GIS/GIS_figures/calib_extent.shp"
sf = ps.Reader(outline_fp)
shapes = sf.shapes()
outline_segments = []
for shape in shapes:
    points = shape.points
    x, y = zip(*points)
    x = np.array(x) - x_offs
    y = np.array(y) - y_offs
    segment = np.array((x, y)).T
    outline_segments.append(segment)

mg = RasterModelGrid((415, 438), xy_spacing=24)
ft2m = 0.3048
set_keys = ["only842", "all800s"]
fs = (3, 2.4)

std_lim = 33 * ft2m
ero_lim = 100
ero_col = "RdBu"
std_colorbar = "viridis_r"

set_keys = ["all800s", "only842"]
plot_text = {"only842": "Only BasicChRtTh", "all800s": "Eight Rt variants"}

text_x = 0.03 * (mg.x_of_node.max() - mg.x_of_node.min())
text_y = 1 * (mg.y_of_node.max() - mg.y_of_node.min())



xtext = 300
ytext = 0

w = 1000
h=30 
dx = 24
times = np.arange(0, 10100, 1000)


#full size = 190 mm x 230
for col, set_key in enumerate(set_keys):

    letters = list("abcdefghijklmnopqrstuvwxyz")
    
    fig = plt.figure(figsize=(3, 7), dpi=300)  # convert mm to inches

    spec = gridspec.GridSpec(
        ncols=2,
        nrows=6,
        left=0, right=1, top=1, bottom=0,
        figure=fig,
        wspace=0.0,
        hspace=0.0,
        width_ratios=[1,1],
        height_ratios=[1, 1, 1, 1, 1, 1],
    )
    
    for i, time in enumerate(times):

        time = str(time)
        
        file = np.sort(
                glob.glob(
                    os.path.abspath(
                        os.path.join(
                            in_folder, set_key + "_synthesis_*{time}.nc".format(time=time)
                        )
                    )
                )
            )[0]
        
        ds = xr.open_dataset(file, engine="netcdf4")
        mg.set_nodata_nodes_to_closed(
            ds.expected_topographic__elevation.values.flatten(), -9999
        )

        cv =  ds.topo_total_star_std.values.flatten() / np.abs(ds.expected_cumulative_erosion__depth.values.flatten())  
        
        # CV 
        
        ax = fig.add_subplot(spec[np.unravel_index(i, (6,2))])
        ax.patch.set_alpha(0. )
        imshow_grid(
            mg,
            np.log10(cv),
            color_for_closed=None,
            allow_colorbar=False,
            vmin=0,
            vmax=1,
            cmap="cividis_r",
        )
        
        plt.axis("off")
        plt.annotate(r"\textbf{"+letters.pop(0)+"}. +"+time+" ka", xy=(0.05, 0.75), xycoords='axes fraction')
        
            
        if i == 9:
            rect = patches.Rectangle(
                (xtext*dx, ytext*dx-h/2),
                w / ft2m ,
                h*dx,
                linewidth=0.5,
                edgecolor="k",
                facecolor="w",
            )
            ax.add_patch(rect)
            ax.text(
                xtext*dx + (w / ft2m) / 2,
                ytext*dx + 1.5 * h*dx,
                str(w) + " m",
                ha="center",
                va="center",
            )

      
        # place colorbars:
        inner_grid = spec[-1, -1].subgridspec(
            ncols=3, nrows=3, wspace=0.1, hspace=1, height_ratios = [1, 2, 1], width_ratios=[1, 4, 1]
        )
        ax = fig.add_subplot(inner_grid[1,1])
        ax.patch.set_alpha(0. )

        sm = plt.cm.ScalarMappable(
            cmap="cividis_r", norm=plt.Normalize(vmin=0, vmax=1)
        )
        cbar = fig.colorbar(sm, cax=ax, orientation="horizontal",extend="max")
        cbar.ax.set_title("Coefficient of Variation", size=5, pad=-0.75)
        cbar.ax.tick_params(labelsize=5)

      
    plt.savefig(set_key + "_param_combined_cv.png")
    plt.savefig(set_key + "_param_combined_cv.pdf")

    plt.close("all")

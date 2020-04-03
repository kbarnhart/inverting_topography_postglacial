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

set_keys = ["only842", "all800s"]
fs = (3, 2.4)

std_lim = 33
ero_lim = 100
ero_col = "RdBu"
std_colorbar = "viridis_r"

set_keys = ["all800s", "only842"]
plot_text = {"only842": "Only BasicChRtTh", "all800s": "Eight Rt variants"}

text_x = 0.03 * (mg.x_of_node.max() - mg.x_of_node.min())
text_y = 1 * (mg.y_of_node.max() - mg.y_of_node.min())

ft2m = 0.3048

xtext = 300
ytext = 0

w = 1000
h=30 
dx = 24
times = [time for time in np.arange(0, 10100, 1000) if time!=4000]


#full size = 190 mm x 230

for time in times:
    letters = list("abcdefghijklmnopqrstuvwxyz")

    fig = plt.figure(figsize=(3, 7), dpi=300)  # convert mm to inches

    spec = gridspec.GridSpec(
        ncols=4,
        nrows=9,
        left=0, right=1, top=1, bottom=0,
        figure=fig,
        wspace=0.0,
        hspace=0.0,
        width_ratios=[0.15, 0.15, 1, 1],
        height_ratios=[0.15, 1, 1, 1, 1, 1, 1, 1, 1],
    )

    ax = fig.add_subplot(spec[2:, 0])
    ax.patch.set_alpha(0. )
    plt.text(
               0.5,
               0.5,
               "Uncertainty derived from:",
               transform=ax.transAxes,
               color="k",
               rotation="vertical",
               ha="center",
               va="center",
           )
    ax.set_xticks([])
    ax.set_yticks([])
    for col, set_key in enumerate(set_keys):

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
        time = str(time)
        basic_fig_name = os.path.join(out_folder, time)

        ax = fig.add_subplot(spec[0, col + 2])
        ax.patch.set_alpha(0. )
        plt.text(
            0.5,
            0.5,
            plot_text[set_key],
            transform=ax.transAxes,
            color="k",
            ha="center",
            va="center",
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # Expected Erosion
        ax = fig.add_subplot(spec[1, col + 2])
        ax.patch.set_alpha(0. )
        imshow_grid(
            mg,
            ds.expected_cumulative_erosion__depth.values.flatten() * ft2m,
            color_for_closed=None,
            allow_colorbar=False,
            vmin=-ero_lim * ft2m,
            vmax=ero_lim * ft2m,
            cmap=ero_col,
        )
        plt.axis("off")
        plt.annotate(r"\textbf{"+letters.pop(0)+"}.", xy=(0.05, 0.55), xycoords='axes fraction')
        plot_name = "Expected Erosion"
        line_segments = LineCollection(segments, colors="k", linewidth=0.1)
        # ax.add_collection(line_segments)
        outline_line_segments = LineCollection(
            outline_segments, colors="k", linewidth=0.2
        )
        ax.add_collection(outline_line_segments)

        if col == 0:
            plt.text(
               0.05,
               0.85 ,
               "Model time:\n+{time} yr".format(time=time),
               transform=ax.transAxes,
               color="k",
               ha="left",
               va="center",
               fontsize=SMALL_SIZE-2
           )
                
            ax = fig.add_subplot(spec[1, :2])
            ax.patch.set_alpha(0. )
            plt.text(
                0.5,
                0.5,
                plot_name,
                transform=ax.transAxes,
                color="k",
                rotation="vertical",
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
        
            
        if col == 1:
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

        # Model Uncertainty and Key
        if set_key == "all800s":
            ax = fig.add_subplot(spec[2, col + 2])
            ax.patch.set_alpha(0. )

            imshow_grid(
                mg,
                ds.std_topo_model.values.flatten() * ft2m,
                color_for_closed=None,
                allow_colorbar=False,
                vmin=0,
                vmax=std_lim * ft2m,
                cmap=std_colorbar,
            )
            plt.axis("off")
            plt.annotate(r"\textbf{"+letters.pop(0)+"}.", xy=(0.05, 0.55), xycoords='axes fraction')
            plot_name = "Model Choice"
            line_segments = LineCollection(segments, colors="k", linewidth=0.1)
            # ax.add_collection(line_segments)
            outline_line_segments = LineCollection(
                outline_segments, colors="k", linewidth=0.2
            )
            ax.add_collection(outline_line_segments)
            
            ax = fig.add_subplot(spec[2, col+1])
            ax.patch.set_alpha(0. )
            plt.text(
                0.5,
                0.5,
                plot_name,
                color="k",
                transform=ax.transAxes,
                rotation="vertical",
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # place colorbars:
            inner_grid = spec[2, col + 2].subgridspec(
                ncols=3, nrows=5, wspace=0.1, hspace=1, height_ratios = [1, 2, 2, 2, 1], width_ratios=[1, 4, 1]
            )
            ax = fig.add_subplot(inner_grid[1,1])
            ax.patch.set_alpha(0. )

            sm = plt.cm.ScalarMappable(
                cmap=ero_col, norm=plt.Normalize(vmin=-ero_lim * ft2m, vmax=ero_lim * ft2m)
            )
            cbar = fig.colorbar(sm, cax=ax, orientation="horizontal",extend="both")
            cbar.ax.set_title("Change in Elevation [m]", size=5, pad=-0.75)
            cbar.ax.tick_params(labelsize=5)

            ax = fig.add_subplot(inner_grid[3,1])
            sm = plt.cm.ScalarMappable(
                cmap=std_colorbar, norm=plt.Normalize(vmin=0, vmax=std_lim * ft2m)
            )
            cbar = fig.colorbar(sm, cax=ax, orientation="horizontal",extend="max")
            cbar.ax.set_title("Standard Deviation [m]", size=5, pad=-0.5)
            cbar.ax.tick_params(labelsize=5)

        # parameter calibration
        ax = fig.add_subplot(spec[3, col + 2])
        ax.patch.set_alpha(0. )
        
        imshow_grid(
            mg,
            ds.std_topo_param.values.flatten() * ft2m,
            color_for_closed=None,
            allow_colorbar=False,
            vmin=0,
            vmax=std_lim * ft2m,
            cmap=std_colorbar,
        )
        plt.axis("off")
        plt.annotate(r"\textbf{"+letters.pop(0)+"}.", xy=(0.05, 0.55), xycoords='axes fraction')
        plot_name = "Calibration"
        line_segments = LineCollection(segments, colors="k", linewidth=0.1)
        # ax.add_collection(line_segments)
        outline_line_segments = LineCollection(
            outline_segments, colors="k", linewidth=0.2
        )
        ax.add_collection(outline_line_segments)
        if col == 0:
            ax = fig.add_subplot(spec[3, col+1])
            ax.patch.set_alpha(0. )
            plt.text(
                0.5,
                0.5,
                plot_name,
                color="k",
                transform=ax.transAxes,
                rotation="vertical",
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
        
        # lowering
        ax = fig.add_subplot(spec[4, col + 2])
        ax.patch.set_alpha(0. )
        
        imshow_grid(
            mg,
            ds.std_topo_lower.values.flatten() * ft2m,
            color_for_closed=None,
            allow_colorbar=False,
            vmin=0,
            vmax=std_lim * ft2m,
            cmap=std_colorbar,
        )
        plt.axis("off")
        plt.annotate(r"\textbf{"+letters.pop(0)+"}.", xy=(0.05, 0.55), xycoords='axes fraction')
        plot_name = "Downcutting"
        line_segments = LineCollection(segments, colors="k", linewidth=0.1)
        # ax.add_collection(line_segments)
        outline_line_segments = LineCollection(
            outline_segments, colors="k", linewidth=0.2
        )
        ax.add_collection(outline_line_segments)
        if col == 0:
            ax = fig.add_subplot(spec[4, col+1])
            ax.patch.set_alpha(0. )
            plt.text(
                0.5,
                0.5,
                plot_name,
                color="k",
                transform=ax.transAxes,
                rotation="vertical",
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])

        # climate
        ax = fig.add_subplot(spec[5, col + 2])
        ax.patch.set_alpha(0. )
        imshow_grid(
            mg,
            ds.std_topo_clim.values.flatten() * ft2m,
            color_for_closed=None,
            allow_colorbar=False,
            vmin=0,
            vmax=std_lim * ft2m,
            cmap=std_colorbar,
        )
        plt.axis("off")
        plt.annotate(r"\textbf{"+letters.pop(0)+"}.", xy=(0.05, 0.55), xycoords='axes fraction')
        plot_name = "Climate"
        line_segments = LineCollection(segments, colors="k", linewidth=0.1)
        # ax.add_collection(line_segments)
        outline_line_segments = LineCollection(
            outline_segments, colors="k", linewidth=0.2
        )
        ax.add_collection(outline_line_segments)
        if col == 0:
            ax = fig.add_subplot(spec[5, col+1])
            ax.patch.set_alpha(0. )
            plt.text(
                0.5,
                0.5,
                plot_name,
                color="k",
                transform=ax.transAxes,
                rotation="vertical",
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])

        # ic
        ax = fig.add_subplot(spec[6, col + 2])
        ax.patch.set_alpha(0. )
        imshow_grid(
            mg,
            ds.std_topo_ic.values.flatten() * ft2m,
            color_for_closed=None,
            allow_colorbar=False,
            vmin=0,
            vmax=std_lim * ft2m,
            cmap=std_colorbar,
        )
        plt.axis("off")
        plt.annotate(r"\textbf{"+letters.pop(0)+"}.", xy=(0.05, 0.55), xycoords='axes fraction')
        plot_name = "Initial Condition"
        line_segments = LineCollection(segments, colors="k", linewidth=0.1)
        # ax.add_collection(line_segments)
        outline_line_segments = LineCollection(
            outline_segments, colors="k", linewidth=0.2
        )
        ax.add_collection(outline_line_segments)
        if col == 0:
            ax = fig.add_subplot(spec[6, col+1])
            plt.text(
                0.5,
                0.5,
                plot_name,
                transform=ax.transAxes,
                color="k",
                rotation="vertical",
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])

        # interactions
        ax = fig.add_subplot(spec[7, col + 2])    
        ax.patch.set_alpha(0. )
        imshow_grid(
            mg,
            ds.std_interactions_topo.values.flatten() * ft2m,
            color_for_closed=None,
            allow_colorbar=False,
            vmin=0,
            vmax=std_lim,
            cmap=std_colorbar,
        )
        plt.axis("off")
        plt.annotate(r"\textbf{"+letters.pop(0)+"}.", xy=(0.05, 0.55), xycoords='axes fraction')
        line_segments = LineCollection(segments, colors="k", linewidth=0.1)
        # ax.add_collection(line_segments)
        outline_line_segments = LineCollection(
            outline_segments, colors="k", linewidth=0.2
        )
        ax.add_collection(outline_line_segments)

        plot_name = "Interactions"
        if col == 0:
            ax = fig.add_subplot(spec[7, col+1])
            ax.patch.set_alpha(0. )
            plt.text(
                0.5,
                0.5,
                plot_name,
                transform=ax.transAxes,
                color="k",
                rotation="vertical",
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            
        # total std
        ax = fig.add_subplot(spec[8, col + 2])    
        ax.patch.set_alpha(0. )
        imshow_grid(
            mg,
            ds.std_total_star_topo.values.flatten() * ft2m,
            color_for_closed=None,
            allow_colorbar=False,
            vmin=0,
            vmax=std_lim,
            cmap=std_colorbar,
        )
        plt.axis("off")
        plt.annotate(r"\textbf{"+letters.pop(0)+"}.", xy=(0.05, 0.55), xycoords='axes fraction')
        line_segments = LineCollection(segments, colors="k", linewidth=0.1)
        # ax.add_collection(line_segments)
        outline_line_segments = LineCollection(
            outline_segments, colors="k", linewidth=0.2
        )
        ax.add_collection(outline_line_segments)

        plot_name = "Total$^{*}$"
        if col == 0:
            ax = fig.add_subplot(spec[8, col+1])
            ax.patch.set_alpha(0. )
            plt.text(
                0.5,
                0.5,
                plot_name,
                transform=ax.transAxes,
                color="k",
                rotation="vertical",
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
    ds.close()

    #ax = plt.axes([0, 0, 1, 1], frame_on=False) 
    ax = fig.add_subplot(spec[1:, 2:])
    ax.plot([0, 0.5, 0.5, 1], [7, 7, 6.45, 6.45], linewidth=1, linestyle="dashed", color="k")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 8)
    ax.patch.set_alpha(0. )
    plt.axis("off")
    
    plt.savefig(basic_fig_name + "_param_combined.png")
    plt.savefig(basic_fig_name + "_param_combined.pdf")

    plt.close("all")

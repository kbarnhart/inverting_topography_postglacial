#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 10:11:23 2017

@author: barnhark
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:58:15 2017

# do CHI elev analysis across Gully/SEW/VALID consistently
# ALSO consider weighting.


@author: barnhark
"""

import os
import numpy as np
from landlab import RasterModelGrid
from landlab.components import FlowRouter, ChiFinder, DepressionFinderAndRouter, FlowAccumulator
from landlab.io.netcdf import read_netcdf
from landlab.io import read_esri_ascii, write_esri_ascii
from landlab.plot import imshow_grid
import matplotlib.pylab as plt
import glob


from landlab import FIXED_GRADIENT_BOUNDARY

df = {'sew': {'outlet_id': 178576,
              'chi':True,
              'roads': True},
      'gully': {'outlet_id': 397031,
                'chi':False,
                'roads': False},
      'validation': {'outlet_id': 60722,
                     'chi':True,
                     'roads': False}}

for site in df.keys():
    plt.close('all')
    plt.cla()

    print(site)
    path =  '/work/WVDP_EWG_STUDY3/study3py/auxillary_inputs/' + 'dems/' + site

    observed_topo_file_name = glob.glob(path+os.sep+'modern'+os.sep+'*.txt')[0]
    outlet_id = df[site]['outlet_id']
    (grid, z) = read_esri_ascii(observed_topo_file_name, name='topographic__elevation', halo=1)
    grid.set_watershed_boundary_condition_outlet_id(outlet_id, z, nodata_value=-9999)
    fa = FlowAccumulator(grid,
                         flow_director='D8',
                         depression_finder = 'DepressionFinderAndRouter')
    fa.run_one_step()
    area = grid.dx**2

    ch = ChiFinder(grid, min_drainage_area=0.0)
    ch.calculate_chi()

    # initial condition topography
    initial_topo_file_name = np.sort(glob.glob(path+os.sep+'initial_conditions'+os.sep+'*.txt'))[0]

    (igrid, iz) = read_esri_ascii(initial_topo_file_name, name='topographic__elevation', halo=1)
    igrid.set_watershed_boundary_condition_outlet_id(outlet_id, iz, nodata_value=-9999)
    ifa = FlowAccumulator(igrid,
                          flow_director='D8',
                          depression_finder = 'DepressionFinderAndRouter')
    ifa.run_one_step()
    area = grid.dx**2
    plt.figure(dpi=300)
    imshow_grid(grid, z, cmap='viridis', plot_name='Modern Topography')
    plt.axis('off')
    plt.savefig(site+'.topo.png')

    ich = ChiFinder(igrid, min_drainage_area=0.0)
    ich.calculate_chi()

    # get chi if it exists
    if df[site]['chi']:
        chi_path =  '/work/WVDP_EWG_STUDY3/study3py/auxillary_inputs/' + 'chi_mask/' + site

        # create mask
        mask = np.zeros_like(z)
        mask[grid.core_nodes] = 1

        chi_file = np.sort(glob.glob(chi_path+os.sep+'*.txt'))[0]
        (cgrid, zmask) = read_esri_ascii(chi_file, name='topographic__elevation', halo=1)
        chi_mask = (zmask>0)*1
        mask += chi_mask

        mask[grid.status_at_node > 0] = 0
        #imshow_grid(cgrid, mask)
        #plt.show()

    else:
        mask = np.zeros_like(z)
        mask[grid.core_nodes] = 2

    # get roads if it exists
    if df[site]['roads']:
        road_path =  '/work/WVDP_EWG_STUDY3/study3py/auxillary_inputs/' + 'roads/' + site

        # create mask
        road_mask = np.zeros_like(z)
        road_mask[grid.core_nodes] = 1

        road_file = np.sort(glob.glob(road_path+os.sep+'*.txt'))[0]
        (rgrid, rmask) = read_esri_ascii(road_file, name='topographic__elevation', halo=1)
        r_mask = (rmask>0)*1
        road_mask += r_mask

        road_mask[grid.status_at_node > 0] = 0

        plt.figure(dpi=300)
        imshow_grid(rgrid, road_mask, plot_name='Road Mask')
        plt.axis('off')
        plt.savefig(site+'.roads.png')

    else:
        road_mask = np.zeros_like(z)
        road_mask[grid.core_nodes] = 1

    z_change = iz-z

    plt.figure(dpi=300)
    imshow_grid(grid, z_change, cmap='RdBu', limits=(-2, 2), plot_name='Topographic Change')
    plt.axis('off')
    plt.savefig(site+'.change_thresh.png')

    plt.figure(dpi=300)
    imshow_grid(grid, grid['node']['topographic__steepest_slope'], 
                cmap='inferno', 
                limits=(0, np.nanmax(grid['node']['topographic__steepest_slope'])), 
                color_for_closed='white',
                plot_name='Slope')
        
    plt.axis('off')
    plt.savefig(site+'.slope.png')

    plt.figure(dpi=300)
    imshow_grid(grid, z_change, cmap='RdBu', limits=(-130, 130), plot_name='Topographic Change')
    plt.axis('off')
    plt.savefig(site+'.change_all.png')

    # plot distribtuion of elevation
    plt.figure(dpi=300)
    plt.hist(z[grid.core_nodes], bins=100, histtype='step', normed=True)
    plt.hist(iz[grid.core_nodes], bins=100, histtype='step', normed=True)
    plt.title('Distribution of Elevation ' + site)
    plt.legend(['Observations (var = '+str(round(np.var(z[grid.core_nodes]), 0)) + ')',
                'Initial (var = '+str(round(np.var(iz[grid.core_nodes]), 0))+ ')'])
    plt.savefig(site+'.hist_elev.png')
    #plt.savefigfig('Distibution of Elevation.pdf')

#    # plot distribtuion of elevation change
#    plt.figure(dpi=300)
#    plt.hist(z_change[grid.core_nodes], bins=100, histtype='step', normed=True)
#    plt.title('Distribution of Elevation Change ' + site)
#    plt.show()

    # show distribution of elevation in and out of the chi area, which is functionally
    # the area where we have "good confidence" in changes
    plt.figure(dpi=300)
    plt.hist(z_change[mask==1], bins=100, histtype='step', normed=True)
    plt.hist(z_change[mask==2], bins=100, histtype='step', normed=True)
    plt.title('Distribution of Elevation Change ' + site)
    plt.legend(['Confident IC: mean = ' + str(round(np.mean(z_change[mask==2]), 0)) + ', std = ' + str(round(np.std(z_change[mask==2]), 0)),
                'Unconfident IC: mean = ' + str(round(np.mean(z_change[mask==1]), 0)) + ', std = ' + str(round(np.std(z_change[mask==1]), 0))])
    plt.savefig(site+'.hist_change.png')
    # make chi for validation.

    # we want to weight by something like (the difference that our IC implies)/(the difference that could have occured)
    # this means we need some constraint on our uncertainty in the IC.

    # in the lower areas of sew/validation (mask = 2) and in the gully, confidence
    # in IC surface is high. The 95% CI on the IC surface is maybe +/- 10 feet.
    # in the interfluves of the upper part of the watershed we are probably similarly
    # confident in the IC surface.
    # in the drainages of the upper watershed, the confidence is low. The valleys may have
    # been much more filled in (e.g 80 feet), or they may have eroded very little in the last 13 k years.
    # We'll call this an uncertainty of plus/minus 40.
    #
    # Thus, we'll assign rough weights of 1/w, or 1/s**2, where s is a standard deviation and s = 5  in the channelized parts of the upper
    # watershed and w=20 in the remainder of the watershed.
    #
    # (10/(1.96)) ~ 5
    #
    # (40/(1.96)) ~ 20

    low_std = 5
    high_std = 20
    st_dev = np.zeros_like(z)
    # set all zone two to 5
    st_dev[mask==2] = low_std

    # smoothly change zone 1 from 5 to 20 based on the amount of elevation change
    # if there was less than 2 meters of change have it asymtote to 5 and if there
    # is greater than 6 feet of change have it asymtote to 20

    st_dev[mask==1] = low_std + (high_std - low_std) /(1 + np.exp(-(np.abs(z_change[mask==1])-low_std)))

    # finally, down-weight the road locations
    st_dev[road_mask == 2] = high_std

    plt.figure(dpi=300)
    imshow_grid(grid, st_dev, cmap='magma_r', limits=(0, 20), color_for_closed='white', plot_name='Uncertainty in Initial Topography')
    plt.axis('off')
    plt.savefig(site+'.st_dev.png')
    #%%
    plt.figure(dpi=300)
    plt.hist(ch.chi[grid.core_nodes], bins=100, normed=True)
    plt.savefig(site+'.chi_dist.png')

    #%%

    # first bin by chi, then within chi, bin by elevation
    is_core = np.zeros_like(z)
    is_core[grid.core_nodes] = True
    chi_percentiles = [0, 5, 20, 50, 100]
    elev_percentiles = [0, 20, 40, 60, 80, 100]

    # calculate the percentiles of the chi distribution 
    chi_edges = np.percentile(ch.chi[is_core==True], chi_percentiles)

    # work through each bin and label them.
    cat = np.zeros_like(z)
    H = np.zeros((len(chi_percentiles)-1, len(elev_percentiles)-1))
    val = 1

    cat_bin = np.zeros_like(z)

    f, axarr = plt.subplots(len(chi_edges)-1, figsize = (4, 16), dpi=300)
    for i in range(len(chi_edges) - 1):

        cat_plot = np.zeros_like(z)

        # selected nodes
        x_min = chi_edges[i]
        x_max = chi_edges[i+1]

        if i != len(chi_edges) - 2:
            x_sel = (ch.chi >= x_min) & (ch.chi < x_max) & (is_core==True)
        else:
            x_sel = (ch.chi >= x_min) & (is_core==True)

        # get the edges for this particular part of chi-space
        z_sel = z[x_sel]
        elev_edges = np.percentile(z_sel, elev_percentiles)
        for j in range(len(elev_edges) - 1):

            elev_min = elev_edges[j]
            elev_max = elev_edges[j+1]

            if j != len(elev_edges) - 2:
                y_sel = (z >= elev_min) & (z < elev_max) & (x_sel)
            else:
                y_sel = (z >= elev_min) & (x_sel)

            sel_nodes = np.where(y_sel)[0]

            H[i,j] = int(len(sel_nodes))
            #sel_nodes = grid.core_nodes
            if len(sel_nodes) > 0:
                cat[sel_nodes] = val
                cat_bin[sel_nodes] = len(sel_nodes)
                cat_plot[sel_nodes] = cat[sel_nodes]
                val +=1

        cat_plot[cat_plot == 0 ] = cat_plot[cat_plot>0].min() - 1
        plt.sca(axarr[i])
        imshow_grid(grid, cat_plot, cmap='viridis')
        plt.axis('off')


    plt.savefig(site+'.cats.png')

    plt.figure(figsize=(4,3), dpi=300)
    imshow_grid(grid, cat, cmap='tab20', limits=(0.5, 20.5))#, plot_name='Chi-Elevation Category')
    plt.axis('off')
    plt.savefig(site+'.all_cats.png')

    plt.figure(dpi=300)
    imshow_grid(grid, cat_bin, cmap='inferno', plot_name='Number of Nodes in Each Category Bin')
    plt.axis('off')
    plt.savefig(site+'.catbin.png')

    plt.figure(dpi=300)
    effective_weight = 1./(cat_bin * (st_dev**2))
    effective_weight[np.isinf(effective_weight)] = 0
    imshow_grid(grid, effective_weight, cmap='inferno', color_for_closed='white', plot_name='Effective Weight')
    plt.axis('off')
    plt.savefig(site+'.effective_weight.png')
    
    
    #%%
    if site=='sew':
        simple_cat_802 = cat
        
        simple_cat_802[simple_cat_802==1] = -1
        simple_cat_802[simple_cat_802==2] = -1
        simple_cat_802[simple_cat_802==3] = -1
        simple_cat_802[simple_cat_802==4] = -1
        simple_cat_802[simple_cat_802==5] = -2
        simple_cat_802[simple_cat_802==6] = -1
        simple_cat_802[simple_cat_802==7] = -1
        simple_cat_802[simple_cat_802==8] = -1
        simple_cat_802[simple_cat_802==9] = -2
        simple_cat_802[simple_cat_802==10] = -2
        simple_cat_802[simple_cat_802==11] = -1
        simple_cat_802[simple_cat_802==12] = -1
        simple_cat_802[simple_cat_802==13] = -2
        simple_cat_802[simple_cat_802==14] = -2
        simple_cat_802[simple_cat_802==15] = -2
        simple_cat_802[simple_cat_802==16] = -1
        simple_cat_802[simple_cat_802==17] = -2
        simple_cat_802[simple_cat_802==18] = -2
        simple_cat_802[simple_cat_802==19] = -2
        simple_cat_802[simple_cat_802==20] = -2
        
        simple_cat_802 = np.abs(simple_cat_802)
        
        plt.figure(dpi=300)
        imshow_grid(grid, simple_cat_802, vmin = 0.7, vmax=2.2, cmap='PuOr', color_for_closed='white', plot_name='802 Sensitivity Categories', allow_colorbar=False)
        plt.axis('off')
        plt.savefig(site+'.simple_cat_802.png')
        
    #%%

    # verify that there are no zeros in the core nodes

    # calculate the number of catagories and include in filename
    num_cat = len(np.unique(cat[cat>0]))

    out_file = os.path.join(os.path.abspath(os.sep), *['work', 'WVDP_EWG_STUDY3', 'study3py','auxillary_inputs', 'chi_elev_categories', site+'.chi_elev_cat.'+str(num_cat)+'.txt'])
    np.savetxt(out_file, cat)

    # save out the catagory-weight file
    weight_out_file = os.path.join(os.path.abspath(os.sep), *['work', 'WVDP_EWG_STUDY3', 'study3py','auxillary_inputs', 'weights', site+'.chi_elev_weight.'+str(num_cat)+'.txt'])
    np.savetxt(weight_out_file, st_dev)
    
    # save the effective weight file
    eff_weight_out_file = os.path.join(os.path.abspath(os.sep), *['work', 'WVDP_EWG_STUDY3', 'study3py','auxillary_inputs', 'weights', site+'.chi_elev_effective_weight.'+str(num_cat)+'.txt'])
    np.savetxt(eff_weight_out_file, effective_weight)
    #%%

    plt.hist(cat)

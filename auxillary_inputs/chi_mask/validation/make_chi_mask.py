#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:47:14 2017

@author: barnhark
"""
# make chi mask for validation site

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

df = {'validation': {'outlet_id': 60722, 'chi':False}}

for site in df.keys():
    print(site)
    path =  '/work/WVDP_EWG_STUDY3/study3py/auxillary_inputs/' + 'dems/' + site
        
    observed_topo_file_name = glob.glob(path+os.sep+'modern'+os.sep+'*.txt')[0]
    outlet_id = df[site]['outlet_id']
    (grid, z) = read_esri_ascii(observed_topo_file_name, name='topographic__elevation')
    grid.set_watershed_boundary_condition_outlet_id(outlet_id, z)
    # initial condition topography
    initial_topo_file_name = np.sort(glob.glob(path+os.sep+'initial_conditions'+os.sep+'*.txt'))[0]
    (igrid, iz) = read_esri_ascii(initial_topo_file_name, name='topographic__elevation')

    imshow_grid(grid, z, cmap='viridis')
    plt.show()
   
    imshow_grid(grid, iz-z, cmap='RdBu', limits=(-40, 40))
    plt.show()
    
    mask = np.zeros_like(z)
    core = z>0
    
    mask[core] = 1
    mask[grid.node_x > 5000] = 0
    
    imshow_grid(grid, mask)
    plt.show()
    
    grid.add_field('node', 'chi_mask', mask)
    
    chi_path =  '/work/WVDP_EWG_STUDY3/study3py/auxillary_inputs/' + 'chi_mask/' + site+ '/chi_mask.txt'
    write_esri_ascii(chi_path, grid, ['chi_mask'], clobber=True)

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

from landlab.io.netcdf import read_netcdf

from landlab.plot import imshow_grid

ncs = glob.glob(os.path.join(*['*','*', 'model_**_0130.nc']))

def replot(nc):    
    print(nc)
    grid = read_netcdf(nc)
    
    z = grid.at_node['topographic__elevation']
      
    fig_name = glob.glob(os.path.split(nc)[0] + os.sep + '*png')[0]
    inputs = glob.glob(os.path.split(nc)[0] + os.sep + 'inputs.txt')[0]

    with open(inputs, 'r') as f:
        params = yaml.load(f)
        outlet_id =params['outlet_id']
        
    grid.set_watershed_boundary_condition_outlet_id(outlet_id, z, nodata_value=-9999)
    
    #fig_name
    imshow_grid(grid, 'topographic__elevation', vmin=1161, vmax=1802, cmap='viridis', output=fig_name)

    return True

ncores = 23
output = Parallel(n_jobs=ncores)(delayed(replot)(nc) for nc in ncs)

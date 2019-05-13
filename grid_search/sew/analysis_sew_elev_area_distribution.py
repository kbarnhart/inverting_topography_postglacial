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

@author: barnhark
"""
import os
import time 
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from yaml import load
#from matplotlib.backends.backend_pdf import PdfPages
#from mpl_toolkits.mplot3d import Axes3D

from joblib import Parallel, delayed

from landlab.io import read_esri_ascii, write_esri_ascii
from landlab.io.netcdf import read_netcdf
from landlab.components import FlowAccumulator, ChiFinder
from glob import glob
from landlab.plot import imshow_grid


##############################################################################
#                                                                            #
#        Part 0: Name of compiled outputs                                    #
#                                                                            #
##############################################################################
run_dir_list = ['work', 'WVDP_EWG_STUDY3', 'study3py','grid_search', 'sew', 'model_800', 'lowering_history_1.pg24f_0etch', 'random']
run_dir = os.path.join(os.path.abspath(os.sep), *run_dir_list)

##### MODEL 800
 
# grid
model_grid_800_file = run_dir+os.path.sep+os.path.join(*['sew_model_800_random.dat'])
df_grid = pd.read_csv(model_grid_800_file, sep='\s+')
df_grid.drop_duplicates(inplace=True)
size = int(np.cbrt(df_grid.shape[0]))

# inputs
model_grid_800_inputs = run_dir+os.path.sep+'inputs_template.txt'
with open(model_grid_800_inputs, 'r') as f:
    inputs = load(f)


#%%
# observed_topography
observed_topo_file_name = inputs['modern_dem_name']
(grid, z) = read_esri_ascii(observed_topo_file_name, name='topographic__elevation', halo=1)
grid.set_watershed_boundary_condition_outlet_id(inputs['outlet_id'], z, nodata_value=-9999)
fa = FlowAccumulator(grid,
                     flow_director='D8',
                     depression_finder = 'DepressionFinderAndRouter')
fa.run_one_step()
area = grid.dx**2
imshow_grid(grid, np.log10(fa.drainage_area/area+1))
plt.show()

ch = ChiFinder(grid, min_drainage_area=0.0)
ch.calculate_chi()
#%%
# initial condition topography
initial_topo_file_name = inputs['DEM_filename']
(igrid, iz) = read_esri_ascii(initial_topo_file_name, name='topographic__elevation', halo=1)
igrid.set_watershed_boundary_condition_outlet_id(inputs['outlet_id'], iz, nodata_value=-9999)
ifa = FlowAccumulator(igrid,
                      flow_director='D8',
                      depression_finder = 'DepressionFinderAndRouter')
ifa.run_one_step()
area = grid.dx**2
imshow_grid(grid, np.log10(ifa.drainage_area/area+1))
plt.show()

ich = ChiFinder(igrid, min_drainage_area=0.0)
ich.calculate_chi()
#%% modeled topo_files
modeled_topo_files = np.sort(glob(run_dir+os.path.sep+os.path.join(*['GRID', 'run*', 'model_*.nc'])))
topo_file = modeled_topo_files[0]
mgrid = read_netcdf(topo_file)
mz = mgrid.at_node['topographic__elevation']
mgrid.set_watershed_boundary_condition_outlet_id(inputs['outlet_id'], mz, nodata_value=-9999)
mfa = FlowAccumulator(mgrid,
             flow_director='D8',
             depression_finder = 'DepressionFinderAndRouter')

mfa.run_one_step()

imshow_grid(grid, np.log10(mfa.drainage_area/area+1))
plt.show()

mch = ChiFinder(mgrid, min_drainage_area=0.0)
mch.calculate_chi()
#%%

# plot distribtuions
plt.figure()
plt.hist(z[grid.core_nodes], bins=100, histtype='step')
plt.hist(iz[grid.core_nodes], bins=100, histtype='step')
plt.hist(mz[grid.core_nodes], bins=100, histtype='step')
plt.title('Distribution of Elevation')
plt.legend(['Observations (var = '+str(round(np.var(z[grid.core_nodes]), 0)) + ')', 
            'Initial (var = '+str(round(np.var(iz[grid.core_nodes]), 0))+ ')', 
            'Modeled (var = '+str(round(np.var(mz[grid.core_nodes]), 0))+ ')']) 
plt.savefig('Distibution of Elevation.pdf')

#%%
imshow_grid(grid, iz-z, cmap='RdBu', limits=(-2, 2))
plt.show()


imshow_grid(grid, iz-z, cmap='RdBu', limits=(-130, 130))
plt.show()

av_diff = np.mean(iz[grid.core_nodes]-z[grid.core_nodes])

#%%
# 1 cell = 0
# 2 cells = 0.3
# 3 cells = 0.47
# 4 cells = 0.60
# 5 cells = 0.69
# 6 cells = 0.77
# 7 cells = 0.84
# 8 cells = 0.90
# 9 cells = 0.95

area_bin_edges = np.log10([0.5, 1.5, 2.5, 3.5, 5.5, 7.5,
                           10.5, 15.5, 20.5, 30.5,
                           50.5, 75.5, 100.5, 
                           200.5, 500.5, 750.5, 
                           1000.0, 10000.0, 100000.0])
plt.figure()
n, bins, patches = plt.hist(np.log10(fa.drainage_area[grid.core_nodes]/grid.dx**2), bins=area_bin_edges, histtype='step')
plt.hist(np.log10(ifa.drainage_area[grid.core_nodes]/grid.dx**2), bins=area_bin_edges, histtype='step')
plt.hist(np.log10(mfa.drainage_area[grid.core_nodes]/grid.dx**2), bins=area_bin_edges, histtype='step')
plt.title('Distribution of Log10 Drainage Area')
plt.legend(['Observations (var = '+str(round(np.var(fa.drainage_area[grid.core_nodes]), 0)) + ')', 
                           'Initial (var = '+str(round(np.var(ifa.drainage_area[grid.core_nodes]), 0))+ ')', 
                           'Modeled (var = '+str(round(np.var(mfa.drainage_area[grid.core_nodes]), 0))+ ')']) 
plt.show()

#%%
plt.figure()
plt.hist(z[grid.core_nodes], bins=100, histtype='step', cumulative=True)
plt.hist(iz[grid.core_nodes], bins=100, histtype='step', cumulative=True)
plt.hist(mz[grid.core_nodes], bins=100, histtype='step', cumulative=True)
plt.title('Distribution of Elevation')
plt.legend(['Observations', 'Initial', 'Modeled'])
plt.savefig('Cumulative Distibution of Elevation.pdf')
#%%
plt.figure()
plt.hist(df_grid.var_elevation+np.var(z[grid.core_nodes]), bins=30, histtype='step')
plt.vlines(np.var(z[grid.core_nodes]), 0, 100)
plt.legend(['Modeled  800 Distribution', 'Observations'])
plt.savefig('Distribution of Var Elevation.pdf')

#%%
plt.figure()
plt.hist(ch.chi[grid.core_nodes], bins=100)
plt.show()

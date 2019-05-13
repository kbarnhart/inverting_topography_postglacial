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

from landlab.io import read_esri_ascii
from landlab.io.netcdf import read_netcdf
from landlab.components import FlowAccumulator
from glob import glob
from landlab.plot import imshow_grid


##############################################################################
#                                                                            #
#        Part 0: Name of compiled outputs                                    #
#                                                                            #
##############################################################################
run_dir_list = ['work', 'WVDP_EWG_STUDY3', 'study3py','grid_search', 'sew', 'model_800', 'lowering_history_1.pg24f_0etch', 'fine']
run_dir = os.path.join(os.path.abspath(os.sep), *run_dir_list)

##### MODEL 800
 
# grid
model_grid_800_file = run_dir+os.path.sep+os.path.join(*['sew_model_800_grid.dat'])
df_grid = pd.read_csv(model_grid_800_file, sep='\s+')
df_grid.drop_duplicates(inplace=True)
size = int(np.cbrt(df_grid.shape[0]))

# weights
model_grid_800_inputs = run_dir+os.path.sep+'inputs_template.txt'
with open(model_grid_800_inputs, 'r') as f:
    inputs = load(f)

X = df_grid.K_rock_sp_exp.values.reshape((size,size, size))
Y = df_grid.linear_diffusivity_exp.values.reshape((size,size, size))
Z = df_grid.K_till_sp_exp.values.reshape((size,size, size))

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
#%%
# using 1/(2xlog10(DA/cell-area)+1) gives a weight of 0.3 to nodes that have no contributing
# area and a weight of 10x to the outlet node. we'll try it. 

w1 = (2.*np.log10((fa.drainage_area/area)+1))
imshow_grid(grid, w1, limits=(0, 10))
plt.show()

#%%
topo_change = iz-z
w2 = (np.abs(topo_change)-np.abs(topo_change)[grid.core_nodes].min())/(4.*(topo_change[grid.core_nodes].max()-topo_change[grid.core_nodes].min()))+ 0.25
imshow_grid(grid, w2, limits=(0, 1))
plt.show()

#%%

def find_misfit(topo_file, z, w1, w2):    
    try:
        mgrid = read_netcdf(topo_file)
        mz = mgrid.at_node['topographic__elevation']
        mgrid.set_watershed_boundary_condition_outlet_id(inputs['outlet_id'], mz, nodata_value=-9999)
        fa = FlowAccumulator(mgrid,
                     flow_director='D8',
                     depression_finder = 'DepressionFinderAndRouter')
        
        fa.run_one_step()
        
        # create three residuals. topography residual
        # 
        misfit0 = np.sum(((z[mgrid.core_nodes]-mz[mgrid.core_nodes])**2))
        misfit1 = np.sum(((z[mgrid.core_nodes]-mz[mgrid.core_nodes])**2)*w1[mgrid.core_nodes])
        misfit2 = np.sum(((z[mgrid.core_nodes]-mz[mgrid.core_nodes])**2)*w2[mgrid.core_nodes])

        ind = int(topo_file.split(os.sep)[-2].split('.')[-1]) - 1
        
        #print(time.time()-start_time)
        return (ind, (misfit0, misfit1, misfit2))

    except:
        return (np.nan, (np.nan, np.nan, np.nan))

#%%
# modeled topography
modeled_topo_files = np.sort(glob(run_dir+os.path.sep+os.path.join(*['GRID', 'run*', 'model_*.nc'])))

#%%

df_grid['topo_residual'] = np.nan
df_grid['da_weighted_topo_residual'] = np.nan
df_grid['erosion_weighted_topo_residual'] = np.nan

if os.path.exists(run_dir+os.sep+'df_grid.csv'):
    df_grid = pd.read_csv(run_dir+os.sep+'df_grid.csv')

while np.any(np.isnan(df_grid['topo_residual'])):
    
    # figure out which files are needed to reprocess. then only do those. 
    to_run = np.where(np.isnan(df_grid['topo_residual']))[0]+1 # dakota uses base 1. 
    files_to_run = [ftr for ftr in modeled_topo_files if int(ftr.split(os.sep)[-2].split('.')[-1]) in to_run]
    
    nc = 6
    n_per_chunk = 10*nc
    chunks = int(np.ceil(float(len(files_to_run))/float(n_per_chunk)))
    for ci in range(chunks):
        sel_topo = []
        for i in range(n_per_chunk):
            try:
                sel_topo.append(files_to_run.pop())
            except IndexError:
                pass
            
        start_time = time.time()
        output = Parallel(n_jobs=nc)(delayed(find_misfit)(topo_file, z, w1, w2) for topo_file in sel_topo)
        # this should take ~16.8 minutes with 6 cores (5 sec per process)
        print(ci, chunks, round((time.time()- start_time)/60, 2), round((time.time()- start_time)/60, 2)*chunks)
        
        for out in output:
            ind = out[0]
            misfits = out[1]
            if np.isnan(ind) == False:
                df_grid.loc[ind, 'topo_residual'] = misfits[0]
                df_grid.loc[ind, 'da_weighted_topo_residual'] = misfits[1]
                df_grid.loc[ind, 'erosion_weighted_topo_residual'] = misfits[2]

        df_grid.to_csv(run_dir+os.sep+'df_grid.csv')

#%%
topo_misfit_min_index = np.where(df_grid['topo_residual'].values==np.nanmin(df_grid['topo_residual']))[0]
topo_misfit_DA_min_index = np.where(df_grid['da_weighted_topo_residual'].values==np.nanmin(df_grid['da_weighted_topo_residual']))[0]
topo_misfit_ER_min_index = np.where(df_grid['erosion_weighted_topo_residual'].values==np.nanmin(df_grid['erosion_weighted_topo_residual']))[0]

df_mins = df_grid.iloc[[topo_misfit_min_index[0], topo_misfit_DA_min_index[0], topo_misfit_ER_min_index[0]],:]

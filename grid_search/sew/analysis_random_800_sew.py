#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:17:39 2017

@author: barnhark
"""

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
run_dir_list = ['work', 'WVDP_EWG_STUDY3', 'study3py','grid_search', 'sew', 'model_800', 'lowering_history_1.pg24f_0etch', 'random']
run_dir = os.path.join(os.path.abspath(os.sep), *run_dir_list)

##### MODEL 800
 
# grid
model_grid_800_file = run_dir+os.path.sep+os.path.join(*['sew_model_800_random.dat'])
df_grid = pd.read_csv(model_grid_800_file, sep='\s+')
df_grid.drop_duplicates(inplace=True)
size = int(np.cbrt(df_grid.shape[0]))

# weights
model_grid_800_inputs = run_dir+os.path.sep+'inputs_template.txt'
with open(model_grid_800_inputs, 'r') as f:
    inputs = load(f)

#%%
# calculate variance, and compare with error based weighting .
output_df = df_grid[df_grid.columns[-32:]]
model_var = np.var(output_df)
model_mean = np.mean(output_df)

# make a histogram
plt.figure()
output_df.hist(figsize=(8.5, 20), layout=(8,4))
plt.savefig('model_800_random_histograms.pdf')
#%% 
weights_filepath_sew = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'auxillary_inputs', 'weights', 'sew_variance.txt']
df_sew = pd.read_csv(os.path.join(os.sep, *weights_filepath_sew), header=None, names=['Error_Based_Variance'])
df_sew['Model_800_Based_Variance'] = model_var

new_weights_filepath_sew = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'auxillary_inputs', 'weights', 'sew_variance_with_model800.txt']
df_sew.to_csv(os.path.join(os.sep, *new_weights_filepath_sew))

df_sew['Factor_Difference'] = df_sew['Model_800_Based_Variance']/df_sew['Error_Based_Variance']

#%%
plt.figure()

plt.loglog([1e-8, 1e10], [1e-8, 1e10], 'k-') 
plt.loglog(df_sew['Model_800_Based_Variance'], df_sew['Error_Based_Variance'], 'b.')
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 21:55:30 2018

@author: barnhark
"""

import os
import glob


search_path = '/work/WVDP_EWG_STUDY3/study3py/prediction/sew/BEST_PARAMETERS/model_*/*.RCP*/*.png'
files = glob.glob(search_path)
for file in files:
    os.remove(file)

search_path = '/work/WVDP_EWG_STUDY3/study3py/prediction/sew/IC_UNCERTAINTY/model_*/*.RCP*/*/*.png'
files = glob.glob(search_path)
for file in files:
    os.remove(file)
    
search_path = '/work/WVDP_EWG_STUDY3/study3py/prediction/sew/BREACHING/model_*/*.RCP*/*/*.png'
files = glob.glob(search_path)
for file in files:
    os.remove(file)
    
search_path = '/work/WVDP_EWG_STUDY3/results/prediction/sew/PARAMETER_UNCERTAINTY/model_*/*.RCP*/run*/elevation_at_points_df.csv'
files = glob.glob(search_path)
for file in files:
    os.remove(file)
  
search_path = '/work/WVDP_EWG_STUDY3/results/prediction/sew/PARAMETER_UNCERTAINTY/model_*/*.RCP*/run*/fail_log.txt'
files = glob.glob(search_path)
for file in files:
    os.remove(file)

search_path = '/work/WVDP_EWG_STUDY3/study3py/prediction/sew/PARAMETER_UNCERTAINTY/model_*/*.RCP*/dakota_pred.rst'
files = glob.glob(search_path)
for file in files:
    os.remove(file)

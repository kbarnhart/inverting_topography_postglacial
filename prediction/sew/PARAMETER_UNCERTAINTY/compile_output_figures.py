#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 07:42:36 2018

@author: barnhark
"""

import os
import glob
import shutil


model_paths = ['work', 'WVDP_EWG_STUDY3', 'results', 'prediction',  'sew' , 'PARAMETER_UNCERTAINTY', 'model_*']
results_paths = glob.glob(os.path.join(os.path.abspath(os.sep), *model_paths))
for result_path in results_paths:
    model_name = os.path.split(result_path)[-1]
    print(model_name)
    # make target folder
    fig_path = os.path.join(os.path.abspath(os.sep), *['work', 'WVDP_EWG_STUDY3', 'study3py', 'prediction',  'sew' , 'PARAMETER_UNCERTAINTY', 'topography_figures', model_name])
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        
    #/work/WVDP_EWG_STUDY3/results/prediction/sew/PARAMETER_UNCERTAINTY/model_800/lowering_future_1.dem24fil_ext.constant_climate/run.1/*png
    search_path = os.path.join(result_path, *['*', 'run.*', '*png'])

    search_path = ['work', 'WVDP_EWG_STUDY3', 'results', 'prediction',  'sew' , 'PARAMETER_UNCERTAINTY', model_name, '*', '*', '*png']
    figure_files = glob.glob(os.path.join(os.path.abspath(os.sep), *search_path))

    # copy all figures into folder. 
    for fig in figure_files:
        shutil.copy(fig, fig_path)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:23:51 2017

@author: barnhark
"""

import os
import glob

domain = ['sew', 'gully']

for d in domain:
    results_dir = ['work', 'WVDP_EWG_STUDY3', 'results','sensitivity_analysis', d, 'MOAT', 'model_*', '*', 'run*', '*.model']

    files_to_remove = glob.glob(os.path.join(os.sep, *results_dir))
    for file in files_to_remove:
        os.remove(file)
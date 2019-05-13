#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:42:34 2017

@author: barnhark
"""

import glob
import os
from subprocess import call

# remove restart files
search_path = ['work', 'WVDP_EWG_STUDY3', 'study3py','prediction', '*', 'PARAMETER_UNCERTAINTY', 'model_*', '*', '*.rst']
restart_files = glob.glob(os.path.join(os.path.abspath(os.sep), *search_path))

for r_f in restart_files:
    call(['rm', r_f])

# remove fail logs
search_path = ['work', 'WVDP_EWG_STUDY3', 'results','prediction', '*', 'PARAMETER_UNCERTAINTY', 'model_*', '*', 'run*', 'fail_log.txt']
fail_files = glob.glob(os.path.join(os.path.abspath(os.sep), *search_path))
for f_f in fail_files:
    call(['rm', f_f])

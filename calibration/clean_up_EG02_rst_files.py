#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:42:34 2017

@author: barnhark
"""

import glob
import shutil
import os
from subprocess import call

search_path = ['work', 'WVDP_EWG_STUDY3', 'study3py','calibration', '*', 'EGO2', 'model_*', '*', 'dakota.rst']
restart_files = glob.glob(os.path.join(os.path.abspath(os.sep), *search_path))
for r_f in restart_files:
    
    cr_f = os.path.join(os.path.split(r_f)[0],'dakota_calib.rst')
    calib_restart_exists = os.path.exists(cr_f)
    
    if calib_restart_exists:
        # combine and rename
        print('combining files ', r_f)

        temp_file = os.path.join(os.path.split(r_f)[0],'temp.rst')

        call(['dakota_restart_util', 'cat', r_f, cr_f, temp_file])
        call(['rm', r_f])
        shutil.move(temp_file, cr_f)
        call(['rm', temp_file])

    else:
        # rename only
        print('only ', r_f, ' exists, renaming')
        shutil.move(r_f, cr_f)
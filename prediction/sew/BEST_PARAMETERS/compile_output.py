#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 07:42:36 2018

@author: barnhark
"""

import os
import shutil
import glob

pngs = glob.glob(os.path.join(*['*','*', '*.png']))

if os.path.exists('topo_figures') is False:
    os.mkdir('topo_figures')


for png in pngs:
    print(png)
    shutil.copy(png, 'topo_figures')

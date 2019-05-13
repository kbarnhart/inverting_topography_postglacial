#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:37:37 2018

@author: barnhark
"""

import pandas as pd
import numpy as np

df = pd.read_csv('/work/WVDP_EWG_STUDY3/study3py/prediction/sew/output_summary_file.csv', index_col=[1,0], usecols=np.arange(1,11))
idx = df.columns.str.split('.', expand=True)
df.columns = idx
df = df.swaplevel(i=-2, j=-1, axis=1)

df.to_latex('summary_table_latex.tex', escape=False, multicolumn=True, multirow=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:30:21 2017

@author: barnhark
"""

# Run one evaluation from each model, using the best parameters

import os
import pandas as pd
import codecs
import math

#from dakotathon.utils import add_dyld_library_path
#add_dyld_library_path()

# get the current files filepath
dir_path = os.path.dirname(os.path.realpath(__file__))
loc = dir_path.split(os.sep)[-2]

# Define filepaths. Here these are given as lists, for cross platform
# compatability


parameter_dict_folderpath = ['..', 'auxillary_inputs']


# loop within models.

# Get model space and parameter ranges (these will be loaded in from a file)

model_parameter_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_parameter_match_calibration_sew.csv'])))

model_param_df = pd.read_csv(model_parameter_input_file)
model_param_df = model_param_df.dropna() # Remove those rows with models that are not yet complete


parameter_dict_folderpath = ['..', 'auxillary_inputs']
parameter_range_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['parameter_ranges.csv'])))
param_range_df = pd.read_csv(parameter_range_input_file)
param_range_df.set_index(['Short Name'], inplace=True)

mids = {i: str(i).rjust(3, '0') for i in model_param_df['ID'].values}

mids = ['800', '802', '804', '808', '810',  '842', 'A00', 'C00'] #'840', # only do prediction parameter uncertainty for a few models


param_replace_dict = {}
for index, row in param_range_df.iterrows():
    if str(row['Latex_Symbol']).startswith('$'):
        symbol = row['Latex_Symbol']
        param_replace_dict[row.name] = symbol

def frexp10(x, decimals=2):
    exp = int(math.floor(math.log10(abs(x))))
    val = x / 10**exp
    return round(val, 2), exp 

with open('posterior_latex_tables.tex', 'w') as fout:

    for mid in mids:
        dakota_out_folderpath = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'calibration',  'sew' , 'QUESO_DRAM', 'model_'+mid, 'lowering_history_0.pg24f_ic5etch', 'dakota_queso_dram.out']
        dakota_out_file = os.path.join(os.path.abspath(os.sep), *dakota_out_folderpath)
    
        with codecs.open(dakota_out_file, "r",encoding='utf-8', errors='ignore') as f:
            dakota_out_lines = f.readlines()
    
        dakota_out_text = ''.join(dakota_out_lines)
    
        parameter_distribution_lines = dakota_out_text.split('Sample moment statistics for each posterior variable:')[-1].split('Sample moment statistics for each response function:')[0].strip().split('\n')
    
    
        temp_dict = []
        for line in parameter_distribution_lines[1:]:
            vals = line.strip().split()
            param_name = param_replace_dict[vals[0]]
            mean = float(vals[1])
            std = float(vals[2])
            skew = float(vals[3])
            kurt = float(vals[4])
            
            temp_dict.append(pd.DataFrame({'val':{'parameter_name': param_name, 
                              'mean': '{0}^{{{1:+03}}}'.format(*frexp10(mean)), 
                              'standard_deviation': '{0}^{{{1:+03}}}'.format(*frexp10(std)),
                              'skewness': '{0}^{{{1:+03}}}'.format(*frexp10(skew)),
                              'kurtosis':'{0}^{{{1:+03}}}'.format(*frexp10(kurt))}}))
    
        
        df = pd.concat(temp_dict, axis=1).T
        df.reset_index(level=0, inplace=True)
        df = df[['parameter_name', 'mean', 'standard_deviation', 'skewness', 'kurtosis']]
        df.rename(columns={'parameter_name':'Parameter Name', 'mean':'Mean', 'standard_deviation':'Standard Deviation',
                           'skewness': 'Skewness', 'kurtosis':'Kurtosis'}, inplace=True)
        
        latex_text = df.to_latex(escape=False, index=False)
        
        fout.write('% MCMC posterior for model number' + mid + '\n')
        fout.write('\\begin{table}\n')
        fout.write('\caption{First for moments of the posterior distribution estimated with QUESO-DRAM for model '+ mid + ' in Upper Franks Creek Watershed (SEW domain).}\n')
        fout.write('   \label{tab:mcmc_param_'+ mid +'}\n%') 
        fout.write(' \centering\n')
        fout.write(latex_text )
        fout.write('\end{table}\n\n')

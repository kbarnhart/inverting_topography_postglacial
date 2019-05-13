#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:29:54 2018

@author: barnhark
"""

import pandas as pd
import os
import numpy as np

parameter_dict_folderpath = ['..', 'auxillary_inputs']

sa_model_parameter_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_parameter_match.csv'])))

ca_model_parameter_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['model_parameter_match_calibration_sew.csv'])))

sa_df = pd.read_csv(sa_model_parameter_input_file, index_col=0)
sa_df = sa_df.dropna() # Remove those rows with models that are not yet complete
ca_df = pd.read_csv(ca_model_parameter_input_file, index_col=0)
ca_df = ca_df.dropna() # Remove those rows with models that are not yet complete
ca_df.drop(['dakota_ga_seed'], axis = 1, inplace=True)


parameter_range_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['parameter_ranges.csv'])))
param_range_df = pd.read_csv(parameter_range_input_file)

param_replace_dict = {}
for index, row in param_range_df.iterrows():
    if str(row['Latex_Symbol']).startswith('$'):
        symbol = row['Latex_Symbol']
        param_name = row['Short Name']
        param_replace_dict[param_name] = symbol
        
#%%
mids = {i: str(i).rjust(3, '0') for i in sa_df['ID'].values}

model_names = sa_df.index.values
dfs = []

# for model in model
for mn in model_names:
    sa = sa_df.loc[mn]
    ca = ca_df.loc[mn]
    
    comp = sa!=ca
    
    if np.any(comp):
        changed_values = ca.loc[comp].T
        changed_values.rename('Fixed Value', inplace=True)
        temp_df = pd.DataFrame(changed_values)
        temp_df['Model'] = mn
        temp_df.set_index('Model', append=True, inplace=True)
        dfs.append(temp_df)

df = pd.concat(dfs)
df.reset_index(inplace=True)

df.rename(columns={'level_0': 'pname'}, inplace=True)

df['Parameter Name'] = df.pname.map(param_replace_dict)

df.drop(['pname'], axis = 1, inplace=True)

df.set_index(['Parameter Name', 'Fixed Value'], inplace=True)
df.sort_index(level=-2, inplace=True)

latex_file = os.path.join('misc_tables', 'param_values_fixed_calibration_table.txt')
df.to_latex(latex_file, escape=False, multirow=True)

#%%

with open(latex_file, 'r') as f:
    temp_lines = f.readlines()
label = 'tab:calib_fixed_param_vals'

latex_lines = []
table_lines = []

caption = 'Parameter values fixed for calibration runs.'

column_format = temp_lines[0].strip().split('\\begin{tabular}')[-1]
temp_lines = temp_lines[1:-1]

header = temp_lines[:4]
header.append('\endfirsthead')
# add      \endhead to correct line in temp_lines
temp_lines.insert(4, '\endhead')

# add page break after 3 varibles of text 
nvar = 4
inds = [ i for i in range(len(temp_lines)) if temp_lines[i].startswith('\multirow')]
if len(inds) > nvar:
    
    fix_inds = inds[::3][1:]
    for insert_ind in fix_inds[::-1]:
        temp_lines[insert_ind-1] = '\pagebreak'
                    
table_lines = ['\\begin{center}',
               '\\begin{longtable}' + column_format,
               '\\caption{' + caption + '} \label{' + label + '}\\\\']
table_lines.extend([line.strip() for line in header])
table_lines.append('\\caption[]{(continued)}\\\\')

end_lines = ['\\end{longtable}',
             '\\end{center}',
             '']

table_lines.extend([line.strip() for line in temp_lines])
table_lines.extend(end_lines)
latex_lines.extend(table_lines)

with open('param_values_fixed_calibration_table.tex' ,'w') as f:
    for line in latex_lines:
        f.write(line + '\n')  
        
os.remove(latex_file)
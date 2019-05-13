#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:14:24 2017

@author: katybarnhart
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import ggplot
from ggplot import *

import os
import glob

##############################################################################
#                                                                            #
#        Part 0: Name of compiled outputs and the initial Dakota .in file    #
#                                                                            #
##############################################################################
run_dir_list = ['work', 'WVDP_EWG_STUDY3', 'study3py','sensitivity_analysis', 'sew', 'MOAT']
run_dir = os.path.join(os.path.abspath(os.sep), *run_dir_list)

results_dir_list = ['work', 'WVDP_EWG_STUDY3', 'results','sensitivity_analysis', 'sew', 'MOAT']
results_dir = os.path.join(os.path.abspath(os.sep), *results_dir_list)

output_filename = 'moat_combined_output.csv'
dakota_in = 'dakota_moat.in'
    
# loop within models. 
all_models_datfile = os.path.join(results_dir, output_filename)
all_models_df = pd.read_csv(all_models_datfile)

color_list = ['#1f78b4','#33a02c','#e31a1c', '#ff7f00','#6a3d9a', '#a6cee3','#b2df8a','#fb9a99','#fdbf6f','#cab2d6', '#ffff99']
across_color_list = ['#e5e5e5', '#1f78b4','#33a02c','#e31a1c', '#ff7f00','#6a3d9a', '#a6cee3','#b2df8a','#fb9a99','#fdbf6f','#cab2d6', '#ffff99']

models = all_models_df.model.unique()

#p = ggplot(all_models_df, aes(x='elapsed_time')) +\
#        geom_histogram() +\
#        facet_wrap(x='model', ncol=ncol, scales='free')+theme_bw()
#p.save(filename=os.path.join(output_folder_figures, 'model_timing.pdf'), width=24.5, height=32)
#
#
#p = ggplot(all_models_df, aes(x=1, y='elapsed_time')) +\
#        geom_boxplot() + theme_bw()+ facet_wrap(x='model')
#p.show()
#
#all_models_df_plot = all_models_df.dropna(subset =['number_of_sub_time_steps'])
#p = ggplot(all_models_df_plot, aes(x ='number_of_sub_time_steps', y='elapsed_time', color = 'model')) +geom_point()+theme_bw()
#p.show()
#
#
#all_models_df_soil = all_models_df.dropna(subset =['soil_production_decay_depth'])

cat_columns = [col for col in all_models_df.columns if col.startswith('chi_elev')]
all_models_df['cat_objective_function'] = np.sum(np.square(all_models_df[cat_columns]), axis=1)

latex_model_dict = {}
#%%
# replacement dictionaries
parameter_dict_folderpath = ['..', '..', '..', 'auxillary_inputs']
parameter_range_input_file = os.path.abspath(os.path.join(*(parameter_dict_folderpath+['parameter_ranges.csv'])))
param_range_df = pd.read_csv(parameter_range_input_file)

param_replace_dict = {}
for index, row in param_range_df.iterrows():
    if str(row['Latex_Symbol']).startswith('$'):
        symbol = row['Latex_Symbol']
        param_replace_dict[row['Short Name']] = symbol

lowering_replace_dict = {'lowering_history_1': '1',
                         'lowering_history_2': '2'}

initial_replace_dict = {'pg24f_0etch':    '0$\%$ etching',
                        'pg24f_3pt5etch': '3.5$\%$ etching',
                        'pg24f_7etch':    '7$\%$ etching',
                        'pg24f_14etch':   '14$\%$ etching',
                        'pg24f_randetch': '7$\%$ etching with noise',
                        'pg24f_ic5etch': '7$\%$, no filling in upper watershed'}

initial_order = ['0$\%$ etching',
                 '3.5$\%$ etching',
                 '7$\%$ etching', 
                 '7$\%$ etching with noise',
                 '7$\%$, no filling in upper watershed',
                 '14$\%$ etching']

#%%
output_folder_tables = os.sep+os.path.join(*['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'sensitivity_analysis', 'sew'])
if os.path.exists(output_folder_tables) is False:
    os.makedirs(output_folder_tables)
    
output_folder_figures = os.sep+os.path.join(*['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_figures', 'sensitivity_analysis', 'sew'])
if os.path.exists(output_folder_figures) is False:
    os.makedirs(output_folder_figures)    
    

#%%
for model in models:
    plt.close('all')
    
    dat_all = all_models_df.loc[all_models_df['model'] == model].dropna(axis=1, how='all')
    dat_all.rename(columns = {'lowering': "Lowering History"}, inplace = True)    
    ##############################################################################
    #                                                                            #
    #        Part 1: Read in the compiled Dakota output                          #
    #                                                                            #
    ##############################################################################
    
    # FOR EACH LOWERING/INITIAL CONDITION...
    bcics = np.unique(dat_all['bc_ic'])
    
    # Get the names of all outputs and the total number of outputs.
    orig_outputs = [col for col in dat_all.columns if col.startswith('ASV')]
    cat_outputs = [col for col in all_models_df.columns if col.startswith('chi_elev')]
    of_outputs = [col for col in all_models_df.columns if col.startswith('cat_objective_function')]
    
    output_groups = {'orig': {'outputs':orig_outputs, 'ncol': 4},
                     'cat': {'outputs':cat_outputs, 'ncol': 5},
                     'cat_of': {'outputs':of_outputs, 'ncol': None}}
    
    latex_temp = {}
    for ogk in output_groups.keys():
        outputs = output_groups[ogk]['outputs']
        ncol = output_groups[ogk]['ncol']
        num_outputs = len(outputs)
        
        elementary_effects_list = {}
        positive_elementary_effects_list = {}
        morris_list = []
        
        for bcic in bcics:
             
            # filter dat
            dat = dat_all[dat_all['bc_ic']==bcic].reset_index(drop=True)
            
            del dat['model']
            del dat["Lowering History"]
            del dat['initial_condition']
            del dat['bc_ic']
            
            # find in file
            in_file = os.path.join(run_dir, model, bcic, dakota_in)
            
            ##############################################################################
            #                                                                            #
            #    Part 2: Use the Dakota .in file to get information about the runs       #
            #                                                                            #
            ##############################################################################
            
            # Get the number of partitions, the names of inputs (which dakota calles
            # 'descriptors') and the input minimums and maximus. 
            with open(in_file, 'r') as f:
                lines = []
                for l in f.readlines():
                    l = l.strip()
                    if l.split(' ')[0] == 'partitions':
                        num_partitions = int(l.split(' ')[-1])
                    if l.split(' ')[0] == 'descriptors':
                        descriptors = l.split('=')[-1].strip().split(' ')
                    if l.split(' ')[0] == 'lower_bounds':
                        lower_bounds = l.split('=')[-1].strip().split(' ')
                    if l.split(' ')[0] == 'upper_bounds':
                        upper_bounds = l.split('=')[-1].strip().split(' ')
    
            # remove the "" around each input name
            input_order = [d[1:-1] for d in descriptors]
            
            # construct an inputs dictionary that has information about the minimum, 
            # maximum and range for each input parameter. 
            inputs = {}
            for i in range(len(input_order)):
                idict = {'min': float(lower_bounds[i]),
                         'max': float(upper_bounds[i])}
                idict['range']=idict['max']-idict['min']
                inputs[input_order[i]] = idict
            
            # Calculate the number of inputs and the number of levels. 
            num_inputs = len(inputs)
            num_levels = num_partitions + 1
            
            ##############################################################################
            #                                                                            #
            # Part 3: Use the output table to calculate other information about the runs #
            #                                                                            #
            ##############################################################################
            
            # Get the number of total model runs or "samples"
            num_samples = dat.shape[0]
            
            # Calculate the number of trajectories through parameter space.
            # this is given by number of samples divided by (number of inputs + 1)
            num_trajectories = int(num_samples/(num_inputs+1))
            
            # Calculate the size of the elementary effect step, delta. Importantly Dakota
            # uses the normalized delta (as if all input Input have been scaled 
            # between zero and one).
            normalized_delta_size = (num_levels)/(2.*(num_levels-1.))
            
            ##############################################################################
            #                                                                            #
            #    Part 4: Scale the the value of each input parameter to fall between     #
            #    zero and one. This permits the delta value to be the same for each      #
            #    parameter and is a critical step to reproduce the values that Dakota    #
            #    generates for the modified mean and standard deviation                  #
            #                                                                            #
            ##############################################################################
            
            # For each input.
            for ik in inputs.keys():
                # Scale the input values to be between zero and one. 
                dat[ik] = (dat[ik]- inputs[ik]['min'])/(inputs[ik]['range'])
            
            ##############################################################################
            #                                                                            #
            # Part 5: Calculate the difference between each subsequent model run         #
            #                                                                            #
            ##############################################################################
            
            # Calculate differences
            dat_diff = dat.diff()
            
            # Construct the index of the differences that represent differences across
            # the boundary of two different trajectories. 
            bad_diffs = np.arange(0, num_samples, num_inputs+1)
            
            # Set those cross-trajectory differences to NaN as they are irrelevant to this
            # analysis. 
            
            dat_diff[dat_diff.index.isin(bad_diffs)] = np.nan
             
            ##############################################################################
            #                                                                            #
            #   Note: The subsequent analysis is done internaly by Dakota and piped to   #
            #   moat.log. I have broken down the post-processing steps so that it is     #
            #   possible to see exactly what Dakota does.                                # 
            #                                                                            #
            ##############################################################################
            
            ##############################################################################
            #                                                                            #
            # Part 6: Calculate the difference between each subsequent model run         #
            #                                                                            #
            ##############################################################################
            
            # Calculate differences
            dat_diff = dat.diff()
            
            # Construct the index of the differences that represent differences across
            # the boundary of two different trajectories. 
            irrelevant_diffs = np.arange(0, num_samples, num_inputs+1)
            
            # Set those cross-trajectory differences to NaN as they are irrelevant to this
            # analysis. 
            dat_diff[dat_diff.index.isin(irrelevant_diffs)] = np.nan
            
            ##############################################################################
            #                                                                            #
            #  Part 7: Loop through each input and calcuate the distribution of          #
            #  differences between each output (this is what the Moris Method terms      #
            #  elementary effects.                                                      #
            #                                                                            #
            ##############################################################################   
            
            # Initialize a dictionary to contain information about each set of elementary
            # effects.   
            elementary_effects_dictionary = {}
            
            # Loop through each input paramter to calculate the distribution of elementary
            # effects. 
            for ik in input_order:
            
                # Identify which differences are associated with a change in this 
                # parameter. This is necessary because the order in which each parameter
                # is changed is different in each trajecory.        
                # Choose those differences where the absolute value of the difference is
                # equal to the normalized delta size. 
                dat_sel_inds = np.isclose(np.abs(dat_diff[ik]), normalized_delta_size, rtol=0.01)
                
                # Select the outputs differences where this input changed (there should
                # be one per trajectory).
                
                just_selected_outputs_diffs = dat_diff[outputs][dat_sel_inds].reset_index(drop=True)
                
                # Calculate the elementary effect by dividing the difference by the 
                # normalized delta. 
                
                # Thus the elementary effect represents the difference in output per
                # change in normalized parameter space and has the same units as the
                # output variable. 
                elementary_effects = just_selected_outputs_diffs/normalized_delta_size
                
                # Save these elementary effects. 
                elementary_effects_dictionary[ik] = elementary_effects.reset_index(drop=True) 
            
            ##############################################################################
            #                                                                            #
            # Part 8: Use the output table to calculate other information about the runs #
            #                                                                            #
            ##############################################################################
            
            # The Method of Morris creates two sets of statistics. The first set is
            # the mean and standard deviation of the elementary effects for a given
            # input and output. The mean, termed 'mu' gives an indication of the 
            # average effect changing the parameter by a large proportion of the 
            # parameter space, while the standard deviation gives a sense of how 
            # variable the effect is depending on where it is taken in parameter space. 
            elementary_effects_df = pd.concat(elementary_effects_dictionary)
            elementary_effects_df.index.names = ['Input', 'Trajectory']
            
            elementary_effects_df.groupby(level=['Input']).mean().dropna()
            
            mu = elementary_effects_df.groupby(level=['Input']).mean().dropna()
            sigma = elementary_effects_df.groupby(level=['Input']).std().dropna()
            
            # This method also produces a modified mean and stardard deviation, termed
            # mu_star and sigma_star. These differ from mu and sigma in that they are 
            # derived from the absolute value of the elementary effects and thus are only
            # positive. 
            positive_elementary_effects_df = np.abs(elementary_effects_df)
            mu_star = positive_elementary_effects_df.groupby(level=['Input']).mean().dropna()
            sigma_star = positive_elementary_effects_df.groupby(level=['Input']).std().dropna()
            
            # Dakota returns mu_star and sigma_star in the moat.log file. 
            
            ##############################################################################
            #                                                                            #
            # Part 9: Dataframe housekeeping for easy plotting                           #
            #                                                                            #
            ##############################################################################
            
            # Dataframe re-organization so that we have a long-form dataframe for plotting.
            
            # First melt each dataframe
            mu = mu.reindex(input_order)
            mu['Input'] = mu.index.values
            mu_melt = pd.melt(mu, id_vars='Input', var_name='Metric', value_name='mu')
            
            sigma = sigma.reindex(input_order)
            sigma['Input'] = sigma.index.values
            sigma_melt = pd.melt(sigma, id_vars='Input',  var_name='Metric', value_name='sigma')
            
            mu_star = mu_star.reindex(input_order)
            mu_star['Input'] = mu_star.index.values
            mu_star_melt = pd.melt(mu_star,  id_vars='Input',  var_name='Metric', value_name='mu_star')
            
            sigma_star = sigma_star.reindex(input_order)
            sigma_star['Input'] = sigma_star.index.values
            sigma_star_melt = pd.melt(sigma_star, id_vars='Input',  var_name='Metric', value_name='sigma_star')
            
            # Then merge the melted dataframes. 
            morris_df = pd.merge(pd.merge(mu_melt, sigma_melt, on=['Input', 'Metric']),
                                 pd.merge(mu_star_melt, sigma_star_melt, on=['Input', 'Metric']),  
                                 on=['Input', 'Metric'])
            bc = bcic.split('.')[0]
            ic = bcic.split('.')[1]
            
            morris_df["Lowering History"] = bc
            morris_df['Initial Condition'] = ic
            
            morris_list.append(morris_df)
            
            index = tuple(bcic.split('.'))
            
            elementary_effects_list[index] = elementary_effects_df
            positive_elementary_effects_list[index] = positive_elementary_effects_df
     
        ##############################################################################
        #                                                                            #
        #        Part 10: Save Morris DF to a csv                                    #
        #                                                                            #
        ##############################################################################
        
        morris_df = pd.concat(morris_list)
        
        # sort
        morris_df.index = range(1,len(morris_df) + 1)
        morris_df = morris_df.sort_values(['Input', "Lowering History", 'Initial Condition'])

        # save just mu* and sigma* with re-ordered rows for clarity for the report
        if len(outputs) == 1:
            morris_df_short =  morris_df[['Input', "Lowering History", 'Initial Condition', 'mu_star', 'sigma_star']]
        else:
            morris_df_short =  morris_df[['Input', "Lowering History", 'Initial Condition', 'Metric', 'mu_star', 'sigma_star']]
        
        morris_df_short["Input"].replace(param_replace_dict, inplace=True)    
        morris_df_short["Initial Condition"].replace(initial_replace_dict, inplace=True)
        morris_df_short["Initial Condition"] = morris_df_short["Initial Condition"].astype('category').cat.set_categories(initial_order, ordered=True)
        morris_df_short["Lowering History"].replace(lowering_replace_dict, inplace=True)    
        
        morris_df_short = morris_df_short.sort_values(['Input', "Lowering History", 'Initial Condition'])
        morris_df_short.index = range(1,len(morris_df_short) + 1)

        morris_df_short.to_csv(os.path.join(output_folder_tables, ogk+'.morris_df_short.'+model+'.parameter.csv'), index=False)
        
        # Change some names for LaTeX Output
        for cn in ['mu_star', 'sigma_star']:
            vals = morris_df_short[cn].map(lambda x: "{:.3e}".format(x))
            strs = vals.str.split('e', expand = True)
            isnan = strs.iloc[:,0].apply(float).apply(np.isnan)
            strs.loc[isnan,1] = '0'
            strs.iloc[:,1] = strs.iloc[:,1].apply(int).apply(str)
            strs.loc[isnan,1] = ''
            morris_df_short.loc[:, cn] = r'$ ' + strs.iloc[:,0] + ' \\times 10^{' + strs.iloc[:,1] + '} $'
        
        morris_df_short.rename(columns = {'mu_star':'$\\mu^*$', 'sigma_star':'$\\sigma^*$'}, inplace = True)
        morris_df_short.set_index(list(morris_df_short.columns[:-2]), inplace=True)
        
        latex_file = os.path.join(output_folder_tables, ogk+'.morris_df_short.'+model+'.txt')
        morris_df_short.to_latex(latex_file, 
                               escape=False,
                               multirow=True)
        if ogk.startswith('cat_of'):
            latex_temp['Parameter'] = latex_file
        ##############################################################################\n",
        #                                                                            #\n",
        # Plot 1: Mu* against Sigma Plots for each output variable                   #\n",
        #                                                                            #\n",
        ##############################################################################\n",
    
#        p = ggplot(morris_df, aes(x='mu_star', y='sigma_star', color='Input', size="Lowering History", shape='Initial Condition')) +\
#                geom_point(size=30, color='k') +\
#                geom_point(size=20) +\
#                scale_color_manual(values=color_list)+\
#                theme_bw()+\
#                xlim(low=0) + ylim(low=0)+\
#                ggtitle(model)
#                
#        if ncol:
#            p = p + facet_wrap(x='Metric', ncol=ncol, scales='free')
#        
#        p.save(filename=os.path.join(output_folder_figures, ogk+'.mu_star_sigma_star_plots.'+model+'.pdf'), width=24.5, height=32)
#        
#        if ogk.startswith('cat_of'):
#         
#            p = ggplot(morris_df, aes(x='mu_star', y='sigma_star', color='Input', size="Lowering History", shape='Initial Condition')) +\
#                geom_point(size=20) +\
#                scale_color_manual(values=color_list)+\
#                theme_bw()+\
#                xlim(low=0) + ylim(low=0)+\
#                ggtitle(model)
#            p.margins = {'left'  : 0.125,  # the left side of the subplots of the figure
#                         'right' : 0.9,    # the right side of the subplots of the figure
#                         'bottom' : 0.1,   # the bottom of the subplots of the figure
#                         'top' : 0.9,      # the top of the subplots of the figure
#                         'wspace' : 0.2,   # the amount of width reserved for blank space between subplots,
#                                       # expressed as a fraction of the average axis width
#                         'hspace' : 0.}
#            p.apply_coords()
#            
#            p.save(filename=os.path.join(output_folder_figures, ogk+'.report.mu_star_sigma_star_plots.'+model+'.pdf'))
                    

##############################################################################
        #                                                                            #
        # Part 11: Difference across lowering histories and initial conditions       #
        #                                                                            #
        ##############################################################################         
                
        # take dat_all, and split into chunks by IC and BC
        # first, lowering histories. 
        lowerings = np.unique(dat_all["Lowering History"])
        
        lowering_frames = {}
        for lowering in lowerings:
            # select correct portion of dat_all
            dat_sel = dat_all[dat_all["Lowering History"]==lowering]
            
            # remove non-numeric columns and index by initial condition and run
            dat_sel.set_index(['initial_condition', 'run'], inplace=True)
            
            drop = ['Unnamed: 0', 'model', 'memory_used', 'elapsed_time', "Lowering History", 'bc_ic', 'eval_id']
            dat_sel = dat_sel.drop(drop, axis=1)
            dat_sel = dat_sel[outputs]
            lowering_frames[lowering] = dat_sel
            
        reference_lowering = 'lowering_history_1'
        reference_lowering_frame = lowering_frames[reference_lowering]
        other_lowerings = [lowering for lowering in lowerings if lowering != reference_lowering]
                
        lowering_diffs = []
        for o_l in other_lowerings:
            ol_frame = lowering_frames[o_l]
            abs_difference = np.abs(ol_frame - reference_lowering_frame)
            
            mu_star = abs_difference.groupby(level=['initial_condition']).mean().dropna()
            mu_star['Input'] = mu_star.index.values
            mu_star_melt = pd.melt(mu_star,  id_vars='Input', var_name='Metric', value_name='mu_star')
            
            sigma_star = abs_difference.groupby(level=['initial_condition']).std().dropna()
            sigma_star['Input'] = sigma_star.index.values
            sigma_star_melt = pd.melt(sigma_star,  id_vars='Input', var_name='Metric', value_name='sigma_star')    
            
            df = pd.merge(mu_star_melt, sigma_star_melt, on=['Input', 'Metric'])
            df["Lowering History"] = o_l 
            lowering_diffs.append(df)
            
            #this joining is not yet done correctly. alphas are wrong, and need to differentiate between loering/ic in lowering vs morris frame
            
        lowering_frame = pd.concat(lowering_diffs)
        morris_for_join = morris_df.drop(['mu', 'sigma', 'Initial Condition'], axis=1)
        morris_for_join['Input'] = 'Continuous Parameter'
        morris_for_join["Lowering History"] = 'Continuous Parameter'
        
        plot_frame = pd.concat([lowering_frame, morris_for_join])
        
#        p = ggplot(lowering_frame, aes(x='mu_star', y='sigma_star', color="Lowering History", shape='Input')) +\
#            geom_point(size=30, color='k') +\
#            geom_point(size=20) +\
#            scale_color_manual(values=color_list)+\
#            theme_bw()+\
#            xlim(low=0) + ylim(low=0)+\
#            ggtitle(model)
#            
#        if ncol:
#            p = p + facet_wrap(x='Metric', ncol=ncol, scales='free')
#            
#        p.save(filename=os.path.join(output_folder_figures, ogk+'.across_lowering.'+model+'.pdf'), width=24.5, height=32)
#
#        # put sensitivity to paramters in gray behind
#        p = ggplot(plot_frame, aes(x='mu_star', y='sigma_star', color="Lowering History", shape='Input')) +\
#                geom_point(size=30, color='k') +\
#                geom_point(size=20) +\
#                scale_color_manual(values=across_color_list)+\
#                theme_bw()+\
#                xlim(low=0) + ylim(low=0)+\
#                ggtitle(model)
#                
#        if ncol:
#            p = p + facet_wrap(x='Metric', ncol=ncol, scales='free')
#            
#        p.save(filename=os.path.join(output_folder_figures, ogk+'.across_lowering_with_reference.'+model+'.pdf'), width=24.5, height=32)
        
        # second, initial_conditions
        initials = np.unique(dat_all['initial_condition'])
        
        initials_frames = {}
        for initial in initials:
            # select correct portion of dat_all
            dat_sel = dat_all[dat_all['initial_condition']==initial]
            
            # remove non-numeric columns and index by initial condition and run
            dat_sel.set_index(["Lowering History", 'run'], inplace=True)
            
            drop = ['Unnamed: 0', 'model', 'memory_used', 'elapsed_time', 'initial_condition', 'bc_ic', 'eval_id']
            dat_sel = dat_sel.drop(drop, axis=1)
            dat_sel = dat_sel[outputs]
            initials_frames[initial] = dat_sel
            
        
        reference_initial = 'pg24f_7etch'
        reference_initial_frame = initials_frames[reference_initial]
        other_initials = [initial for initial in initials if initial != reference_initial]
        initial_diffs = []
        for o_i in other_initials:
            oi_frame = initials_frames[o_i]
            abs_difference = np.abs(oi_frame - reference_initial_frame)
            
            mu_star = abs_difference.groupby(level=["Lowering History"]).mean().dropna()
            mu_star['Input'] = mu_star.index.values
            mu_star_melt = pd.melt(mu_star,  id_vars='Input', var_name='Metric', value_name='mu_star')
            
            sigma_star = abs_difference.groupby(level=["Lowering History"]).std().dropna()
            sigma_star['Input'] = sigma_star.index.values
            sigma_star_melt = pd.melt(sigma_star,  id_vars='Input', var_name='Metric', value_name='sigma_star')    
            
            df = pd.merge(mu_star_melt, sigma_star_melt, on=['Input', 'Metric'])
            df['Initial Condition'] = o_i 
            initial_diffs.append(df)
        
        initial_frame = pd.concat(initial_diffs)
        morris_for_join = morris_df.drop(['mu', 'sigma', "Lowering History"], axis=1)
        morris_for_join['Input'] = 'Continuous Parameter'
        morris_for_join['Initial Condition'] = 'Continuous Parameter'
        
        plot_frame = pd.concat([initial_frame, morris_for_join])
        
#         # put sensitivity to paramters in gray behind
#        p = ggplot(initial_frame, aes(x='mu_star', y='sigma_star', shape='Input', color='Initial Condition')) +\
#                geom_point(size=30, color='k') +\
#                geom_point(size=20) +\
#                scale_color_manual(values=color_list)+\
#                theme_bw()+\
#                xlim(low=0) + ylim(low=0)+\
#                ggtitle(model)
#                
#        if ncol:
#            p = p + facet_wrap(x='Metric', ncol=ncol, scales='free')
#            
#        p.save(filename=os.path.join(output_folder_figures, ogk+'.across_initial.'+model+'.pdf'), width=24.5, height=32)
#    
#        # put sensitivity to paramters in gray behind
#        p = ggplot(plot_frame, aes(x='mu_star', y='sigma_star', shape='Input', color='Initial Condition')) +\
#                geom_point(size=30, color='k') +\
#                geom_point(size=20) +\
#                scale_color_manual(values=across_color_list)+\
#                theme_bw()+\
#                xlim(low=0) + ylim(low=0)+\
#                ggtitle(model)
#        if ncol:
#            p = p + facet_wrap(x='Metric', ncol=ncol, scales='free')
#            
#        p.save(filename=os.path.join(output_folder_figures, ogk+'.across_initial_with_reference.'+model+'.pdf'), width=24.5, height=32)
        
        #%%
        # save the initial and lowering dataframes
     
        if len(outputs) == 1:
            initial_frame =  initial_frame[['Initial Condition', 'Input', 'mu_star', 'sigma_star']]
            
        else:
            initial_frame =  initial_frame[['Initial Condition', 'Input', 'Metric', 'mu_star', 'sigma_star']]
        initial_frame["Input"].replace(lowering_replace_dict, inplace=True)    
        initial_frame["Initial Condition"].replace(initial_replace_dict, inplace=True)
        initial_frame.rename(columns = {'Input': "Lowering History"}, inplace=True)
        initial_frame.to_csv(os.path.join(output_folder_tables, ogk+'.morris_df_short.'+model+'.initial.csv'), index=False)
        
        initial_frame["Initial Condition"] = initial_frame["Initial Condition"].astype('category').cat.set_categories(initial_order, ordered=True)
        initial_frame = initial_frame.sort_values(['Initial Condition', "Lowering History"])
        initial_frame.index = range(1,len(initial_frame) + 1)
        
        initial_frame.rename(columns = {'Initial Condition': 'Initial Condition\\\\(Reference: 7$\%$ etch)'}, inplace = True)
        initial_frame.index = range(1,len(initial_frame) + 1)
        
        # Change some names for LaTeX Output
        for cn in ['mu_star', 'sigma_star']:
            vals = initial_frame[cn].map(lambda x: "{:.3e}".format(x))
            strs = vals.str.split('e', expand = True)
            isnan = strs.iloc[:,0].apply(float).apply(np.isnan)
            strs.loc[isnan,1] = '0'
            strs.iloc[:,1] = strs.iloc[:,1].apply(int).apply(str)
            strs.loc[isnan,1] = ''
            initial_frame.loc[:, cn] = r'$ ' + strs.iloc[:,0] + ' \\times 10^{' + strs.iloc[:,1] + '} $'
        
        initial_frame.rename(columns = {'mu_star':'$\\mu^*$', 'sigma_star':'$\\sigma^*$'}, inplace = True)
        initial_frame.set_index(list(initial_frame.columns[:-2]), inplace=True)
        
        latex_file = os.path.join(output_folder_tables, ogk+'.morris_df_short.'+model+'.initial.txt')
        initial_frame.to_latex(latex_file, 
                               escape=False,
                               multirow=True)
        if ogk.startswith('cat_of'):
            latex_temp['Initial Condition'] = latex_file
        #%%
        if len(outputs) == 1:
            lowering_frame =  lowering_frame[['Lowering History', 'Input', 'mu_star', 'sigma_star']]
        else:
            lowering_frame =  lowering_frame[['Lowering History', 'Input', 'Metric', 'mu_star', 'sigma_star']]

        lowering_frame["Input"].replace(initial_replace_dict, inplace=True)    
        lowering_frame['Lowering History'].replace(lowering_replace_dict, inplace=True)
        lowering_frame.rename(columns = {'Input': "Initial Condition"}, inplace=True)
        
        lowering_frame["Initial Condition"] = lowering_frame["Initial Condition"].astype('category').cat.set_categories(initial_order, ordered=True)
        lowering_frame = lowering_frame.sort_values(["Lowering History", 'Initial Condition'])
        lowering_frame.index = range(1,len(lowering_frame) + 1)
        
        lowering_frame.to_csv(os.path.join(output_folder_tables, ogk+'.morris_df_short.'+model+'.lowering.csv'), index=False)
        lowering_frame.rename(columns = {'Lowering History': 'Lowering History\\\\(Reference: History 1)'}, inplace = True)
        lowering_frame.index = range(1,len(lowering_frame) + 1)
        
        # Change some names for LaTeX Output
        for cn in ['mu_star', 'sigma_star']:
            vals = lowering_frame[cn].map(lambda x: "{:.3e}".format(x))
            strs = vals.str.split('e', expand = True)
            isnan = strs.iloc[:,0].apply(float).apply(np.isnan)
            strs.loc[isnan,1] = '0'
            strs.iloc[:,1] = strs.iloc[:,1].apply(int).apply(str)
            strs.loc[isnan,1] = ''
            lowering_frame.loc[:, cn] = r'$ ' + strs.iloc[:,0] + ' \\times 10^{' + strs.iloc[:,1] + '} $'
        
        lowering_frame.rename(columns = {'mu_star':'$\\mu^*$', 'sigma_star':'$\\sigma^*$'}, inplace = True)
        lowering_frame.set_index(list(lowering_frame.columns[:-2]), inplace=True)

        latex_file = os.path.join(output_folder_tables, ogk+'.morris_df_short.'+model+'.lowering.txt')
        lowering_frame.to_latex(latex_file, 
                                escape=False,
                                multirow=True)
        if ogk.startswith('cat_of'):
            latex_temp['Lowering History'] = latex_file
    latex_model_dict[model] = latex_temp
 
#%%               
# construct latex file     
model_order_filepath = ['..', '..', '..', 'auxillary_inputs']
model_order_input_file = os.path.abspath(os.path.join(*(model_order_filepath+['model_name_element_match.csv'])))
model_order_df = pd.read_csv(model_order_input_file)

latex_lines = []

table_order =  ['Parameter', 
                'Lowering History',
                'Initial Condition']

table_label_keys = {'Parameter': 'param', 
                    'Lowering History': 'lower', 
                    'Initial Condition': 'initial'}

for ID in model_order_df.ID:
    model = 'model_' + ID
    
    if model in latex_model_dict:
        latex_dict = latex_model_dict[model]
        
        for table in table_order:
            table_lines = []
            file = latex_dict[table]
            
            with open(file, 'r') as f:
                temp_lines = f.readlines()
            label = '_'.join(['moat', 
                              model.split('_')[1],
                              'sew',
                              table_label_keys[table]])
            model_full_name =  model_order_df.loc[model_order_df.ID == ID, 'Model Name'].values[0]    
            caption = table+' Sensitivity for Model ' + model.split('_')[1] + ', ' +  model_full_name + '\\\\South East Watershed Domain'
            if table.startswith('Param'):
                column_format = temp_lines[0].strip().split('\\begin{tabular}')[-1]
                temp_lines = temp_lines[1:-1]
                
                header = temp_lines[:4]
                header.append('\endfirsthead')
                # add      \endhead to correct line in temp_lines
                temp_lines.insert(4, '\endhead')
                
                # add page break after 3 varibles of text 
                nvar = 3
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
                             '\\clearpage'
                             '']
            else:
                table_lines = ['\\begin{table} \label{' + label + '}',
                               '\\begin{centering}',
                               '\\caption{' + caption + '}']
                end_lines = ['\\end{centering}',
                             '\\end{table}' ,
                             '']
            table_lines.extend([line.strip() for line in temp_lines])
            
            
            table_lines.extend(end_lines)
            
            latex_lines.extend(table_lines)
        latex_lines.append('\\clearpage')

with open('MOAT_SEW_latex_combined.tex', 'w') as f:
    for line in latex_lines:
        f.write(line + '\n')   
    






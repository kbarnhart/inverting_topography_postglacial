#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 08:10:31 2017

@author: barnhark
"""

# analyze OPT output
# dakota.out and OPT_DEFAULT

# get best Obj Fxn, evaluation values, and confidence interval on values

# did it complete?
# Best photo?

# 808 is an example that did not complete. 

# Start by importing the necessary python libraries
import os
import glob
import numpy as np
import pandas as pd
import shutil
import yaml
import scipy.stats as stats

import matplotlib.pylab as plt
from landlab.io import read_esri_ascii
from landlab.io.netcdf import read_netcdf
from landlab.components import FlowAccumulator, ChiFinder
from landlab.plot import imshow_grid
from metric_calculator import GroupedDifferences

##############################################################################
#                                                                            #
#        Part 0: Name of results directory, params and results files         #
#                                                                            #
##############################################################################
# get the current files filepath
dir_path = os.path.dirname(os.path.realpath(__file__))

# additional file paths
work_dir = ['work', 'WVDP_EWG_STUDY3', 'study3py','calibration', 'sew', 'EGO']
results_dir = ['work', 'WVDP_EWG_STUDY3', 'results','calibration', 'sew', 'EGO']

# file names
output_file = 'dakota_hybrid_calibration.out'
optim_file = 'OPT_DEFAULT.out'

# load chi catagories
cat_file = os.path.join(os.path.abspath(os.sep), *['work', 'WVDP_EWG_STUDY3', 'study3py','auxillary_inputs', 'chi_elev_categories', 'sew.chi_elev_cat.20.txt'])
cat = np.loadtxt(cat_file)

catagory_weight_file = os.path.join(os.path.abspath(os.sep), *['work', 'WVDP_EWG_STUDY3', 'study3py','auxillary_inputs', 'weights', 'sew.chi_elev_weight.20.txt'])
cat_wt = np.loadtxt(catagory_weight_file)

#%%
input_file = glob.glob(os.path.join(os.sep, *(work_dir+['model**', '**', 'inputs_template.txt'])))[0]
with open(input_file, 'r') as f:
    inputs = yaml.load(f)

outlet_id = inputs['outlet_id']
# observed_topography
observed_topo_file_name = inputs['modern_dem_name']
(grid, z) = read_esri_ascii(observed_topo_file_name, name='topographic__elevation', halo=1)
grid.set_watershed_boundary_condition_outlet_id(inputs['outlet_id'], z, nodata_value=-9999)
fa = FlowAccumulator(grid,
                     flow_director='D8',
                     depression_finder = 'DepressionFinderAndRouter')
fa.run_one_step()
#hs = grid.calc_hillshade_at_node()
ch = ChiFinder(grid, min_drainage_area=10*grid.dx**2)
ch.calculate_chi()

# initial condition topography
initial_topo_file_name = inputs['DEM_filename']
(igrid, iz) = read_esri_ascii(initial_topo_file_name, name='topographic__elevation', halo=1)
igrid.set_watershed_boundary_condition_outlet_id(inputs['outlet_id'], iz, nodata_value=-9999)
ifa = FlowAccumulator(igrid,
                      flow_director='D8',
                      depression_finder = 'DepressionFinderAndRouter')
ifa.run_one_step()
#ihs = grid.calc_hillshade_at_node()
ich = ChiFinder(igrid, min_drainage_area=10*grid.dx**2)
ich.calculate_chi()

#%%

# Use glob to find all folders with a 'params.in' file.
work_dir.extend(['model**', '**',  output_file])
output_files = glob.glob(os.path.join(os.path.abspath(os.sep), *work_dir))

if os.path.exists('best_runs') is False:
    os.mkdir('best_runs')
    
if os.path.exists('comparison_figures') is False:
    os.mkdir('comparison_figures')

if os.path.exists(os.sep+os.path.join(*['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', 'sew'])) is False:
    os.makedirs(os.sep+os.path.join(*['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', 'sew']))


conf_intervals = {}
outputs = {}
needs_restart = []
num_cores_total = 0
cmnd_lines = []

diff_factor = {}

calibration_logs = {}

cmnd_lines = []

index_order = ['chi_elev_'+str(i) for i in range(1, 21)]
observed_values = np.zeros(len(index_order))
#%%
for o_f in np.sort(output_files):
    print(o_f)
   # o_f ='/work/WVDP_EWG_STUDY3/study3py/calibration/sew/GAUSSNEWTON/model_600/lowering_history_0.pg24f_7etch/dakota_calibration.out'
    model = o_f.split(os.sep)[-3]
    lowering = o_f.split(os.sep)[-2]
    
    outs = {}
    outs['jacobian_check'] = True
    with open(o_f, 'r') as f:
        lines = f.readlines()
        
    confidence_intervals = {}
    best_parameters = {}
    best_residuals = {}
    d_Vb = {}
    d_SEb = {}
    all_lines = ''.join(lines)
    input_file_lines = all_lines.split('End DAKOTA input file')[0].split('\n')
    temp = all_lines.split('End DAKOTA input file')[1]
    
    del all_lines
    # get EGO portion
    ego_lines = temp.split('<<<<< Iterator efficient_global completed.')[0]
    ego_gn_working_lines = ego_lines.split('<<<<< Function evaluation summary')[0]
    try:
        ego_summary_lines = ego_lines.split('<<<<< Function evaluation summary')[-1]
    except IndexError:
        ego_summary_lines = None

    # get GN Portion
    try:
        gn_lines = temp.split('<<<<< Iterator efficient_global completed.')[1]
    except IndexError:
        gn_lines = None
    
    if gn_lines:
        gn_gn_summary_lines = gn_lines.split('<<<<< Function evaluation summary')
        
        gn_working_lines = gn_gn_summary_lines[0]
        if len(gn_gn_summary_lines) > 1:
            # calibration is  complete:
            gn_summary_lines = gn_gn_summary_lines[1].split('\n')
        else:
            gn_summary_lines = None
            
        del gn_gn_summary_lines
    else:
        gn_working_lines = None
        gn_summary_lines = None
        
    del temp
    
    # get some basic information from the input file lines
    for line in input_file_lines:  
        # get parameter names and order:
        if line.strip()[:11] == 'descriptors':
            parameter_names = [eval(p) for p in line.strip().split('=')[-1].split(' ') if len(p)>0]
              
        # get metric names and order
        if line.strip()[:20] == 'response_descriptors':
            metric_names = [eval(m) for m in line.strip().split('=')[-1].split(' ') if len(m)>0]    
        
        # get metric weights and order
        if line.strip()[:7] == 'weights':
            metric_weights = pd.Series([float(w) for w in line.strip().split('=')[-1].split(' ') if len(w)>0], index=metric_names)
            standard_deviation = np.sqrt(1./metric_weights)
        
        # get number of cores used by the calibration runs. 
        if line.strip()[:22] == 'evaluation_concurrency':
            num_cores = int(line.split('=')[-1].strip()) 
            
    del (line)
    # if calibration completed analyse the summary lines and the working lines
    dof = float(len(observed_values) - len(parameter_names))
    
    if ego_summary_lines:
        grouped_ego_summary_lines = ego_summary_lines.split('<<<<<')
        for grouped_lines in grouped_ego_summary_lines:
            for j in range(len(grouped_lines.split('\n'))):
                g_lines = grouped_lines.split('\n')[j]
                
                # best parameter values
                if g_lines.strip()[:15] == 'Best parameters':
                    best_parameter_lines = [c for c in grouped_lines.split('=')[-1].strip().split() if len(c)>0]
                    n_param = int(len(best_parameter_lines)/2)
                    for p in range(n_param):
                        param = best_parameter_lines.pop()
                        val = float(best_parameter_lines.pop())
                        best_parameters[param] = val
                ego_best_parameters = pd.Series(best_parameters) 
                
                # best residual terms
                if g_lines.strip()[:19] == 'Best residual terms':
                    best_residual_lines = grouped_lines.split('=')[-1].split(' ')
                    # the metric name is not listed, because it is implied 
                    # by the order. 
                    best_resids = []
                    for brl in best_residual_lines:
                        if len(brl.strip())>0:
                            best_resids.append(float(brl.strip().split(' ')[0]))
                    
                    for k in range(len(best_resids)):
                        best_residuals[metric_names[k]] = best_resids[k]
                    
                    ego_best_residual_values = pd.Series(best_residuals)
                
                    outs['ego_objective_function'] = np.sum(np.square(ego_best_residual_values))
        
    if gn_summary_lines:
        # from within the summary lines, extract best fit information
        for sline in gn_summary_lines:
            
            # best objective function value
            if sline[:24] == '<<<<< Best residual norm':
                outs['objective_function'] = 2.* float(sline.split('=')[-1].strip())
                
            # best function evaluation number
            if sline[:47] == '<<<<< Best data captured at function evaluation':
                outs['best_function_evaluation'] = int(sline.split(' ')[-1])
        
            # confidence intervals
            if sline.strip()[:23] == 'Confidence Interval for':
                ci_line = sline[24:]
                parameter_name = ci_line.strip().split('is')[0].strip()
                confidence_interval = eval(ci_line.strip().split('is')[1].strip(''))
                confidence_intervals[parameter_name] = confidence_interval
                
                t_statistic = stats.t.ppf(0.975, dof, loc=0, scale=1)
                
                d_SEb[parameter_name] = (confidence_interval[1]-confidence_interval[0])/(2.*t_statistic)
                
        del sline
        
        SEb = pd.Series(d_SEb)
        Vb = SEb**2.
        
        # next, split the summary lines by '<<<<<' to get the best parameter 
        # values and the best residual terms. 
        
        all_gn_summary_lines = ''.join(gn_summary_lines) 
        grouped_gn_summary_lines = all_gn_summary_lines.split('<<<<<')
        for grouped_lines in grouped_gn_summary_lines:
            for j in range(len(grouped_lines.split('\n'))):
                g_lines = grouped_lines.split('\n')[j]
                
                # best parameter values
                if g_lines.strip()[:15] == 'Best parameters':
                    best_parameter_lines = [c for c in grouped_lines.split('=')[-1].strip().split() if len(c)>0]
                    n_param = int(len(best_parameter_lines)/2)
                    for p in range(n_param):
                        param = best_parameter_lines.pop()
                        val = float(best_parameter_lines.pop())
                        best_parameters[param] = val
                best_parameters = pd.Series(best_parameters)            
                
                # best residual terms
                if g_lines.strip()[:19] == 'Best residual terms':
                    best_residual_lines = grouped_lines.split('=')[-1].split(' ')
                    # the metric name is not listed, because it is implied 
                    # by the order. 
                    best_resids = []
                    for brl in best_residual_lines:
                        if len(brl)>0:
                            best_resids.append(float(brl.strip().split(' ')[0]))
                    
                    for k in range(len(best_resids)):
                        best_residuals[metric_names[k]] = best_resids[k]
                    
                    best_residual_values = pd.Series(best_residuals)
                    
        #del (grouped_lines, best_resids, best_residuals, confidence_interval,
         #    ci_line, parameter_name, g_lines, all_gn_summary_lines, grouped_gn_summary_lines,
        #     j)
        
        # create an output table with (A) Modeled, wtd residual and 
        # (B) Observed, Wt, Std, Modeled, Residual, Weighted Residual
        coeff_of_variation = standard_deviation/observed_values
        best_simulated_values =  best_residual_values + observed_values
        best_weighted_residuals = np.sqrt(metric_weights) * best_residual_values
        proportion_of_objective_fxn = best_weighted_residuals**2/np.sum(best_weighted_residuals**2)
        best_evaluation_df = pd.DataFrame(data=dict(observed_values=observed_values, 
                                                    metric_weights = metric_weights,
                                                    standard_deviation = standard_deviation,
                                                    coeff_of_variation = coeff_of_variation,
                                                    ego_best_residual_values = ego_best_residual_values,
                                                    best_simulated_values = best_simulated_values,
                                                    best_residual_values = best_residual_values,
                                                    best_weighted_residuals = best_weighted_residuals,
                                                    best_weighted_residuals_squared = best_weighted_residuals**2,
                                                    proportion_of_objective_fxn = proportion_of_objective_fxn,))
        best_evaluation_df = best_evaluation_df.reindex(index=index_order, 
                                                        columns=['observed_values',
                                                                 'metric_weights',
                                                                 'standard_deviation',
                                                                 'coeff_of_variation',
                                                                 'ego_best_residual_values',
                                                                 'best_simulated_values',
                                                                 'best_residual_values',
                                                                 'best_weighted_residuals',
                                                                 'best_weighted_residuals_squared',
                                                                 'proportion_of_objective_fxn'])
            
        output_filepath = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', 'sew', 'ego.sew.residuals.full.'+model+'.csv']
        best_evaluation_df.to_csv(os.path.join(os.sep, *output_filepath), float_format='%.3e')
        
        best_evaluation_df_short = best_evaluation_df[['best_simulated_values',
                                                      'best_residual_values',
                                                      'best_weighted_residuals']]
        output_filepath_short = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', 'sew', 'ego.sew.residuals.short.'+model+'.csv']
        best_evaluation_df_short.to_csv(os.path.join(os.sep, *output_filepath_short), float_format='%.3e')
        
#        f, axarr = plt.subplots(1,2)
#        axarr[0].hist(best_evaluation_df.best_weighted_residuals)
#        axarr[0].set_title('Shapiro-Wilk p = ' + str(round(stats.shapiro(best_evaluation_df.best_weighted_residuals)[1], 3)))
#        stats.probplot(best_evaluation_df.best_weighted_residuals, dist="norm", plot=axarr[1])        
#        output_filepath = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', 'sew', 'ego.sew.residuals.hist.'+model+'.pdf']
#        plt.savefig(os.path.join(os.sep, *output_filepath))
#        
     
    # for all models, 
    
    # Parse the working lines to determine the parameter values used in each
    # evaluation. 
    if gn_summary_lines:#gn_working_lines:
        try:
            split_working = gn_working_lines.split('Parameters for evaluation')[1:] # the second of these will not have useful text
            evaluation_dict = {}
            for i in range(len(split_working)):
                evaluation_text = split_working[i].strip()
                eval_id = int(evaluation_text.split(':')[0])
                param_lines = [pt.strip().split(' ') for pt in evaluation_text.split(':')[1].split('\n\n')[0].splitlines() if len(pt)>0]
                param_dict = {}
                for pt in param_lines:
                    param_dict[pt[1]] = pt[0]
                evaluation_dict[eval_id] = param_dict
            evaluation_df = pd.DataFrame(evaluation_dict).T
            
            # Parse the working lines to determine the simulated values from each
            # evaluation. 
            split_working = gn_working_lines.split('Active response data')[1:] # the second of these will not have useful text
            results_dict = {}
            for i in range(len(split_working)):
                evaluation_text = split_working[i]
                eval_id = int(evaluation_text.split(':')[0].split('evaluation')[-1].strip())
                results_lines = [rv.strip().split(' ') for rv in evaluation_text.split(':')[1].split('}')[1].split('\n\n')[0].strip().splitlines() if len(rv)>0]
                re_dict = {}
                for rv in results_lines:
                    re_dict[rv[1]] = rv[0]
                results_dict[eval_id] = re_dict
            results_df = pd.DataFrame(results_dict).T    
            results_df = pd.concat([evaluation_df, results_df], axis=1)
            
            # Parse the working lines to determine the jacobians and the best run's
            # jacobian
            split_working = gn_working_lines.split('Begin Dakota derivative estimation routine')[1:] # the first of these will not have useful text
            calibration_itteration = {}
            for i in range(len(split_working)):
                evaluation_text = split_working[i]
                evaluation_ids = []
                
                for line in evaluation_text.split('>>>>> Total response returned to iterator:')[0].splitlines():
                    if line.startswith('Begin'):
                        evaluation_ids.append(int(line.split('Evaluation')[-1]))
                                        
                    response_lines = evaluation_text.split('>>>>> Total response returned to iterator:')[-1].splitlines()
                    response_lines = [r_l.strip() for r_l in response_lines]
                    response_text = ' '.join(response_lines)
                    if response_lines[-1] != 'Waiting on completed jobs':
                        
                        # based on active set keys, identify if jacobian or base value were calculated
                        active_set_lines = response_text.split('Active set vector')[1].strip().split('{')[1].split('}')[0]
                        active_set_keys = [int(k.strip()) for k in active_set_lines.split(' ') if len(k)>0]
                        
                        if active_set_keys[0] == 2:
                            calc_jacobian = True
                            calc_base_value = False
                        if active_set_keys[0] == 3:
                            calc_jacobian = True
                            calc_base_value = True
                        
                        if calc_base_value:
                            active_set_index = [ind for ind in range(len(response_lines)) if response_lines[ind].startswith('Active set vector')][0]
                            base_value_start = active_set_index+1
                            base_value_lines = response_lines[base_value_start:base_value_start+len(metric_names)]
                            
                            base_values = [float(line.split(' ')[0].strip()) for line in base_value_lines if line.split(' ')[1].strip() in metric_names]
                            
                        else:
                            base_values = None
                        
                        # get the jacobian
                        if calc_jacobian:
                            
                            jacobian_lines = response_text.split('Deriv vars vector')[-1].split('[')[1:]
                            jacobian = []
                            for j_l in jacobian_lines:
                                je = j_l.strip()
                                part1 = je.split(']')[0]
                                try:
                                    part2 = je.split(']')[1].strip().split(' ')[0] 
                                    if part2 in metric_names:
                                        temp = []
                                        for j in part1.split(' '): 
                                            if len(j) > 0:
                                                temp.append(float(j))
                                        jacobian.append(temp)
                                except IndexError:
                                    pass
                            jacobian_df = pd.DataFrame(jacobian, index=metric_names, columns=parameter_names)
                            
                            if np.any(jacobian_df.abs().sum(axis=0) == 0):
                                outs['jacobian_check'] = False
                                print(model, 'broken jacobian')
                        else:
                            jacobian = None
                        
                        calibration_itteration[i] = {'ids': evaluation_ids,
                                                     'base_values': base_values,
                                                     'jacobian': jacobian_df}
                        if gn_summary_lines:
                            if outs['best_function_evaluation'] in evaluation_ids:
                                best_jacobian = jacobian_df
            calibration_logs[model] = calibration_itteration   
        except:
            pass
        
    # if calibration finished, then  
    if gn_summary_lines:
        # use to reproduce confidence interval. 
        objective_function = np.sum(np.square(best_weighted_residuals))
        
        of_checks_out = np.allclose(outs['objective_function'], objective_function, atol=0.001, rtol=0.001)
        
        print('objective function: ', model, of_checks_out)
                
        
        parameter_estimate_df = pd.DataFrame(data = dict(best_parameters = best_parameters,
                                                         best_ego_parameters = ego_best_parameters,
                                                         parameter_variance = Vb,
                                                         parameter_standard_deviation = SEb,
                                                         lower_bound = best_parameters - (t_statistic*SEb),
                                                         upper_bound = best_parameters + (t_statistic*SEb)))
        
        output_filepath = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', 'sew', 'ego.sew.parameters.full.'+model+'.csv']
        parameter_estimate_df.to_csv(os.path.join(os.sep, *output_filepath), float_format='%.3e')
        
        # get best jacobian to calculate full variance covariance matrix and calculate the beales measure points    

        # calculate XtWX
        X = best_jacobian
        w = pd.DataFrame(np.diag(metric_weights),index=metric_weights.index, columns=metric_weights.index) 
        wX = w.dot(X)
        XTwX = X.T.dot(wX)
        
        maximum_likelihood_of = (float(len(observed_values) + len(parameter_names))*np.log(2.*np.pi) - 
                                 np.log(np.linalg.det(w)) +
                                 objective_function)
        
        # calculate s^2
        s_squared = np.sum(np.square(best_weighted_residuals))/(dof)
        
        # dakota uses unweighted residuals...
        dakota_s_squared = np.sum(np.square(best_residual_values))/(dof)
        
        # calculate ml of
        outs['maximum_likelihood_objective_function'] = maximum_likelihood_of
        
        # calculate aic 
        NP = len(parameter_names)
        NOBS = len(metric_names)
        NPR = 0
        Sprime = maximum_likelihood_of
        
        outs['AIC'] = Sprime + (2. * NP)  
        outs['AICc'] = Sprime + (2. * NP) + ((2. * NP * (NP + 1))/(NOBS + NPR - NP - 1))

        outs['BIC'] = Sprime + NP * np.log(NOBS + NPR)

        # attempt to calculate V(b)
        var_checks_out = False
        try:
            XtwX1 = np.linalg.inv(XTwX)
            COV = pd.DataFrame(s_squared * XtwX1, index=parameter_names, columns=parameter_names)
            Vb_2 = pd.Series(s_squared * XtwX1.diagonal(), index=parameter_names)
            SEb_2 = np.sqrt(Vb_2)
            
            dakota_Vb = pd.Series(dakota_s_squared * XtwX1.diagonal(), index=parameter_names)
            dakota_SEb = np.sqrt(dakota_Vb)
            
            dakota_factor = dakota_Vb/Vb
            var_checks_out = np.allclose(dakota_factor, np.ones(dakota_factor.shape, dtype=float))
            print('variance: ', model, var_checks_out)
             
            # save variance covariance matrix
            output_filepath = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', 'sew', 'ego.sew.COV.'+model+'.csv']
            COV.to_csv(os.path.join(os.sep, *output_filepath), float_format='%.4e')
        except:
            print(model, 'failed to invert')
            print(X)  
        
        
            # calculate beales points, create dakota file to run them. 
        if var_checks_out:
            beales_list = []
            factor = [-1., 1]
            level = 0.05
            for param in parameter_names:
                
                dfn = NP
                dfd = NOBS + NPR - NP
                F_statistic = stats.f.ppf(0.975, dfn, dfd, loc=0, scale=1)
                pre_factor = np.sqrt(NP * F_statistic)/dakota_SEb[param]
            
                difference = pre_factor * COV[param]
                for f in factor:
                    beales_list.append(best_parameters + difference * f)
            beales = pd.DataFrame(beales_list)
            
            new_folder_path = os.path.split(o_f)[0]
            list_of_points = ''
            for ii in range(beales.shape[0]):
                for pp in parameter_names:
                    list_of_points += str(beales.loc[ii, pp])+'\t'
                list_of_points += '\n'
            
            with open('dakota_beale_template.in', 'r') as f:
                dakota_lines = f.readlines()
            
            dakota_file = []
            for line in dakota_lines:
                line = line.replace('{model_name}', model)
                
                line = line.replace('{lowering_history}', o_f.split(os.sep)[-2].split('.')[0])
                line = line.replace('{initial_condition}', o_f.split(os.sep)[-2].split('.')[1])
                line = line.replace('{loc}', o_f.split(os.sep)[-5])
                
                line = line.replace('{num_variables}', str(len(parameter_names)))
                line = line.replace('{variable_names}', ' '.join(parameter_names))

                line = line.replace('{num_responses}', str(len(metric_names)))
                line = line.replace('{responses_names}', ' '.join(metric_names))
                line = line.replace('{responses_weights}', ' '.join([str(1) for i in range(len(metric_names))]))

                line = line.replace('{list_of_points}', list_of_points)
                dakota_file.append(line)
                
                        # Write dakota input file 
            with open(os.path.join(new_folder_path, 'dakota_beale.in'), 'w') as dakota_f:
                dakota_f.writelines(dakota_file)
            
                    # to actually submit the jobs to summit, create a unique submit script
            # for each cmd_lines chunk.
            script_contents = ['#!/bin/sh',
                               '#SBATCH --job-name beale'+model,
                               '#SBATCH --ntasks-per-node 24',
                               '#SBATCH --partition shas',
                               '#SBATCH --mem-per-cpu 4GB',
                               '#SBATCH --nodes 1',
                               '#SBATCH --time 24:00:00',
                               '#SBATCH --account ucb19_summit1',
                               '',
                               '# load environment modules',
                               'module load intel/16.0.3',
                               'module load openmpi/1.10.2',
                               'module load cmake/3.5.2',
                               '#module load perl',
                               'module load mkl',
                               'module load gsl',
                               '',
                               '# make sure environment variables are set correctly',
                               'source ~/.bash_profile',
                               '## run dakota using a restart file if it exists.',
                               'if [ -e dakota.rst ]',
                               'then',
                               'dakota -i dakota_beale.in -o dakota_beale.out --read_restart dakota_beale.rst &> dakota_beale.log',
                               'else',
                               'dakota -i dakota_beale.in -o dakota_beale.out --write_restart dakota_beale.rst &> dakota_beale.log',
                               'fi']
                
            script_path = os.path.join(new_folder_path,'start_beale.sh')
            with open(script_path, 'w') as f:
                for line in script_contents:
                    f.write(line+"\n") 
            beale_cmnd_line = 'cd ' + new_folder_path + '\nsbatch start_beale.sh'
                
         
        
        
         
        
    outputs[model] = outs
    if 'objective_function' not in outs:
        new_folder_path = os.path.split(o_f)[0]
        cmnd_line = 'cd ' + new_folder_path + '\nsbatch start_dakota.sh'
        cmnd_lines.append(cmnd_line)
        
    # copy the best image into a folder for comparisons  
    if 'best_function_evaluation' in outs:
        best_folderpath = results_dir+[model, lowering, 'run.'+str(outs['best_function_evaluation']), '*.png']
        best = glob.glob(os.path.join(os.path.abspath(os.sep), *best_folderpath))
        if len(best) == 1:
            shutil.copy(best[-1], os.path.join('best_runs', 'objective_fxn_'+str(round(objective_function))+'.'+os.path.split(best[-1])[-1]))
        
#        # get modeled topography, open the file, and create a comparison figure. 
#        best_nc_folderpath = results_dir+[model, '**', 'run.'+str(outs['best_function_evaluation']), '*.nc']
#        topo_file = glob.glob(os.path.join(os.path.abspath(os.sep), *best_nc_folderpath))[0]
#        
#        
#        mgrid = read_netcdf(topo_file)
#        mz = mgrid.at_node['topographic__elevation']
#        mgrid.set_watershed_boundary_condition_outlet_id(inputs['outlet_id'], mz, nodata_value=-9999)
#        mfa = FlowAccumulator(mgrid,
#                     flow_director='D8',
#                     depression_finder = 'DepressionFinderAndRouter')
#        
#        mfa.run_one_step()
#        mch = ChiFinder(mgrid, min_drainage_area=10*grid.dx**2)
#        mch.calculate_chi()
#        
#        # calc grouped differences:
#        gd = GroupedDifferences(topo_file, observed_topo_file_name,  outlet_id=outlet_id, category_values=cat, weight_values=cat_wt)
#        gd.calculate_metrics()
#        cat_resids = gd.metric
#        
#        cat_diff[model] = cat_resids
#        
#        # for modeled and True, 
#        # elevation, slope, erosion, shaded, chi
#        f, axarr = plt.subplots(3, 2, sharex=True, figsize=(8.5,11))
#        
#        # elevation
#        plt.sca(axarr[0,0])
#        plt.title('Observations')
#        plt.text(0.2, 0.9, 'Elevation', color='w', ha='center', va='center', transform=axarr[0,0].transAxes)
#        imshow_grid(grid, z, vmin=1230, vmax=1940, cmap='jet')#, color_for_closed='w')
#        
#        plt.sca(axarr[0,1])
#        plt.title(model)
#        imshow_grid(mgrid, mz, vmin=1230, vmax=1940, cmap='jet')#, color_for_closed='w')
#
#        # erosion
#        plt.sca(axarr[1,0])
#        plt.text(0.2, 0.9, 'Erosion Depth', color='w', ha='center', va='center', transform=axarr[1,0].transAxes)
#        imshow_grid(grid, iz-z, vmin=-120, vmax=120, cmap='RdBu_r')
#        
#        plt.sca(axarr[1,1])
#        imshow_grid(mgrid, iz-mz, vmin=-120, vmax=120, cmap='RdBu_r')
#        
#        # slope
#        plt.sca(axarr[2,0])
#        plt.text(0.2, 0.9, 'Slope', color='w', ha='center', va='center', transform=axarr[2,0].transAxes)
#        imshow_grid(grid, grid['node']['topographic__steepest_slope'], vmin=0, vmax=1.2, cmap='plasma_r')
#        
#        plt.sca(axarr[2,1])
#        imshow_grid(mgrid, mgrid['node']['topographic__steepest_slope'], vmin=0, vmax=1.2, cmap='plasma_r')
#
#        for ni in range(2):
#            for nj in range(3):
#                axarr[nj,ni].set_aspect('equal')
#                axarr[nj,ni].set_xlabel('')
#                axarr[nj,ni].set_ylabel('')
#                axarr[nj,ni].xaxis.set_major_formatter(plt.NullFormatter())
#                axarr[nj,ni].yaxis.set_major_formatter(plt.NullFormatter())
#        plt.tight_layout()
#        plt.savefig(os.path.join('ego.comparison_figures', model + '.png'), dpi=300)
        #plt.savefig(os.path.join('ego.comparison_figures', model + '.pdf')) # pdfs take about 3 minutes to render and 10 
            

df = pd.DataFrame(outputs).T

output_filepath = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', 'sew', 'ego.sew.calibration.summary.csv']
df.to_csv(os.path.join(os.sep, *output_filepath), float_format='%.3e')


#%%
#plt.figure()
#plt.scatter(df.AIC, df.BIC, c = df.AICc)
#plt.show
#
#plt.figure()
#plt.scatter(df.ego_objective_function, df.objective_function, c = df.AICc)
#plt.show
# 
#%%

# create restart file for those models that are not yet done. 
not_yet_done = df.index[df.objective_function.isnull()]
folder_paths = [os.path.split(pth)[0] for pth in output_files]    
with open('restart_calibration.sh', 'w') as f:
    for line in cmnd_lines:
        f.write(line + '\n')
#%%
#diff_df = pd.DataFrame(diff_factor).T
##diff_df = diff_df.loc[diff_df.number_of_parameters==4,:]
#slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(diff_df.objective_function), np.log(diff_df.OF_factor))
#
#print(slope, intercept)
#
#plt.figure()
#X = np.sort(np.log(diff_df.objective_function))
#plt.plot(X, intercept+slope*X, 'k-', alpha=0.5)
#plt.scatter(np.log(diff_df.objective_function), np.log(diff_df.OF_factor), c=diff_df.number_of_parameters)
#plt.show()
#
#guess_factor = np.log(diff_df.objective_function) - (32. - diff_df.number_of_parameters)*np.log(2.*np.pi) - np.log(np.linalg.det(w))
#
#plt.figure()
#R = np.log(diff_df.OF_factor) - guess_factor
#plt.scatter(np.log(diff_df.objective_function), R, c=diff_df.number_of_parameters)
#plt.show()


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
work_dir = ['work', 'WVDP_EWG_STUDY3', 'study3py','calibration', 'sew', 'GAUSSNEWTON']
results_dir = ['work', 'WVDP_EWG_STUDY3', 'results','calibration', 'sew', 'GAUSSNEWTON']

# file names
output_file = 'dakota_calibration.out'
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
# open the modern metric file to get modern observed values
modern_values_filepath_sew = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'auxillary_inputs', 'modern_metric_files', 'dem24fil_ext.metrics.txt']
with open(os.path.join(os.sep, *modern_values_filepath_sew), 'r') as f:
    modern_sew = yaml.load(f)
    modern_sew.pop('Topo file')
    # the chi_density sum of squares is not stored in this file since it is
    # a surface misfit. It has an effective modern value of 0.0
    modern_sew['chi_density_sum_squares'] = 0.0
    
observed_values = pd.Series(modern_sew)  
observed_values

#
index_order = ['chi_density_sum_squares', 'chi_gradient', 'chi_intercept', 
               'one_cell_nodes', 'two_cell_nodes', 'three_cell_nodes', 'four_cell_nodes',
               'cumarea95', 'cumarea96', 'cumarea97', 'cumarea98', 'cumarea99', 
               'elev02', 'elev08', 'elev23', 'elev30', 'elev36', 
               'elev50', 'elev75', 'elev85',
               'elev90', 'elev96', 'elev100',
               'hypsometric_integral',
               'mean_elevation', 'var_elevation',
               'mean_elevation_chi_area', 'var_elevation_chi_area',
               'mean_gradient', 'var_gradient',
               'mean_gradient_chi_area','var_gradient_chi_area']
# Use glob to find all folders with a 'params.in' file.
work_dir.extend(['model**', '**',  output_file])
output_files = glob.glob(os.path.join(os.path.abspath(os.sep), *work_dir))

if os.path.exists('best_runs') is False:
    os.mkdir('best_runs')
    
if os.path.exists('comparison_figures') is False:
    os.mkdir('comparison_figures')

conf_intervals = {}
outputs = {}
needs_restart = []
num_cores_total = 0
cmnd_lines = []

diff_factor = {}

cat_diff = {}

calibration_logs = {}

for o_f in output_files:
   # o_f ='/work/WVDP_EWG_STUDY3/study3py/calibration/sew/GAUSSNEWTON/model_600/lowering_history_0.pg24f_7etch/dakota_calibration.out'
    model = o_f.split(os.sep)[-3]
    outs = {}
    outs['jacobian_check'] = True
    with open(o_f, 'r') as f:
        lines = f.readlines()
        
    confidence_intervals = {}
    best_parameters = {}
    best_residuals = {}
    dakota_parameter_variance = {}
    dakota_parameter_std = {}
    all_lines = ''.join(lines)
    input_file_lines = all_lines.split('End DAKOTA input file')[0].split('\n')
    temp = all_lines.split('End DAKOTA input file')[1]
    remaining_lines = temp.split('<<<<< Function evaluation summary')
    working_lines = remaining_lines[0]
    if len(remaining_lines) > 1:
        # calibration is  complete:
        summary_lines = remaining_lines[1].split('\n')
    else:
        summary_lines = None
    del temp, remaining_lines
    
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
    if summary_lines:
        # from within the summary lines, extract best fit information
        for sline in summary_lines:
            
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
                
                dakota_parameter_std[parameter_name] = (confidence_interval[1]-confidence_interval[0])/(2.*t_statistic)
                
        del sline
        
        dakota_parameter_std = pd.Series(dakota_parameter_std)
        dakota_parameter_variance = dakota_parameter_std**2.
        
        # next, split the summary lines by '<<<<<' to get the best parameter 
        # values and the best residual terms. 
        
        all_summary_lines = ''.join(summary_lines) 
        grouped_summary_lines = all_summary_lines.split('<<<<<')
        for grouped_lines in grouped_summary_lines:
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
         #    ci_line, parameter_name, g_lines, all_summary_lines, grouped_summary_lines,
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
                                                    best_simulated_values = best_simulated_values,
                                                    best_residual_values = best_residual_values,
                                                    best_weighted_residuals = best_weighted_residuals,
                                                    best_weighted_residuals_squared = best_weighted_residuals**2,
                                                    proportion_of_objective_fxn = proportion_of_objective_fxn))
        best_evaluation_df = best_evaluation_df.reindex(index=index_order, 
                                                        columns=['observed_values',
                                                                 'metric_weights',
                                                                 'standard_deviation',
                                                                 'coeff_of_variation',
                                                                 'best_simulated_values',
                                                                 'best_residual_values',
                                                                 'best_weighted_residuals',
                                                                 'best_weighted_residuals_squared',
                                                                 'proportion_of_objective_fxn'])
            
        output_filepath = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', 'sew', 'sew.residuals.full.'+model+'.csv']
        best_evaluation_df.to_csv(os.path.join(os.sep, *output_filepath), float_format='%.3e')
        
        best_evaluation_df_short = best_evaluation_df[['best_simulated_values',
                                                      'best_residual_values',
                                                      'best_weighted_residuals']]
        output_filepath_short = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', 'sew', 'sew.residuals.short.'+model+'.csv']
        best_evaluation_df_short.to_csv(os.path.join(os.sep, *output_filepath_short), float_format='%.3e')
        
        f, axarr = plt.subplots(1,2)
        axarr[0].hist(best_evaluation_df.best_weighted_residuals)
        axarr[0].set_title('Shapiro-Wilk p = ' + str(round(stats.shapiro(best_evaluation_df.best_weighted_residuals)[1], 3)))
        stats.probplot(best_evaluation_df.best_weighted_residuals, dist="norm", plot=axarr[1])        
        output_filepath = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', 'sew', 'sew.residuals.hist.'+model+'.pdf']
        plt.savefig(os.path.join(os.sep, *output_filepath))
        
     
    # for all models, 
    
    # Parse the working lines to determine the parameter values used in each
    # evaluation. 
    split_working = working_lines.split('Parameters for evaluation')[1:] # the second of these will not have useful text
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
    split_working = working_lines.split('Active response data')[1:] # the second of these will not have useful text
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
    split_working = working_lines.split('Begin Dakota derivative estimation routine')[1:] # the first of these will not have useful text
    calibration_itteration = {}
    for i in range(len(split_working)):
        evaluation_text = split_working[i]
        evaluation_ids = []
        for line in evaluation_text.split('>>>>> Total response returned to iterator:')[0].splitlines():
            if line[:20] == 'Active response data':
                evaluation_ids.append(int(line.strip().strip(':').split(' ')[-1]))
                
        response_lines = evaluation_text.split('>>>>> Total response returned to iterator:')[-1].splitlines()
        response_lines = [r_l.strip() for r_l in response_lines]
        
        if response_lines[-1] != 'Waiting on completed jobs':
        
            response_text = ' '.join(response_lines)
            
            jacobian_lines = response_text.split('[')[1:]
            jacobian = []
            for j_l in jacobian_lines:
                jacobian.append([float(je) for je in j_l.strip().split(']')[0].split(' ') if len(je)>0])
    
            jacobian_df = pd.DataFrame(jacobian, index=metric_names, columns=parameter_names)
            
            if np.any(jacobian_df.abs().sum(axis=0) == 0):
                outs['jacobian_check'] = False
                print(model, 'broken jacobian')
                
            pre_jacobian_lines = evaluation_text.split('>>>>> Total response returned to iterator:')[0].splitlines()
            
            base_values = response_text.split('}')
            base_values = None
            calibration_itteration[i] = {'ids': evaluation_ids,
                                         'base_values': base_values,
                                         'jacobian': jacobian_df}
            if summary_lines:
                if outs['best_function_evaluation'] in evaluation_ids:
                    best_jacobian = jacobian_df
    calibration_logs[model] = calibration_itteration   
    # if calibration finished, then  
    if summary_lines:
        # use to reproduce confidence interval. 
        objective_function = np.sum(np.square(best_weighted_residuals))
        
        of_checks_out = np.allclose(outs['objective_function'], objective_function, atol=0.001, rtol=0.001)
        
        print('objective function: ', model, of_checks_out)
        
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
        
        # calculate V(b)
        try:
            XtwX1 = np.linalg.inv(XTwX)
            Vb = pd.Series(s_squared * XtwX1.diagonal(), index=parameter_names)
            SEb = np.sqrt(Vb)
            
            dakota_Vb = pd.Series(dakota_s_squared * XtwX1.diagonal(), index=parameter_names)
            dakota_SEb = np.sqrt(dakota_Vb)
            
            dakota_factor = dakota_Vb/dakota_parameter_variance
            var_checks_out = np.allclose(dakota_factor, np.ones(dakota_factor.shape, dtype=float))
            print('variance: ', model, var_checks_out)
            
            parameter_estimate_df = pd.DataFrame(data = dict(best_parameters = best_parameters,
                                                             parameter_variance = Vb,
                                                             parameter_standard_deviation = SEb,
                                                             lower_bound = best_parameters - (t_statistic*SEb),
                                                             upper_bound = best_parameters + (t_statistic*SEb)))
            
            output_filepath = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', 'sew', 'sew.parameters.full.'+model+'.csv']
            parameter_estimate_df.to_csv(os.path.join(os.sep, *output_filepath), float_format='%.3e')
            
#            diff_factor[model] = {'number_of_parameters':len(parameter_names),
#                                  'objective_function': objective_function,
#                                  'OF_factor': parameter_estimate_df.diff_factor.values[0],
#                                  'maximum_likelihood_objective_function' : maximum_likelihood_of,
#                                  's_squared': s_squared}
                        
        except:
            print(model, 'failed to invert')
            print(X)
        
        
    if outs == {} or model in ['model_300', 'model_400', 'model_410', 'model_440', 'model_600', 'model_C00',] :
        needs_restart.append({model:num_cores})
        num_cores_total += num_cores
        cmnd_line = 'cd ' + os.path.split(o_f)[0] + '; sh start_dakota.sh'
        cmnd_lines.append(cmnd_line)
        
    outs['num_params'] = num_cores - 1
    outputs[model] = outs
    
    # copy the best image into a folder for comparisons  
    if 'best_function_evaluation' in outs:
        
        best_folderpath = results_dir+[model, '**', 'run.'+str(outs['best_function_evaluation']), '*.png']
        best = glob.glob(os.path.join(os.path.abspath(os.sep), *best_folderpath))
        if len(best) == 1:
            shutil.copy(best[-1], os.path.join('best_runs', 'objective_fxn_'+str(round(objective_function))+'.'+os.path.split(best[-1])[-1]))
            
        # get modeled topography, open the file, and create a comparison figure. 
        best_nc_folderpath = results_dir+[model, '**', 'run.'+str(outs['best_function_evaluation']), '*.nc']
        topo_file = glob.glob(os.path.join(os.path.abspath(os.sep), *best_nc_folderpath))[0]
        
        
        mgrid = read_netcdf(topo_file)
        mz = mgrid.at_node['topographic__elevation']
        mgrid.set_watershed_boundary_condition_outlet_id(inputs['outlet_id'], mz, nodata_value=-9999)
        mfa = FlowAccumulator(mgrid,
                     flow_director='D8',
                     depression_finder = 'DepressionFinderAndRouter')
        
        mfa.run_one_step()
        mch = ChiFinder(mgrid, min_drainage_area=10*grid.dx**2)
        mch.calculate_chi()
        
        # calc grouped differences:
        gd = GroupedDifferences(topo_file, observed_topo_file_name,  outlet_id=outlet_id, category_values=cat, weight_values=cat_wt)
        gd.calculate_metrics()
        cat_resids = gd.metric
        
        cat_diff[model] = cat_resids
        
        # for modeled and True, 
        # elevation, slope, erosion, shaded, chi
        f, axarr = plt.subplots(3, 2, sharex=True, figsize=(8.5,11))
        
        # elevation
        plt.sca(axarr[0,0])
        plt.title('Observations')
        plt.text(0.2, 0.9, 'Elevation', color='w', ha='center', va='center', transform=axarr[0,0].transAxes)
        imshow_grid(grid, z, vmin=1230, vmax=1940, cmap='jet')#, color_for_closed='w')
        
        plt.sca(axarr[0,1])
        plt.title(model)
        imshow_grid(mgrid, mz, vmin=1230, vmax=1940, cmap='jet')#, color_for_closed='w')

        # erosion
        plt.sca(axarr[1,0])
        plt.text(0.2, 0.9, 'Erosion Depth', color='w', ha='center', va='center', transform=axarr[1,0].transAxes)
        imshow_grid(grid, iz-z, vmin=-120, vmax=120, cmap='RdBu_r')
        
        plt.sca(axarr[1,1])
        imshow_grid(mgrid, iz-mz, vmin=-120, vmax=120, cmap='RdBu_r')
        
        # slope
        plt.sca(axarr[2,0])
        plt.text(0.2, 0.9, 'Slope', color='w', ha='center', va='center', transform=axarr[2,0].transAxes)
        imshow_grid(grid, grid['node']['topographic__steepest_slope'], vmin=0, vmax=1.2, cmap='plasma_r')
        
        plt.sca(axarr[2,1])
        imshow_grid(mgrid, mgrid['node']['topographic__steepest_slope'], vmin=0, vmax=1.2, cmap='plasma_r')

        for ni in range(2):
            for nj in range(3):
                axarr[nj,ni].set_aspect('equal')
                axarr[nj,ni].set_xlabel('')
                axarr[nj,ni].set_ylabel('')
                axarr[nj,ni].xaxis.set_major_formatter(plt.NullFormatter())
                axarr[nj,ni].yaxis.set_major_formatter(plt.NullFormatter())
        plt.tight_layout()
        plt.savefig(os.path.join('comparison_figures', model + '.png'), dpi=300)
        #plt.savefig(os.path.join('comparison_figures', model + '.pdf')) # pdfs take about 3 minutes to render and 10 
            

df = pd.DataFrame(outputs).T

output_filepath = ['work', 'WVDP_EWG_STUDY3', 'study3py', 'result_tables', 'calibration', 'sew', 'sew.calibration.summary.csv']
df.to_csv(os.path.join(os.sep, *output_filepath), float_format='%.3e')

#%%
cat_df = pd.DataFrame(cat_diff).T
cat_of = np.sum(np.square(cat_df), axis=1)
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


#%%
num_nodes = int(np.ceil(num_cores_total/24))
# write out command lines
with open('cmd_lines_restart', 'w') as f:
    for line in cmnd_lines:
        f.write(line + '\n')

# create launch script
script_contents = ['#!/bin/sh',
                   '#SBATCH --job-name rst_sew_calib',
                   '#SBATCH --ntasks-per-node 24',
                   '#SBATCH --partition shas',
                   '#SBATCH --mem-per-cpu 4GB',
                   '#SBATCH --nodes ' + str(num_nodes),
                   '#SBATCH --time 24:00:00',
                   '#SBATCH --account ucb19_summit1',
                   '',
                   'module purge',
                   'module load intel',
                   'module load impi',
                   'module load loadbalance',
                   'mpirun lb cmd_lines_restart']

with open('launch_dakota_calibration_restart.sh', 'w') as f:
    for line in script_contents:
        f.write(line + '\n')
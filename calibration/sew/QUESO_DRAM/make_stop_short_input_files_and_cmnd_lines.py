#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:07:38 2017

@author: barnhark
"""
import os
import shutil
import glob

restart_files = glob.glob(os.path.join(*['model_*', '*', 'posterior.dat']))

#%%
all_source_lines = []
for rf in restart_files:
    path = os.path.split(rf)[0]
    cmnd_lines = []
    start_all_individually = []
    
    
    dakota_file = path + os.sep + 'dakota_queso_dram.in'

    with open(dakota_file, 'r') as f:
        dakota_lines = f.readlines()
    
    for value in range(1,11):
        key = '{0:02d}'.format(value)
        
        work_dir = os.path.join(path, 'short_stop', key)
        in_file_name = 'dakota_queso_dram_short_'+key+'.in'
        out_file_name = 'dakota_queso_dram_short_'+key+'.out'
        log_file_name = 'dakota'+key+'.log'
        
        start_dakota_file = os.path.join(work_dir, 'start_posterior_stop_'+key+'.sh')
        
        if os.path.exists(work_dir):
            pass
        else:
            os.makedirs(work_dir)
        files_to_copy = ['dakota_mcmc.rst','data.dat', 'driver.py', 'inputs_template.txt']
        for ftc in files_to_copy:
            shutil.copy(os.path.join(path, ftc), os.path.join(work_dir, ftc))
            
        write_out_file = work_dir + os.sep + in_file_name
        
        new_lines  = []
        for line in dakota_lines:
            if line.strip().startswith('tabular_data_file'):
                line = line.replace('mcmc.dat', 'mcmc_'+key+'.dat')
            if line.strip().startswith('max_iterations'):
                line = line.replace('10', str(int(key)))
            if line.strip().startswith('export_chain_points_file'):
                line = line.replace('posterior.dat', 'posterior_'+ key+ '.dat')
            new_lines.append(line)
            
            
            
            
            
        with open(write_out_file, 'w') as f:
            f.writelines(new_lines)
        
        start_dakota_lines = ['#!/bin/sh',
                              '#SBATCH --job-name '+key+'_queso',
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
                              '',
                              '## run dakota.',
                              'dakota -i ' + in_file_name + ' -o ' + out_file_name + ' --read_restart dakota_mcmc.rst &> ' + log_file_name]
        start_dakota_lines = [line + '\n' for line in start_dakota_lines]
        with open(start_dakota_file, 'w') as f:
            f.writelines(start_dakota_lines)    
        cmnd_lines.append('cd ' + os.path.abspath(work_dir) +'; sh ' + 'start_posterior_stop_'+key+'.sh' + '\n')
        start_all_individually.append('cd ' + os.path.abspath(work_dir) + '; sbatch ' + 'start_posterior_stop_'+key+'.sh' + '\n')
    
    with open(os.path.join(path, 'cmnd_lines'), 'w') as f:
        f.writelines(cmnd_lines) 
    
    with open(os.path.join(path, 'start_all_individually.sh'), 'w') as f:
        f.writelines(start_all_individually)    
     
    all_source_lines.append('source ' + os.path.abspath(os.path.join(path, 'start_all_individually.sh\n')))

with open('start_all_short_stops.sh', 'w') as f:
    f.writelines(all_source_lines) 
     
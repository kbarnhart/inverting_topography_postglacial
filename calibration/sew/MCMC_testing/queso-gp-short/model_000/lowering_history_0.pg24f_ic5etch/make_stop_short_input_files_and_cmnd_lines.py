#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:07:38 2017

@author: barnhark
"""
import os
import shutil

dakota_file = '/work/WVDP_EWG_STUDY3/study3py/calibration/sew/MCMC_testing/queso-gp-short/model_000/lowering_history_0.pg24f_ic5etch/dakota_queso_dram_short_XX.in'

cmnd_lines = []
start_all_individually = []
with open(dakota_file, 'r') as f:
    dakota_lines = f.readlines()

for value in range(1,19):
    key = '{0:02d}'.format(value)
    
    in_file_name = 'dakota_queso_dram_short_'+key+'.in'
    out_file_name = 'dakota_queso_dram_short_'+key+'.out'
    log_file_name = 'dakota'+key+'.log'
    start_dakota_file = os.path.join(key, 'start_posterior_stop_'+key+'.sh')
    
    if os.path.exists(key):
        pass
    else:
        os.mkdir(key)
    files_to_copy = ['dakota.rst','data.dat', 'driver.py', 'inputs_template.txt']
    for ftc in files_to_copy:
        shutil.copy(ftc, os.path.join(key, ftc))
    write_out_file = '/work/WVDP_EWG_STUDY3/study3py/calibration/sew/MCMC_testing/queso-gp-short/model_000/lowering_history_0.pg24f_ic5etch/' + key + '/' + in_file_name
    
    new_lines  = []
    for line in dakota_lines:
        if line.strip().startswith('max_iterations'):
            line = line.replace('{XX}', str(int(key)))
        else:
            line = line.replace('{XX}', key)
        new_lines.append(line)
    with open(write_out_file, 'w') as f:
        f.writelines(new_lines)
    
    start_dakota_lines = ['#!/bin/sh',
                          '#SBATCH --job-name '+key+'_queso',
                          '#SBATCH --ntasks-per-node 24',
                          '#SBATCH --partition shas',
                          '#SBATCH --mem-per-cpu 4GB',
                          '#SBATCH --nodes 1',
                          '#SBATCH --time 15:00:00',
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
                          'dakota -i ' + in_file_name + ' -o ' + out_file_name + ' --read_restart dakota.rst &>' + log_file_name]
    start_dakota_lines = [line + '\n' for line in start_dakota_lines]
    with open(start_dakota_file, 'w') as f:
        f.writelines(start_dakota_lines)    
    cmnd_lines.append('cd /work/WVDP_EWG_STUDY3/study3py/calibration/sew/MCMC_testing/queso-gp-short/model_000/lowering_history_0.pg24f_ic5etch/' + key +' ; source ' + 'start_posterior_stop_'+key+'.sh' + '\n')
    start_all_individually.append('cd /work/WVDP_EWG_STUDY3/study3py/calibration/sew/MCMC_testing/queso-gp-short/model_000/lowering_history_0.pg24f_ic5etch/' + key +' ; sbatch ' + 'start_posterior_stop_'+key+'.sh' + '\n')
    
    
with open('cmnd_lines', 'w') as f:
    f.writelines(cmnd_lines) 

    
with open('start_all_individually.sh', 'w') as f:
    f.writelines(start_all_individually)    
                      